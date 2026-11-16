"""Stage 4: cleaned blocks -> one record per article, with its place in the code.

Input : data/interim/clean/blocks.jsonl
Output: data/processed/articles.jsonl      one article per line (672), the dataset for IR / RAG
        data/processed/structure.json      the table of contents: books > titles > chapters > ...

The Criminal Code is organised in 5 levels above the articles:
    គន្ថី (book) > មាតិកា (title) > ជំពូក (chapter) > ផ្នែក (section) > កថាភាគ (sub-section)
Each level starts with a centered marker line ("ជំពូកទី ២") followed by its name.
Numbering restarts inside each parent (every book has its own មាតិកាទី ១, ...).

Layout quirks handled here (they come from clean.py's position-based block types):
    - a long hierarchy name does not start centered, so it arrives as a "paragraph"
      -> the first block after a marker is always its name
    - a wrapped article title puts its 2nd line indented, so it arrives as a "heading"
      -> a non-marker heading right after an article heading continues that title
    - other non-marker "headings" are short, right-shifted text lines -> part of the paragraph

Usage (from backend/, venv active):
    python -m pipeline.segment
"""

import json
import re
from collections import Counter

from pipeline.clean import KHMER_DIGITS, from_khmer_number
from pipeline.config import ARTICLES_FILE, CLEAN_DIR, LAST_ARTICLE

LEVELS = ["book", "title", "chapter", "section", "subsection"]
LEVEL_NAMES_KM = {"book": "គន្ថី", "title": "មាតិកា", "chapter": "ជំពូក",
                  "section": "ផ្នែក", "subsection": "កថាភាគ"}

# Marker keywords, tolerant to OCR damage: គន្ថី comes out as គន្ថីនី, គន្ីិីន, គន្លឹន, ...
MARKER = re.compile(r"^(?:(?P<book>គន\S*)|(?P<title>មាតិកា\S*)|(?P<chapter>ជំពូក\S*)"
                    r"|(?P<section>ផ្នែក\S*)|(?P<subsection>កថាភាគ\S*))\s*(?P<num>[០-៩]*)\s*$")
SOLE = "ទោល"   # "ជំពូកទោល" = the only chapter -> number 1


def parse_marker(text: str) -> tuple[str, int | None] | None:
    """'ជំពូកទី ២' -> ('chapter', 2); 'ជំពូកទោល' -> ('chapter', 1); not a marker -> None."""
    m = MARKER.match(text)
    if not m or len(text) > 25:
        return None
    level = next(lvl for lvl in LEVELS if m.group(lvl))
    if m.group("num"):
        return level, from_khmer_number(m.group("num"))
    if SOLE in m.group(level):
        return level, 1
    # No number and not "ទោល": ordinary text that happens to start with a keyword,
    # e.g. "ផ្នែកទៀត" (= "another part"), the 2nd line of the title of article 465
    return None


def to_khmer(n: int) -> str:
    return "".join(KHMER_DIGITS[int(c)] for c in str(n))


def join_lines(a: str, b: str) -> str:
    return (a + b) if a else b    # Khmer: no space between wrapped lines


def segment(blocks: list[dict]) -> tuple[list[dict], dict, list[str]]:
    articles: list[dict] = []
    path: dict[str, dict | None] = {lvl: None for lvl in LEVELS}   # current position in the code
    last_no: dict[str, int] = {lvl: 0 for lvl in LEVELS}           # last number seen per level
    structure: list[dict] = []   # every hierarchy node, in order
    warnings: list[str] = []

    naming: dict | None = None   # hierarchy node whose name is still being read
    article: dict | None = None
    prev_type = None

    for b in blocks:
        text, page = b["text"], b["pages"][0]
        marker = parse_marker(text) if b["type"] == "heading" else None

        if marker:
            level, number = marker
            depth = LEVELS.index(level)
            expected = last_no[level] + 1
            if number is None or not 1 <= number <= expected + 1:
                warnings.append(f"p{page}: marker {text!r} read as {number}, using {expected}")
                number = expected
            node = {"level": level, "no": number, "name": "", "page": page}
            path[level] = node
            last_no[level] = number
            for lower in LEVELS[depth + 1:]:          # a new chapter resets section, sub-section
                path[lower] = None
                last_no[lower] = 0
            structure.append(node)
            naming, article = node, None
            prev_type = "marker"
            continue

        if naming is not None and b["type"] != "article":
            # The name of the hierarchy node: the first block after the marker, plus any
            # centered continuation lines directly after it
            if not naming["name"] or b["type"] == "heading":
                naming["name"] = join_lines(naming["name"], text)
                prev_type = "name"
                continue
            naming = None

        if b["type"] == "article":
            naming = None
            article = {
                "article_no": b["article_no"],
                "title": re.sub(r"^មាត្រា\s*[០-៩]+\s*", "", text),
                "paragraphs": [],
                "pages": list(b["pages"]),
                "path": {lvl: (dict(no=n["no"], name=n["name"]) if n else None)
                         for lvl, n in path.items()},
                "_conf": [b["conf"]],
                "_notes": Counter(b["notes"]),
                "_raw_heading": b.get("raw_text"),
            }
            articles.append(article)
            prev_type = "article"
            continue

        if article is None:
            warnings.append(f"p{page}: text outside any article: {text[:50]!r}")
            continue

        if b["type"] == "heading" and prev_type in ("article", "title_cont"):
            article["title"] = join_lines(article["title"], text)      # wrapped article title
            prev_type = "title_cont"
        elif b["type"] == "heading" and article["paragraphs"]:
            article["paragraphs"][-1] = join_lines(article["paragraphs"][-1], text)
            prev_type = "paragraph"
        else:
            article["paragraphs"].append(text)
            prev_type = "paragraph"
        for p in b["pages"]:
            if p not in article["pages"]:
                article["pages"].append(p)
        article["_conf"].append(b["conf"])
        article["_notes"].update(b["notes"])

    return [finalize(a) for a in articles], build_structure(structure), warnings


def finalize(a: dict) -> dict:
    """Turn the working dict into the published record."""
    no = a["article_no"]
    notes = a["_notes"]
    path = a["path"]
    breadcrumb = " › ".join(
        f"{LEVEL_NAMES_KM[lvl]}ទី {to_khmer(n['no'])} {n['name']}".strip()
        for lvl, n in path.items() if n
    )
    return {
        "id": f"art_{no:03d}",
        "article_no": no,
        "title": a["title"].strip(),
        "text": "\n".join(a["paragraphs"]),
        "paragraphs": a["paragraphs"],
        "path": path,
        "breadcrumb": breadcrumb,                 # ready to show in the UI or prepend for RAG
        "pages": sorted(a["pages"]),
        "ocr": {
            "min_conf": round(min(a["_conf"]), 4),
            "heading_fixed": "article_no_fixed" in notes,
            "raw_heading": a["_raw_heading"],
            "list_markers_fixed": notes.get("list_marker_fixed", 0),
            "low_conf_blocks": notes.get("low_conf", 0),
        },
    }


def build_structure(nodes: list[dict]) -> list[dict]:
    """Nest the flat list of hierarchy nodes into a tree (the table of contents)."""
    root: list[dict] = []
    stack: list[tuple[int, dict]] = []
    for n in nodes:
        depth = LEVELS.index(n["level"])
        item = {**n, "children": []}
        while stack and stack[-1][0] >= depth:
            stack.pop()
        (stack[-1][1]["children"] if stack else root).append(item)
        stack.append((depth, item))
    return root


def validate(articles: list[dict]) -> list[str]:
    problems = []
    numbers = [a["article_no"] for a in articles]
    if numbers != list(range(1, LAST_ARTICLE + 1)):
        missing = sorted(set(range(1, LAST_ARTICLE + 1)) - set(numbers))
        problems.append(f"article numbers are not 1..{LAST_ARTICLE}; missing {missing}")
    for a in articles:
        if not a["text"]:
            problems.append(f"article {a['article_no']} (p{a['pages'][0]}) has no text")
        if not a["title"]:
            problems.append(f"article {a['article_no']} (p{a['pages'][0]}) has no title")
        if not a["path"]["book"]:
            problems.append(f"article {a['article_no']} has no book")
    return problems


def run():
    blocks = [json.loads(line) for line in
              (CLEAN_DIR / "blocks.jsonl").read_text(encoding="utf-8").splitlines()]
    articles, structure, warnings = segment(blocks)
    problems = validate(articles)

    ARTICLES_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(ARTICLES_FILE, "w", encoding="utf-8") as f:
        for a in articles:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")
    structure_file = ARTICLES_FILE.parent / "structure.json"
    structure_file.write_text(json.dumps(structure, ensure_ascii=False, indent=1), encoding="utf-8")

    per_book = Counter(a["path"]["book"]["no"] for a in articles if a["path"]["book"])
    print(f"{len(articles)} articles -> {ARTICLES_FILE}")
    print("  per book: " + ", ".join(f"book {k}: {v}" for k, v in sorted(per_book.items())))
    print(f"  table of contents -> {structure_file}")
    for w in warnings:
        print("  warning:", w)
    for p in problems:
        print("  PROBLEM:", p)
    if not problems:
        print("  validation passed: articles 1-672 in order, each with title, text and book")


if __name__ == "__main__":
    run()
