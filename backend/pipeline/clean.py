"""Stage 3: raw OCR lines -> cleaned, typed text blocks.

Input : data/interim/ocr/page_XXX.json
Output: data/interim/clean/blocks.jsonl   one block per line, read by segment.py
        data/interim/clean/body.txt       the same text, easy to read and check by eye

Block types:
    heading    centered line: គន្ថី / មាតិកា / ជំពូក / ផ្នែក / កថាភាគ or their titles
    article    an article heading line: "មាត្រា ១ ដែនអនុវត្តនៃច្បាប់ព្រហ្មទណ្ឌ"
    paragraph  body text, with wrapped lines (also across pages) joined together

What gets cleaned:
    - only the law body (pages 3-247); cover, royal decree, signatures and table of contents are dropped
    - page-number footers removed
    - Unicode NFC, zero-width characters removed, whitespace collapsed
    - list markers repaired: OCR reads "៤- ការ..." as "៤១ការ...", "១- ..." as "ខប..."

Usage (from backend/, venv active):
    python -m pipeline.clean
"""

import json
import re
import unicodedata
from dataclasses import asdict, dataclass, field

from pipeline.config import (
    BODY_END_MARKER, BODY_FIRST_PAGE, BODY_LAST_PAGE, CENTER_X, CLEAN_DIR,
    FOOTER_Y, FULL_WIDTH_X2, INDENT_X, LAST_ARTICLE, LOW_CONF_THRESHOLD, OCR_DIR,
)
from pipeline.ocr import group_into_lines

KHMER_DIGITS = "០១២៣៤៥៦៧៨៩"
ARTICLE_WORD = "មាត្រា"
SENTENCE_END = ("។", "៕", "៖")

ZERO_WIDTH = re.compile(r"[​‌‍⁠﻿]")
SPACES = re.compile(r"\s+")

# Characters the OCR produces in place of the "-" after a list number
DASH_MARKS = "-–—.ៈះ៖។ៗ"
DASH_LETTERS = "ងពបខ"   # only a dash when NOT followed by a vowel sign / coeng (else it starts a word)
KHMER_COMBINING = re.compile(r"[ា-៓៝]")
# Letters the OCR produces in place of a one-digit list number ("ខប..." is really "១- ...")
DIGIT_LOOKALIKES = {"ខ": 1, "ឧ": 1, "ព": 3, "ជ": 8}
LIST_MARKER = re.compile(r"^([០-៩]{1,3})([^\s០-៩])")
MAX_LIST_ITEM = 30

# Article headings are printed "មាត្រា ១.-". KhmerOCR has no "." or "-" in its alphabet,
# so ".-" often comes out as an extra digit: "មាត្រា ១៧", "មាត្រា ៣៣០".
ARTICLE_HEAD = re.compile(r"^មាត្រា\s*([០-៩]+)\s*(.*)$")
# Digit pairs this OCR model confuses in the bold heading font (found in article_report.json)
CONFUSED_DIGITS = {"៤": "៨", "៨": "៤", "៣": "៧", "៧": "៣"}


def to_khmer_number(n: int) -> str:
    return "".join(KHMER_DIGITS[int(c)] for c in str(n))


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = ZERO_WIDTH.sub("", text)
    text = SPACES.sub(" ", text).strip()
    text = re.sub(r"(?:\s*។){2,}", " ។", text)  # "។ ។" -> "។"
    return text


@dataclass
class Block:
    type: str
    text: str
    pages: list[int]
    conf: float
    lines: int = 1
    notes: list[str] = field(default_factory=list)
    article_no: int | None = None   # only for type == "article"
    raw_text: str | None = None     # original OCR text, kept when the text was corrected

    def add_line(self, text: str, page: int, conf: float):
        # Khmer is written without spaces between words, so wrapped lines join directly
        self.text += text
        if page not in self.pages:
            self.pages.append(page)
        self.conf = min(self.conf, conf)
        self.lines += 1


def load_body_lines() -> list[dict]:
    """Every OCR line of the law body, in reading order, footers removed."""
    out = []
    for page in range(BODY_FIRST_PAGE, BODY_LAST_PAGE + 1):
        path = OCR_DIR / f"page_{page:03d}.json"
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing; run: python -m pipeline.ocr {page}")
        page_data = json.loads(path.read_text(encoding="utf-8"))
        # Re-group the saved segments with our own line grouping, which fixes rows that
        # KhmerOCR merged (e.g. "មាត្រា ច្បាប់បញ្ញត្តថា ២ ..."), without re-running OCR
        segments = [s for line in page_data["lines"] for s in line["segments"]]
        for line in group_into_lines(segments):
            if line["bbox"][1] >= FOOTER_Y:
                continue
            text = normalize(line["text"])
            if page == BODY_LAST_PAGE and text.startswith(BODY_END_MARKER):
                return out  # signature block reached, end of the law
            if text:
                out.append({"page": page, "text": text, "conf": line["conf"],
                            "x1": line["bbox"][0], "x2": line["bbox"][2]})
    return out


def classify(line: dict, is_article: bool) -> str:
    if line["x1"] >= CENTER_X:
        return "heading"
    if line["x1"] >= INDENT_X:
        return "paragraph_start"
    if is_article:
        return "article"
    return "continuation"


def from_khmer_number(digits: str) -> int:
    return int("".join(str(KHMER_DIGITS.index(c)) for c in digits))


def repair_list_marker(text: str, expected: int) -> tuple[str, int | None]:
    """If a paragraph starts with a (misread) list number, rewrite it as "N- ".

    The OCR often reads the "-" after the number as another digit or a letter:
        "៤១ការ..."  -> "៤- ការ..."      "១២១ការ..." -> "១២- ការ..."
        "១០ៈការ..." -> "១០- ការ..."     "ខបការ..."  -> "១- ការ..."
    `expected` (the next number in the current list) decides ambiguous cases.
    Returns (text, list number or None).
    """
    def is_dash(i: int, letters: bool = True) -> bool:
        if i >= len(text):
            return False
        if text[i] in DASH_MARKS:
            return True
        return (letters and text[i] in DASH_LETTERS and i + 1 < len(text)
                and not KHMER_COMBINING.match(text[i + 1])
                and text[i:i + 2] != "បទ")          # "បទល្មើស", "បទឧក្រិដ្ឋ" are words

    def result(n: int, cut: int, dash_letters: bool = True):
        if is_dash(cut, dash_letters):
            cut += 1
        return f"{to_khmer_number(n)}- {text[cut:].lstrip(' -')}", n

    # 1) A letter misread for a one-digit number, followed by a dash: "ខបការ" -> "១- ការ"
    n = DIGIT_LOOKALIKES.get(text[:1])
    if n in (expected, 1) and is_dash(1):
        return result(n, 1)

    # 2) Real digits with no space after them
    m = LIST_MARKER.match(text)
    if not m:
        return text, None
    digits = m.group(1)
    whole = from_khmer_number(digits)
    head = from_khmer_number(digits[:-1]) if len(digits) > 1 else None
    valid = lambda x: x is not None and 1 <= x <= MAX_LIST_ITEM

    if text[len(digits)] in DASH_MARKS and valid(whole):  # "១០ៈការ": a clear dash mark
        return result(whole, len(digits))
    if head == expected:                       # "៤១ការ" while expecting 4: last digit was the dash
        return result(head, len(digits), dash_letters=False)
    if whole == expected:                      # "១១ការ" while expecting 11
        return result(whole, len(digits))
    if is_dash(len(digits)) and valid(whole):  # "១០បការ": the dash was read as a letter
        return result(whole, len(digits))
    if valid(head):                            # otherwise the last digit is the misread dash
        return result(head, len(digits), dash_letters=False)
    if valid(whole):                           # "២កាល": one digit, the dash was lost
        return result(whole, len(digits))
    return text, None


def build_blocks(lines: list[dict], articles: dict[int, int]) -> list[Block]:
    """Group lines into blocks. `articles` maps line index -> article number."""
    blocks: list[Block] = []
    current: Block | None = None   # the paragraph that continuation lines are added to
    list_counter = 0

    for idx, line in enumerate(lines):
        kind = classify(line, idx in articles)
        text, page, conf = line["text"], line["page"], line["conf"]

        if kind == "heading":
            blocks.append(Block("heading", text, [page], conf))
            current = None
        elif kind == "article":
            number = articles[idx]
            readings, digits, title = article_readings(text)
            block = Block("article", text, [page], conf, article_no=number)
            if digits is None or from_khmer_number(digits) != number:
                block.raw_text = text
                block.text = f"{ARTICLE_WORD} {to_khmer_number(number)} {title}".strip()
                block.notes.append("article_no_fixed")
                if number not in readings:
                    block.notes.append("number_from_sequence")
            blocks.append(block)
            current, list_counter = None, 0
        elif kind == "paragraph_start" or current is None:
            in_list = list_counter > 0 or (current is not None and current.text.endswith("៖"))
            if text.startswith("៖") and in_list:
                # The whole number was lost and only the dash survived as "៖":
                # "...ដូចខាងក្រោមនេះ ៖" then "៖បទល្មើស..." is item 1 of a list
                text = f"{to_khmer_number(list_counter + 1)}- {text[1:].lstrip()}"
            fixed, number = repair_list_marker(text, list_counter + 1)
            current = Block("paragraph", fixed, [page], conf)
            if number is not None:
                list_counter = number
                if fixed != text:
                    current.notes.append("list_marker_fixed")
            if kind == "continuation":
                current.notes.append("starts_unindented")
            blocks.append(current)
        else:
            current.add_line(text, page, conf)

    for b in blocks:
        b.conf = round(b.conf, 4)
        if b.conf < LOW_CONF_THRESHOLD:
            b.notes.append("low_conf")
    return blocks


def article_readings(text: str) -> tuple[list[int], str | None, str]:
    """Possible article numbers for a heading line: (readings, OCR digits, title).

    "មាត្រា ៣៣០ ..." -> [330, 33, ...]: the number as read, with the misread ".-"
    (one or two trailing characters) dropped, and with one commonly confused digit swapped.
    """
    m = ARTICLE_HEAD.match(text)
    if not m:
        return [], None, text[len(ARTICLE_WORD):].strip()
    digits, title = m.groups()
    title = re.sub(r"^[ខបងព៖ៈ។.\-][០-៩]?\s+", "", title)   # leftover of a misread ".-"
    readings = set()
    for k in range(max(1, len(digits) - 2), len(digits) + 1):
        variant = digits[:k]
        readings.add(from_khmer_number(variant))
        for pos, ch in enumerate(variant):
            if ch in CONFUSED_DIGITS:
                swapped = variant[:pos] + CONFUSED_DIGITS[ch] + variant[pos + 1:]
                readings.add(from_khmer_number(swapped))
    return sorted(r for r in readings if 1 <= r <= LAST_ARTICLE), digits, title


def best_increasing_chain(candidates: list[list[int]]) -> dict[int, int]:
    """Pick at most one reading per heading so the chosen numbers strictly increase.

    Dynamic programming (a weighted longest increasing subsequence): every heading in
    the chain scores 1, plus 1 more when it is exactly the previous number + 1.
    Returns {heading index: number}.
    """
    nodes = [(i, v) for i, vs in enumerate(candidates) for v in vs]
    score, back = [], []
    for a, (i, v) in enumerate(nodes):
        best, best_prev = 1 + (v == 1), None
        for b in range(a):
            j, u = nodes[b]
            if j < i and u < v:
                s = score[b] + 1 + (v == u + 1)
                if s > best:
                    best, best_prev = s, b
        score.append(best)
        back.append(best_prev)

    chain, node = {}, max(range(len(nodes)), key=score.__getitem__, default=None)
    while node is not None:
        i, v = nodes[node]
        chain[i] = v
        node = back[node]
    return chain


def find_article_headings(lines: list[dict]) -> tuple[dict[int, int], dict]:
    """Decide which lines are article headings and what their numbers are.

    Articles are numbered 1, 2, 3, ... in order, so the sequence decides, not the layout:
    1. Candidates: every left-margin line starting with "មាត្រា". Some are real headings,
       others are wrapped sentences that mention an article ("...មាត្រា ១២ នៃក្រមនេះ").
    2. Each candidate gets its possible readings ("៣៣០" -> 330 or 33).
    3. The readings of ALL candidates are chosen together so the most of them form an
       increasing sequence (best_increasing_chain). Deciding one line at a time fails:
       a single wrong number pushes every later heading out of order.
    4. Candidates left over are filled in when they fit a gap exactly:
       one candidate between 212 and 214 must be 213.
    5. Anything still left is a sentence that mentions an article, not a heading.

    Returns ({line index: article number}, report).
    """
    cand = [i for i, l in enumerate(lines)
            if l["x1"] < INDENT_X and l["text"].startswith(ARTICLE_WORD)]
    readings = [article_readings(lines[i]["text"])[0] for i in cand]
    chain = best_increasing_chain(readings)          # {candidate position: number}

    bounds = [(-1, 0)] + sorted(chain.items()) + [(len(cand), LAST_ARTICLE + 1)]
    for (k_left, n_left), (k_right, n_right) in zip(bounds, bounds[1:]):
        run = range(k_left + 1, k_right)
        if len(run) and len(run) == n_right - n_left - 1:
            for offset, k in enumerate(run):
                chain[k] = n_left + 1 + offset

    articles = {cand[k]: n for k, n in chain.items()}
    found = set(articles.values())
    report = {
        "candidates": len(cand),
        "not_headings": [(lines[i]["page"], lines[i]["text"]) for k, i in enumerate(cand) if k not in chain],
        "missing": [n for n in range(1, LAST_ARTICLE + 1) if n not in found],
    }
    return articles, report


def write_outputs(blocks: list[Block]):
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    with open(CLEAN_DIR / "blocks.jsonl", "w", encoding="utf-8") as f:
        for i, b in enumerate(blocks):
            f.write(json.dumps({"id": i, **asdict(b)}, ensure_ascii=False) + "\n")

    with open(CLEAN_DIR / "body.txt", "w", encoding="utf-8") as f:
        for b in blocks:
            if b.type == "heading":
                f.write(f"\n# {b.text}\n")
            elif b.type == "article":
                f.write(f"\n## {b.text}    [p{b.pages[0]}]\n")
            else:
                f.write(f"{b.text}\n")


def run():
    lines = load_body_lines()
    articles, report = find_article_headings(lines)
    blocks = build_blocks(lines, articles)
    write_outputs(blocks)
    (CLEAN_DIR / "article_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    count = lambda t: sum(1 for b in blocks if b.type == t)
    noted = lambda n: sum(1 for b in blocks if n in b.notes)
    print(f"{len(lines)} lines from pages {BODY_FIRST_PAGE}-{BODY_LAST_PAGE} -> {len(blocks)} blocks")
    print(f"  headings: {count('heading')}  articles: {count('article')}  paragraphs: {count('paragraph')}")
    print(f"  list markers fixed: {noted('list_marker_fixed')}  low-confidence blocks: {noted('low_conf')}")
    print(f"  article numbers fixed: {noted('article_no_fixed')}  "
          f"filled from sequence: {noted('number_from_sequence')}  "
          f"'មាត្រា' lines that are not headings: {len(report['not_headings'])}")
    print(f"  articles found: {LAST_ARTICLE - len(report['missing'])}/{LAST_ARTICLE}  "
          f"missing: {report['missing'] or 'none'}")
    print(f"wrote {CLEAN_DIR / 'blocks.jsonl'}, body.txt and article_report.json")


if __name__ == "__main__":
    run()
