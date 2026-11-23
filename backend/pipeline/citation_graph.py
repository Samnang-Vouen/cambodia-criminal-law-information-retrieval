"""
Stage 6: citation graph of the Criminal Code, with PageRank and HITS.

The code has no web pages and no hyperlinks, but its articles cite each other:
    "... តាមបែបបទកំណត់ដោយមាត្រា ១៧៩ (ការរឹបអូស និងសិទ្ធិតតិយជន) នៃក្រមនេះ ។"
    ("... in the manner set by Article 179 (Confiscation and rights of third parties) of this code")
Every citation is an edge  citing article -> cited article.  See docs/citation_graph.md.

1. extract_citations()  find "មាត្រា N" in every article text and check N against the cited
                        article's title, which the code repeats in brackets after the number
                        (OCR turned the bracket into a letter: "(ការ" -> "កោរ"). Citations
                        of other laws (Constitution, ...) are dropped; "A ... ដល់ (to)
                        មាត្រា B" ranges are expanded.
2. pagerank(), hits()   power iteration, written out so every step can be explained;
                        tests check them against networkx.

Output: data/processed/citations.json   every citation with its evidence
        data/processed/graph_scores.json PageRank, hub, authority, in/out degree per article

Run from backend/:  python -m pipeline graph      (or: python -m pipeline.citation_graph)
"""
import json
import re
from dataclasses import asdict, dataclass

from pipeline.config import ARTICLES_FILE, DATA_DIR

CITATIONS_FILE = DATA_DIR / "processed" / "citations.json"
SCORES_FILE = DATA_DIR / "processed" / "graph_scores.json"

CITATION = re.compile(r"មាត្រា\s*([០-៩]+)")
KHMER_DIGITS = str.maketrans("០១២៣៤៥៦៧៨៩", "0123456789")
KHMER_LETTERS = re.compile(r"[ក-៓]")
THIS_CODE = "នៃក្រមនេះ"                       # "of this code": ends the bracketed title
OTHER_LAWS = ("រដ្ឋធម្មនុញ្ញ", "នៃច្បាប់", "ក្រមនីតិវិធី")   # Constitution, "of the law ...", procedure code
RANGE_WORD = "ដល់"                            # "to": "Article A ... to Article B"
CONFUSED_DIGITS = {"3": "7", "7": "3", "4": "8", "8": "4"}   # same pairs as in clean.py
MATCH_THRESHOLD = 0.5
WINDOW = 90                                    # characters after (and before) the number

DAMPING = 0.85
TOLERANCE = 1e-12
MAX_ITERATIONS = 1000


# ---------------------------------------------------------------- 1. citations

@dataclass
class Citation:
    source: int          # citing article
    target: int          # cited article
    raw: str             # number as written in the text
    status: str          # verified | corrected | range | unverified
    title_match: float   # similarity between the bracketed text and the cited article's title
    context: str         # text after the number (the bracketed title, OCR-damaged)


def _bigrams(text: str) -> set[str]:
    s = "".join(KHMER_LETTERS.findall(text))
    return {s[i:i + 2] for i in range(len(s) - 1)}


def title_match(title: str, context: str) -> float:
    """How well the text around a citation matches an article title: the share of the
    title's character bigrams found in the context, or the share of the context's bigrams
    found in the title (when the bracketed title is cut short), whichever is higher."""
    t, c = _bigrams(title), _bigrams(context)
    if not t or not c:
        return 0.0
    return max(len(t & c) / len(t), len(t & c) / len(c) if len(c) >= 6 else 0.0)


def _digit_variants(number: str) -> list[str]:
    out = []
    for i, ch in enumerate(number):
        if ch in CONFUSED_DIGITS:
            out.append(number[:i] + CONFUSED_DIGITS[ch] + number[i + 1:])
    return out


def extract_citations(articles: list[dict]) -> tuple[list[Citation], list[dict]]:
    """All citations between articles of this code, plus the mentions that were dropped
    (other laws, numbers outside the code) with the reason."""
    titles = {a["article_no"]: a["title"] for a in articles}
    citations, dropped = [], []
    for a in articles:
        text, src = a["text"], a["article_no"]
        previous: Citation | None = None
        for m in CITATION.finditer(text):
            raw = m.group(1).translate(KHMER_DIGITS)
            after = text[m.end(): m.end() + WINDOW]
            if THIS_CODE in after:
                after = after[: after.index(THIS_CODE)]
            before = text[max(0, m.start() - WINDOW): m.start()]
            if any(law in after for law in OTHER_LAWS):
                dropped.append({"source": src, "raw": raw, "reason": "other law", "context": after[:60]})
                continue

            def score(n: int) -> float:
                if n not in titles:
                    return 0.0
                return max(title_match(titles[n], after), title_match(titles[n], before))

            number, status = int(raw), "verified"
            match = score(number)
            if match < MATCH_THRESHOLD:
                best = max(((score(int(v)), int(v)) for v in _digit_variants(raw)), default=(0.0, 0))
                if best[0] >= MATCH_THRESHOLD:
                    match, number, status = best[0], best[1], "corrected"
                else:
                    status = "unverified"
            if number not in titles:
                dropped.append({"source": src, "raw": raw, "reason": "not an article of this code",
                                "context": after[:60]})
                continue

            cite = Citation(src, number, raw, status, round(match, 3), after[:60])
            # "មាត្រា A (...) ដល់មាត្រា B (...)": every article from A to B is cited
            if previous and RANGE_WORD in text[max(0, m.start() - 4): m.start()] \
                    and previous.target < number:
                for between in range(previous.target + 1, number):
                    citations.append(Citation(src, between, "", "range", 0.0, f"{previous.target}-{number}"))
            citations.append(cite)
            previous = cite
    return citations, dropped


# ---------------------------------------------------------------- 2. graph algorithms

def build_edges(citations: list[Citation]) -> set[tuple[int, int]]:
    """Distinct citing -> cited pairs; an article citing itself is not an edge."""
    return {(c.source, c.target) for c in citations if c.source != c.target}


def pagerank(nodes: list[int], edges: set[tuple[int, int]], d: float = DAMPING) -> dict[int, float]:
    """PR(v) = (1-d)/N + d · ( Σ_{u→v} PR(u)/out(u)  +  Σ_{u dangling} PR(u)/N )

    A random reader starts at any article, follows a citation with probability d, or
    jumps to a random article with probability 1-d. Articles that cite nothing
    ("dangling", most of the code) send the reader to a random article. PageRank is the
    share of time the reader spends on each article. Sums to 1."""
    n = len(nodes)
    out = {v: [] for v in nodes}
    for u, v in edges:
        out[u].append(v)
    rank = {v: 1 / n for v in nodes}
    for _ in range(MAX_ITERATIONS):
        dangling = sum(rank[u] for u in nodes if not out[u])
        new = {v: (1 - d) / n + d * dangling / n for v in nodes}
        for u in nodes:
            for v in out[u]:
                new[v] += d * rank[u] / len(out[u])
        if sum(abs(new[v] - rank[v]) for v in nodes) < TOLERANCE:
            return new
        rank = new
    return rank


def hits(nodes: list[int], edges: set[tuple[int, int]]) -> tuple[dict[int, float], dict[int, float]]:
    """HITS (Kleinberg 1999). A good AUTHORITY is cited by good hubs; a good HUB cites
    good authorities:
        authority(v) = Σ_{u→v} hub(u)        hub(u) = Σ_{u→v} authority(v)
    repeated, normalizing both to sum 1, until they stop changing. Articles without any
    citation get 0."""
    hub = {v: 1.0 for v in nodes}
    auth = {v: 0.0 for v in nodes}
    for _ in range(MAX_ITERATIONS):
        auth = {v: 0.0 for v in nodes}
        for u, v in edges:
            auth[v] += hub[u]
        new_hub = {v: 0.0 for v in nodes}
        for u, v in edges:
            new_hub[u] += auth[v]
        sa, sh = sum(auth.values()) or 1.0, sum(new_hub.values()) or 1.0
        auth = {v: x / sa for v, x in auth.items()}
        new_hub = {v: x / sh for v, x in new_hub.items()}
        if sum(abs(new_hub[v] - hub[v]) for v in nodes) < TOLERANCE:
            return new_hub, auth
        hub = new_hub
    return hub, auth


# ---------------------------------------------------------------- stage

def run():
    articles = [json.loads(line) for line in ARTICLES_FILE.open(encoding="utf-8")]
    nodes = [a["article_no"] for a in articles]
    citations, dropped = extract_citations(articles)
    edges = build_edges(citations)
    pr = pagerank(nodes, edges)
    hub, auth = hits(nodes, edges)
    in_deg = {v: 0 for v in nodes}
    out_deg = {v: 0 for v in nodes}
    for u, v in edges:
        out_deg[u] += 1
        in_deg[v] += 1

    CITATIONS_FILE.write_text(json.dumps({"citations": [asdict(c) for c in citations], "dropped": dropped},
                                         ensure_ascii=False, indent=1), encoding="utf-8")
    SCORES_FILE.write_text(json.dumps({
        "damping": DAMPING,
        "edges": sorted(edges),
        "articles": {str(v): {"pagerank": pr[v], "hub": hub[v], "authority": auth[v],
                              "in_degree": in_deg[v], "out_degree": out_deg[v]} for v in nodes},
    }, ensure_ascii=False, indent=1), encoding="utf-8")

    status = {s: sum(c.status == s for c in citations) for s in ("verified", "corrected", "range", "unverified")}
    print(f"citations: {len(citations)} {status}, dropped: {len(dropped)} "
          f"({', '.join(sorted({d['reason'] for d in dropped}))})")
    print(f"graph: {len(nodes)} articles, {len(edges)} edges, "
          f"{sum(1 for v in nodes if out_deg[v])} citing, {sum(1 for v in nodes if in_deg[v])} cited")
    for name, scores in (("PageRank", pr), ("authority", auth), ("hub", hub)):
        top = sorted(nodes, key=lambda v: (-scores[v], v))[:8]
        print(f"top {name}: " + ", ".join(f"{v} ({scores[v]:.4f}, in {in_deg[v]}, out {out_deg[v]})" for v in top))
    print(f"written: {CITATIONS_FILE.name}, {SCORES_FILE.name}")


if __name__ == "__main__":
    run()
