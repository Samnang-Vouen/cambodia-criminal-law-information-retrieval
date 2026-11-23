"""
Build the judging pool and the labelling sheet (TREC-style pooling, docs/evaluation.md §3).

For every query in queries.json, the pool is the union of
  - the top POOL_DEPTH articles of every ranked model, and
  - the Boolean result set (query text, implicit AND); when it has more than BOOLEAN_CAP
    articles, a random sample of BOOLEAN_CAP (fixed seed), so Boolean precision can still be
    estimated without bias.

Output (evaluation/labels/):
  labels.csv  the sheet for the labellers: one row per (query, article), sorted by
              article number, WITHOUT model names or ranks (blind judging). Labellers
              fill `relevance` (0/1/2) and may ADD rows for relevant articles that no
              model found. Open it in Excel or Google Sheets (UTF-8 with BOM).
  pool.json   which model found which article at which rank (for analysis only).

Re-running keeps every label already in labels.csv and only adds new rows.

Run from backend/:  python -m evaluation.pool
"""
import csv
import json
import random
from pathlib import Path

from pipeline.boolean import QuerySyntaxError
from pipeline.build_index import load
from pipeline.config import ARTICLES_FILE
from pipeline.search import RANKED_MODELS, Searcher

EVAL_DIR = Path(__file__).resolve().parent
QUERIES_FILE = EVAL_DIR / "queries.json"
LABELS_DIR = EVAL_DIR / "labels"
LABELS_CSV = LABELS_DIR / "labels.csv"
POOL_FILE = LABELS_DIR / "pool.json"

POOL_DEPTH = 10
BOOLEAN_CAP = 20
COLUMNS = ["query_id", "query", "query_en", "article_no", "title", "text", "relevance", "notes"]


def load_queries() -> list[dict]:
    return json.loads(QUERIES_FILE.read_text(encoding="utf-8"))


def build_pool(searcher: Searcher, queries: list[dict]) -> dict:
    pool = {}
    for q in queries:
        found: dict[int, dict[str, int]] = {}      # article_no -> {model: rank}
        for model in RANKED_MODELS:
            for rank, h in enumerate(searcher.search(q["query"], k=POOL_DEPTH, model=model), 1):
                found.setdefault(h.article_no, {})[model] = rank
        try:
            hits = [h.article_no for h in searcher.search(q["query"], k=10_000, model="boolean")]
        except QuerySyntaxError:
            hits = []
        sample = hits if len(hits) <= BOOLEAN_CAP else random.Random(q["id"]).sample(hits, BOOLEAN_CAP)
        for a in sample:
            found.setdefault(a, {})["boolean"] = 0       # 0 = unranked
        pool[q["id"]] = {
            "boolean_results": len(hits),
            "boolean_sampled": len(hits) > BOOLEAN_CAP,
            "articles": {str(a): found[a] for a in sorted(found)},
        }
    return pool


def read_labels(path: Path | None = None) -> list[dict]:
    path = path or LABELS_CSV          # looked up at call time, so it can be redirected
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_sheet(queries: list[dict], pool: dict, articles: dict[int, dict]) -> tuple[int, int]:
    existing = read_labels()
    keyed = {(r["query_id"], int(r["article_no"])): r for r in existing if r.get("article_no", "").strip()}
    added = 0
    for q in queries:
        for a in map(int, pool[q["id"]]["articles"]):
            if (q["id"], a) not in keyed:
                keyed[(q["id"], a)] = {"query_id": q["id"], "relevance": "", "notes": ""}
                added += 1
    by_id = {q["id"]: q for q in queries}
    rows = []
    for (qid, a), row in sorted(keyed.items(), key=lambda x: (x[0][0], x[0][1])):
        q, art = by_id.get(qid, {}), articles.get(a, {})
        rows.append({**row, "query": q.get("query", row.get("query", "")),
                     "query_en": q.get("en", row.get("query_en", "")), "article_no": a,
                     "title": art.get("title", ""), "text": art.get("text", "")})
    LABELS_DIR.mkdir(exist_ok=True)
    with LABELS_CSV.open("w", encoding="utf-8-sig", newline="") as f:    # BOM: Excel shows Khmer
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return len(rows), added


def main():
    queries = load_queries()
    articles = {}
    for line in ARTICLES_FILE.open(encoding="utf-8"):
        a = json.loads(line)
        articles[a["article_no"]] = a
    pool = build_pool(Searcher(load()), queries)
    LABELS_DIR.mkdir(exist_ok=True)
    POOL_FILE.write_text(json.dumps(pool, ensure_ascii=False, indent=1), encoding="utf-8")
    total, added = write_sheet(queries, pool, articles)

    sizes = [len(p["articles"]) for p in pool.values()]
    print(f"queries: {len(queries)}   pool per query: min {min(sizes)}, "
          f"mean {sum(sizes) / len(sizes):.1f}, max {max(sizes)}   total pairs: {sum(sizes)}")
    sampled = [qid for qid, p in pool.items() if p["boolean_sampled"]]
    print(f"Boolean results sampled ({BOOLEAN_CAP} of more) for: {', '.join(sampled) or 'none'}")
    print(f"labels.csv: {total} rows ({added} new)  ->  {LABELS_CSV}")


if __name__ == "__main__":
    main()
