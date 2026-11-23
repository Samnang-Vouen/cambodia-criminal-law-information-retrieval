"""
Evaluate all retrieval models on the labelled test queries (project criterion
"Evaluation and experiments"). Method and metric definitions: docs/evaluation.md.

  Ranked models (tfidf, cosine, bm25, bm25_ngram): P@5, P@10, R@10, F1@10, MAP, MRR, nDCG@10
  Boolean model (query text with implicit AND):    set precision, recall, F1
  Significance: paired randomization test, bm25 vs every other ranked model (AP, nDCG@10)
  Failure analysis: hardest queries, relevant articles that no model ranks in its top 10

Input : evaluation/queries.json, evaluation/labels/labels.csv (made with evaluation.pool and
        filled in by the labellers), backend/index/index.json
Output: evaluation/labels/qrels.json      the labels as {query_id: {article_no: grade}}
        evaluation/results/evaluation.md  report;  evaluation.json  all numbers

Only queries whose pooled articles are ALL labelled are evaluated (unless --partial), so a
half-finished labelling never produces misleading numbers.

Run from backend/:  python -m evaluation.evaluate [--partial]
"""
import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

from evaluation import metrics as M
from evaluation.pool import LABELS_CSV, LABELS_DIR, POOL_FILE, load_queries, read_labels
from pipeline.boolean import QuerySyntaxError
from pipeline.build_index import load
from pipeline.search import RANKED_MODELS, Searcher

EVAL_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EVAL_DIR / "results"
QRELS_FILE = LABELS_DIR / "qrels.json"
K = 10
RANKED_METRICS = ("P@5", "P@10", "R@10", "F1@10", "AP", "RR", "nDCG@10")
SIGNIFICANCE_METRICS = ("AP", "nDCG@10")
BASELINE = "bm25"


# ---------------------------------------------------------------- labels

def parse_labels(rows: list[dict], query_ids: set[str], n_articles: int = 672):
    """labels.csv rows -> (qrels, unlabelled count per query, problems)."""
    qrels: dict[str, dict[int, int]] = {}
    unlabelled: Counter = Counter()
    problems = []
    for i, row in enumerate(rows, start=2):            # row 1 is the header
        qid, art, grade = row.get("query_id", "").strip(), row.get("article_no", "").strip(), \
            row.get("relevance", "").strip()
        if qid not in query_ids:
            problems.append(f"row {i}: unknown query_id {qid!r}")
            continue
        if not art.isdigit() or not 1 <= int(art) <= n_articles:
            problems.append(f"row {i}: article_no {art!r} is not 1-{n_articles}")
            continue
        if grade == "":
            unlabelled[qid] += 1
            continue
        if grade not in ("0", "1", "2"):
            problems.append(f"row {i}: relevance {grade!r} must be 0, 1 or 2")
            continue
        previous = qrels.setdefault(qid, {}).get(int(art))
        if previous is not None and previous != int(grade):
            problems.append(f"row {i}: {qid} article {art} labelled {previous} and {grade}")
        qrels[qid][int(art)] = int(grade)
    return qrels, unlabelled, problems


# ---------------------------------------------------------------- runs

def run_models(searcher: Searcher, queries: list[dict]) -> dict:
    """Full rankings (all matching articles, for AP) for the ranked models, and the
    Boolean result set."""
    runs = {m: {} for m in (*RANKED_MODELS, "boolean")}
    for q in queries:
        for m in RANKED_MODELS:
            runs[m][q["id"]] = [h.article_no for h in searcher.search(q["query"], k=10_000, model=m)]
        try:
            runs["boolean"][q["id"]] = [h.article_no for h in
                                        searcher.search(q["query"], k=10_000, model="boolean")]
        except QuerySyntaxError:
            runs["boolean"][q["id"]] = []
    return runs


# ---------------------------------------------------------------- scoring

def score_ranking(ranking: list[int], qrels: dict[int, int]) -> dict:
    return {
        "P@5": M.precision_at(ranking, qrels, 5),
        "P@10": M.precision_at(ranking, qrels, K),
        "R@10": M.recall_at(ranking, qrels, K),
        "F1@10": M.f1_at(ranking, qrels, K),
        "AP": M.average_precision(ranking, qrels),
        "RR": M.reciprocal_rank(ranking, qrels),
        "nDCG@10": M.ndcg_at(ranking, qrels, K),
        "judged@10": M.judged_at(ranking, qrels, K),
    }


def score_boolean(results: list[int], qrels: dict[int, int]) -> dict:
    """Precision over the JUDGED results: exact when all results are in the pool, an
    unbiased estimate when they were randomly sampled (pool.py). Recall is exact with
    respect to the labels, since the full result set is known."""
    judged = [d for d in results if d in qrels]
    hits = sum(M.is_rel(qrels[d]) for d in judged)
    total = M.n_relevant(qrels)
    p = hits / len(judged) if judged else 0.0
    r = sum(M.is_rel(qrels.get(d, 0)) for d in results) / total if total else 0.0
    return {"P": p, "R": r, "F1": 2 * p * r / (p + r) if p + r else 0.0,
            "results": len(results), "judged": len(judged)}


def evaluate(runs: dict, qrels: dict, queries: list[dict]) -> dict:
    evaluated = [q for q in queries if q["id"] in qrels and M.n_relevant(qrels[q["id"]])]
    per_query = {}
    for q in evaluated:
        rel = qrels[q["id"]]
        per_query[q["id"]] = {m: score_ranking(runs[m][q["id"]], rel) for m in RANKED_MODELS}
        per_query[q["id"]]["boolean"] = score_boolean(runs["boolean"][q["id"]], rel)

    def mean(model, metric, ids=None):
        ids = ids if ids is not None else list(per_query)
        return statistics.mean(per_query[i][model][metric] for i in ids) if ids else 0.0

    summary = {m: {x: mean(m, x) for x in (*RANKED_METRICS, "judged@10")} for m in RANKED_MODELS}
    summary["boolean"] = {x: mean("boolean", x) for x in ("P", "R", "F1")}
    summary["boolean"]["empty_results"] = sum(per_query[i]["boolean"]["results"] == 0 for i in per_query)

    by_type = {}
    for t in sorted({q["type"] for q in evaluated}):
        ids = [q["id"] for q in evaluated if q["type"] == t]
        by_type[t] = {"n": len(ids), **{m: {x: mean(m, x, ids) for x in ("AP", "nDCG@10")}
                                        for m in RANKED_MODELS}}
        by_type[t]["boolean"] = {"F1": mean("boolean", "F1", ids)}

    significance = {
        f"{BASELINE} vs {m}": {
            x: {"diff": mean(BASELINE, x) - mean(m, x),
                "p": M.randomization_test([per_query[i][BASELINE][x] for i in per_query],
                                          [per_query[i][m][x] for i in per_query])}
            for x in SIGNIFICANCE_METRICS}
        for m in RANKED_MODELS if m != BASELINE
    }

    hardest = sorted(per_query, key=lambda i: statistics.mean(per_query[i][m]["AP"] for m in RANKED_MODELS))
    missed = {}
    for q in evaluated:
        top = set().union(*(runs[m][q["id"]][:K] for m in RANKED_MODELS))
        lost = sorted(a for a, g in qrels[q["id"]].items() if M.is_rel(g) and a not in top)
        if lost:
            missed[q["id"]] = {str(a): qrels[q["id"]][a] for a in lost}

    return {
        "k": K,
        "rel_threshold": M.REL_THRESHOLD,
        "n_queries": len(queries),
        "evaluated": [q["id"] for q in evaluated],
        "n_relevant": {q["id"]: M.n_relevant(qrels[q["id"]]) for q in evaluated},
        "no_relevant": [q["id"] for q in queries if q["id"] in qrels and not M.n_relevant(qrels[q["id"]])],
        "summary": summary,
        "by_type": by_type,
        "significance": significance,
        "hardest_queries": hardest[:5],
        "missed_by_all": missed,
        "per_query": per_query,
    }


# ---------------------------------------------------------------- report

def to_markdown(r: dict, queries: list[dict], skipped: dict) -> str:
    by_id = {q["id"]: q for q in queries}
    out = ["# Evaluation results", "",
           "Generated by `python -m evaluation.evaluate`. Method: `docs/evaluation.md`.", "",
           f"Evaluated queries: {len(r['evaluated'])} of {r['n_queries']}. "
           f"Relevant = grade ≥ {r['rel_threshold']}. Cut-off k = {r['k']}.", ""]
    if skipped:
        out += ["Not evaluated: " + ", ".join(f"{q} ({why})" for q, why in skipped.items()), ""]
    out += ["## 1. Ranked models (mean over queries)", "",
            "| Model | P@5 | P@10 | R@10 | F1@10 | MAP | MRR | nDCG@10 | judged@10 |",
            "|---|---|---|---|---|---|---|---|---|"]
    for m in RANKED_MODELS:
        s = r["summary"][m]
        out.append(f"| {m} | " + " | ".join(f"{s[x]:.3f}" for x in RANKED_METRICS)
                   + f" | {s['judged@10']:.0%} |")
    b = r["summary"]["boolean"]
    out += ["", "## 2. Boolean model (query text, implicit AND)", "",
            "| Precision | Recall | F1 | queries with no result |", "|---|---|---|---|",
            f"| {b['P']:.3f} | {b['R']:.3f} | {b['F1']:.3f} | {b['empty_results']} |"]
    out += ["", f"## 3. Significance: {BASELINE} vs other models (paired randomization test)", "",
            "| Comparison | ΔMAP | p | ΔnDCG@10 | p |", "|---|---|---|---|---|"]
    for pair, s in r["significance"].items():
        out.append(f"| {pair} | {s['AP']['diff']:+.3f} | {s['AP']['p']:.3f} | "
                   f"{s['nDCG@10']['diff']:+.3f} | {s['nDCG@10']['p']:.3f} |")
    out += ["", "## 4. By query type (MAP / nDCG@10; Boolean F1)", "",
            "| Type | n | " + " | ".join(RANKED_MODELS) + " | boolean F1 |",
            "|---|---|" + "---|" * len(RANKED_MODELS) + "---|"]
    for t, s in r["by_type"].items():
        out.append(f"| {t} | {s['n']} | " + " | ".join(
            f"{s[m]['AP']:.2f} / {s[m]['nDCG@10']:.2f}" for m in RANKED_MODELS) + f" | {s['boolean']['F1']:.2f} |")
    out += ["", "## 5. Hardest queries (lowest mean AP over the ranked models)", ""]
    for qid in r["hardest_queries"]:
        pq = r["per_query"][qid]
        out.append(f"- **{qid}** `{by_id[qid]['query']}` ({by_id[qid]['en']}): " + ", ".join(
            f"{m} AP {pq[m]['AP']:.2f}" for m in RANKED_MODELS))
    out += ["", "## 6. Relevant articles that no model ranks in its top 10", ""]
    out += [f"- **{qid}** ({by_id[qid]['en']}): articles " + ", ".join(
        f"{a} (grade {g})" for a, g in arts.items()) for qid, arts in r["missed_by_all"].items()] or ["- none"]
    out += ["", "## 7. Per query", "",
            "| Query | type | " + " | ".join(f"{m} AP" for m in RANKED_MODELS) + " | boolean F1 | relevant |",
            "|---|---|" + "---|" * len(RANKED_MODELS) + "---|---|"]
    for qid in r["evaluated"]:
        pq = r["per_query"][qid]
        out.append(f"| {qid} {by_id[qid]['en']} | {by_id[qid]['type']} | " + " | ".join(
            f"{pq[m]['AP']:.2f}" for m in RANKED_MODELS) + f" | {pq['boolean']['F1']:.2f} | "
            f"{r['n_relevant'][qid]} |")
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser(description="Evaluate the retrieval models on the labelled queries")
    ap.add_argument("--partial", action="store_true",
                    help="also evaluate queries that still have unlabelled pooled articles")
    args = ap.parse_args()

    queries = load_queries()
    if not LABELS_CSV.exists():
        raise SystemExit("no labels yet: run `python -m evaluation.pool` and fill in labels.csv")
    qrels, unlabelled, problems = parse_labels(read_labels(), {q["id"] for q in queries})
    if problems:
        raise SystemExit("fix labels.csv first:\n  " + "\n  ".join(problems))

    pool = json.loads(POOL_FILE.read_text(encoding="utf-8")) if POOL_FILE.exists() else {}
    skipped = {}
    for q in queries:
        if q["id"] not in qrels:
            skipped[q["id"]] = "no labels"
        elif unlabelled[q["id"]] and not args.partial:
            skipped[q["id"]] = f"{unlabelled[q['id']]} unlabelled"
    if pool and set(pool) != {q["id"] for q in queries}:
        print("warning: queries.json changed since the pool was built - run `python -m evaluation.pool`")
    usable = {qid: rel for qid, rel in qrels.items() if qid not in skipped}
    if qrels:
        QRELS_FILE.write_text(
            json.dumps({q: {str(a): g for a, g in sorted(r.items())} for q, r in qrels.items()},
                       ensure_ascii=False, indent=1), encoding="utf-8")
    if not any(M.n_relevant(rel) for rel in usable.values()):
        raise SystemExit(f"no fully labelled query with a relevant article yet "
                         f"({len(qrels)} queries have some labels). Use --partial to evaluate anyway.")

    r = evaluate(run_models(Searcher(load()), queries), usable, queries)
    RESULTS_DIR.mkdir(exist_ok=True)
    report = to_markdown(r, queries, skipped)
    (RESULTS_DIR / "evaluation.json").write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
    (RESULTS_DIR / "evaluation.md").write_text(report, encoding="utf-8")
    print(report.split("## 5.")[0])
    print(f"-> {RESULTS_DIR / 'evaluation.md'}")


if __name__ == "__main__":
    main()
