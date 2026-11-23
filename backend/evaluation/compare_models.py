"""
Compare how TF-IDF, cosine and BM25 rank the articles for the same queries (project
criterion "Classical search methods"). See docs/model_comparison.md for the analysis.

No relevance labels are needed: this measures HOW the rankings differ and WHICH part of
each formula causes the difference. Which model is BETTER is measured with relevance
labels in the evaluation step.

  1. Agreement   for every pair of models: overlap@10, average overlap (AO@10), same top-1
  2. Profiles    what the top 10 of each model look like: article length, title matches,
                 how many query terms are matched, how often they are repeated
  3. Ablation    BM25 with one component switched off at a time (length normalization,
                 TF saturation, TF, title weight): which one makes BM25 differ from TF-IDF
  4. Known-item  every article with a unique title, searched by its title: rank of that
                 article (MRR, success@1, success@10). An automatic sanity check, not a
                 relevance evaluation.

Input : evaluation/queries.json, backend/index/index.json (python -m pipeline index)
Output: evaluation/results/model_comparison.json and model_comparison.md

Run from backend/:  python -m evaluation.compare_models
"""
import json
import statistics
import time
from collections import Counter
from itertools import combinations
from pathlib import Path

from pipeline.analyzer import word_terms
from pipeline.build_index import load
from pipeline.config import ARTICLES_FILE
from pipeline.search import Params, Searcher

EVAL_DIR = Path(__file__).resolve().parent
QUERIES_FILE = EVAL_DIR / "queries.json"
RESULTS_DIR = EVAL_DIR / "results"
K = 10
MODELS = ("tfidf", "cosine", "bm25", "bm25_ngram")

# BM25 with one component changed at a time (model "bm25", different Params)
ABLATIONS = {
    "bm25 (default)": Params(),
    "b=0: no length normalization": Params(b={"title": 0.0, "text": 0.0, "ngram": 0.0}),
    "k1=1000: no TF saturation": Params(k1=1000.0),
    "k1=0: no TF at all (IDF only)": Params(k1=0.0),
    "w_title=1: no title boost": Params(weight={"title": 1.0, "text": 1.0}),
}


# ---------------------------------------------------------------- ranking agreement

def overlap(a: list[int], b: list[int]) -> float:
    """Share of the (longer) result list that both models return."""
    depth = max(len(a), len(b))
    return len(set(a) & set(b)) / depth if depth else 1.0


def average_overlap(a: list[int], b: list[int]) -> float:
    """AO: mean of overlap@d for d = 1..depth. Rewards agreement at the top more than
    overlap@10 does (Webber et al. 2010, the basis of rank-biased overlap)."""
    depth = max(len(a), len(b))
    if not depth:
        return 1.0
    return sum(len(set(a[:d]) & set(b[:d])) / d for d in range(1, depth + 1)) / depth


# ---------------------------------------------------------------- result profiles

class Profiler:
    """Properties of a result list, used to explain WHY models rank differently."""

    def __init__(self, searcher: Searcher):
        self.s = searcher
        self.f = searcher.index.fields

    def tf(self, term: str, doc: int) -> int:
        return sum(self.f[n].postings.get(term, {}).get(doc, 0) for n in ("title", "text"))

    def profile(self, query: str, docs: list[int]) -> dict:
        terms = [t for t in dict.fromkeys(word_terms(query)) if self.s.word_df(t)]
        if not docs or not terms:
            return {}
        return {
            "text_length": statistics.mean(self.f["text"].lengths[d] for d in docs),
            "title_match": statistics.mean(
                any(d in self.f["title"].postings.get(t, {}) for t in terms) for d in docs),
            "terms_matched": statistics.mean(
                sum(self.tf(t, d) > 0 for t in terms) / len(terms) for d in docs),
            "tf_per_term": statistics.mean(
                sum(self.tf(t, d) for t in terms) / max(1, sum(self.tf(t, d) > 0 for t in terms))
                for d in docs),
        }


# ---------------------------------------------------------------- experiment

def ranking(searcher: Searcher, query: str, model: str, k: int = K) -> list[int]:
    return [h.doc for h in searcher.search(query, k=k, model=model)]


def tied_share(searcher: Searcher, query: str, model: str) -> float:
    """Share of the top K whose score equals another top-K score: their order is then
    decided by the tie-breaker (article number), not by the model."""
    scores = [round(h.score, 9) for h in searcher.search(query, k=K, model=model)]
    counts = Counter(scores)
    return sum(counts[s] > 1 for s in scores) / len(scores) if scores else 0.0


def known_item(searchers: dict[str, tuple[Searcher, str]], articles: list[dict]) -> dict:
    titles = Counter(a["title"] for a in articles)
    items = [(doc, a["title"]) for doc, a in enumerate(articles)
             if titles[a["title"]] == 1 and word_terms(a["title"])]
    out = {"n_queries": len(items)}
    for name, (searcher, model) in searchers.items():
        ranks = []
        for doc, title in items:
            r = ranking(searcher, title, model, k=len(articles))
            ranks.append(r.index(doc) + 1 if doc in r else None)
        rr = [1 / r if r else 0.0 for r in ranks]
        out[name] = {
            "mrr": statistics.mean(rr),
            "success@1": statistics.mean(r == 1 for r in ranks),
            "success@10": statistics.mean(r is not None and r <= 10 for r in ranks),
            "not_found": sum(r is None for r in ranks),
        }
    return out


def run_experiment() -> dict:
    index = load()
    articles = [json.loads(line) for line in ARTICLES_FILE.open(encoding="utf-8")]
    queries = json.loads(QUERIES_FILE.read_text(encoding="utf-8"))
    base = Searcher(index)
    ablated = {name: Searcher(index, p) for name, p in ABLATIONS.items()}
    profiler = Profiler(base)

    per_query = []
    for q in queries:
        runs = {m: ranking(base, q["query"], m) for m in MODELS}
        runs.update({name: ranking(s, q["query"], "bm25") for name, s in ablated.items()})
        per_query.append({
            **q,
            "terms": word_terms(q["query"]),
            "rankings": {m: [articles[d]["article_no"] for d in r] for m, r in runs.items()},
            "profiles": {m: {**profiler.profile(q["query"], runs[m]),
                             "tied": tied_share(base, q["query"], m)} for m in MODELS},
            "_docs": runs,
        })

    def mean_over_queries(fn):
        return statistics.mean(fn(pq) for pq in per_query)

    agreement = {
        f"{a} vs {b}": {
            "overlap@10": mean_over_queries(lambda pq: overlap(pq["_docs"][a], pq["_docs"][b])),
            "AO@10": mean_over_queries(lambda pq: average_overlap(pq["_docs"][a], pq["_docs"][b])),
            "same_top1": mean_over_queries(lambda pq: pq["_docs"][a][:1] == pq["_docs"][b][:1]),
        }
        for a, b in combinations(MODELS, 2)
    }
    profiles = {
        m: {key: statistics.mean(pq["profiles"][m][key] for pq in per_query
                                 if "text_length" in pq["profiles"][m])   # query had results
            for key in ("text_length", "title_match", "terms_matched", "tf_per_term", "tied")}
        for m in MODELS
    }
    ablation = {
        name: {
            other: mean_over_queries(lambda pq: average_overlap(pq["_docs"][name], pq["_docs"][other]))
            for other in ("bm25 (default)", "tfidf", "cosine")
        }
        for name in ABLATIONS
    }
    for name, s in ablated.items():
        profiles[name] = {
            key: statistics.mean(v[key] for pq in per_query if (v := profiler.profile(pq["query"], pq["_docs"][name])))
            for key in ("text_length", "title_match", "terms_matched", "tf_per_term")
        }
        profiles[name]["tied"] = statistics.mean(tied_share(s, pq["query"], "bm25") for pq in per_query)

    searchers = {m: (base, m) for m in MODELS}
    searchers.update({name: (s, "bm25") for name, s in ablated.items() if name != "bm25 (default)"})
    for pq in per_query:
        pq["AO@10"] = {f"{a} vs {b}": average_overlap(pq["_docs"][a], pq["_docs"][b])
                       for a, b in combinations(MODELS, 2)}
        del pq["_docs"]
    return {
        "k": K,
        "corpus_avg_text_length": index.fields["text"].avg_length,
        "agreement": agreement,
        "profiles": profiles,
        "ablation_AO@10": ablation,
        "known_item": known_item(searchers, articles),
        "queries": per_query,
    }


# ---------------------------------------------------------------- report

def to_markdown(r: dict) -> str:
    out = ["# Model comparison results", "",
           "Generated by `python -m evaluation.compare_models`. Analysis: `docs/model_comparison.md`.", ""]
    out += ["## 1. Agreement between models (mean over queries, top 10)", "",
            "| Pair | overlap@10 | AO@10 | same top-1 |", "|---|---|---|---|"]
    out += [f"| {p} | {v['overlap@10']:.2f} | {v['AO@10']:.2f} | {v['same_top1']:.0%} |"
            for p, v in r["agreement"].items()]
    out += ["", "## 2. What the top 10 look like (mean over queries)", "",
            f"Average article text length in the corpus: {r['corpus_avg_text_length']:.1f} terms.", "",
            "| Model | text length | has query term in title | query terms matched | tf per matched term | tied scores |",
            "|---|---|---|---|---|---|"]
    out += [f"| {m} | {p['text_length']:.1f} | {p['title_match']:.0%} | {p['terms_matched']:.0%} "
            f"| {p['tf_per_term']:.2f} | {p['tied']:.0%} |"
            for m, p in r["profiles"].items()]
    out += ["", "## 3. BM25 ablation: AO@10 with each model", "",
            "| BM25 variant | vs bm25 (default) | vs tfidf | vs cosine |", "|---|---|---|---|"]
    out += [f"| {n} | {v['bm25 (default)']:.2f} | {v['tfidf']:.2f} | {v['cosine']:.2f} |"
            for n, v in r["ablation_AO@10"].items()]
    ki = r["known_item"]
    out += ["", f"## 4. Known-item search: {ki['n_queries']} unique article titles as queries", "",
            "| Model | MRR | success@1 | success@10 | not found |", "|---|---|---|---|---|"]
    out += [f"| {m} | {v['mrr']:.3f} | {v['success@1']:.1%} | {v['success@10']:.1%} | {v['not_found']} |"
            for m, v in ki.items() if m != "n_queries"]
    out += ["", "## 5. Per query: top 5 and agreement", ""]
    for q in r["queries"]:
        out += [f"### {q['id']} `{q['query']}` ({q['en']}, {q['type']})", "",
                f"Terms: {' '.join(q['terms'])}", "",
                "| Model | top 5 (article numbers) |", "|---|---|"]
        out += [f"| {m} | {', '.join(map(str, q['rankings'][m][:5]))} |" for m in MODELS]
        out += ["", "AO@10: " + " · ".join(f"{p} {v:.2f}" for p, v in q["AO@10"].items()), ""]
    return "\n".join(out) + "\n"


def main():
    t0 = time.perf_counter()
    r = run_experiment()
    report = to_markdown(r)
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "model_comparison.json").write_text(
        json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
    (RESULTS_DIR / "model_comparison.md").write_text(report, encoding="utf-8")
    print(report.split("## 5.")[0])
    print(f"done in {time.perf_counter() - t0:.1f} s -> {RESULTS_DIR}")


if __name__ == "__main__":
    main()
