"""
Retrieval quality metrics. See docs/evaluation.md §4 for definitions and examples.

Relevance labels are graded: 2 = relevant, 1 = partially relevant, 0 = not relevant.
Unjudged articles count as not relevant (the usual convention for pooled test sets).
Binary metrics (precision, recall, F1, AP, RR) count an article as relevant when its
grade is >= REL_THRESHOLD; nDCG uses the grades themselves.
"""
import math
import random

REL_THRESHOLD = 1


def is_rel(grade: int) -> bool:
    return grade >= REL_THRESHOLD


def n_relevant(qrels: dict[int, int]) -> int:
    return sum(is_rel(g) for g in qrels.values())


# ---------------------------------------------------------------- set metrics

def set_precision_recall_f1(retrieved: list[int], qrels: dict[int, int]) -> tuple[float, float, float]:
    """For an unranked result set (Boolean retrieval) or a ranking cut at k."""
    hits = sum(is_rel(qrels.get(d, 0)) for d in retrieved)
    total = n_relevant(qrels)
    p = hits / len(retrieved) if retrieved else 0.0
    r = hits / total if total else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1


# ---------------------------------------------------------------- ranked metrics

def precision_at(ranking: list[int], qrels: dict[int, int], k: int) -> float:
    """Relevant in the top k / k. Fewer than k results still divide by k
    (a model that returns nothing more is not rewarded for it)."""
    return sum(is_rel(qrels.get(d, 0)) for d in ranking[:k]) / k


def recall_at(ranking: list[int], qrels: dict[int, int], k: int) -> float:
    total = n_relevant(qrels)
    return sum(is_rel(qrels.get(d, 0)) for d in ranking[:k]) / total if total else 0.0


def f1_at(ranking: list[int], qrels: dict[int, int], k: int) -> float:
    p, r = precision_at(ranking, qrels, k), recall_at(ranking, qrels, k)
    return 2 * p * r / (p + r) if p + r else 0.0


def average_precision(ranking: list[int], qrels: dict[int, int]) -> float:
    """Mean of precision@i over the ranks i of the relevant articles; relevant articles
    never retrieved contribute 0. MAP = mean of AP over queries."""
    total = n_relevant(qrels)
    if not total:
        return 0.0
    hits, s = 0, 0.0
    for i, d in enumerate(ranking, 1):
        if is_rel(qrels.get(d, 0)):
            hits += 1
            s += hits / i
    return s / total


def reciprocal_rank(ranking: list[int], qrels: dict[int, int]) -> float:
    for i, d in enumerate(ranking, 1):
        if is_rel(qrels.get(d, 0)):
            return 1 / i
    return 0.0


def dcg(grades: list[int]) -> float:
    """Gain 2^grade - 1 (a "relevant" article is worth 3× a "partially relevant" one),
    discounted by log2(rank + 1)."""
    return sum((2 ** g - 1) / math.log2(i + 1) for i, g in enumerate(grades, 1))


def ndcg_at(ranking: list[int], qrels: dict[int, int], k: int) -> float:
    ideal = dcg(sorted(qrels.values(), reverse=True)[:k])
    return dcg([qrels.get(d, 0) for d in ranking[:k]]) / ideal if ideal else 0.0


def judged_at(ranking: list[int], qrels: dict[int, int], k: int) -> float:
    """Share of the top k that has a label. Low values mean the metrics of this model
    are unreliable (unjudged articles are counted as not relevant)."""
    top = ranking[:k]
    return sum(d in qrels for d in top) / len(top) if top else 1.0


# ---------------------------------------------------------------- significance

def randomization_test(a: list[float], b: list[float], trials: int = 10_000, seed: int = 0) -> float:
    """Two-sided paired randomization (permutation) test on per-query scores.
    H0: the two models are interchangeable, so the sign of each per-query difference is
    random. p = share of random sign flips with a mean difference at least as large as
    the observed one (Smucker, Allan & Carterette, 2007)."""
    diffs = [x - y for x, y in zip(a, b)]
    observed = abs(sum(diffs))
    if observed == 0:
        return 1.0
    rng = random.Random(seed)
    extreme = 0
    for _ in range(trials):
        if abs(sum(d if rng.random() < 0.5 else -d for d in diffs)) >= observed - 1e-12:
            extreme += 1
    return (extreme + 1) / (trials + 1)
