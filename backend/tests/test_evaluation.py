"""Tests for evaluation/metrics.py, evaluate.py and pool.py. Run from backend/:  python -m pytest

Metric values are computed by hand in the comments, so the tests double as worked examples
for docs/evaluation.md."""
import math

import pytest

from evaluation import metrics as M
from evaluation import pool
from evaluation.evaluate import evaluate, parse_labels, score_boolean
from pipeline.search import RANKED_MODELS

# ranking 1..5; relevant: 1 (grade 2), 3 (grade 1), 9 (grade 2, never retrieved); 2 judged 0
RANKING = [1, 2, 3, 4, 5]
QRELS = {1: 2, 2: 0, 3: 1, 9: 2}


# ---------------------------------------------------------------- metrics

def test_precision_recall_f1_at_k():
    assert M.precision_at(RANKING, QRELS, 5) == pytest.approx(2 / 5)     # 1 and 3
    assert M.recall_at(RANKING, QRELS, 5) == pytest.approx(2 / 3)        # 9 missing
    p, r = 2 / 5, 2 / 3
    assert M.f1_at(RANKING, QRELS, 5) == pytest.approx(2 * p * r / (p + r))


def test_precision_divides_by_k_even_with_fewer_results():
    assert M.precision_at([1], QRELS, 10) == pytest.approx(0.1)


def test_unjudged_counts_as_not_relevant():
    assert M.precision_at([4, 5], QRELS, 2) == 0.0


def test_average_precision():
    # relevant at ranks 1 and 3: (1/1 + 2/3) / 3 relevant in total (9 is never found)
    assert M.average_precision(RANKING, QRELS) == pytest.approx((1 + 2 / 3) / 3)


def test_reciprocal_rank():
    assert M.reciprocal_rank(RANKING, QRELS) == 1.0
    assert M.reciprocal_rank([2, 4, 3], QRELS) == pytest.approx(1 / 3)
    assert M.reciprocal_rank([4, 5], QRELS) == 0.0


def test_ndcg():
    # DCG  = (2²-1)/log2(2) + (2¹-1)/log2(4)                = 3 + 0.5
    # IDCG = 3/log2(2) + 3/log2(3) + 1/log2(4)   (grades 2, 2, 1)
    dcg = 3 + 0.5
    idcg = 3 + 3 / math.log2(3) + 0.5
    assert M.ndcg_at(RANKING, QRELS, 5) == pytest.approx(dcg / idcg)
    assert M.ndcg_at([1, 9, 3], QRELS, 3) == pytest.approx(1.0)          # ideal order


def test_grade_one_counts_as_relevant():
    assert M.is_rel(1) and M.is_rel(2) and not M.is_rel(0)


def test_judged_at():
    assert M.judged_at(RANKING, QRELS, 5) == pytest.approx(3 / 5)        # 1, 2, 3 judged


def test_set_metrics():
    p, r, f1 = M.set_precision_recall_f1([1, 2, 3], QRELS)
    assert (p, r) == (pytest.approx(2 / 3), pytest.approx(2 / 3)) and f1 == pytest.approx(2 / 3)
    assert M.set_precision_recall_f1([], QRELS) == (0.0, 0.0, 0.0)


def test_randomization_test():
    assert M.randomization_test([0.5] * 10, [0.5] * 10) == 1.0           # no difference
    assert M.randomization_test([1.0] * 20, [0.0] * 20) < 0.01           # always better
    assert M.randomization_test([1, 0] * 10, [0, 1] * 10) == 1.0         # cancels out


# ---------------------------------------------------------------- labels

def rows(*items):
    return [{"query_id": q, "article_no": a, "relevance": g} for q, a, g in items]


def test_parse_labels():
    qrels, unlabelled, problems = parse_labels(
        rows(("q01", "33", "2"), ("q01", "34", "0"), ("q01", "35", ""), ("q02", "1", "1")),
        {"q01", "q02"})
    assert qrels == {"q01": {33: 2, 34: 0}, "q02": {1: 1}}
    assert unlabelled == {"q01": 1}
    assert problems == []


@pytest.mark.parametrize("row, message", [
    (("q99", "1", "1"), "unknown query_id"),
    (("q01", "999", "1"), "not 1-672"),
    (("q01", "x", "1"), "not 1-672"),
    (("q01", "1", "3"), "must be 0, 1 or 2"),
    (("q01", "1", "yes"), "must be 0, 1 or 2"),
])
def test_parse_labels_reports_bad_rows(row, message):
    _, _, problems = parse_labels(rows(row), {"q01"})
    assert len(problems) == 1 and message in problems[0]


def test_parse_labels_reports_conflicting_duplicates():
    _, _, problems = parse_labels(rows(("q01", "1", "2"), ("q01", "1", "0")), {"q01"})
    assert "labelled 2 and 0" in problems[0]


def test_write_sheet_keeps_existing_labels(tmp_path, monkeypatch):
    monkeypatch.setattr(pool, "LABELS_DIR", tmp_path)
    monkeypatch.setattr(pool, "LABELS_CSV", tmp_path / "labels.csv")
    queries = [{"id": "q01", "query": "x", "en": "x", "type": "t"}]
    articles = {n: {"title": f"t{n}", "text": f"text {n}"} for n in (1, 2, 3)}
    pool.write_sheet(queries, {"q01": {"articles": {"1": {}, "2": {}}}}, articles)

    labelled = pool.read_labels()
    labelled[0]["relevance"] = "2"
    labelled.append({"query_id": "q01", "article_no": "3", "relevance": "1", "notes": "found by hand"})
    with pool.LABELS_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        import csv
        w = csv.DictWriter(f, fieldnames=pool.COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(labelled)

    total, added = pool.write_sheet(queries, {"q01": {"articles": {"1": {}, "2": {}}}}, articles)
    again = {r["article_no"]: r for r in pool.read_labels()}
    assert (total, added) == (3, 0)
    assert again["1"]["relevance"] == "2"
    assert again["3"]["relevance"] == "1" and again["3"]["title"] == "t3"   # manual row kept, filled


# ---------------------------------------------------------------- evaluation

def test_boolean_precision_uses_judged_results_only():
    # 30 results, only 2 judged (sampled): 1 relevant of 2 judged -> P = 0.5
    s = score_boolean(list(range(1, 31)), {1: 2, 2: 0, 40: 2})
    assert s["P"] == 0.5 and s["R"] == 0.5 and s["judged"] == 2


def test_evaluate_end_to_end():
    queries = [{"id": "a", "query": "", "en": "", "type": "t1"},
               {"id": "b", "query": "", "en": "", "type": "t2"},
               {"id": "c", "query": "", "en": "", "type": "t2"}]      # no relevant -> excluded
    qrels = {"a": {1: 2, 2: 1}, "b": {5: 2}, "c": {7: 0}}
    runs = {m: {"a": [1, 3, 2], "b": [4, 6], "c": [7]} for m in RANKED_MODELS}
    runs["bm25"] = {"a": [1, 2], "b": [5], "c": [7]}
    runs["boolean"] = {"a": [1], "b": [], "c": []}
    r = evaluate(runs, qrels, queries)

    assert r["evaluated"] == ["a", "b"] and r["no_relevant"] == ["c"]
    assert r["summary"]["bm25"]["AP"] == pytest.approx(1.0)
    other = "tfidf"
    # a: relevant 1, 2 at ranks 1 and 3 -> (1/1 + 2/3) / 2;  b: relevant 5 not retrieved -> 0
    assert r["summary"][other]["AP"] == pytest.approx(((1 + 2 / 3) / 2 + 0) / 2)
    assert r["missed_by_all"] == {}                        # bm25 found everything
    assert r["summary"]["boolean"]["empty_results"] == 1
    assert r["by_type"]["t2"]["n"] == 1
    assert r["significance"][f"bm25 vs {other}"]["AP"]["diff"] > 0
