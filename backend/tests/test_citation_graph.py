"""Tests for pipeline/citation_graph.py. Run from backend/:  python -m pytest"""
import json
import random

import networkx as nx
import pytest

from pipeline.citation_graph import (build_edges, extract_citations, hits, pagerank,
                                     title_match)
from pipeline.config import ARTICLES_FILE

# toy code: titles are real Khmer phrases, so the title check works as on the corpus
TITLES = {1: "និយមន័យ", 2: "បទទារុណកម្ម", 3: "បទឃាតកម្មគិតទុកជាមុន", 4: "បទរំលោភសេពសន្ថវៈ",
          5: "ការរឹបអូសទ្រព្យសម្បត្តិ", 6: "ការបិទផ្សាយសេចក្តីសម្រេច", 7: "បទលួចរត់ប្រព្រឹត្តដោយអ្នកយាម"}


def code(texts: dict[int, str]) -> list[dict]:
    return [{"article_no": n, "title": TITLES[n], "text": texts.get(n, "")} for n in sorted(TITLES)]


def cites(texts):
    citations, dropped = extract_citations(code(texts))
    return [(c.source, c.target, c.status) for c in citations], dropped


# ---------------------------------------------------------------- extraction

def test_title_match():
    assert title_match("បទទារុណកម្ម", " កេបទទារុណកម្ម ") == 1.0
    assert title_match("បទទារុណកម្ម", " ការរឹបអូសទ្រព្យ ") < 0.5
    assert title_match("ការបិទផ្សាយសេចក្តីសម្រេច", " ការបិទផ្សាយ ") == 1.0   # title cut short


def test_citation_verified_by_bracketed_title():
    found, _ = cites({1: "ដូចមានចែងក្នុងមាត្រា ២ បេទទារុណកម្ម នៃក្រមនេះ ។"})
    assert found == [(1, 2, "verified")]


def test_ocr_digit_corrected_by_title():
    # written ៧ (7) but the bracketed title is that of article 3: ៣ and ៧ are confused by the OCR
    found, _ = cites({1: "តាមមាត្រា ៧ បេទឃាតកម្មគិតទុកជាមុន នៃក្រមនេះ ។"})
    assert found == [(1, 3, "corrected")]


def test_range_is_expanded():
    found, _ = cites({1: "ក្នុងមាត្រា ៣ បេទឃាតកម្មគិតទុកជាមុនដល់មាត្រា ៦ ការបិទផ្សាយសេចក្តីសម្រេច នៃក្រមនេះ"})
    assert found == [(1, 3, "verified"), (1, 4, "range"), (1, 5, "range"), (1, 6, "verified")]


def test_other_law_and_unknown_numbers_dropped():
    found, dropped = cites({1: "តាមមាត្រា ២ នៃរដ្ឋធម្មនុញ្ញនៃព្រះរាជាណាចក្រកម្ពុជា ។ មាត្រា ៩៩ អ្វីមួយ ។"})
    assert found == []
    assert [d["reason"] for d in dropped] == ["other law", "not an article of this code"]


def test_citation_without_matching_title_is_kept_but_flagged():
    found, _ = cites({1: "តាមមាត្រា ២ ខាងលើ ។"})
    assert found == [(1, 2, "unverified")]


def test_self_citation_is_not_an_edge():
    citations, _ = extract_citations(code({2: "បទប្បញ្ញត្តិនៃមាត្រា ២ បេទទារុណកម្ម នៃក្រមនេះ"}))
    assert citations and build_edges(citations) == set()


# ---------------------------------------------------------------- algorithms

def random_graph(seed, n=30, m=60):
    rng = random.Random(seed)
    nodes = list(range(1, n + 1))
    edges = {(rng.choice(nodes), rng.choice(nodes)) for _ in range(m)}
    return nodes, {(u, v) for u, v in edges if u != v}


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_pagerank_matches_networkx(seed):
    nodes, edges = random_graph(seed)
    g = nx.DiGraph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    expected = nx.pagerank(g, alpha=0.85, tol=1e-12)
    ours = pagerank(nodes, edges)
    assert sum(ours.values()) == pytest.approx(1.0)
    for v in nodes:
        assert ours[v] == pytest.approx(expected[v], abs=1e-8)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_hits_matches_networkx(seed):
    nodes, edges = random_graph(seed)
    g = nx.DiGraph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    exp_hub, exp_auth = nx.hits(g, max_iter=1000, tol=1e-12)
    hub, auth = hits(nodes, edges)
    for v in nodes:
        assert hub[v] == pytest.approx(exp_hub[v], abs=1e-6)
        assert auth[v] == pytest.approx(exp_auth[v], abs=1e-6)


def test_exclusive_citation_counts_more_than_shared_one():
    # 2 is cited by one article that cites only 2; 3 is cited by one article citing 3 and 4
    nodes = [1, 2, 3, 4, 5, 6]
    pr = pagerank(nodes, {(1, 2), (5, 3), (5, 4)})
    assert pr[3] == pytest.approx(pr[4])
    assert pr[2] > pr[3] > pr[6]


def test_hub_and_authority():
    # 1 and 2 cite 5 and 6 (hubs); 3 cites only 5
    hub, auth = hits([1, 2, 3, 4, 5, 6], {(1, 5), (1, 6), (2, 5), (2, 6), (3, 5)})
    assert auth[5] > auth[6] > auth[4] == 0
    assert hub[1] == hub[2] > hub[3] > hub[4] == 0


# ---------------------------------------------------------------- real corpus

@pytest.fixture(scope="module")
def corpus():
    articles = [json.loads(line) for line in ARTICLES_FILE.open(encoding="utf-8")]
    citations, dropped = extract_citations(articles)
    return articles, citations, dropped


def test_corpus_citations(corpus):
    _, citations, dropped = corpus
    assert all(c.status in ("verified", "range") for c in citations)    # every number checked
    assert {d["reason"] for d in dropped} == {"other law"}               # Constitution
    assert {(c.source, c.target) for c in citations if c.status == "range"} >= {(199, 201), (199, 204)}


def test_corpus_scores(corpus):
    articles, citations, _ = corpus
    nodes = [a["article_no"] for a in articles]
    edges = build_edges(citations)
    pr = pagerank(nodes, edges)
    hub, auth = hits(nodes, edges)
    top = lambda s: sorted(nodes, key=lambda v: -s[v])
    assert top(pr)[0] == 210                       # torture, cited only by 211-215
    assert {42, 181} <= set(top(auth)[:5])          # legal persons + additional penalties
    assert top(hub)[0] == 230                       # a "legal persons" article citing 14
