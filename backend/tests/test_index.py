"""Tests for pipeline/build_index.py and pipeline/search.py. Run from backend/:  python -m pytest

The small tests use Latin words: the analyzer passes them through unchanged, so the
expected terms (and therefore the scores) are exact and easy to compute by hand."""
import json
import math

import pytest

from pipeline.build_index import build, load, save
from pipeline.config import ARTICLES_FILE
from pipeline.search import Params, Searcher, bm25_idf


def art(no, title, text, book=1):
    return {"id": f"art_{no:03d}", "article_no": no, "title": title, "text": text,
            "path": {"book": {"no": book, "name": ""}}}


TOY = [
    art(1, "alpha", "beta gamma"),
    art(2, "beta", "alpha gamma"),
    art(3, "delta", "gamma gamma gamma gamma", book=2),
    art(4, "delta", "epsilon", book=2),
]
# avg title length = 1, avg text length = (2 + 2 + 4 + 1) / 4 = 2.25


@pytest.fixture(scope="module")
def toy():
    return Searcher(build(TOY))


def scores(searcher, query, **kw):
    return {h.article_no: h.score for h in searcher.search(query, k=100, **kw)}


# ---------------------------------------------------------------- index structure

def test_postings_and_lengths():
    index = build(TOY)
    text = index.fields["text"]
    assert text.postings["gamma"] == {0: 1, 1: 1, 2: 4}
    assert text.lengths == [2, 2, 4, 1]
    assert index.fields["title"].postings["delta"] == {2: 1, 3: 1}
    assert text.avg_length == 2.25
    assert index.docs[2] == {"id": "art_003", "article_no": 3, "book": 2}


def test_save_load_roundtrip(tmp_path):
    index = build(TOY, "abc")
    save(index, tmp_path / "index.json")
    loaded = load(tmp_path / "index.json", check_source=False)
    assert loaded.docs == index.docs
    assert loaded.source_sha256 == "abc"
    for name in index.fields:
        assert loaded.fields[name].postings == index.fields[name].postings
        assert loaded.fields[name].lengths == index.fields[name].lengths


# ---------------------------------------------------------------- BM25 properties

def test_bm25_matches_formula(toy):
    # query "epsilon": only doc 4, in text (tf 1, length 1)
    p = Params()
    idf = math.log(1 + (4 - 1 + 0.5) / (1 + 0.5))
    tf = 1.0 * 1 / (1 - 0.75 + 0.75 * 1 / 2.25)          # text weight 1, length-normalized
    expected = idf * tf * (p.k1 + 1) / (tf + p.k1)
    assert scores(toy, "epsilon")[4] == pytest.approx(expected)


def test_rare_terms_weigh_more():
    assert bm25_idf(4, 1) > bm25_idf(4, 3) > 0
    assert bm25_idf(672, 672) > 0            # never negative, even in every document


def test_rare_term_beats_frequent_term(toy):
    ranked = [h.article_no for h in toy.search("epsilon gamma")]
    assert ranked[0] == 4                    # epsilon (df 1) > gamma ×4 (df 3)


def test_term_frequency_saturates(toy):
    p = Params()
    s = scores(toy, "gamma")
    assert s[3] > s[2]                       # tf 4 beats tf 1 ...
    assert s[3] < 4 * s[2]                   # ... but not 4 times
    assert s[3] < bm25_idf(4, 3) * (p.k1 + 1)  # upper bound of BM25


def test_title_match_beats_text_match(toy):
    s = scores(toy, "alpha")
    assert s[1] > s[2]                       # alpha in title (doc 1) vs in text (doc 2)


def test_shorter_document_wins_with_same_tf():
    index = build([art(1, "x", "money a b c d e f g"), art(2, "y", "money a")])
    s = scores(Searcher(index), "money")
    assert s[2] > s[1]


def test_bm25f_term_in_both_fields_saturates_once():
    # BM25F combines TF before saturation, so the score stays below the single-term bound
    index = build([art(1, "money", "money money"), art(2, "x", "y")])
    s = scores(Searcher(index), "money")
    assert s[1] < bm25_idf(2, 1) * (Params().k1 + 1)


# ---------------------------------------------------------------- search behaviour

def test_no_match_and_empty_query(toy):
    assert toy.search("zzz") == []
    assert toy.search("") == []
    assert toy.search("។ ៖") == []


def test_book_filter(toy):
    assert {h.article_no for h in toy.search("gamma", book=2)} == {3}


def test_results_explain_score(toy):
    for h in toy.search("alpha gamma"):
        assert h.score == pytest.approx(sum(h.terms.values()) + h.ngram_score)


def test_unknown_model(toy):
    with pytest.raises(ValueError):
        toy.search("alpha", model="bm26")


def test_cosine_scores_between_0_and_1(toy):
    s = scores(toy, "alpha gamma", model="cosine")
    assert s and all(0 < v <= 1 + 1e-9 for v in s.values())


def test_tfidf_matches_formula(toy):
    # "gamma" (df 3 of 4): doc 3 has tf 4 in its text → (1 + ln 4) · ln(4/3)
    s = scores(toy, "gamma", model="tfidf")
    assert s[3] == pytest.approx((1 + math.log(4)) * math.log(4 / 3))
    assert s[1] == pytest.approx(math.log(4 / 3))


def test_tfidf_has_no_length_normalization_cosine_has():
    # third doc without "money", so its idf ln(N/df) is not 0
    index = build([art(1, "x", "money a b c d e f g"), art(2, "y", "money a"), art(3, "z", "a")])
    s = Searcher(index)
    tfidf, cos =scores(s, "money", model="tfidf"), scores(s, "money", model="cosine")
    assert tfidf[1] == tfidf[2]          # same tf, length ignored
    assert cos[2] > cos[1]               # shorter document wins


def test_ranking_does_not_depend_on_term_order(monkeypatch):
    # Found by random search: without rounding scores before tie-breaking, articles 2 and 6
    # (equal scores up to ~1e-16) swapped places depending on the order in which the term
    # contributions were summed, and that order used to come from set(), which changes
    # between Python runs.
    import itertools
    import pipeline.search as search
    docs = [art(1, "zeta", "theta zeta zeta"), art(2, "epsilon eta", "delta zeta epsilon zeta"),
            art(3, "beta", "alpha alpha alpha delta"), art(4, "delta", "beta gamma eta"),
            art(5, "delta zeta", "delta zeta alpha gamma"), art(6, "delta zeta", "eta delta beta")]
    s = Searcher(build(docs), Params(k1=0.0))
    rankings = set()
    for order in itertools.permutations(["eta", "delta", "zeta", "theta"]):
        monkeypatch.setattr(search, "unique", lambda terms, o=order: list(o))
        rankings.add(tuple(h.article_no for h in s.search("eta delta zeta theta")))
    assert rankings == {(1, 2, 6, 4, 5, 3)}


# ---------------------------------------------------------------- real corpus

@pytest.fixture(scope="module")
def corpus():
    articles = [json.loads(line) for line in ARTICLES_FILE.open(encoding="utf-8")]
    return Searcher(build(articles))


def test_corpus_index_has_every_article(corpus):
    assert corpus.n == 672
    assert [d["article_no"] for d in corpus.index.docs] == list(range(1, 673))


def test_title_query_finds_its_article(corpus):
    for model in ("bm25", "bm25_ngram", "tfidf", "cosine"):
        assert corpus.search("ការការពារស្របច្បាប់", model=model)[0].article_no == 33, model


def test_self_defence_query(corpus):
    top3 = [h.article_no for h in corpus.search("ការពារខ្លួន", k=3)]
    assert 33 in top3


def test_ngrams_recover_ocr_damaged_articles(corpus):
    # Articles 193 and 267 contain ទ្រព្យសម្បត្តិ only as the OCR misspelling ទ្រព្យសម្ប្ិ
    words = {h.article_no for h in corpus.search("ទ្រព្យសម្បត្តិ", k=672, model="bm25")}
    grams = {h.article_no for h in corpus.search("ទ្រព្យសម្បត្តិ", k=672, model="bm25_ngram")}
    assert not {193, 267} & words
    assert {193, 267} <= grams
