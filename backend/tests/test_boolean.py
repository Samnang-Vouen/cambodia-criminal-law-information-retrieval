"""Tests for pipeline/boolean.py. Run from backend/:  python -m pytest

Toy documents use Latin words (the analyzer keeps them unchanged), so the expected
result sets can be read directly from TOY."""
import json
import random

import pytest

from pipeline.boolean import (BooleanRetriever, QuerySyntaxError, difference, intersect,
                              parse, positive_terms, union)
from pipeline.build_index import build
from pipeline.config import ARTICLES_FILE
from pipeline.search import Searcher


def art(no, title, text, book=1):
    return {"id": f"art_{no:03d}", "article_no": no, "title": title, "text": text,
            "path": {"book": {"no": book, "name": ""}}}


TOY = [
    art(1, "theft", "money taken by force"),          # doc 0
    art(2, "murder", "killing with poison"),           # doc 1
    art(3, "fraud", "money taken by lies", book=2),    # doc 2
    art(4, "murder", "killing a minor", book=2),       # doc 3
    art(5, "poison", "selling poison"),                # doc 4
]
DOC_WORDS = [set((a["title"] + " " + a["text"]).split()) for a in TOY]


@pytest.fixture(scope="module")
def toy():
    return BooleanRetriever(build(TOY))


def match(toy, query):
    return toy.search(query).docs


# ---------------------------------------------------------------- operators

def test_single_term_searches_title_and_text(toy):
    assert match(toy, "poison") == [1, 4]          # text of doc 1, title + text of doc 4
    assert match(toy, "money") == [0, 2]


def test_and_or_not(toy):
    assert match(toy, "murder AND poison") == [1]
    assert match(toy, "theft OR fraud") == [0, 2]
    assert match(toy, "murder AND NOT poison") == [3]
    assert match(toy, "NOT money") == [1, 3, 4]


def test_operators_are_case_insensitive(toy):
    assert match(toy, "murder and not poison") == match(toy, "murder AND NOT poison")


def test_adjacent_operands_mean_and(toy):
    assert match(toy, "killing minor") == match(toy, "killing AND minor") == [3]
    assert match(toy, "murder NOT poison") == [3]


def test_precedence_not_and_or(toy):
    # AND binds tighter than OR: theft OR (murder AND poison)
    assert match(toy, "theft OR murder AND poison") == [0, 1]
    assert match(toy, "(theft OR murder) AND poison") == [1]
    # NOT binds tighter than AND: (NOT money) AND killing
    assert match(toy, "NOT money AND killing") == [1, 3]
    assert match(toy, "NOT (money AND killing)") == [0, 1, 2, 3, 4]


def test_double_negation(toy):
    assert match(toy, "NOT NOT poison") == match(toy, "poison")


def test_unknown_word_matches_nothing(toy):
    assert match(toy, "zebra") == []
    assert match(toy, "zebra OR theft") == [0]
    assert match(toy, "NOT zebra") == [0, 1, 2, 3, 4]


def test_results_are_in_document_order(toy):
    docs = match(toy, "poison OR theft OR fraud")
    assert docs == sorted(docs) == [0, 1, 2, 4]


# ---------------------------------------------------------------- parsing

def test_parsed_query_is_fully_bracketed():
    tree, _ = parse("a OR b c AND NOT d")
    assert str(tree) == "(a OR (b AND c AND NOT d))"


@pytest.mark.parametrize("query", ["", "   ", "AND theft", "theft AND", "theft OR", "NOT",
                                   "(theft", "theft)", "()", "theft AND OR fraud"])
def test_syntax_errors(query):
    with pytest.raises(QuerySyntaxError):
        parse(query)


def test_stopword_operands_are_ignored_and_reported():
    tree, ignored = parse("ការពារ និង ខ្លួន")
    assert str(tree) == "(ការពារ AND ខ្លួន)"
    assert ignored == ["និង"]


def test_error_mentions_ignored_words():
    with pytest.raises(QuerySyntaxError, match="និង"):
        parse("ការពារ AND និង")


def test_khmer_operand_with_several_words_needs_all():
    tree, _ = parse("ថ្នាំពុល")
    assert tree.terms == ("ថ្នាំ", "ពុល")


def test_prefixed_noun_operand_only_requires_its_stem():
    tree, _ = parse("ការរំលោភសេពសន្ថវៈ")
    assert tree.terms == ("រំលោភ", "សេពសន្ថវៈ")       # not also "ការរំលោភ"
    tree, _ = parse("ការពារ")                          # not a prefixed noun: kept
    assert tree.terms == ("ការពារ",)


def test_positive_terms_skip_negated_parts():
    tree, _ = parse("murder AND NOT (poison OR NOT minor)")
    assert positive_terms(tree) == ["murder", "minor"]   # NOT NOT minor is positive


# ---------------------------------------------------------------- merge algorithms

def test_merges_match_set_operations():
    rng = random.Random(0)
    for _ in range(300):
        a = sorted(rng.sample(range(50), rng.randint(0, 20)))
        b = sorted(rng.sample(range(50), rng.randint(0, 20)))
        assert intersect(a, b) == sorted(set(a) & set(b))
        assert union(a, b) == sorted(set(a) | set(b))
        assert difference(a, b) == sorted(set(a) - set(b))


def _random_query(rng, depth=0):
    """Random query string + the Python-set function that defines its meaning."""
    vocab = sorted(set().union(*DOC_WORDS)) + ["zebra"]
    kind = rng.choice(["term"] * 3 + (["and", "or", "not"] if depth < 3 else []))
    if kind == "term":
        word = rng.choice(vocab)
        return word, lambda words: word in words
    if kind == "not":
        q, f = _random_query(rng, depth + 1)
        return f"NOT ({q})", lambda words: not f(words)
    (q1, f1), (q2, f2) = _random_query(rng, depth + 1), _random_query(rng, depth + 1)
    if kind == "and":
        return f"({q1}) AND ({q2})", lambda words: f1(words) and f2(words)
    return f"({q1}) OR ({q2})", lambda words: f1(words) or f2(words)


def test_random_queries_agree_with_set_semantics(toy):
    rng = random.Random(42)
    for _ in range(300):
        query, meaning = _random_query(rng)
        expected = [d for d, words in enumerate(DOC_WORDS) if meaning(words)]
        assert match(toy, query) == expected, query


# ---------------------------------------------------------------- Searcher + real corpus

@pytest.fixture(scope="module")
def corpus():
    articles = [json.loads(line) for line in ARTICLES_FILE.open(encoding="utf-8")]
    return Searcher(build(articles))


def test_searcher_boolean_model(corpus):
    hits = corpus.search("ឃាតកម្ម AND ពុល", k=100, model="boolean")
    assert [h.article_no for h in hits] == [201]          # murder by poison
    assert hits[0].score == 1.0


def test_noun_query_finds_article_using_other_form(corpus):
    # Art. 239 is titled បទរំលោភសេពសន្ថវៈ; the query uses the noun ការរំលោភ
    hits = corpus.search("ការរំលោភសេពសន្ថវៈ", k=100, model="boolean")
    assert 239 in [h.article_no for h in hits]


def test_searcher_boolean_book_filter_and_syntax_error(corpus):
    hits = corpus.search("ឃាតកម្ម", k=100, model="boolean", book=2)
    assert hits and all(corpus.index.docs[h.doc]["book"] == 2 for h in hits)
    with pytest.raises(QuerySyntaxError):
        corpus.search("ឃាតកម្ម AND", model="boolean")


def test_corpus_set_identities(corpus):
    b = corpus.boolean
    a, c = set(b.search("លួច").docs), set(b.search("ចោរកម្ម").docs)
    assert set(b.search("លួច OR ចោរកម្ម").docs) == a | c
    assert set(b.search("លួច AND ចោរកម្ម").docs) == a & c
    assert set(b.search("លួច AND NOT ចោរកម្ម").docs) == a - c
    assert len(b.search("NOT លួច").docs) == 672 - len(a)
    assert 201 not in {corpus.index.docs[d]["article_no"] for d in b.search("ឃាតកម្ម NOT ពុល").docs}
