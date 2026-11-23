"""Tests for pipeline/analyzer.py. Run from backend/:  python -m pytest"""
import json

import pytest

from pipeline.analyzer import ngram_terms, normalize, segment, word_terms
from pipeline.config import ARTICLES_FILE


# ---------------------------------------------------------------- normalization

def test_spelling_variants_are_unified():
    assert normalize("សេចក្ដី") == normalize("សេចក្តី") == "សេចក្តី"
    assert normalize("ឲ្យ") == "ឱ្យ"


def test_invisible_characters_removed():
    assert normalize("ការ​ពារ﻿") == "ការពារ"


def test_query_with_variant_spelling_matches_document():
    assert word_terms("សេចក្ដីសម្រេច") == word_terms("សេចក្តីសម្រេច")


# ---------------------------------------------------------------- segmentation

def test_offsets_point_into_normalized_text():
    text = "ការការពារស្របច្បាប់ ត្រូវមានលក្ខខណ្ឌដូចខាងក្រោមនេះ ៖\n១- បទល្មើស"
    norm = normalize(text)
    for tok in segment(text):
        assert norm[tok.start:tok.end] == tok.text


def test_line_break_separates_words():
    # khmernltk deletes "\n", which would glue the last word of a paragraph to the next one
    tokens = [t.text for t in segment("ច្បាប់\nបុគ្គល")]
    assert tokens == ["ច្បាប់", "បុគ្គល"]


def test_punctuation_is_not_a_token():
    tokens = [t.text for t in segment("ទេ ។ ដូចខាងក្រោមនេះ ៖ ១- ក")]
    assert "។" not in tokens and "៖" not in tokens and "-" not in tokens


# ---------------------------------------------------------------- word terms

def test_stopwords_removed_but_legal_words_kept():
    terms = word_terms("បុគ្គលណាដែលប្រព្រឹត្តបទល្មើស និង ផ្តន្ទាទោស")
    assert "ដែល" not in terms and "និង" not in terms and "ណា" not in terms
    assert {"បុគ្គល", "ប្រព្រឹត្ត", "បទល្មើស", "ផ្តន្ទាទោស"} <= set(terms)


def test_prefixed_noun_also_indexes_its_stem():
    terms = word_terms("ការការពារស្របច្បាប់")
    assert "ការការពារ" in terms and "ការពារ" in terms


def test_verb_query_matches_noun_in_document():
    doc = set(word_terms("ការការពារស្របច្បាប់"))
    assert set(word_terms("ការពារ")) & doc


def test_short_remainder_is_not_a_stem():
    assert "ណ៍" not in word_terms("ការណ៍")


def test_one_syllable_verb_is_a_stem():
    assert word_terms("ការធ្វើ") == ["ការធ្វើ", "ធ្វើ"]
    assert "ឃុំ" in word_terms("ការឃុំខ្លួន")


def test_word_that_only_looks_prefixed_is_not_stemmed():
    assert word_terms("ការពារ") == ["ការពារ"]          # not ការ + ពារ
    assert word_terms("ការងារ") == ["ការងារ"]


def test_stem_must_be_a_word():
    assert word_terms("ការិយាល័យ") == ["ការិយាល័យ"]    # no "ិយាល័យ"


def test_ocr_fragments_dropped():
    assert word_terms("្នា") == []     # starts with a coeng: broken ឆ្នាំ
    assert word_terms("រ") == []       # lone consonant


def test_numbers_become_ascii():
    assert word_terms("មាត្រា ៣៣") == ["មាត្រា", "33"]
    assert word_terms("មាត្រា 33") == ["មាត្រា", "33"]


def test_amount_with_thousands_separator_is_one_number():
    assert "50000000" in word_terms("ពី ៥០ ០០០ ០០០ រៀល")
    assert word_terms("ចំណុច ១ ២០ ៣") == ["ចំណុច", "1", "20", "3"]


# ---------------------------------------------------------------- KCC n-grams

def test_kcc_bigrams():
    assert ngram_terms("ព្រហ្មទណ្ឌ") == ["ព្រហ្ម", "ហ្មទ", "ទណ្ឌ"]


def test_short_run_is_one_gram():
    assert ngram_terms("ឬ") == ["ឬ"]


def test_ngrams_match_ocr_damaged_word():
    # OCR dropped the final ឌ: word terms no longer match, bigrams still do
    assert not set(word_terms("ព្រហ្មទណ្")) & set(word_terms("ព្រហ្មទណ្ឌ"))
    assert set(ngram_terms("ព្រហ្មទណ្")) & set(ngram_terms("ព្រហ្មទណ្ឌ"))


def test_ngrams_do_not_cross_spaces():
    assert "ខបុ" not in "".join(ngram_terms("ច្បាប់ បុគ្គល"))
    assert all(" " not in g for g in ngram_terms("ច្បាប់ បុគ្គល"))


# ---------------------------------------------------------------- whole corpus

@pytest.fixture(scope="module")
def articles():
    return [json.loads(line) for line in ARTICLES_FILE.open(encoding="utf-8")]


def test_every_article_analyzes_with_valid_offsets(articles):
    for a in articles:
        text = a["title"] + "\n" + a["text"]
        norm = normalize(text)
        tokens = segment(text)
        assert tokens, a["id"]
        for tok in tokens:
            assert norm[tok.start:tok.end] == tok.text, a["id"]
        assert word_terms(text), a["id"]
        assert ngram_terms(text), a["id"]


def test_self_defence_article_contains_query_terms(articles):
    art33 = next(a for a in articles if a["article_no"] == 33)
    doc = set(word_terms(art33["title"] + "\n" + art33["text"]))
    assert set(word_terms("ការពារខ្លួន")) <= doc
