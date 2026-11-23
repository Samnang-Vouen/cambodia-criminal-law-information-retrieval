"""
Text analyzer: turns Khmer text into index terms.

Used at index time (build_index.py, on every article) AND at search time (the API, on
every query). Both sides must call the same functions, otherwise a query term can never
be equal to an index term. See docs/text_analysis.md.

Steps
  1. normalize()    NFC, remove invisible characters, unify spelling variants
                    (្ដ → ្ត, ឲ → ឱ), lowercase Latin.
  2. segment()      split the text into runs (Khmer letters / numbers / Latin) at spaces,
                    punctuation and line breaks; split each Khmer run into words with
                    khmernltk's CRF model. Every token keeps its start/end offset.
  3. word_terms()   drop stopwords and OCR fragments, convert Khmer digits to 0-9,
                    add the stem of prefixed nouns (ការការពារ → also ការពារ).
  4. ngram_terms()  overlapping n-grams of Khmer Character Clusters (KCC) inside each run.
                    Robust to OCR misspellings and word-segmentation errors.

Run from backend/:
  python -m pipeline.analyzer "ការការពារស្របច្បាប់"   analyse one text
  python -m pipeline.analyzer --stats                 statistics over articles.jsonl
"""
import json
import logging
import re
import sys
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache

from khmernltk import word_tokenize
from khmernltk.utils.data import seg_kcc

from pipeline.config import ARTICLES_FILE, DATA_DIR

logging.getLogger("khmer-nltk").setLevel(logging.WARNING)   # hide "Loaded model from ..."

# ---------------------------------------------------------------- 1. normalization

# zero-width space/joiners, word joiner, BOM, soft hyphen, Khmer inherent vowels (U+17B4/5)
INVISIBLE = re.compile("[​‌‍⁠﻿­឴឵]")
# Same word, two spellings. Both pairs look identical on screen; the corpus (from OCR)
# uses the right-hand form almost always (្ត 2459× vs ្ដ 16×, ឱ 544× vs ឲ 6×).
SPELLING_VARIANTS = [
    ("្ដ", "្ត"),   # coeng DA ្ដ → coeng TA ្ត   (សេចក្ដី → សេចក្តី)
    ("ឲ", "ឱ"),               # ឲ → ឱ                        (ឲ្យ → ឱ្យ)
]
KHMER_TO_ASCII_DIGITS = str.maketrans("០១២៣៤៥៦៧៨៩", "0123456789")


def normalize(text: str) -> str:
    """Search normalization. Apart from deleting invisible characters it keeps the text
    length, so offsets into articles.jsonl text (already cleaned) stay valid."""
    text = unicodedata.normalize("NFC", text)
    text = INVISIBLE.sub("", text)
    for variant, standard in SPELLING_VARIANTS:
        text = text.replace(variant, standard)
    return text.lower()


# ---------------------------------------------------------------- 2. segmentation

# A Khmer run = consonants, vowels and signs (U+1780–U+17D3, U+17DD). Punctuation
# (។ ៖ ៗ, U+17D4–U+17DA), digits (U+17E0–U+17E9), spaces and "\n" all end a run.
# Amounts use a space (or . ,) as thousands separator: ៥០ ០០០ ០០០ រៀល = 50,000,000 riel,
# so digit groups of exactly 3 after a number belong to that number.
RUN = re.compile(
    r"(?P<khmer>[ក-៓៝]+)"
    r"|(?P<number>[0-9០-៩]+(?:[ .,][0-9០-៩]{3}(?![0-9០-៩]))*)"
    r"|(?P<latin>[a-z]+)"
)
NOT_DIGIT = re.compile(r"[^0-9]")


@dataclass(frozen=True)
class Token:
    text: str    # surface form, == normalize(input)[start:end]
    start: int
    end: int
    kind: str    # "khmer" | "number" | "latin"


@lru_cache(maxsize=100_000)
def _split_words(run: str) -> tuple[str, ...]:
    """Khmer run → words, with khmernltk's CRF word segmenter. Cached: the same
    phrases (ផ្តន្ទាទោសដាក់ពន្ធនាគារ…) repeat hundreds of times in the code."""
    # khmernltk may insert U+200B (its own word mark) via its correction table
    words = tuple(w.replace("​", "") for w in word_tokenize(run))
    words = tuple(w for w in words if w)
    if "".join(words) != run:      # never happened on the corpus; keeps offsets exact
        return (run,)
    return words


def segment(text: str) -> list[Token]:
    """All tokens of normalize(text), in order, with offsets. Nothing is filtered here."""
    text = normalize(text)
    tokens = []
    for m in RUN.finditer(text):
        kind, start = m.lastgroup, m.start()
        if kind != "khmer":
            tokens.append(Token(m.group(), start, m.end(), kind))
            continue
        for word in _split_words(m.group()):
            tokens.append(Token(word, start, start + len(word), kind))
            start += len(word)
    return tokens


# ---------------------------------------------------------------- 3. word terms

# Function words, chosen by hand from the highest document-frequency tokens of the corpus
# (python -m pipeline.analyzer --stats). Frequent LEGAL words (ផ្តន្ទាទោស, ពន្ធនាគារ,
# បទល្មើស, ទោស…) are NOT stopwords: they matter in queries, BM25's IDF already
# gives them a low weight. ខ្លួន ("self") is kept on purpose: ការពារខ្លួន = self-defence.
STOPWORDS = frozenset(normalize(w) for w in """
    ដែល ត្រូវ ត្រូវបាន ឬ ជា ទៅ ពី មាន និង នៃ នេះ នោះ ដោយ ក្នុង នៅក្នុង នៅ របស់
    អាច បាន ចំពោះ នូវ ដល់ តាម លើ ឱ្យ មិន នឹង សម្រាប់ ដើម្បី យ៉ាង ទៀត សោត
    ដូចតទៅ ថា ទេ ណា គេ តែ បើ ក៏ កាលបើ ហើយ គឺ ពុំ ឡើយ ដែរ ទាំង ផង
""".split())

# Prefixes that turn a verb/adjective into a noun. The noun is often one CRF token
# (ការការពារ, ការផ្តន្ទាទោស), so a query for the verb (ការពារ) would not match it.
NOUN_PREFIXES = ("ការ", "សេចក្តី", "អំពើ", "អ្នក", "ភាព")
# A stem needs 2+ clusters or a vowel: ការណ៍ → "ណ៍" is not a stem, ការធ្វើ → "ធ្វើ" is
DEPENDENT_VOWEL = re.compile("[ា-ៅ]")
# Words that start like a prefix but are NOT prefix + word: ការពារ (to defend) is not
# ការ + ពារ. Without this, "ការពារ" would also produce "ពារ" and BM25 would count the
# same concept twice (seen in the term contributions of the query ការពារខ្លួន).
NOT_PREFIXED = frozenset(normalize(w) for w in "ការពារ ការងារ".split())


def _is_word(token: str) -> bool:
    """A Khmer word starts with a consonant or an independent vowel (U+1780–U+17B3) and is
    longer than one character. Rejects OCR fragments like "្នា" (from a broken ឆ្នាំ) and
    lone consonants like "រ"."""
    return len(token) > 1 and "ក" <= token[0] <= "ឳ"


def stem(word: str) -> str | None:
    """The word without its noun prefix (ការការពារ → ការពារ), or None."""
    if word in NOT_PREFIXED:
        return None
    for prefix in NOUN_PREFIXES:
        rest = word[len(prefix):]
        # _is_word: ការិយាល័យ (office) must not give "ិយាល័យ"
        if (word.startswith(prefix) and _is_word(rest)
                and (len(seg_kcc(rest)) >= 2 or DEPENDENT_VOWEL.search(rest))):
            return rest
    return None


def word_terms(text: str) -> list[str]:
    """Index/query terms from words, in text order (duplicates kept: they are term frequency)."""
    terms = []
    for tok in segment(text):
        if tok.kind == "number":
            digits = NOT_DIGIT.sub("", tok.text.translate(KHMER_TO_ASCII_DIGITS))
            terms.append(digits.lstrip("0") or "0")      # "៥០ ០០០ ០០០" → "50000000"
        elif tok.kind == "latin":
            terms.append(tok.text)
        elif _is_word(tok.text) and tok.text not in STOPWORDS:
            terms.append(tok.text)
            base = stem(tok.text)
            if base and base not in STOPWORDS:
                terms.append(base)
    return terms


# ---------------------------------------------------------------- 4. KCC n-grams

def ngram_terms(text: str, n: int = 2) -> list[str]:
    """Overlapping n-grams of Khmer Character Clusters within each Khmer run.
    ព្រហ្មទណ្ឌ → ព្រ|ហ្ម|ទ|ណ្ឌ → ព្រហ្ម, ហ្មទ, ទណ្ឌ. A run shorter than n is one gram."""
    grams = []
    for m in RUN.finditer(normalize(text)):
        if m.lastgroup != "khmer":
            continue
        kcc = seg_kcc(m.group())
        if len(kcc) <= n:
            grams.append(m.group())
        else:
            grams.extend("".join(kcc[i:i + n]) for i in range(len(kcc) - n + 1))
    return grams


@dataclass
class Analysis:
    words: list[str]
    ngrams: list[str]


def analyze(text: str) -> Analysis:
    return Analysis(words=word_terms(text), ngrams=ngram_terms(text))


# ---------------------------------------------------------------- CLI

STATS_FILE = DATA_DIR / "interim" / "analysis" / "term_stats.json"


def show(text: str):
    print("normalized :", normalize(text))
    print("tokens     :", " | ".join(f"{t.text}[{t.start}:{t.end}]" for t in segment(text)))
    print("word terms :", " ".join(word_terms(text)))
    print("kcc bigrams:", " ".join(ngram_terms(text)))


def corpus_stats():
    articles = [json.loads(line) for line in ARTICLES_FILE.open(encoding="utf-8")]
    t0 = time.perf_counter()
    raw, words, grams = Counter(), Counter(), Counter()
    word_df, gram_df = Counter(), Counter()
    stems = 0
    for a in articles:
        text = a["title"] + "\n" + a["text"]
        toks = segment(text)
        raw.update(t.text for t in toks)
        w, g = word_terms(text), ngram_terms(text)
        stems += sum(1 for t in toks if t.kind == "khmer" and _is_word(t.text)
                     and t.text not in STOPWORDS and stem(t.text))
        words.update(w)
        grams.update(g)
        word_df.update(set(w))
        gram_df.update(set(g))
    secs = time.perf_counter() - t0

    raw_khmer = sum(c for t, c in raw.items() if RUN.fullmatch(t).lastgroup == "khmer")
    stop = sum(raw[s] for s in STOPWORDS)
    fragments = sum(c for t, c in raw.items()
                    if RUN.fullmatch(t).lastgroup == "khmer" and not _is_word(t))
    summary = {
        "articles": len(articles),
        "seconds": round(secs, 1),
        "tokens": sum(raw.values()),
        "khmer_tokens": raw_khmer,
        "stopword_tokens_removed": stop,
        "fragment_tokens_removed": fragments,
        "stems_added": stems,
        "word_terms": sum(words.values()),
        "word_vocabulary": len(words),
        "kcc_bigrams": sum(grams.values()),
        "kcc_bigram_vocabulary": len(grams),
        "stopwords_not_in_corpus": sorted(s for s in STOPWORDS if s not in raw),
    }
    for k, v in summary.items():
        print(f"{k:26}: {v}")
    print("top 40 word terms by document frequency:")
    print("  " + " ".join(f"{t}:{c}" for t, c in word_df.most_common(40)))

    STATS_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATS_FILE.write_text(json.dumps({
        "summary": summary,
        "word_df": dict(word_df.most_common()),
        "kcc_bigram_df": dict(gram_df.most_common(2000)),
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"written: {STATS_FILE}")


if __name__ == "__main__":
    if sys.argv[1:] == ["--stats"]:
        corpus_stats()
    elif sys.argv[1:]:
        show(" ".join(sys.argv[1:]))
    else:
        print(__doc__)
