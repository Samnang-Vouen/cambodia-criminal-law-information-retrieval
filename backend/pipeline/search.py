"""
Ranking models over the inverted index built by build_index.py.

  bm25        BM25F over the word fields (title weighted higher than text)
  bm25_ngram  bm25 + a lower-weighted BM25 over KCC bigrams (helps with OCR misspellings)
  tfidf       TF-IDF score: sum of (1 + ln tf) · ln(N/df) over the query terms, no length
              normalization (title + text words)
  cosine      TF-IDF vectors compared with cosine similarity (title + text words)
  boolean     AND / OR / NOT, unranked: matching articles in article order (pipeline/boolean.py)

Formulas and parameter choices: docs/indexing_and_ranking.md, docs/boolean_retrieval.md.
How the ranked models differ on the same queries: docs/model_comparison.md.
The parameters below are standard starting values, NOT tuned yet: the evaluation step will
tune them.

Run from backend/:
  python -m pipeline.search "ការពារខ្លួន"
  python -m pipeline.search "ការពារខ្លួន" --model cosine -k 5 --book 1
  python -m pipeline.search "ឃាតកម្ម AND NOT ពុល" --model boolean
"""
import argparse
import json
import math
from collections import Counter
from dataclasses import dataclass, field

from pipeline.analyzer import ngram_terms, word_terms
from pipeline.boolean import BooleanRetriever, QuerySyntaxError
from pipeline.build_index import Index, load
from pipeline.config import ARTICLES_FILE

MODELS = ("bm25", "bm25_ngram", "tfidf", "cosine", "boolean")
RANKED_MODELS = ("bm25", "bm25_ngram", "tfidf", "cosine")
WORD_FIELDS = ("title", "text")


@dataclass
class Params:
    k1: float = 1.2                                   # TF saturation (common range 1.2–2.0)
    b: dict = field(default_factory=lambda: {"title": 0.75, "text": 0.75, "ngram": 0.75})
    weight: dict = field(default_factory=lambda: {"title": 2.0, "text": 1.0})  # BM25F field weights
    ngram_weight: float = 0.3                         # share of the bigram score in bm25_ngram


@dataclass
class Hit:
    doc: int
    article_no: int
    id: str
    score: float
    terms: dict[str, float]      # word term -> its contribution to the score
    ngram_score: float = 0.0     # total contribution of the bigram field (bm25_ngram only)


def unique(terms: list[str]) -> list[str]:
    """Distinct terms in query order. Not set(): set order changes between runs (string
    hashing is randomized), which changes float summation order and so tie-breaking."""
    return list(dict.fromkeys(terms))


def bm25_idf(n_docs: int, df: int) -> float:
    """Robertson–Spärck Jones IDF, +1 inside the log so it is never negative
    (the variant used by Lucene/Elasticsearch)."""
    return math.log(1 + (n_docs - df + 0.5) / (df + 0.5))


class Searcher:
    def __init__(self, index: Index, params: Params | None = None):
        self.index = index
        self.params = params or Params()
        self.n = index.n_docs
        self.f = index.fields
        self._tfidf_ready = False
        self.boolean = BooleanRetriever(index)

    # ------------------------------------------------------------ BM25F (word fields)

    def word_df(self, term: str) -> int:
        """Documents containing the term in ANY word field (BM25F uses one IDF per term)."""
        docs = set()
        for name in WORD_FIELDS:
            docs.update(self.f[name].postings.get(term, ()))
        return len(docs)

    def _bm25f(self, terms: list[str]) -> dict[int, dict[str, float]]:
        """BM25F: per document, first combine the length-normalized, weighted TF of all
        fields, THEN saturate once with k1. (Summing per-field BM25 scores instead would
        saturate each field separately and over-reward a term found in both.)"""
        p, contrib = self.params, {}
        for term in unique(terms):
            df = self.word_df(term)
            if df == 0:
                continue
            idf = bm25_idf(self.n, df)
            tf_combined: dict[int, float] = {}
            for name in WORD_FIELDS:
                f, b = self.f[name], p.b[name]
                avg = f.avg_length
                for doc, tf in f.postings.get(term, {}).items():
                    norm = 1 - b + b * f.lengths[doc] / avg
                    tf_combined[doc] = tf_combined.get(doc, 0.0) + p.weight[name] * tf / norm
            for doc, tf in tf_combined.items():
                contrib.setdefault(doc, {})[term] = idf * tf * (p.k1 + 1) / (tf + p.k1)
        return contrib

    def _bm25_field(self, name: str, terms: list[str]) -> dict[int, float]:
        """Classic single-field BM25."""
        p, f, scores = self.params, self.f[name], {}
        b, avg = p.b[name], self.f[name].avg_length
        for term in unique(terms):
            postings = f.postings.get(term)
            if not postings:
                continue
            idf = bm25_idf(self.n, len(postings))
            for doc, tf in postings.items():
                norm = 1 - b + b * f.lengths[doc] / avg
                scores[doc] = scores.get(doc, 0.0) + idf * tf * (p.k1 + 1) / (tf + p.k1 * norm)
        return scores

    # ------------------------------------------------------------ TF-IDF and cosine

    def _prepare_tfidf(self):
        """Document vectors over title + text: weight = (1 + ln tf) · ln(N / df)."""
        tf: dict[str, dict[int, int]] = {}
        for name in WORD_FIELDS:
            for term, postings in self.f[name].postings.items():
                t = tf.setdefault(term, {})
                for doc, n in postings.items():
                    t[doc] = t.get(doc, 0) + n
        self._idf = {term: math.log(self.n / len(p)) for term, p in tf.items()}
        self._dw = {term: {d: (1 + math.log(n)) * self._idf[term] for d, n in p.items()}
                    for term, p in tf.items()}
        sq = [0.0] * self.n
        for p in self._dw.values():
            for d, w in p.items():
                sq[d] += w * w
        self._norm = [math.sqrt(s) for s in sq]
        self._tfidf_ready = True

    def _tfidf(self, terms: list[str]) -> dict[int, dict[str, float]]:
        """TF-IDF score (Manning et al., IIR eq. 6.9): sum of the document's tf-idf weights
        of the query terms. No length normalization and no saturation beyond the log."""
        if not self._tfidf_ready:
            self._prepare_tfidf()
        contrib = {}
        for term in unique(terms):
            for doc, dw in self._dw.get(term, {}).items():
                if dw:
                    contrib.setdefault(doc, {})[term] = dw
        return contrib

    def _cosine(self, terms: list[str]) -> dict[int, dict[str, float]]:
        """Cosine similarity between the query's and the document's tf-idf vectors."""
        if not self._tfidf_ready:
            self._prepare_tfidf()
        q = {t: (1 + math.log(n)) * self._idf[t] for t, n in Counter(terms).items() if t in self._idf}
        q_norm = math.sqrt(sum(w * w for w in q.values()))
        contrib = {}
        if q_norm == 0:                  # only terms that occur in every document
            return contrib
        for term, qw in q.items():
            for doc, dw in self._dw[term].items():
                if self._norm[doc]:
                    contrib.setdefault(doc, {})[term] = qw * dw / (q_norm * self._norm[doc])
        return contrib

    # ------------------------------------------------------------ public

    def search(self, query: str, k: int = 10, model: str = "bm25", book: int | None = None) -> list[Hit]:
        """Ranked models: best first. "boolean": every match has score 1.0, in article
        order (Boolean retrieval does not rank); raises QuerySyntaxError on a bad query."""
        if model not in MODELS:
            raise ValueError(f"unknown model {model!r}, choose from {MODELS}")
        if model == "boolean":
            hits = [Hit(d, self.index.docs[d]["article_no"], self.index.docs[d]["id"], 1.0, {})
                    for d in self.boolean.search(query).docs
                    if book is None or self.index.docs[d]["book"] == book]
            return hits[:k]
        terms = word_terms(query)
        scorer = {"tfidf": self._tfidf, "cosine": self._cosine}.get(model, self._bm25f)
        contrib = scorer(terms)
        ngram = {}
        if model == "bm25_ngram":
            ngram = self._bm25_field("ngram", ngram_terms(query))

        hits = []
        for doc in contrib.keys() | ngram.keys():
            meta = self.index.docs[doc]
            if book is not None and meta["book"] != book:
                continue
            word_part = contrib.get(doc, {})
            ng = self.params.ngram_weight * ngram.get(doc, 0.0)
            hits.append(Hit(doc, meta["article_no"], meta["id"],
                            score=sum(word_part.values()) + ng, terms=word_part, ngram_score=ng))
        # rounded: equal scores summed in a different order can differ by ~1e-16, which
        # must not decide the order; real ties are broken by article number
        hits.sort(key=lambda h: (-round(h.score, 9), h.article_no))
        return hits[:k]


# ---------------------------------------------------------------- CLI

def main():
    ap = argparse.ArgumentParser(description="Search the Criminal Code index")
    ap.add_argument("query")
    ap.add_argument("--model", choices=MODELS, default="bm25")
    ap.add_argument("-k", type=int, default=10)
    ap.add_argument("--book", type=int)
    args = ap.parse_args()

    titles = {}
    for line in ARTICLES_FILE.open(encoding="utf-8"):
        a = json.loads(line)
        titles[a["article_no"]] = a["title"]
    searcher = Searcher(load())
    if args.model == "boolean":
        try:
            res = searcher.boolean.search(args.query)
        except QuerySyntaxError as e:
            raise SystemExit(f"query error: {e}")
        print(f"parsed as   : {res.query}")
        if res.ignored:
            print(f"ignored     : {' '.join(res.ignored)}  (no searchable words)")
        print(f"matches     : {len(res.docs)} articles")
    else:
        print(f"query terms : {' '.join(word_terms(args.query))}")
    for rank, h in enumerate(searcher.search(args.query, args.k, args.model, args.book), 1):
        parts = " ".join(f"{t}={s:.2f}" for t, s in sorted(h.terms.items(), key=lambda x: -x[1]))
        if h.ngram_score:
            parts += f" ngram={h.ngram_score:.2f}"
        print(f"{rank:2}. art {h.article_no:3}  {h.score:6.2f}  {titles[h.article_no][:40]:40}  [{parts}]")


if __name__ == "__main__":
    main()
