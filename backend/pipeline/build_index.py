"""
Stage 5: build the inverted index from data/processed/articles.jsonl.

For every article, three fields are analyzed with pipeline.analyzer:
  title   word terms of the article title
  text    word terms of the article text
  ngram   KCC bigrams of title + text   (robust to OCR errors, see docs/text_analysis.md)

The index stores only COUNTS (postings with term frequencies, field lengths). Ranking
formulas (BM25, TF-IDF) are computed at search time in pipeline/search.py, so their
parameters can be changed without rebuilding. See docs/indexing_and_ranking.md.

Output: backend/index/index.json (regenerable, not in git)

Run from backend/:  python -m pipeline index      (or: python -m pipeline.build_index)
"""
import hashlib
import json
import os
import time
import warnings
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from pipeline.analyzer import ngram_terms, word_terms
from pipeline.config import ARTICLES_FILE, INDEX_DIR

INDEX_FILE = INDEX_DIR / "index.json"
FORMAT_VERSION = 1
FIELDS = ("title", "text", "ngram")


@dataclass
class Field:
    postings: dict[str, dict[int, int]]   # term -> {doc number: term frequency}
    lengths: list[int]                    # number of terms in this field, per doc

    @property
    def avg_length(self) -> float:
        return sum(self.lengths) / len(self.lengths)

    def df(self, term: str) -> int:
        return len(self.postings.get(term, ()))


@dataclass
class Index:
    docs: list[dict]          # doc number -> {"id", "article_no", "book"}
    fields: dict[str, Field]
    source_sha256: str

    @property
    def n_docs(self) -> int:
        return len(self.docs)


# ---------------------------------------------------------------- build

def analyze_article(article: dict) -> dict[str, list[str]]:
    return {
        "title": word_terms(article["title"]),
        "text": word_terms(article["text"]),
        # "\n" so no bigram is formed across the title/text boundary
        "ngram": ngram_terms(article["title"] + "\n" + article["text"]),
    }


def build(articles: list[dict], source_sha256: str = "") -> Index:
    fields = {name: Field(postings={}, lengths=[]) for name in FIELDS}
    docs = []
    for doc, article in enumerate(articles):
        docs.append({
            "id": article["id"],
            "article_no": article["article_no"],
            "book": (article.get("path", {}).get("book") or {}).get("no"),
        })
        for name, terms in analyze_article(article).items():
            field = fields[name]
            field.lengths.append(len(terms))
            for term, tf in Counter(terms).items():
                field.postings.setdefault(term, {})[doc] = tf
    return Index(docs=docs, fields=fields, source_sha256=source_sha256)


# ---------------------------------------------------------------- save / load

def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(index: Index, path: Path = INDEX_FILE):
    data = {
        "format_version": FORMAT_VERSION,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": {"file": ARTICLES_FILE.name, "sha256": index.source_sha256},
        "n_docs": index.n_docs,
        "docs": index.docs,
        "fields": {
            name: {
                "avg_length": round(f.avg_length, 3),
                "lengths": f.lengths,
                # postings as [[doc, tf], ...] sorted by doc (JSON keys must be strings)
                "postings": {t: sorted(p.items()) for t, p in sorted(f.postings.items())},
            }
            for name, f in index.fields.items()
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")                 # atomic write, as in ocr.py
    tmp.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    os.replace(tmp, path)


def load(path: Path = INDEX_FILE, check_source: bool = True) -> Index:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run: python -m pipeline index")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data["format_version"] != FORMAT_VERSION:
        raise ValueError(f"index format {data['format_version']} != {FORMAT_VERSION}, rebuild it")
    index = Index(
        docs=data["docs"],
        fields={
            name: Field(
                postings={t: {d: tf for d, tf in p} for t, p in f["postings"].items()},
                lengths=f["lengths"],
            )
            for name, f in data["fields"].items()
        },
        source_sha256=data["source"]["sha256"],
    )
    if check_source and ARTICLES_FILE.exists() and file_sha256(ARTICLES_FILE) != index.source_sha256:
        warnings.warn(f"{ARTICLES_FILE.name} changed since the index was built - "
                      "run: python -m pipeline index")
    return index


# ---------------------------------------------------------------- stage

def validate(index: Index, articles: list[dict]):
    assert index.n_docs == len(articles), "one doc per article"
    for name in ("title", "text"):
        empty = [index.docs[d]["article_no"] for d, n in enumerate(index.fields[name].lengths) if n == 0]
        assert not empty, f"articles with an empty {name} field: {empty}"
    for name, field in index.fields.items():
        for term, postings in field.postings.items():
            assert postings and all(tf > 0 for tf in postings.values()), (name, term)
        assert sum(field.lengths) == sum(sum(p.values()) for p in field.postings.values()), name


def run():
    articles = [json.loads(line) for line in ARTICLES_FILE.open(encoding="utf-8")]
    t0 = time.perf_counter()
    index = build(articles, file_sha256(ARTICLES_FILE))
    secs = time.perf_counter() - t0
    validate(index, articles)
    save(index)

    print(f"documents : {index.n_docs}   (built in {secs:.1f} s, validation passed)")
    print(f"{'field':6} {'vocabulary':>10} {'postings':>9} {'terms':>7} {'avg len':>8} {'max len':>8}")
    for name, f in index.fields.items():
        n_post = sum(len(p) for p in f.postings.values())
        print(f"{name:6} {len(f.postings):>10} {n_post:>9} {sum(f.lengths):>7} "
              f"{f.avg_length:>8.1f} {max(f.lengths):>8}")
    print(f"written   : {INDEX_FILE}  ({INDEX_FILE.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    run()
