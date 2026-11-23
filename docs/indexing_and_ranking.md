# Indexing and Ranking: Inverted Index, BM25 and TF-IDF

How the 672 articles are stored in an inverted index and how a query is scored against them with BM25F (main model), BM25 over character bigrams (OCR safety net), and the TF-IDF and cosine-similarity baselines. How these models differ on the same queries: [model_comparison.md](model_comparison.md).

_Last updated: 2026-11-23_

---

## Contents

1. [Overview](#1-overview)
2. [What an Inverted Index Is](#2-what-an-inverted-index-is)
3. [Index Design](#3-index-design)
4. [Ranking Models](#4-ranking-models)
5. [Parameters](#5-parameters)
6. [Worked Example: Why Article 33 Ranks First](#6-worked-example-why-article-33-ranks-first)
7. [Example Results](#7-example-results)
8. [Challenges and Solutions](#8-challenges-and-solutions)
9. [Numbers](#9-numbers)
10. [Python API and CLI](#10-python-api-and-cli)
11. [Tests](#11-tests)
12. [Known Limitations and Future Work](#12-known-limitations-and-future-work)
13. [How to Run](#13-how-to-run)

---

## 1. Overview

| | |
|---|---|
| **Goal** | Given a Khmer query, return the most relevant articles, ranked, with an explanation of each score |
| **Build** | `backend/pipeline/build_index.py`, stage `index`: `python -m pipeline index` |
| **Search** | `backend/pipeline/search.py`: `Searcher.search(query, k, model, book)` |
| **Input** | `data/processed/articles.jsonl` (672 articles) |
| **Output** | `backend/index/index.json` (1.0 MB, regenerable, not in git) |
| **Text analysis** | `pipeline/analyzer.py` (see [text_analysis.md](text_analysis.md)) |
| **Models** | `bm25` (BM25F over title + text), `bm25_ngram` (+ KCC bigrams), `tfidf` (TF-IDF score), `cosine` (TF-IDF + cosine similarity); plus `boolean` (AND/OR/NOT, unranked, see [boolean_retrieval.md](boolean_retrieval.md)) |
| **Speed** | build 8.2 s; load 0.13 s; query ≈ 1–1.5 ms (`bm25`, `tfidf`, `cosine`), 3 ms (`bm25_ngram`) |
| **Tests** | `backend/tests/test_index.py` (18 tests) |

Why BM25: it rewards rare terms (IDF), stops rewarding repetition after a few occurrences (TF saturation) and corrects for article length, needs no training data, and every score can be explained term by term. It is also the standard baseline in IR research, so later models (embeddings, hybrid) can be compared against it.

---

## 2. What an Inverted Index Is

A book index tells you, for a word, which pages it is on. An **inverted index** does the same for documents: for every **term**, it stores the list of documents that contain it (**postings**), with how often it occurs (**term frequency, tf**).

```
term          postings (doc number : tf)
ការពារ    →  32:1  33:1  40:1  441:1  444:1  445:1  446:1  473:1 …     (title field)
ខ្លួន      →  …                                                       (text field)
```

To answer a query, the search engine only reads the postings of the query terms, not all 672 articles. It is called *inverted* because it maps term → documents instead of document → terms.

Doc numbers are positions in `articles.jsonl` (0-based): doc 32 = Article 33.

---

## 3. Index Design

### 3.1 Three fields per article

| Field | Content (from `analyzer.py`) | Purpose |
|---|---|---|
| `title` | `word_terms(title)` | a match in the title is strong evidence, so it gets a higher weight |
| `text` | `word_terms(text)` | main body |
| `ngram` | `ngram_terms(title + "\n" + text)` (KCC bigrams) | finds articles where OCR damaged the query word |

The `"\n"` between title and text makes sure no bigram is formed across the boundary.

### 3.2 What is stored: counts only

For each field: the postings `term → {doc: tf}`, and the length (number of terms) of every document. Per document: `id`, `article_no`, `book` (for filtering).

**No scores are stored.** IDF, length normalization and the BM25/TF-IDF formulas are computed at search time from these counts. Consequences:
- `k1`, `b` and field weights can be changed (and tuned in the evaluation step) **without rebuilding**.
- Several ranking models share the same index.
- With 672 documents, computing scores at query time costs about 1 ms.

Word positions are **not** stored: BM25 doesn't use them, and highlighting can re-run `analyzer.segment()` on the few articles shown (it returns exact offsets).

### 3.3 File format (`backend/index/index.json`)

```json
{
  "format_version": 1,
  "built_at": "2026-11-23T…+00:00",
  "source": {"file": "articles.jsonl", "sha256": "…"},
  "n_docs": 672,
  "docs": [{"id": "art_001", "article_no": 1, "book": 1}, …],
  "fields": {
    "title": {"avg_length": 5.2, "lengths": [3, …], "postings": {"ការពារ": [[32, 1], [33, 1], …], …}},
    "text":  {…},
    "ngram": {…}
  }
}
```

Decisions:
- **JSON, not pickle:** readable, can be opened to check a posting list, and loading it never runs code (pickle can). At 1.0 MB it loads in 0.13 s.
- **Atomic write:** written to `index.tmp`, then renamed, as in `ocr.py`. A crash never leaves a half-written index.
- **Staleness check:** the SHA-256 of `articles.jsonl` is stored. `load()` warns if the dataset changed since the index was built ("run: python -m pipeline index").
- **`format_version`:** `load()` refuses an index written by an incompatible version of the code.
- Location `backend/index/` was already configured (`INDEX_DIR` in `config.py`) and ignored by git (`.gitignore`). The index is regenerated from `articles.jsonl` in 8 s.

### 3.4 Validation (run on every build)

- one document per article (672);
- no article has an empty `title` or `text` field;
- every posting has tf > 0;
- for every field, the sum of lengths equals the sum of all tf values (nothing lost between counting and storing).

---

## 4. Ranking Models

Notation: `N` = number of articles (672), `df(t)` = number of articles containing term `t`, `tf(t,d)` = occurrences of `t` in article `d`, `|d|` = length of `d` in terms, `avgdl` = average length.

### 4.1 BM25

$$
\text{score}(d, q) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{tf(t,d) \cdot (k_1 + 1)}{tf(t,d) + k_1 \cdot \left(1 - b + b \cdot \frac{|d|}{avgdl}\right)}
\qquad
\text{IDF}(t) = \ln\left(1 + \frac{N - df(t) + 0.5}{df(t) + 0.5}\right)
$$

The three parts, and what they mean for the Criminal Code:

| Part | Effect | Example from the corpus |
|---|---|---|
| **IDF** | rare terms count more | `ផ្តន្ទាទោស` (punish) is in 491/672 articles: IDF ≈ 0.31. `ការពារ` (defend) is in 28: IDF ≈ 3.16 |
| **TF saturation** (`k1`) | the first occurrence counts most; the score for one term can never exceed `IDF · (k1 + 1)` | an article repeating `ទោស` (penalty) 10 times is not 10× more relevant |
| **Length normalization** (`b`) | the same tf counts more in a short article | article text lengths range from a few to 334 terms (average 49) |

The IDF has `1 +` inside the log (the Lucene/Elasticsearch variant). The original formula becomes **negative** for terms in more than half of the documents, which would *punish* an article for containing a query word like `ផ្តន្ទាទោស`. With `1 +` the IDF is always positive and simply small for frequent terms.

### 4.2 BM25F: title and text together (model `bm25`)

A simple approach would compute BM25 for the title and for the text separately and add them. Robertson et al. (2004, *Simple BM25 extension to multiple weighted fields*) showed that this over-rewards a term found in both fields, because each field is saturated separately: the score can then exceed the single-term limit.

**BM25F** first combines the weighted, length-normalized tf of all fields, and **then** saturates once:

$$
\tilde{tf}(t,d) = \sum_{f \in \{title,\,text\}} w_f \cdot \frac{tf_f(t,d)}{1 - b_f + b_f \cdot \frac{|d_f|}{avgdl_f}}
\qquad
\text{score}(d,q) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{\tilde{tf}(t,d) \cdot (k_1+1)}{\tilde{tf}(t,d) + k_1}
$$

- `w_title = 2`, `w_text = 1`: one occurrence in the (short) title counts like two in the text.
- `df(t)` for the IDF counts articles that contain `t` in **any** of the two fields.

This is implemented in `Searcher._bm25f()`.

### 4.3 BM25 + KCC bigrams (model `bm25_ngram`)

$$
\text{score} = \text{BM25F}_{words}(d, q) + w_{ngram} \cdot \text{BM25}_{ngram}(d, q), \qquad w_{ngram} = 0.3
$$

The bigram field has a **different vocabulary** (bigrams, not words), so it cannot be merged into BM25F. It gets its own classic BM25 score, added with a low weight. A query produces many more bigrams than words, so without a low weight the bigram score would dominate. The effect is shown in [§8.3](#83-do-bigrams-really-recover-ocr-damaged-articles).

### 4.4 TF-IDF (model `tfidf`) and cosine similarity (model `cosine`)

Both use the same **tf-idf weight** over title + text words as one bag (title not weighted higher):

$$
w(t,d) = (1 + \ln tf(t,d)) \cdot \ln\frac{N}{df(t)}
$$

Sublinear tf (`1 + ln tf`) instead of raw tf is the usual choice (SMART `l` weighting). A term in every article gets weight 0.

**`tfidf`: TF-IDF score** (Manning et al., *IIR* eq. 6.9), the sum of the article's weights for the query terms:

$$
\text{score}(d,q) = \sum_{t \in q} w(t,d)
$$

No length normalization: a long article that repeats a term scores as high as or higher than a short one.

**`cosine`: vector space model.** Query and article are vectors of tf-idf weights; the score is the cosine of the angle between them:

$$
\text{score}(d,q) = \cos(\vec{q}, \vec{d}) = \frac{\sum_t w(t,q)\, w(t,d)}{\lVert\vec{q}\rVert\, \lVert\vec{d}\rVert}
$$

- Scores are between 0 and 1.
- Dividing by `‖d‖` normalizes for length, but `‖d‖` grows with **every** term of the article, not just the query terms. Matching a second, less important query word adds very little (see [model_comparison.md](model_comparison.md)).

The two models differ **only** in the normalization, which makes it possible to measure exactly what cosine normalization changes. Both are baselines for BM25.

### 4.5 Score explanations

Every result (`Hit`) carries `terms`: the contribution of each query term, plus `ngram_score`. They add up exactly to `score` (tested). This is shown in the CLI and can be shown in the UI ("matched: ការពារ 6.06, ខ្លួន 1.57").

---

## 5. Parameters

| Parameter | Value | Meaning | Source |
|---|---|---|---|
| `k1` | 1.2 | TF saturation; higher = repetition counts longer | standard default (range 1.2–2.0) |
| `b` (title, text, ngram) | 0.75 | length normalization; 0 = none, 1 = full | standard default |
| `w_title` | 2.0 | title weight in BM25F | chosen: a title is a short summary of the article |
| `w_text` | 1.0 | text weight | reference |
| `w_ngram` | 0.3 | share of the bigram score | chosen so bigrams support, not dominate, the word score |

**None of these is tuned yet.** They are reasonable starting values. The evaluation step (test queries with known relevant articles) will measure P@k / MRR / nDCG and tune them. They are all in `search.Params`, and changing them needs no rebuild.

---

## 6. Worked Example: Why Article 33 Ranks First

Query `ការពារខ្លួន` (self-defence) → analyzer → terms `ការពារ`, `ខ្លួន`.

Article 33 (`ការការពារស្របច្បាប់`, legitimate self-defence): title length 4 (avg 5.2), text length 52 (avg 49.03).

| | `ការពារ` (defend) | `ខ្លួន` (self) |
|---|---|---|
| df (of 672) | 28 | 134 |
| IDF = ln(1 + (672 − df + 0.5)/(df + 0.5)) | **3.162** | **1.610** |
| tf in title | 1 (stem of `ការការពារ`) | 0 |
| tf in text | 6 | 1 |
| title part: 2 · 1 / (0.25 + 0.75 · 4/5.2) | 2.419 | 0 |
| text part: tf / (0.25 + 0.75 · 52/49.03) | 6 / 1.045 = 5.739 | 1 / 1.045 = 0.957 |
| combined tf̃ | 8.158 | 0.957 |
| saturation: tf̃ · 2.2 / (tf̃ + 1.2) | 1.918 | 0.976 |
| **contribution** = IDF × saturation | **6.06** | **1.57** |

**Score = 7.64.** Observations:
- `ការពារ` contributes 4× more than `ខ្លួន`: it is rarer (IDF 3.16 vs 1.61) and more frequent in the article.
- The saturation step turns tf̃ = 8.2 into 1.92 (the limit is 2.2): the 6 occurrences in the text add little beyond the first few.
- The title match comes from the **prefix stem** (`ការការពារ` → `ការពារ`, see [text_analysis.md §4.3](text_analysis.md#light-stemming-noun-prefixes)). Without it, the title would not match at all.

The CLI shows the same numbers:
```
 1. art  33    7.64  ការការពារស្របច្បាប់     [ការពារ=6.06 ខ្លួន=1.57]
```

---

## 7. Example Results

Top 3 per model (`python -m pipeline.search "<query>" -k 3 --model <model>`):

| Query | `bm25` | `bm25_ngram` | `cosine` |
|---|---|---|---|
| `ការការពារស្របច្បាប់` (legitimate defence, exact title of Art. 33) | **33**, 34, 477 | **33**, … | **33**, … |
| `ការពារខ្លួន` (self-defence) | **33**, 35, 34 | **33**, 35, 34 | **33**, 477, 35 |
| `ព្រហ្មទណ្ឌ` (criminal) | 37, 668, 42 | 37, 668, 42 | 37, 24, 42 |
| `ការឃុំខ្លួន` (detention) | 51, 591, 131 | 51, 591, 131 | 591, 51, 131 |
| `ឃាតកម្ម` (murder) | 433, 199, 200, 201, 204 (all murder articles) | | |
| `ចោរកម្ម` (theft) | 399, 402, 400, 401 (articles on receiving stolen goods, all mention ចោរកម្ម) | | |
| `ការរំលោភសេពសន្ថវៈ` (rape) | **239** (`បទរំលោភសេពសន្ថវៈ`), 391, 205 | | |

- Art. 34 (presumption of legitimate defence) and 35 (state of necessity) right after 33 are sensible neighbours.
- `cosine` ranks Art. 477 (`…ការការពារការសម្ងាត់នៃការការពារ`, *national defence* secrets) second for "self-defence", although it does not contain `ខ្លួន` (self). The cause is cosine's normalization, which makes the second query word nearly worthless (contribution 0.02); BM25 adds 1.6 for it. Full analysis in [model_comparison.md §5.2](model_comparison.md).
- For the rape query, Art. 391 (`ការរំលោភលើទំនុកចិត្ត`, *abuse of trust*) is 2nd: `រំលោភ` means "violate/abuse" in general, and bag-of-words matching cannot tell that the user meant the phrase `រំលោភសេពសន្ថវៈ`. Phrase/proximity scoring would help ([§12](#12-known-limitations-and-future-work)).
- These are spot checks. A systematic comparison of the models on 24 queries is in [model_comparison.md](model_comparison.md); which model is *better* needs relevance labels (evaluation step).

---

## 8. Challenges and Solutions

Each challenge lists: **symptom → root cause → solution → result**.

### 8.1 The same concept counted twice (found through score explanations)

- **Symptom:** In the first version, the query `ការពារខ្លួន` produced the terms `ការពារ ពារ ខ្លួន`, and `ពារ` contributed **6.77** to Article 33: "defend" was counted twice, while `ខ្លួន` (self) only got 1.5.
- **Root cause:** The analyzer's prefix rule treated `ការពារ` (to defend) as `ការ` + `ពារ`. But `ការពារ` and `ការងារ` (work) are words in their own right. The rule only makes sense for `ការ` + a real word.
- **Solution:** In `analyzer.py`, an exception list `NOT_PREFIXED = {ការពារ, ការងារ}`.
- **Result:** Query terms are now `ការពារ ខ្លួន`. A test makes sure `ការពារ` gives no stem.

While reviewing **all** stems produced on the corpus for this fix, two more problems turned up and were fixed:
- `ការិយាល័យ` (office) would give the stem `ិយាល័យ`, which starts with a vowel sign. Now the stem must itself pass the word check.
- One-syllable verbs were never stemmed (`ការធ្វើ` → `ធ្វើ`, `ការប្រើ` → `ប្រើ`, `ការឃុំ` → `ឃុំ` detention), because the rule required 2 clusters. The rule is now "2+ clusters **or** contains a vowel", which still rejects `ការណ៍` → `ណ៍`.

This is why score explanations are worth having: the bug was invisible in the ranked list, but obvious in the per-term contributions.

### 8.2 Summing per-field BM25 over-rewards terms in both fields

- **Problem:** `BM25(title) + BM25(text)` saturates each field separately: a term in the title and the text can score up to 2 × `IDF · (k1+1)`.
- **Solution:** BM25F ([§4.2](#42-bm25f-title-and-text-together-model-bm25)), which saturates once after combining fields.
- **Result:** Test `test_bm25f_term_in_both_fields_saturates_once`: the score of one term stays below `IDF · (k1+1)` even when it is in both fields.

### 8.3 Do bigrams really recover OCR-damaged articles?

- **Question:** The bigram field is meant to find articles where OCR misspelled the query word. Does it?
- **Experiment:** Articles 193 and 267 contain *property* only as the OCR misspelling `ទ្រព្យសម្ប្ិ`, never as `ទ្រព្យសម្បត្តិ`. Query: the correct spelling `ទ្រព្យសម្បត្តិ`.

  | Model | Results | Rank of Art. 193 | Rank of Art. 267 |
  |---|---|---|---|
  | `bm25` | 54 | **not found** | **not found** |
  | `bm25_ngram` | 87 | 57 | 63 |

- **Result:** Bigrams **do** recover the damaged articles (recall ↑), but they rank low, and the query now returns 33 extra articles, some of them only weakly related (precision ↓). Whether this trade is worth it, and the best `w_ngram`, is decided in the evaluation step. Until then, the default model is `bm25`. Test `test_ngrams_recover_ocr_damaged_articles`.

### 8.4 Negative IDF

- **Problem:** The original Robertson–Spärck Jones IDF `ln((N − df + 0.5)/(df + 0.5))` is negative when `df > N/2`. `ផ្តន្ទាទោស` is in 491/672 articles: an article containing it would score **lower** than one that doesn't.
- **Solution:** `ln(1 + …)` (Lucene variant). Test `test_rare_terms_weigh_more`: IDF > 0 even for a term in every article.

### 8.5 Measuring speed correctly

- **Symptom:** The first latency measurement gave 18.9 ms/query for `bm25` but 2.4 ms for the (more expensive) `bm25_ngram`.
- **Root cause:** The `bm25` run was first and included loading khmer-nltk's CRF model (~4 s, see [text_analysis.md §5.8](text_analysis.md#58-model-loading-time)).
- **Solution:** Warm up (one query per model) before timing. Measured over 4 queries × 100 runs: **1.0–1.5 ms** (`bm25`, varies between runs), **3.1 ms** (`bm25_ngram`), **≈1 ms** (`tfidf`, `cosine`).
- **For the API:** load the index and run one query at startup.

---

## 9. Numbers

`python -m pipeline index`:

| Field | Vocabulary (distinct terms) | Postings | Terms | Avg length | Max length |
|---|---|---|---|---|---|
| `title` | 842 | 3,458 | 3,496 | 5.2 | 16 |
| `text` | 2,447 | 23,882 | 32,945 | 49.0 | 334 |
| `ngram` | 6,381 | 73,609 | 111,904 | 166.5 | 1,019 |

| | |
|---|---|
| Documents | 672 |
| Build time (analysis of all articles) | 8.2 s |
| Index file | 1.0 MB JSON |
| Load time | 0.13 s |
| Query time (warm) | 1.0–1.5 ms `bm25` · 3.1 ms `bm25_ngram` · ≈1 ms `tfidf` / `cosine` |

---

## 10. Python API and CLI

```python
from pipeline.build_index import load
from pipeline.search import Searcher, Params

searcher = Searcher(load())                                # or Searcher(load(), Params(k1=1.5))
hits = searcher.search("ការពារខ្លួន", k=10, model="bm25", book=1)
for h in hits:
    print(h.article_no, h.score, h.terms, h.ngram_score)
```

| Name | What it is |
|---|---|
| `build_index.build(articles)` | `Index` from a list of article dicts (in memory) |
| `build_index.save(index, path)` / `load(path)` | write / read `index.json` (`load` warns if `articles.jsonl` changed) |
| `Index.docs`, `Index.fields[name].postings / lengths / avg_length / df(term)` | raw statistics |
| `search.Params` | `k1`, `b`, field `weight`, `ngram_weight` |
| `Searcher.search(query, k=10, model="bm25", book=None)` | list of `Hit(doc, article_no, id, score, terms, ngram_score)`, best first; ties broken by article number |
| `search.MODELS` | `("bm25", "bm25_ngram", "tfidf", "cosine", "boolean")` |

CLI:

```bash
python -m pipeline.search "ការពារខ្លួន"                        # bm25, top 10
python -m pipeline.search "ការពារខ្លួន" --model cosine -k 5 --book 1
```

---

## 11. Tests

`backend/tests/test_index.py` (21 tests; the whole suite of 93 tests runs in ~13 s with `python -m pytest`).

The small tests use a **toy corpus of Latin words** (`alpha`, `beta`, `gamma`…). The analyzer leaves Latin words unchanged, so the expected terms and scores are exact and easy to check by hand.

| Group | What is checked |
|---|---|
| Index structure | postings, lengths, average length, doc metadata; save → load gives the same index |
| BM25 properties | score equals the formula computed by hand; rarer term weighs more; IDF never negative; TF saturates (tf 4 < 4 × tf 1, below the upper bound); title beats text; shorter document wins with the same tf; BM25F saturates once across fields |
| Search behaviour | no match / empty / punctuation-only query → no results; book filter; term contributions add up to the score; unknown model → error; cosine scores in (0, 1]; TF-IDF score equals its formula and ignores length while cosine does not; ranking independent of the order in which term scores are summed |
| Real corpus | 672 documents in article order; Art. 33's title query ranks Art. 33 first in all three models; `ការពារខ្លួន` has Art. 33 in the top 3; bigrams recover Arts. 193/267 that plain BM25 misses |

---

## 12. Known Limitations and Future Work

| Limitation | Impact | Possible improvement |
|---|---|---|
| Parameters not tuned | ranking quality unknown beyond spot checks | **next step:** test queries + relevance judgments → P@k, MRR, nDCG; grid-search `k1`, `b`, `w_title`, `w_ngram` |
| Only word matching | synonyms and everyday wording are missed (`លួច` steal vs `ចោរកម្ម` theft) | semantic embeddings + hybrid with BM25 (reciprocal rank fusion) |
| Noun query counts its stem too (`ការឃុំ` + `ឃុំ`) | noun forms get about 2× the weight of other query words | give stems a lower weight (e.g. 0.5); measure in evaluation |
| Bigrams add noise | `bm25_ngram` returns many weakly related articles | tune `w_ngram`; or use bigrams only when word matching finds few results |
| No phrase / proximity scoring | `ការពារ … ខ្លួន` far apart scores the same as `ការពារខ្លួន` | store positions; add a proximity bonus |
| Whole articles as documents | a long article matching in one paragraph is diluted by length normalization | paragraph-level index (also needed for RAG chunks) |
| JSON index loaded fully into memory | fine for 672 articles (1 MB); not for millions of documents | compressed binary postings, or a search engine (Lucene/Elasticsearch) |

---

## 13. How to Run

From `backend/` with the venv active (`PYTHONUTF8=1` on Windows to print Khmer):

```bash
python -m pipeline index                     # build backend/index/index.json (~8 s)
python -m pipeline.search "ការពារខ្លួន"        # try a query
python -m pipeline.search "ចោរកម្ម" --model bm25_ngram -k 5
python -m pytest                             # all tests
```

Rebuild the index whenever `articles.jsonl` or `analyzer.py` changes. `load()` warns when the dataset changed, but it cannot detect analyzer changes: rebuild after editing the analyzer.
