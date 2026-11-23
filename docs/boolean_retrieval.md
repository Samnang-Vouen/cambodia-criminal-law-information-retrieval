# Boolean Retrieval: AND, OR, NOT

How exact-match queries with `AND`, `OR`, `NOT` and parentheses are parsed and answered with the inverted index.

_Last updated: 2026-11-23_

---

## Contents

1. [Overview](#1-overview)
2. [Query Syntax](#2-query-syntax)
3. [Examples on the Criminal Code](#3-examples-on-the-criminal-code)
4. [How It Works](#4-how-it-works)
5. [Design Decisions](#5-design-decisions)
6. [Boolean vs Ranked Retrieval](#6-boolean-vs-ranked-retrieval)
7. [Python API and CLI](#7-python-api-and-cli)
8. [Tests](#8-tests)
9. [Known Limitations and Future Work](#9-known-limitations-and-future-work)

---

## 1. Overview

| | |
|---|---|
| **Goal** | Exact-match search: "articles that contain X and Y but not Z" |
| **Code** | `backend/pipeline/boolean.py` (parser + merge algorithms), used by `search.py` as model `boolean` |
| **Index** | the same inverted index as BM25 (`backend/index/index.json`, word fields `title` + `text`), see [indexing_and_ranking.md](indexing_and_ranking.md) |
| **Operators** | `AND`, `OR`, `NOT`, parentheses; two operands next to each other = `AND` |
| **Result** | the set of matching articles, in article order (not ranked) |
| **Speed** | ≈ 0.3 ms per query |
| **Tests** | `backend/tests/test_boolean.py` (30 tests, incl. 300 random queries checked against set logic) |

---

## 2. Query Syntax

| Query | Meaning |
|---|---|
| `ឃាតកម្ម` | articles containing ឃាតកម្ម (murder) |
| `ឃាតកម្ម AND ពុល` | both words |
| `ឃាតកម្ម OR ចោរកម្ម` | at least one of them |
| `ឃាតកម្ម AND NOT ពុល` | ឃាតកម្ម, but not ពុល |
| `NOT ទោស` | every article without ទោស |
| `(លួច OR ចោរកម្ម) AND NOT អនីតិជន` | parentheses group |
| `ឃាតកម្ម ពុល` | = `ឃាតកម្ម AND ពុល` (implicit AND) |
| `ឃាតកម្ម NOT ពុល` | = `ឃាតកម្ម AND NOT ពុល` |

Rules:

- **Operators** are the English words `AND`, `OR`, `NOT`, in any case (`and` works too), **separated by spaces** from the search words. Khmer text has no spaces between words, so `ឃាតកម្មANDពុល` is read as one word.
- **Precedence:** `NOT` > `AND` > `OR`, as in most search engines and programming languages:
  - `a OR b AND c` = `a OR (b AND c)`
  - `NOT a AND b` = `(NOT a) AND b`
  - use parentheses for anything else.
- **Operands are analyzed** exactly like ranked queries (`analyzer.word_terms`, [text_analysis.md](text_analysis.md)): normalization, segmentation, stopwords, prefix stems. So:
  - `សេចក្ដី` and `សេចក្តី` match the same articles (spelling normalization);
  - `ការពារ` also matches `ការការពារ` (the index stores the stem of prefixed nouns);
  - an operand that is segmented into several words must match **all** of them: `ថ្នាំពុល` → `ថ្នាំ AND ពុល`;
  - a prefixed noun only requires its **stem**: `ការរំលោភសេពសន្ថវៈ` → `រំលោភ AND សេពសន្ថវៈ`, so it also finds Art. 239 `បទរំលោភសេពសន្ថវៈ`, which never uses the noun `ការរំលោភ` (see §5).
- **Ignored words:** an operand with no searchable word (a stopword such as `និង`, `ឬ`, `មិន`, or punctuation) is skipped and reported: `ការពារ និង ខ្លួន` → `(ការពារ AND ខ្លួន)`, *ignored: និង*.
- **Matching** is on the title and text of each article; a word may be in either.

### Error messages

| Query | Error |
|---|---|
| *(empty)* | `empty query: type at least one search word` |
| `ឃាតកម្ម AND` | `the query ends where a search word is expected` |
| `AND ឃាតកម្ម` | `'AND' where a search word is expected` |
| `(ឃាតកម្ម OR ចោរកម្ម` | `missing ')'` |
| `ឃាតកម្ម)` | `unexpected ')' (missing '('?)` |
| `ការពារ AND និង` | `the query ends where a search word is expected (ignored, not searchable: និង)` |

Errors are raised as `QuerySyntaxError` (a `ValueError`); the API will turn them into a "400 Bad Request" with the message.

---

## 3. Examples on the Criminal Code

Output of `python -m pipeline.search "<query>" --model boolean`:

| Query | Parsed as | Matches | Notes |
|---|---|---|---|
| `ឃាតកម្ម` (murder) | `ឃាតកម្ម` | 10 | Arts. 188, 193, 199–205, 433 |
| `ឃាតកម្ម AND ពុល` (murder, poison) | `(ឃាតកម្ម AND ពុល)` | 1 | Art. 201, *murder by poisoning* |
| `ឃាតកម្ម AND NOT ពុល` | `(ឃាតកម្ម AND NOT ពុល)` | 9 | the 10 murder articles minus Art. 201 |
| `លួច OR ចោរកម្ម` (steal, theft) | `(លួច OR ចោរកម្ម)` | 29 | 25 + 5 − 1 article containing both: a user can cover **synonyms** by hand |
| `ពិន័យ AND NOT ពន្ធនាគារ` (fine, not prison) | `(ពិន័យ AND NOT ពន្ធនាគារ)` | 27 | articles with a fine but no prison sentence |
| `សេពសន្ថវៈ NOT អនីតិជន` (sexual intercourse, not minor) | `(សេពសន្ថវៈ AND NOT អនីតិជន)` | 9 | implicit AND before NOT |
| `NOT (ពិន័យ OR ពន្ធនាគារ)` | `NOT (ពិន័យ OR ពន្ធនាគារ)` | 278 | articles with no penalty at all: definitions, general rules |
| `ការពារ និង ខ្លួន` | `(ការពារ AND ខ្លួន)` | 6 | `និង` ignored; Arts. 33, 34, 35, … |
| `ហិង្សា AND អនីតិជន` (violence, minor) | | 0 | both words exist (38 and 29 articles) but never in the same article |

The "parsed as" column is printed by the CLI and returned in `BooleanResult.query`. Showing it to users tells them exactly how their query was understood (e.g. that `AND` was applied before `OR`).

---

## 4. How It Works

```
"ឃាតកម្ម AND NOT ពុល"
      │ 1. tokenize()   split at spaces and ( ); recognise AND/OR/NOT;
      │                 analyze every other word with word_terms(); drop words with no terms
      ▼
[TEXT ឃាតកម្ម] [AND] [NOT] [TEXT ពុល]
      │ 2. parse()      recursive descent parser → query tree
      ▼
And( Term(ឃាតកម្ម), Not(Term(ពុល)) )
      │ 3. evaluate()   walk the tree; combine sorted posting lists with linear merges
      ▼
postings(ឃាតកម្ម) = [187, 192, 198, 199, 200, 201, 202, 203, 204, 432]     (doc numbers)
postings(ពុល)     = [193, 200, 224, 225, 226, 229, 565, 567, 569]
difference        = [187, 192, 198, 199, 201, 202, 203, 204, 432]  → Arts. 188, 193, 199, 200, 202–205, 433
```

(doc number = article number − 1)

### 4.1 Parser

A **recursive descent parser**: one function per precedence level, each calling the next higher level. The grammar:

```
or_expr  := and_expr ("OR" and_expr)*
and_expr := not_expr (["AND"] not_expr)*        ← "AND" is optional: implicit AND
not_expr := "NOT" not_expr | primary
primary  := TEXT | "(" or_expr ")"
```

Because `or_expr` is built from `and_expr`s and `and_expr` from `not_expr`s, `NOT` automatically binds tightest and `OR` loosest. Chains like `a AND b AND c` become one node with three children (`And((a, b, c))`) instead of a deep binary tree, which helps the evaluation in §4.3.

### 4.2 Posting lists

For a term, the posting list is the **sorted** list of documents containing it in the `title` **or** the `text` field: `union(title postings, text postings)`. An operand with several analyzer terms is the intersection of their lists.

### 4.3 Merge algorithms

All operations are the classic **linear merges** over sorted lists (Manning, Raghavan & Schütze, *Introduction to Information Retrieval*, ch. 1): two pointers walk both lists once, so the cost is O(|a| + |b|), never O(|a| × |b|).

| Function | Result | Used for |
|---|---|---|
| `intersect(a, b)` | docs in both | `AND` |
| `union(a, b)` | docs in either | `OR`, title ∪ text |
| `difference(a, b)` | docs in `a` but not in `b` | `AND NOT` |

Two optimizations from the textbook:

1. **Shortest list first** (`intersect_all`): for `a AND b AND c`, intersect the two shortest lists first. The intermediate result can only shrink, so every later merge is cheap, and evaluation stops as soon as it is empty. Example: `ពន្ធនាគារ AND ពិន័យ AND ពុល` (367, 256 and 9 articles) starts with the 9 articles of `ពុល`.
2. **`AND NOT` without a complement:** `a AND NOT b` is computed as `difference(a, b)`, not as `intersect(a, complement(b))`. The complement of a rare word contains almost all 672 articles; `difference` never builds it. A complement over all documents is only computed for a `NOT` that is not part of an `AND` (`NOT ទោស`, or `x OR NOT y`).

---

## 5. Design Decisions

| Decision | Alternatives | Why |
|---|---|---|
| English operator words `AND/OR/NOT` | Khmer words (`និង`, `ឬ`, `មិន`); symbols (`&`, `\|`, `-`) | `និង`, `ឬ`, `មិន` also occur in normal Khmer text and are stopwords; making them operators would change the meaning of natural queries. English operators are what the course and most search engines use. |
| Operators case-insensitive | uppercase only | the corpus contains no Latin words, so `and` can never be a search word |
| Implicit AND | error, or implicit OR | Khmer users separate phrases with spaces (`ការពារ ខ្លួន`); "all words" is what users expect, as in Google |
| Stopword operands ignored and reported | treat as "matches nothing" / "matches everything" | "nothing" would make `ការពារ និង ខ្លួន` return no results; "everything" would make `x OR និង` return all articles. Ignoring + telling the user is the least surprising. |
| Same analyzer as ranked search | exact string match | consistent behaviour across models; spelling variants and prefix stems work in Boolean queries too |
| Prefixed noun requires only its stem (`operand_terms()`) | AND of all analyzer terms (first version) | found while preparing the evaluation: the analyzer turns `ការរំលោភ` into `ការរំលោភ` + `រំលោភ`, and ANDing both excluded the rape article (Art. 239), which only has `រំលោភ`. Six of the 24 test queries returned nothing; now four (the remaining ones are genuine: question words and word pairs that never occur together). Every article with the noun also has the stem indexed, so requiring the stem loses nothing. |
| Results in article order, score 1.0 | rank by BM25 | pure Boolean retrieval has no notion of "more relevant"; article order is natural for a legal code. Ranked search is available as the other models (see §6). |
| Word fields only (no bigram field) | include KCC bigrams | Boolean queries are about exact words; bigrams would make `NOT` unpredictable (an article could be excluded because it shares two letter clusters with the word) |
| Merge on sorted lists | Python `set` operations | sets would be as fast here (672 documents), but the merge algorithms are what an inverted index uses at scale and what the course teaches; the tests check they agree with set logic |

---

## 6. Boolean vs Ranked Retrieval

| | Boolean (`boolean`) | Ranked (`bm25`, `tfidf`, `cosine`) |
|---|---|---|
| Result | a **set**: match or no match | a **list** ordered by score |
| Query | precise logic, written by the user | free text |
| "Too many / too few results" | common: `ពន្ធនាគារ` → 367 articles; one extra `AND` → 0 | always returns the best-scoring articles first |
| Partial matches | no: `AND` needs every word | yes: an article with 2 of 3 words still ranks |
| Exclusion | **yes** (`NOT`) | no |
| Transparency | complete: the user knows exactly why an article matched | scores need explanation (see [indexing_and_ranking.md §6](indexing_and_ranking.md#6-worked-example-why-article-33-ranks-first)) |
| Good for | experts who know the legal terms: "all articles on fines but not prison" | everyday questions: "self-defence" |
| Evaluation | set metrics: **Precision, Recall, F1** | ranking metrics: P@k, MAP, nDCG |

The last row matters for the evaluation step: Boolean results are a set, so they are measured with Precision / Recall / F1 directly, while ranked models are measured at a cutoff k or over the whole ranking.

---

## 7. Python API and CLI

```python
from pipeline.build_index import load
from pipeline.boolean import BooleanRetriever, QuerySyntaxError
from pipeline.search import Searcher

b = BooleanRetriever(load())
res = b.search("ឃាតកម្ម AND NOT ពុល")
res.docs       # [187, 192, 198, ...]  matching doc numbers, ascending
res.query      # '(ឃាតកម្ម AND NOT ពុល)'  how the query was parsed
res.terms      # ['ឃាតកម្ម']  positive terms (not under NOT), for highlighting
res.ignored    # []  words with nothing searchable

# or through the common interface, like the ranked models:
Searcher(load()).search("ឃាតកម្ម AND NOT ពុល", k=50, model="boolean", book=2)
```

| Name | What it is |
|---|---|
| `parse(query)` | `(tree, ignored)`; tree nodes are `Term`, `And`, `Or`, `Not`; `str(tree)` gives the bracketed form |
| `BooleanRetriever.search(query)` | `BooleanResult(docs, query, terms, ignored)` |
| `BooleanRetriever.postings(term)` | sorted docs containing the term (title ∪ text) |
| `intersect`, `union`, `difference`, `intersect_all` | merge algorithms on sorted lists |
| `positive_terms(tree)` | terms not under an odd number of `NOT`s |
| `QuerySyntaxError` | raised for invalid queries; message is user-readable |

CLI (from `backend/`):

```bash
python -m pipeline.search "ឃាតកម្ម AND NOT ពុល" --model boolean
python -m pipeline.search "(លួច OR ចោរកម្ម) AND NOT អនីតិជន" --model boolean -k 50 --book 2
```

```
parsed as   : (ឃាតកម្ម AND NOT ពុល)
matches     : 9 articles
 1. art 188    1.00  បទឧក្រិដ្ឋប្រឆាំងមនុស្សជាតិ
 2. art 193    1.00  បទឧក្រិដ្ឋសង្គ្រាម
 3. art 199    1.00  ។ បទឃាតកម្ម
 …
```

---

## 8. Tests

`backend/tests/test_boolean.py`, 30 tests (part of `python -m pytest`, 93 tests in total):

| Group | What is checked |
|---|---|
| Operators | single term in title or text; AND, OR, NOT; lowercase operators; implicit AND (also before NOT); precedence `NOT > AND > OR` and parentheses; double negation; unknown words; results in document order |
| Parsing | bracketed output; 10 invalid queries raise `QuerySyntaxError`; stopword operands ignored and reported, also in error messages; multi-word Khmer operand; prefixed noun requires only its stem; positive terms skip negated parts |
| Merge algorithms | `intersect`/`union`/`difference` equal Python set operations on 300 random list pairs |
| Random queries | 300 randomly generated nested queries (up to 3 levels of AND/OR/NOT) give the same result as evaluating their meaning directly with Python sets: tests parser, precedence and evaluation together |
| Real corpus | `ឃាតកម្ម AND ពុល` = [Art. 201]; `ការរំលោភសេពសន្ថវៈ` finds Art. 239; book filter; syntax error through `Searcher`; set identities on `លួច`/`ចោរកម្ម` (OR = ∪, AND = ∩, AND NOT = −, NOT = 672 − n); Art. 201 excluded by `NOT ពុល` |

---

## 9. Known Limitations and Future Work

| Limitation | Example | Possible improvement |
|---|---|---|
| No phrase search | `"ការរំលោភសេពសន្ថវៈ"` matches the words anywhere in the article | store word positions in the index; support quoted phrases and `NEAR/n` |
| No wildcard / prefix search | `ឃាតកម្ម*` | prefix search over the sorted vocabulary |
| Operators must be separated by spaces | `ឃាតកម្មANDពុល` is one word | could split at Latin/Khmer boundaries |
| Multi-word operand = AND of its words | `ថ្នាំពុល` also matches `ថ្នាំ … ពុល` far apart | phrase matching (positions) |
| OCR misspellings are not matched | an article with a damaged `ទ្រព្យសម្ប្ិ` does not match `ទ្រព្យសម្បត្តិ` | fix OCR errors in cleaning; the ranked `bm25_ngram` model covers this partially |
| Unranked results | 367 results for `ពន្ធនាគារ` in article order | optional "ranked Boolean": filter with the Boolean query, order by BM25 of the positive terms (`BooleanResult.terms` is already there) |
