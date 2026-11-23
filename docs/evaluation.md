# Evaluation: Test Queries, Relevance Labels and Metrics

How the search models are evaluated: the test collection, how relevance labels are made, which metrics are computed and why, and how the models are compared.

_Last updated: 2026-11-23_

> **Status:** the framework, the 24 test queries and the labelling sheet are ready. The **relevance labels must be made by the team** (§3). Results appear in `backend/evaluation/results/evaluation.md` after running the evaluation; §9 explains how to read and discuss them.

---

## Contents

1. [Overview](#1-overview)
2. [Test Collection](#2-test-collection)
3. [Making the Relevance Labels](#3-making-the-relevance-labels)
4. [Metrics](#4-metrics)
5. [Evaluating the Boolean Model](#5-evaluating-the-boolean-model)
6. [Comparing Models: Significance](#6-comparing-models-significance)
7. [Failure Analysis](#7-failure-analysis)
8. [How to Run](#8-how-to-run)
9. [Discussing the Results](#9-discussing-the-results)
10. [Challenges and Solutions](#10-challenges-and-solutions)
11. [Limitations](#11-limitations)

---

## 1. Overview

| | |
|---|---|
| **Goal** | Project criterion *Evaluation and experiments*: test queries with relevance labels; Precision, Recall, F1, MAP, nDCG; compare methods on the same test set; discuss weak or incorrect results |
| **Models evaluated** | ranked: `tfidf`, `cosine`, `bm25`, `bm25_ngram`; set-based: `boolean` |
| **Queries** | 24, `backend/evaluation/queries.json` (shared with [model_comparison.md](model_comparison.md)) |
| **Labels** | graded 0 / 1 / 2, made by the team in `backend/evaluation/labels/labels.csv` |
| **Code** | `evaluation/pool.py` (labelling sheet), `evaluation/evaluate.py` (runner + report), `evaluation/metrics.py` (metrics) |
| **Output** | `evaluation/labels/qrels.json` (labels), `evaluation/results/evaluation.md` / `.json` (results) |
| **Tests** | `backend/tests/test_evaluation.py` (20 tests; metric values checked against hand calculations) |

The workflow:

```
queries.json ──► python -m evaluation.pool ──► labels.csv ──► team labels (0/1/2) ──► python -m evaluation.evaluate
   24 queries      pool = top 10 of every model       402 rows,        + adds missed            │
                   + Boolean results (≤ 20)            blind            relevant articles        ▼
                                                                                    qrels.json + evaluation.md
```

---

## 2. Test Collection

A test collection has three parts (the "Cranfield paradigm"):

| Part | Here |
|---|---|
| Documents | the 672 articles (`data/processed/articles.jsonl`) |
| Queries | 24 Khmer queries with English translations and a type |
| Relevance labels | for each query, a grade for every article in its pool (plus articles the labellers add) |

### Queries

| Type | n | Examples | Why included |
|---|---|---|---|
| `single_term` | 8 | `ឃាតកម្ម` murder, `សំណូក` bribe, `អាវុធ` weapon | the most common kind of search; tests TF and length handling |
| `multi_term` | 8 | `ការពារខ្លួន` self-defence, `ការចាប់ជំរិត` hostage-taking | tests how terms with different IDF are combined |
| `question` | 5 | `តើអ្នកបើកបរដែលស្រវឹងត្រូវផ្តន្ទាទោសយ៉ាងដូចម្តេច` how is a drunk driver punished? | how real users ask; noise words, partial matches |
| `frequent_terms` | 3 | `ពន្ធនាគារ` prison (367 articles), `ទោសពិន័យជាប្រាក់` monetary fine | many matches, hard to rank |

Every query word was checked to occur in the corpus. Some queries (`ការពារខ្លួន`, `ការឃុំខ្លួន`, `ឃាតកម្ម`, `ចោរកម្ម`, `ការរំលោភសេពសន្ថវៈ`) were also used for spot checks while the models were built; none was changed to make a model look better, and the query set was fixed before any labelling.

---

## 3. Making the Relevance Labels

### 3.1 Pooling: which articles get labelled

Labelling all 672 articles for 24 queries (16,128 judgments) is impossible. Like TREC, we label a **pool**: the union of what the systems return. `python -m evaluation.pool` builds it:

- the **top 10 of every ranked model** (`tfidf`, `cosine`, `bm25`, `bm25_ngram`), so P@10, nDCG@10 etc. are computed on fully labelled lists;
- the **Boolean results**: all of them when there are ≤ 20, otherwise a **random sample of 20** (fixed seed), so Boolean precision can still be estimated without bias (§5).

Result: **402 query–article pairs**, 10 to 40 per query (mean 16.8). The large pools are the frequent-term queries, where the models disagree most ([model_comparison.md §4.1](model_comparison.md#41-agreement-between-models)).

At about 30 seconds per judgment, the whole sheet takes 3–4 hours, or under an hour each for a team of five.

### 3.2 The labelling sheet

`backend/evaluation/labels/labels.csv`: open it in **Excel or Google Sheets** (it is UTF-8 with a BOM, so Khmer displays correctly). One row per query–article pair:

| Column | Content |
|---|---|
| `query_id`, `query`, `query_en` | the query |
| `article_no`, `title`, `text` | the article to judge (full text) |
| `relevance` | **to fill in: 0, 1 or 2** |
| `notes` | optional: why, especially for hard cases |

Rows are **sorted by article number, without model names or ranks** (blind judging): the labeller cannot tell which model found an article or how high it ranked, so the labels cannot favour a model.

### 3.3 Labelling guidelines

Judge each article **only against the query's meaning** (`query_en` helps): *would a person who typed this query want to read this article?*

| Grade | Meaning | Example for a query *fraud* (not one of the test queries) |
|---|---|---|
| **2: relevant** | The article is **about** the query's topic: it defines the offence or concept, sets its penalty, or states a rule specifically for it. | the article defining fraud; the article setting the penalty for fraud |
| **1: partially relevant** | The article **deals with** the topic but is mainly about something else, or is useful background: an aggravating circumstance, an additional penalty, a general rule that also covers it, a closely related concept. | an article listing fraud among several offences that carry an additional penalty |
| **0: not relevant** | The query word appears only **incidentally or in another meaning**. | an article where the word for *deceive* is used about something other than the offence |

The examples deliberately do not use the 24 test queries: the grades for those are the labellers' decision.

Rules:
1. **Read the article text, not just the title.** Many titles repeat (*applicable penalties* occurs 14 times).
2. **Do not consider where the article came from.** The sheet hides it on purpose.
3. **When unsure between two grades, choose the lower one** and write a note. Consistency matters more than any single decision.
4. **One person per query.** Split the work by query (e.g. q01–q05, q06–q10, …), not by row, so each query is judged consistently.
5. **Add relevant articles that are missing.** If you know or find a relevant article that is not in the sheet (search with `python -m pipeline.search "<words>" --model boolean`, or browse the chapter in `structure.json`), **add a row** with `query_id`, `article_no` and `relevance`. Title and text are filled in the next time `python -m evaluation.pool` runs. This is important for **recall**: without it, an article that every model misses is invisible to the evaluation. Typical cases: the article uses a synonym of the query word, or a different word form ([model_comparison.md §5.5](model_comparison.md#55-failures-shared-by-all-models) shows such misses).
6. Optional, recommended: **two people label the same 3–4 queries** independently, and compare. The share of identical grades shows how subjective the labels are, which is a good point for the slides.

### 3.4 From sheet to labels

`python -m evaluation.evaluate` reads the sheet and checks it:

| Check | Message |
|---|---|
| grade not 0, 1, 2 or empty | `row 57: relevance 'yes' must be 0, 1 or 2` |
| unknown query id / article number | `row 12: unknown query_id 'q25'`, `article_no '999' is not 1-672` |
| the same pair labelled twice, differently | `row 88: q03 article 412 labelled 2 and 0` |

Any problem stops the evaluation with the list of rows to fix. The labels are saved as `labels/qrels.json` (`{query_id: {article_no: grade}}`), the format to submit as "relevance labels".

A query is evaluated only when **all** its pooled rows are labelled (`--partial` overrides this), so a half-finished sheet never produces misleading numbers. Re-running `evaluation.pool` (e.g. after adding queries) **keeps every label** and only adds new rows.

---

## 4. Metrics

Notation: for one query, *R* = number of relevant articles (grade ≥ 1), *rel(i)* = 1 if the article at rank *i* is relevant, *g(i)* its grade (0/1/2). Unlabelled articles count as not relevant. All metrics are computed per query and **averaged over queries** (every query counts equally).

The worked example used in the tests: ranking `[1, 2, 3, 4, 5]`; labels: Art. 1 → 2, Art. 2 → 0, Art. 3 → 1, Art. 9 → 2 (never retrieved). So *R* = 3.

| Metric | Formula | Example | What it tells |
|---|---|---|---|
| **P@k** (precision) | relevant in top k / k | P@5 = 2/5 = 0.40 | how much of what the user sees is useful |
| **R@k** (recall) | relevant in top k / *R* | R@5 = 2/3 = 0.67 | how much of what exists is found; important in law (missing an applicable article is costly) |
| **F1@k** | 2·P·R / (P + R) | 2·0.4·0.67/1.07 = 0.50 | balance of the two |
| **AP** (average precision) | (1/*R*) · Σ P@i over the ranks *i* of relevant articles | (1/1 + 2/3) / 3 = 0.556 | quality of the whole ranking, rewards relevant articles near the top; **MAP** = mean AP |
| **RR** (reciprocal rank) | 1 / rank of the first relevant article | 1/1 = 1.0 | how fast the user finds *something*; **MRR** = mean RR |
| **nDCG@k** | DCG@k / ideal DCG@k, DCG = Σ (2^g(i) − 1) / log₂(i + 1) | (3 + 0.5) / (3 + 3/log₂3 + 0.5) = 0.649 | uses the **grades**: a grade-2 article is worth 3× a grade-1 article, and both are worth less lower down |
| **judged@10** | labelled articles in top 10 / 10 | | sanity check: should be 100% for the pooled models |

Choices:
- **k = 10**: a results page; the pool covers each model's top 10 completely.
- **Relevant = grade ≥ 1** for the binary metrics (P, R, F1, AP, RR): partially relevant articles are still useful to a legal researcher. `REL_THRESHOLD` in `metrics.py` can be set to 2 for a stricter evaluation.
- **P@k divides by k** even if a model returns fewer than k articles: returning less is not rewarded.
- **AP uses the full ranking** (every article the model returns), so recall beyond the top 10 counts.
- Queries with **no relevant article at all** are excluded (recall and AP are undefined) and listed in the report.

---

## 5. Evaluating the Boolean Model

The Boolean model returns an unranked **set**, so ranked metrics do not apply. It is evaluated with **set precision, recall and F1** over its whole result set, using the same query text (implicit AND of the words, see [boolean_retrieval.md](boolean_retrieval.md)).

- **Precision** = relevant / judged results. When the result set is small (≤ 20) all results are in the pool, so this is exact. When it is larger (5 queries: up to 367 results for `ពន្ធនាគារ`), the pool contains a random sample of 20, and precision on that sample is an **unbiased estimate** of the precision of the whole set.
- **Recall** = relevant articles in the result set / *R*: exact with respect to the labels, since the full result set is known.
- The report also counts queries for which Boolean returns **nothing**. Currently 4 of 24: three questions (q17, q18, q21, all words must appear together) and q16 (*violence* and *minor* never occur in the same article). Recall and F1 are 0 for them: the main weakness of strict Boolean AND for natural queries.

---

## 6. Comparing Models: Significance

With 24 queries, a difference in MAP can be due to a few queries. The report tests **bm25 against every other ranked model** with a **paired randomization test** (Smucker, Allan & Carterette, 2007) on per-query AP and nDCG@10:

1. Compute the per-query differences, e.g. AP(bm25) − AP(tfidf).
2. If the models were interchangeable, each difference would be equally likely to be positive or negative. Flip the sign of each difference at random, 10,000 times, and count how often the mean difference is at least as large as the observed one.
3. That share is the **p-value**. p < 0.05: the difference is unlikely to be chance.

It is the test recommended for IR evaluation: it makes no assumption about the distribution of scores. Three comparisons are made per metric, so a p-value just under 0.05 should be read with care.

---

## 7. Failure Analysis

The report lists, for the "discuss weak or incorrect results" part of the criterion:

- **Hardest queries**: the 5 with the lowest AP averaged over the ranked models.
- **Relevant articles that no model ranks in its top 10**: the clearest failures, often from the manually added rows (§3.3 rule 5).
- **Per query**: AP of every model and Boolean F1, to find queries where one model wins or loses clearly.
- **By query type**: MAP and nDCG@10 per type (single term, multi-term, question, frequent terms).

[model_comparison.md §5](model_comparison.md#5-case-studies) already identifies likely failures to check against the labels: synonyms (`ចោរកម្ម` vs `លួច`), word senses (`លួចរត់` escape), and the double-counted noun stem.

---

## 8. How to Run

From `backend/`, with the index built (`python -m pipeline index`):

```bash
python -m evaluation.pool            # (re)build labels/labels.csv, keeps existing labels
# ... the team fills in the relevance column ...
python -m evaluation.evaluate        # check labels, write qrels.json and results/evaluation.md
python -m evaluation.evaluate --partial   # also include queries with unlabelled rows
python -m pytest tests/test_evaluation.py
```

The report (`results/evaluation.md`) has: 1. ranked-model table, 2. Boolean table, 3. significance, 4. by query type, 5. hardest queries, 6. relevant articles missed by all models, 7. per-query AP.

---

## 9. Discussing the Results

To be written once the labels exist. Questions to answer with the numbers:

1. **Which model is best overall** (MAP, nDCG@10), and is the difference significant?
2. **Does the ranking match the model comparison?** [model_comparison.md](model_comparison.md) predicts that TF-IDF favours long articles and cosine short ones; do the labels show that as lower precision?
3. **Which query types are hard**, for which model? Is Boolean competitive on single terms but poor on questions?
4. **Is `bm25_ngram` worth it?** Does it gain recall (OCR-damaged articles) without losing too much precision?
5. **The hardest queries and the missed articles**: what is the cause (synonym, word sense, segmentation, OCR, stem weighting), and which fix would help (stopwords for question words, synonym list, stem weight, dense retrieval)?

---

## 10. Challenges and Solutions

### 10.1 Boolean queries with prefixed nouns found nothing

- **Symptom:** When building the pool, 6 of 24 queries had an empty Boolean result, including `ការរំលោភសេពសន្ថវៈ` (rape), although Art. 239 `បទរំលោភសេពសន្ថវៈ` is the rape article.
- **Root cause:** The analyzer turns the prefixed noun `ការរំលោភ` into two terms (`ការរំលោភ` + stem `រំលោភ`), and a Boolean operand required **all** its terms, so it also required the noun form, which Art. 239 does not contain.
- **Solution:** A Boolean operand keeps only the stem of a prefixed noun (`operand_terms()` in `boolean.py`); every article with the noun has the stem indexed as well.
- **Result:** 4 empty results instead of 6, the remaining ones genuine (§5). Fixed **before** labelling, because Boolean results are part of the pool. Tests `test_prefixed_noun_operand_only_requires_its_stem` and `test_noun_query_finds_article_using_other_form`.

### 10.2 Re-running the pool must not destroy labels

- **Problem:** Queries may be added or models changed after labelling started; rebuilding the sheet must not lose work.
- **Solution:** `pool.py` reads the existing sheet, keeps every label (and every manually added row), adds only new pairs, and fills title/text for manual rows. Test `test_write_sheet_keeps_existing_labels`.

### 10.3 Unbiased Boolean precision without labelling hundreds of results

- **Problem:** `ពន្ធនាគារ` returns 367 articles under Boolean search; labelling all of them is not feasible, but taking the first 20 (in article order) would give a biased estimate (Book 1 only).
- **Solution:** a seeded **random sample** of 20, and precision computed over judged results only (§5).

---

## 11. Limitations

| Limitation | Effect | Mitigation |
|---|---|---|
| Pool depth 10 | relevant articles ranked below 10 by every model are only labelled if a labeller adds them; recall and AP may be overestimated | rule 5 (§3.3); the report lists added articles no model found |
| 24 queries | wide confidence intervals; one query can move MAP | significance test (§6); more queries can be added at any time (`queries.json` → `pool` → label new rows) |
| One labeller per query | labels reflect one person's judgement | guidelines (§3.3); optional double labelling of 3–4 queries |
| Queries written by the team | may not match real users' wording | mix of types, including natural questions |
| Parameters not tuned | BM25 uses standard `k1`, `b` and weights | tuning on 24 queries would overfit; if done, use cross-validation (tune on some queries, report on the others) |
