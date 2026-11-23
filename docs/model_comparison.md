# Model Comparison: TF-IDF vs Cosine vs BM25

How TF-IDF, cosine similarity and BM25 rank the articles of the Criminal Code for the same queries, which part of each formula causes the differences, and where all of them fail.

_Last updated: 2026-11-23_

---

## Contents

1. [Overview](#1-overview)
2. [The Models Side by Side](#2-the-models-side-by-side)
3. [Method](#3-method)
4. [Results](#4-results)
5. [Case Studies](#5-case-studies)
6. [Main Differences: Summary](#6-main-differences-summary)
7. [Challenges and Solutions](#7-challenges-and-solutions)
8. [Limitations](#8-limitations)
9. [How to Run](#9-how-to-run)

---

## 1. Overview

| | |
|---|---|
| **Goal** | Project criterion *Classical search methods*: "Compare how they rank documents for the same queries and explain the main differences" |
| **Models** | `tfidf` (TF-IDF score), `cosine` (TF-IDF + cosine similarity), `bm25` (BM25F), `bm25_ngram` (BM25F + character bigrams); formulas in [indexing_and_ranking.md §4](indexing_and_ranking.md#4-ranking-models) |
| **Queries** | 24 Khmer queries in 4 types: `backend/evaluation/queries.json` |
| **Code** | `backend/evaluation/compare_models.py` |
| **Results** | `backend/evaluation/results/model_comparison.md` (all tables + top 5 per query and model) and `.json` |
| **Reproducible** | same output files byte for byte across runs (checked with two different Python hash seeds) |

This comparison needs **no relevance labels**. It measures **how** the rankings differ and **which component** of each formula causes it. Which model is **better** is measured in the evaluation step with labelled queries. Section 4.4 adds one automatic quality check (known-item search) that does not need labels.

**Main findings:**
1. **Length normalization is the biggest difference.** TF-IDF and cosine use identical term weights, yet they agree least of all pairs (same top result for only 25% of queries). TF-IDF's top 10 are 37% longer than the average article. BM25 with length normalization switched off behaves almost exactly like TF-IDF.
2. **TF-IDF cannot break ties.** For 64% of its top-10 results, another result has exactly the same score, so the order is decided by article number, not by the model. BM25: 7%, cosine: 0%.
3. **Cosine undervalues the second query word.** Its normalization uses the length of the whole article vector, so a less rare query word adds almost nothing. On "self-defence" it ranks a *national defence* article 2nd.
4. **BM25 finds an article by its title most reliably:** MRR 0.971 vs 0.882 (TF-IDF) and 0.871 (cosine) on 489 title queries. Saturation, the title boost and length normalization each contribute.
5. **All three share the same failures**: synonyms (`ចោរកម្ម` vs `លួច`), word senses (`លួច` "steal" in `លួចរត់` "escape"), and the double-counted prefix stem. None of these can be fixed by changing the ranking formula.

---

## 2. The Models Side by Side

| Component | `tfidf` | `cosine` | `bm25` (BM25F) |
|---|---|---|---|
| Term frequency | `1 + ln tf` (grows without limit, slowly) | `1 + ln tf` | `tf · (k1+1) / (tf + k1)` (bounded by `k1 + 1` = 2.2) |
| IDF | `ln(N / df)`, 0 for a term in every article | `ln(N / df)` | `ln(1 + (N − df + 0.5)/(df + 0.5))`, always > 0 |
| Length normalization | **none** | divide by `‖d‖`, the length of the **whole** article vector | `1 − b + b · |d|/avgdl` inside the TF part, `b = 0.75` |
| Title vs text | one bag of words | one bag of words | title weighted ×2 |
| Query term weights | each distinct query term once | query vector, also normalized | each distinct query term once |
| Score range | 0 … ∞ | 0 … 1 | 0 … ∞ |

`tfidf` and `cosine` differ **only** in the normalization. `bm25` differs from `tfidf` in four components, and the ablation in §4.3 switches them off one at a time to see which one matters.

---

## 3. Method

### 3.1 Queries

24 queries (`backend/evaluation/queries.json`), written to cover different situations:

| Type | n | Examples | What it tests |
|---|---|---|---|
| `single_term` | 8 | `ឃាតកម្ម` murder, `សំណូក` bribe, `អាវុធ` weapon | only TF, IDF does not change the order |
| `multi_term` | 8 | `ការពារខ្លួន` self-defence, `ទទួលខុសត្រូវព្រហ្មទណ្ឌ` criminal responsibility | how terms with different IDF are combined |
| `question` | 5 | `តើការលួចទ្រព្យសម្បត្តិត្រូវទទួលទោសអ្វី` "what is the penalty for stealing property?" | natural language with noise words |
| `frequent_terms` | 3 | `ពន្ធនាគារ` prison (in 367 of 672 articles), `ទោសពិន័យជាប្រាក់` monetary fine | low IDF: length and TF decide everything |

Every query word was checked to exist in the corpus. The same file will be used for the evaluation step (with relevance labels added).

### 3.2 Measures

For every query, each model's top 10 (`K = 10`):

| Measure | Definition | Meaning |
|---|---|---|
| **overlap@10** | \|A ∩ B\| / 10 | share of articles both models return, ignoring order |
| **AO@10** (average overlap) | mean over d = 1…10 of \|A₁..d ∩ B₁..d\| / d | agreement weighted towards the top: two lists with the same 10 articles in reverse order score 1.0 on overlap but ≈0.5 on AO (Webber et al., 2010) |
| **same top-1** | A₁ = B₁ | do they agree on the best article? |
| **text length** | mean text length (terms) of the top 10 | length bias (corpus average: 49.0) |
| **title match** | share of the top 10 with a query term in the title | effect of field weighting |
| **query terms matched** | mean share of distinct query terms present | "coordination": all words or just one |
| **tf per matched term** | mean occurrences of a matched query term | effect of TF saturation |
| **tied scores** | share of the top 10 whose score equals another one's | order decided by the tie-breaker (article number), not by the model |

### 3.3 Ablation

To find out *why* BM25 differs from TF-IDF, BM25 is run again with **one component changed at a time** (all through `search.Params`, no rebuild):

| Variant | Change | Switches off |
|---|---|---|
| `b=0` | `b = 0` for all fields | length normalization |
| `k1=1000` | `k1 = 1000` | TF saturation (TF becomes ≈ linear) |
| `k1=0` | `k1 = 0` | TF entirely (score = sum of IDFs of matched terms) |
| `w_title=1` | title weight 2 → 1 | title boost |

### 3.4 Known-item search

The 489 articles whose title is unique (titles like `ទោសដែលត្រូវអនុវត្ត` *applicable penalties* occur up to 46 times and are skipped) are searched by their own title. Measured: rank of that article → **MRR**, **success@1**, **success@10**. This is an automatic check of "can a model find an article by its name", **not** a relevance evaluation: it is biased towards models that weight the title (see [§8](#8-limitations)).

---

## 4. Results

All numbers from `backend/evaluation/results/model_comparison.md`.

### 4.1 Agreement between models

| Pair | overlap@10 | AO@10 | same top-1 |
|---|---|---|---|
| tfidf vs cosine | 0.76 | **0.58** | **25%** |
| tfidf vs bm25 | 0.79 | 0.68 | 50% |
| cosine vs bm25 | 0.85 | 0.71 | 50% |
| bm25 vs bm25_ngram | 0.88 | 0.87 | 83% |

- The models mostly return the **same articles** (overlap 0.76–0.88) but in a **different order** (AO 0.58–0.71). They agree on the best article only half of the time or less.
- **TF-IDF and cosine disagree most**, although they use the same weights. The only difference between them is normalization, so normalization alone changes the top result for 3 out of 4 queries.
- `bm25_ngram` stays close to `bm25` (AO 0.87): with weight 0.3 the bigram field reorders, but rarely replaces, results.

Agreement depends strongly on the query type (AO@10 per query in the results file):

| Query | tfidf/cosine | tfidf/bm25 | cosine/bm25 |
|---|---|---|---|
| `ការចាប់ជំរិត` hostage-taking (rare terms, 9 articles) | 0.88 | 0.91 | 0.91 |
| `ឃាតកម្ម` murder (10 articles) | 0.72 | 0.82 | 0.90 |
| `ទទួលខុសត្រូវព្រហ្មទណ្ឌ` criminal responsibility | 0.31 | **0.39** | 0.80 |
| `ពន្ធនាគារ` prison (367 articles) | **0.20** | 0.62 | 0.41 |
| `ទោសពិន័យជាប្រាក់` monetary fine (3 frequent words) | **0.16** | 0.43 | 0.58 |

The rarer the query terms, the fewer articles match and the more the models agree. With **frequent terms**, IDF is low and similar for all articles, so ranking is decided by TF and length handling: exactly where the models differ.

### 4.2 What each model's top 10 look like

| Model | text length | title match | query terms matched | tf per matched term | tied scores |
|---|---|---|---|---|---|
| `tfidf` | **67.1** | 60% | 82% | **2.17** | **64%** |
| `cosine` | 48.7 | 62% | 80% | 1.91 | 0% |
| `bm25` | 48.4 | **65%** | 80% | 2.00 | 7% |
| `bm25_ngram` | 47.8 | 64% | 77% | 1.97 | 1% |
| *corpus average* | *49.0* | | | | |

- **TF-IDF prefers long articles** (67 terms, 37% above average) that repeat the query terms (2.17 times each). Nothing in its formula penalizes length: a long article has more chances to contain and repeat a term.
- **Cosine and BM25 return articles of average length.** Both normalize, in different ways (§5).
- **BM25 has the most title matches** (65%) because of the title weight.
- **TF-IDF produces ties.** For a single-term query, its score is `(1 + ln tf) · idf`: every article with the same tf gets exactly the same score, and tf takes only a few values (1, 2, 3…). The order among them is then decided by the tie-breaker (article number), which says nothing about relevance. Cosine divides by `‖d‖`, which is different for every article, so it never ties. BM25 includes the article length in the TF part, so ties remain only for articles of the same length.

### 4.3 Ablation: what makes BM25 different from TF-IDF

AO@10 of each BM25 variant with the original models, and the profile of its top 10:

| BM25 variant | vs bm25 | vs **tfidf** | vs cosine | text length | tied |
|---|---|---|---|---|---|
| default | 1.00 | 0.68 | 0.71 | 48.4 | 7% |
| **`b=0`: no length normalization** | 0.73 | **0.88** | 0.57 | **67.1** | **62%** |
| `k1=1000`: no TF saturation | 0.85 | 0.64 | 0.68 | 41.6 | 6% |
| `k1=0`: no TF (IDF only) | 0.54 | 0.59 | 0.50 | 62.8 | 97% |
| `w_title=1`: no title boost | 0.91 | 0.69 | 0.72 | 49.0 | 6% |

- **Length normalization is the main difference.** Switching it off (`b=0`) makes BM25 agree with TF-IDF at 0.88 instead of 0.68, and its top 10 get exactly TF-IDF's average length (67.1) and almost the same share of ties (62% vs 64%). The remaining differences (saturation, IDF formula, title boost) matter much less for these queries.
- **TF saturation matters, but differently than expected.** Without it (`k1=1000`), TF counts linearly, and BM25 moves *away* from TF-IDF (0.64). TF-IDF already dampens TF with the logarithm; linear TF combined with length normalization favours **short articles that repeat a term** (41.6 terms, 2.22 occurrences per term, 70% title matches since title occurrences now count ×2 linearly).
- **TF is needed to break ties.** With `k1=0` only the set of matched terms counts: 97% of top-10 scores are tied, and the ranking is far from every other model (AO ≤ 0.59).
- **The title boost changes little** in the ranking of these queries (AO 0.91), but it matters for finding articles by title (§4.4).

### 4.4 Known-item search: 489 article titles as queries

| Model | MRR | success@1 | success@10 |
|---|---|---|---|
| `tfidf` | 0.882 | 81.6% | 98.0% |
| `cosine` | 0.871 | 80.6% | 98.4% |
| **`bm25`** | **0.971** | **95.3%** | 99.2% |
| `bm25_ngram` | 0.963 | 93.9% | 99.4% |
| `bm25`, `b=0` | 0.951 | 92.6% | 98.8% |
| `bm25`, `k1=1000` | 0.924 | 86.9% | 99.2% |
| `bm25`, `k1=0` | 0.921 | 87.7% | 98.8% |
| `bm25`, `w_title=1` | 0.936 | 89.2% | 99.2% |

- BM25 puts the article first for 95% of its titles; TF-IDF and cosine for about 81%.
- Every BM25 component helps; removing **saturation** costs most (MRR −0.047), then the **title boost** (−0.035) and **length normalization** (−0.020).
- Even BM25 without the title boost (0.936) beats TF-IDF and cosine. So the advantage is not only the title weight.
- All models find every article somewhere; success@10 is ≥ 98% for all. The differences are at the very top.
- `bm25_ngram` is slightly worse than `bm25` here: titles are typed correctly, so the bigram field only adds noise. Its use is OCR-damaged words ([indexing_and_ranking.md §8.3](indexing_and_ranking.md#83-do-bigrams-really-recover-ocr-damaged-articles)).

---

## 5. Case Studies

Top 5 per model; numbers in brackets are article text lengths in terms.

### 5.1 One frequent term: `ពន្ធនាគារ` (prison, in 367 articles)

| Rank | `tfidf` | `cosine` | `bm25` |
|---|---|---|---|
| 1 | 49 (45) 1.58 | 46 (35) 0.10 | 49 (45) 1.10 |
| 2 | 160 (92) **1.58** | 97 (30) 0.09 | 46 (35) 1.08 |
| 3 | 46 (35) 1.44 | 195 (**8**) 0.08 | 95 (24) 1.05 |
| 4 | 85 (55) **1.44** | 47 (27) 0.08 | 97 (30) 1.04 |
| 5 | 87 (56) **1.44** | 95 (24) 0.07 | 85 (55) 1.00 |

AO@10: tfidf/cosine **0.20**, tfidf/bm25 0.62, cosine/bm25 0.41.

With one query term, IDF is the same for every article and only TF and length decide:
- **TF-IDF ranks by tf only:** score 1.58 = tf 5, 1.44 = tf 4. Articles 49 and 160 tie, and so do 46, 85 and 87; their order is just article number.
- **Cosine ranks by tf relative to everything else in the article:** short articles win. Art. 195 (*penalty incurred*) has only 8 terms.
- **BM25 sits in between:** saturation makes tf 4 and tf 5 nearly equal, and length normalization moderates, without letting the shortest article win.

### 5.2 Two terms with different IDF: `ការពារខ្លួន` (self-defence)

Query terms `ការពារ` (defend, df 28, rare) and `ខ្លួន` (self, df 134, common).

| Rank | `tfidf` | `cosine` | `bm25` |
|---|---|---|---|
| 1 | **33** self-defence | **33** self-defence | **33** self-defence |
| 2 | 479 disclosure of *national defence* secrets (92) | 477 *national defence* secrets | **35** state of necessity |
| 3 | **34** presumption of legitimate defence | **35** | **34** |
| 4 | **35** | 474 *national defence* services | 479 |
| 5 | 477 | 478 *national defence* secret | 477 |

- **BM25** returns the three articles on self-defence and necessity (33, 35, 34) on top: `ខ្លួន` adds 1.6–1.8 points to articles that contain both words.
- **Cosine** puts three *national defence* articles (477, 474, 478) in its top 5. They contain `ការពារ` (as in `ការការពារជាតិ`, national defence) but **not** `ខ្លួន`. In cosine the contribution of `ខ្លួន` is only 0.02–0.03, because each term's weight is divided by the length of the whole article vector and `ខ្លួន` has a low IDF. Matching the second word hardly helps.
- **TF-IDF** ranks Art. 479 (92 terms, also about national defence) 2nd: it is long and contains `ការពារ` 3 times and `ខ្លួន` twice (contribution 2.73), and the TF-IDF formula never penalizes the length.

### 5.3 Long, repetitive articles: `ទទួលខុសត្រូវព្រហ្មទណ្ឌ` (criminal responsibility)

The lowest TF-IDF/BM25 agreement of all queries (AO 0.39).

| Rank | `tfidf` | `cosine` | `bm25` |
|---|---|---|---|
| 1 | 42 legal persons (35) | **37** absence of criminal responsibility (7) | **37** (7) |
| 2 | 31 no responsibility, mental disorder (67) | **24** principle of personal responsibility (7) | 42 (35) |
| 3 | 187 legal persons (**173**) | 42 (35) | **24** (7) |
| 4 | 192 legal persons (**177**) | **38** `នីតិភាពនៃការទទួលខុសត្រូវព្រហ្មទណ្ឌ` (11) | **38** (11) |
| 5 | 198 legal persons (**178**) | 31 (67) | 273 legal persons (128) |

- Articles 187, 192 and 198 have the same title (*criminal responsibility of legal persons*, used for 16 articles across the code) and are about 175 terms long. TF-IDF ranks them 3rd to 5th because they are long and repeat the query terms.
- Cosine and BM25 agree (AO 0.80): they rank the short, general articles of Book 1 that **define** criminal responsibility (37, 24, 38). These are what a user asking about "criminal responsibility" most likely wants.

### 5.4 The title boost: `អាវុធ` (weapon)

| Rank | `tfidf` | `cosine` | `bm25` |
|---|---|---|---|
| 1 | **81** *weapons and objects considered weapons* (42) | 564 use of weapons (14) | **81** (42) |
| 2 | 488 manufacturing / trading weapons (85) | 366 aggravation: use of a weapon (21) | 366 (21) |
| 3 | 490 possession / transport of weapons (63) | 493 confiscation (19) | 564 (14) |

Art. 81 (`អាវុធ និងវត្ថុដែលត្រូវបានចាត់ទុកដូចអាវុធ`) **defines** weapons and starts its title with `អាវុធ`. BM25 ranks it first (title ×2), and TF-IDF does too (high tf). Cosine prefers the shortest articles mentioning a weapon; Art. 81 is not in its top 5.

### 5.5 Failures shared by all models

Changing the formula does not fix these; they are limits of matching words.

**Synonyms: `ចោរកម្ម` (theft).** All models return the same 5 articles (399–402, 89), which are about *receiving stolen goods*. The theft articles themselves (353 ff., `និយមន័យនៃអំពើលួច` *definition of theft*) use the verb `លួច`, not the noun `ចោរកម្ម`, and are not found by any model. → synonym list or semantic search.

**Word senses and segmentation: `តើការលួចទ្រព្យសម្បត្តិត្រូវទទួលទោសអ្វី` (what is the penalty for stealing property?).** All three models fill 6–7 of their top 7 with articles from the chapter on **escape** (560–574). Art. 353, the definition of theft, is in **no** model's top 10. Causes:
1. `ការលួចរត់` (escape, literally "steal-run") is segmented as `ការលួច | រត់`, so escape articles contain `ការលួច` *and* its stem `លួច`.
2. The query's `ការលួច` also produces both terms, so the concept "steal" is **counted twice** (see the limitation in [indexing_and_ranking.md §12](indexing_and_ranking.md#12-known-limitations-and-future-work)). Art. 353 has `អំពើលួច`, which is segmented as `អំពើ | លួច` and only matches `លួច`.
3. Bag-of-words models cannot tell the two senses apart.

→ legal dictionary for the segmenter (`លួចរត់` as one word), lower weight for stems, semantic search.

**Double-counted stem: `តើអ្នកបើកបរដែលស្រវឹងត្រូវផ្តន្ទាទោសយ៉ាងដូចម្តេច` (how is a drunk driver punished?).** BM25 ranks Art. 297 (`បទស្រវឹងជាក់ស្តែង`, *manifest drunkenness*, one of the 3 articles with `ស្រវឹង` drunk) **3rd**, behind two articles on drivers refusing orders or fleeing. "Driver" is counted twice (`អ្នកបើកបរ` 5.7 + stem `បើកបរ` 4.1) against "drunk" once (9.3). TF-IDF and cosine rank Art. 297 first. Here BM25's handling of the stems hurts it.

**Question words are not stopwords.** `តើ` (question particle, 25 articles) and `អ្វី` (what, 35) are kept as terms. They contribute little in these examples, but add noise. → add question words to the stopword list ([text_analysis.md](text_analysis.md#stopwords)).

---

## 6. Main Differences: Summary

| | `tfidf` | `cosine` | `bm25` |
|---|---|---|---|
| Long articles | **favoured** (top 10: 67 terms vs 49 average) | neutral | neutral |
| Repeated terms | rewarded slowly (log) without limit | rewarded relative to the article's other words | rewarded, then capped (saturation) |
| Second, more common query word | counts fully | **counts very little** | counts moderately |
| Ties | **frequent** (64% of top 10) | none | rare (7%) |
| Title | no special weight | no special weight | ×2 |
| Finding an article by its title (MRR) | 0.882 | 0.871 | **0.971** |
| Typical mistake | long boilerplate articles (§5.3) | short articles matching only the rarest word (§5.2) | over-weights a concept whose noun and stem both match (§5.5) |

In one sentence each:
- **TF-IDF** measures *how much* of the query an article contains, so long articles win.
- **Cosine** measures *what share* of an article is about the query's rarest word, so short articles win and secondary query words are nearly ignored.
- **BM25** measures *whether* an article contains the query terms (with diminishing returns for repetition), adjusted for length. It is the most balanced of the three, and the main reason is its length normalization.

---

## 7. Challenges and Solutions

### 7.1 TF-IDF and cosine were one model

- **Problem:** The criterion lists "TF-IDF, cosine similarity, and BM25". Our first `tfidf` model was TF-IDF *with* cosine, so the effect of cosine normalization could not be measured.
- **Solution:** Two models: `tfidf` (sum of tf-idf weights, *IIR* eq. 6.9) and `cosine`. Same weights, so any difference comes from the normalization.
- **Result:** the most informative comparison of all (§4.1, §5.1): normalization alone changes the top result for 75% of queries.

### 7.2 Results changed between runs

- **Symptom:** Running the experiment twice gave different numbers for the `k1=0` variant (text length 65.1, then 65.3).
- **Root cause:** The scorers looped over `set(terms)`. Python randomizes string hashing per run, so the set order (and with it the order in which term contributions are summed) changed. Floating-point addition is not associative: the same contributions summed in another order can differ in the last digit (~1e-16). With `k1=0` almost every score is a tie, and this noise decided the tie-break.
- **Solution:** (1) Iterate terms in query order (`unique()` in `search.py`); (2) round scores to 9 decimals before sorting, so that equal scores are real ties, broken by article number.
- **Verification:**
  - A random search found a 6-article corpus where the old code ranked articles 2 and 6 differently depending on the term order. It is now a test (`test_ranking_does_not_depend_on_term_order`).
  - The full experiment gives byte-identical output files with `PYTHONHASHSEED=11` and `22`.

### 7.3 Ties in TF-IDF are invisible in a ranked list

- **Symptom:** TF-IDF's ranking for "prison" looked plausible, but the scores showed groups of identical values.
- **Solution:** The `tied scores` measure (§3.2). It shows that 64% of TF-IDF's top 10 are ordered by the tie-breaker. This is a real weakness of TF-IDF on short queries, not something visible from the result list alone.

---

## 8. Limitations

| Limitation | Consequence |
|---|---|
| No relevance labels | agreement measures *difference*, not *quality*. The evaluation step will measure precision, recall, MAP and nDCG on these same queries. |
| Known-item search favours title-weighted models | BM25's advantage there is partly built in (title ×2). Even without the title boost it scores higher (0.936 vs 0.882), but it is a check of "find by name", not of answering questions. |
| 24 queries, written by the team | enough to show systematic effects (the ablation results are consistent across query types), not a representative sample of real user queries |
| Case-study interpretations ("what the user most likely wants") are our judgement | they will be checked when the relevance labels are made |
| Parameters (`k1`, `b`, weights) not tuned | BM25 could become better or worse after tuning in the evaluation step |

---

## 9. How to Run

From `backend/`, with the index built (`python -m pipeline index`):

```bash
python -m evaluation.compare_models          # ~15 s
```

Writes `backend/evaluation/results/model_comparison.md` (tables §1–4 and the top 5 of every model for every query) and `model_comparison.json` (all rankings, profiles and AO values). Look at single queries with the search CLI:

```bash
python -m pipeline.search "ពន្ធនាគារ" --model tfidf -k 10
python -m pipeline.search "ពន្ធនាគារ" --model cosine -k 10
python -m pipeline.search "ពន្ធនាគារ" -k 10
```

To compare more queries, add them to `backend/evaluation/queries.json` (`id`, `query`, `en`, `type`) and run the script again.
