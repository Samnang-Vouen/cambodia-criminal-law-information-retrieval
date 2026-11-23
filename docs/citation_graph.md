# Citation Graph: PageRank and HITS on the Criminal Code

How the cross-references between articles were extracted into a directed graph, how PageRank and HITS were computed on it, and which articles score high and why.

_Last updated: 2026-11-23_

---

## Contents

1. [Overview](#1-overview)
2. [From Web Graph to Citation Graph](#2-from-web-graph-to-citation-graph)
3. [Extracting the Citations](#3-extracting-the-citations)
4. [The Graph](#4-the-graph)
5. [PageRank](#5-pagerank)
6. [HITS](#6-hits)
7. [Results: Which Articles Score High and Why](#7-results-which-articles-score-high-and-why)
8. [What This Means for Search](#8-what-this-means-for-search)
9. [Challenges and Solutions](#9-challenges-and-solutions)
10. [Tests](#10-tests)
11. [Limitations](#11-limitations)
12. [How to Run](#12-how-to-run)

---

## 1. Overview

| | |
|---|---|
| **Goal** | Project criterion *Web graph analysis (if applicable)*: build a directed graph from links, apply PageRank and HITS, explain which nodes score high and why |
| **Applies because** | articles of the code **cite each other** ("… as set out in Article 179 of this code"), a citation network like the links between web pages |
| **Code** | `backend/pipeline/citation_graph.py`, stage `graph`: `python -m pipeline graph` |
| **Output** | `data/processed/citations.json` (every citation with its evidence), `data/processed/graph_scores.json` (PageRank, hub, authority, in/out degree per article) |
| **Algorithms** | PageRank and HITS written out by hand (power iteration), checked against `networkx` |
| **Tests** | `backend/tests/test_citation_graph.py` (17 tests) |

**Main findings:**
1. The graph has 672 articles and **268 citation edges**; 74 articles cite, 107 are cited. All 273 written citation numbers were verified against the title of the cited article.
2. **HITS** finds a clean hub/authority structure. The **hubs** are 16 articles titled *criminal responsibility of legal persons* (one at the end of many offence chapters); the **authorities** are the general rules they all cite: Art. 42 and the additional penalties for legal persons, Arts. 170–181.
3. **PageRank** ranks different articles first: **base offences** such as Art. 210 *torture and acts of cruelty*, which are cited by a series of follow-up articles that cite **only** them.
4. The difference is explained by the shape of the graph: it is almost **flat** (only 2 citation paths of length 2), so PageRank reduces to "in-links weighted by 1/out-degree of the citing article", while HITS rewards being cited **together** by many articles that cite the same set.

---

## 2. From Web Graph to Citation Graph

| Web graph | Citation graph of the code |
|---|---|
| node = web page | node = article (672) |
| edge = hyperlink from page A to page B | edge = article A cites article B ("មាត្រា B … នៃក្រមនេះ") |
| PageRank: a page is important if important pages link to it | an article is important if (important) articles refer to it |
| HITS hub: a page with links to many good pages (e.g. a link directory) | an article that refers to many key provisions |
| HITS authority: a page linked to by many good hubs | a provision that many articles rely on |

Citation networks of laws and court decisions are a well-known application of these algorithms (the same idea as citation analysis of scientific papers, from which PageRank itself was inspired).

---

## 3. Extracting the Citations

### 3.1 How the code cites

A citation is written as the article number, followed by the cited article's **title in brackets**, followed by "of this code":

```
… តាមបែបបទកំណត់ដោយមាត្រា ១៧៩ (ការរឹបអូស និងសិទ្ធិតតិយជន) នៃក្រមនេះ ។
… in the manner set by Article 179 (Confiscation and rights of third parties) of this code.
```

The OCR alphabet has no brackets, so the opening bracket became a letter that merges with the title: `(ការ…` → `កោរ…`, `(បទ…` → `បេទ…`. The title text itself survives.

### 3.2 Extraction rules

| Rule | Example | Count |
|---|---|---|
| Find every `មាត្រា` + number in the text of each article | `មាត្រា ១៧៩` | 277 mentions |
| **Verify** the number: the text after (or before) the number must match the **title of that article** (≥ 50% of the title's character bigrams) | `១៧៩ កោររឹបអូស និងសិទ្ធិតតិយជន` ↔ title of Art. 179 | **273 verified** |
| If it does not match, try the **OCR digit confusions** found in `clean.py` (`៣↔៧`, `៤↔៨`); accept a variant whose title matches | | 0 needed |
| **Drop citations of other laws**: "of the Constitution", "of the law on …", "Code of Criminal Procedure" | `មាត្រា ២៧ នៃរដ្ឋធម្មនុញ្ញ` | **4 dropped** |
| **Expand ranges**: "Article A (…) **to** (`ដល់`) Article B (…)" cites every article from A to B | Art. 199: `មាត្រា ២០០ … ដល់មាត្រា ២០៥` → 200, 201, …, 205 | **17 added** |
| A number that matches no title is kept but flagged `unverified` | | 0 |
| An article citing itself is not an edge | `បទប្បញ្ញត្តិនៃមាត្រានេះ` ("the provisions of this article") has no number anyway | |

Result: **290 citations** (273 verified + 17 from ranges), every one checked, stored with its evidence in `citations.json`:

```json
{"source": 519, "target": 179, "raw": "179", "status": "verified", "title_match": 0.958,
 "context": " កោររឹបអូស និងសិទ្ធិតតិយជន "}
```

The bracketed title makes this much more reliable than a plain "find `មាត្រា` + number": a wrong number (OCR error, citation of another law) would have to match the title of the wrong article by chance.

---

## 4. The Graph

| | |
|---|---|
| Nodes | 672 (all articles, including those without citations) |
| Edges (distinct citing → cited pairs) | 268 |
| Articles that cite | 74 (45 of them cite exactly one article) |
| Articles that are cited | 107 |
| Articles that both cite and are cited | 2 (Arts. 55 and 269) |
| Paths of length 2 | 2 (153 → 55 → 53 and 273 → 269 → 265) |
| Most citations from one article | 14 (Art. 230) |
| Most cited | 16 (Arts. 42, 171, 172, 180, 181) |

Two citation patterns make up almost the whole graph:

1. **"Legal persons" articles → general rules.** Many offence chapters end with an article titled *ការទទួលខុសត្រូវព្រហ្មទណ្ឌរបស់នីតិបុគ្គល* (criminal responsibility of legal persons). Each states that companies can be liable "under the conditions of Article 42", lists the additional penalties that apply, and cites the article that defines each penalty (Arts. 170–181) plus the offences of its chapter. 16 such articles produce most of the edges.
2. **Follow-up articles → their base offence.** Articles on aggravating circumstances or special cases refer back to the article that defines the offence: Arts. 211–215 each cite Art. 210 (*torture and acts of cruelty*) and nothing else.

The graph is almost **flat**: articles that cite are (except two) never cited themselves. This decides how PageRank behaves (§7.3).

---

## 5. PageRank

### 5.1 Definition

A "random reader" starts at a random article. At every step, with probability *d* = 0.85, they follow one of the current article's citations (chosen at random); otherwise, or when the article cites nothing, they jump to a random article. PageRank is the share of time spent on each article:

$$
PR(v) = \frac{1-d}{N} + d \left( \sum_{u \to v} \frac{PR(u)}{\text{out}(u)} + \sum_{u \text{ dangling}} \frac{PR(u)}{N} \right)
$$

- *N* = 672; out(*u*) = number of articles *u* cites.
- **Dangling** articles (those citing nothing, 598 of 672) spread their rank evenly over all articles. Without this, rank would "leak" out of the graph and the scores would not sum to 1.
- Each citing article **divides its vote** among the articles it cites: one citation from an article citing only you is worth as much as 14 citations from articles citing 14 others each.

### 5.2 Computation

Power iteration (`pagerank()` in `citation_graph.py`): start with 1/N everywhere, apply the formula until the scores change by less than 10⁻¹² (sum of absolute changes). The result matches `networkx.pagerank` to 10⁻⁸ (tested on random graphs).

An article that nobody cites gets the base score **0.00136** (≈ 1/672 = 0.00149, a bit less because some rank flows to cited articles).

---

## 6. HITS

### 6.1 Definition

HITS (Kleinberg, 1999) gives every node two scores that reinforce each other:

$$
\text{authority}(v) = \sum_{u \to v} \text{hub}(u) \qquad \text{hub}(u) = \sum_{u \to v} \text{authority}(v)
$$

- A good **authority** is cited by good hubs.
- A good **hub** cites good authorities.

### 6.2 Computation

Start with hub = 1 everywhere; compute authorities from hubs, hubs from authorities, normalize both to sum 1; repeat until the hub scores change by less than 10⁻¹² (`hits()` in `citation_graph.py`). Matches `networkx.hits` to 10⁻⁶. Articles without any citation get 0.

HITS was designed for the subgraph of pages matching a query; here it is computed on the whole code (the graph is small), which shows its global structure.

---

## 7. Results: Which Articles Score High and Why

### 7.1 HITS authorities: the general rules for companies

| Article | Title | Authority | Cited by |
|---|---|---|---|
| 42 | ការទទួលខុសត្រូវព្រហ្មទណ្ឌរបស់នីតិបុគ្គល *criminal responsibility of legal persons* | 0.0894 | 16 |
| 171 | ការដាក់ឱ្យស្ថិតនៅក្រោមការត្រួតពិនិត្យតាមដានរបស់តុលាការ *placement under judicial supervision* | 0.0894 | 16 |
| 172 | ការហាមឃាត់ចំពោះការធ្វើសកម្មភាព *prohibition from carrying out an activity* | 0.0894 | 16 |
| 180 | ការបិទផ្សាយសេចក្តីសម្រេច *posting of the judgment* | 0.0894 | 16 |
| 181 | ការផ្សាយសេចក្តីសម្រេចតាមមធ្យោបាយទូរគមនាគមន៍សោតទស្សន៍ *publication of the judgment by audiovisual means* | 0.0894 | 16 |
| 170 | ការរំលាយ និងការជម្រះបញ្ជីនៃនីតិបុគ្គល *dissolution and liquidation of a legal person* | 0.0811 | 14 |
| 173 | ការបណ្តេញចេញពីលទ្ធកម្មសាធារណៈ *exclusion from public procurement* | 0.0662 | 11 |
| 174 | ការហាមឃាត់ចំពោះការអំពាវនាវជាសាធារណៈឱ្យធ្វើការសន្សំ *prohibition on public appeals for savings* | 0.0662 | 11 |

All in **Book 1** (general provisions). They are the "toolbox" that the special part of the code reuses: whenever a chapter makes companies liable, it refers back to Art. 42 for the conditions and to Arts. 170–181 for the penalties. The five articles cited by all 16 hubs share the top score; the others are cited by fewer hubs (some chapters list fewer penalties).

### 7.2 HITS hubs: the "legal persons" articles

| Article | Book | Cites | Hub |
|---|---|---|---|
| 230 | 2 | 42, 170–174, 176–181, 226, 227 (14) | 0.0745 |
| 277 | 2 | 42, 170–174, 176–181, 274 (13) | 0.0739 |
| 559 | 4 | 42, 170–174, 178–181, 548, 554, 556 (13) | 0.0697 |
| 625 | 4 | 42, 170–174, 178–181, 605–607 (13) | 0.0697 |
| 644 | 4 | 42, 170–174, 178–181, 638, 640 (12) | 0.0691 |
| 283, 409, 519 | 2, 3, 4 | 42, 170–174, 178–181 + 1 offence (11) | 0.0686 |

All the top hubs have the same title, *criminal responsibility of legal persons*. They are hubs because they cite **the same strong authorities**: a hub's score is the sum of the authorities it cites, so citing the whole penalty list (and the offences of the chapter) gives the highest score. They are not cited by anything (hub ≠ important by itself).

HITS separates the two roles exactly as on the web: the **list pages** (hubs) and the **reference pages** they point to (authorities).

### 7.3 PageRank: base offences with their own follow-up articles

| Rank | Article | Title | PageRank | Cited by (their out-degree) |
|---|---|---|---|---|
| 1 | 210 | បទទារុណកម្ម និងអំពើឃោរឃៅ *torture and acts of cruelty* | 0.0071 | 211, 212, 213, 214, 215 (each 1) |
| 2 | 413 | ស្ថានទម្ងន់ទោសដោយសារការប្រើប្រាស់វិធីគ្រោះថ្នាក់ … *aggravation: use of dangerous means* | 0.0060 | 414–417 (each 1) |
| 3 | 53 | ប្រភេទនៃទោសបន្ថែម *types of additional penalties* | 0.0058 | 55, 71, 152 (each 1); 55 is itself cited by 153 |
| 4 | 265 | បទបដិសេធក្នុងការផ្តល់ទ្រព្យសម្បត្តិ ឬសេវា *refusal to supply goods or services* | 0.0051 | 267, 269, 270 (each 1), 273 (7) |
| 5 | 256 | បទដណ្តើមកាន់កាប់មធ្យោបាយដឹកជញ្ជូន *hijacking a means of transport* | 0.0048 | 257, 258, 259 (each 1) |
| 6–8 | 246, 410, 494 | *indecent assault*, *intentional damage*, *conditions for incitement* | 0.0037 | 2 articles each, citing only them |
| 9 | 42 | *criminal responsibility of legal persons* (top authority) | 0.0031 | 16 articles (each citing 7–14) |

Base score of an uncited article: 0.00136.

**Why PageRank and HITS disagree.** Art. 42 is cited 16 times, Art. 210 only 5 times, yet Art. 210 has 2.3 times Art. 42's PageRank (0.0071 vs 0.0031):

- Each of Arts. 211–215 cites **only** Art. 210, so each passes its whole vote (0.85 × its rank) to it.
- The 16 articles citing Art. 42 (all titled *criminal responsibility of legal persons*; Art. 296's title carries an OCR artefact) cite 7–14 articles each (median 11), so Art. 42 receives only 1/7 to 1/14 of each vote.

So **PageRank rewards exclusive citations**, while **HITS rewards being cited together with the other authorities** by the same hubs. Rank correlation between PageRank and in-degree over the 107 cited articles is only 0.57 (Spearman).

**Why PageRank behaves almost like a weighted in-degree here.** On the web, PageRank's strength is recursion: a link from an important page counts more, and importance flows along long paths. In this graph the citing articles are themselves (except two) never cited, so every citing article has about the same base rank, and there are only 2 paths of length 2. PageRank then reduces to

$$
PR(v) \approx \text{base} + d \cdot \text{base} \cdot \sum_{u \to v} \frac{1}{\text{out}(u)}
$$

i.e. in-links weighted by 1/out-degree of the citing article. This also explains why the ranking is **the same for d = 0.5, 0.85 and 0.95** (top 6 identical, Art. 42 at rank 9 each time): with no long paths, *d* scales the scores but does not reorder them.

### 7.4 What the scores mean legally

| Score | Articles that score high | Legal meaning |
|---|---|---|
| HITS authority | Art. 42, Arts. 170–181 | **general provisions reused across the code**: to understand any offence committed by a company, you need these |
| HITS hub | the 16 "legal persons" articles | **index-like articles** that gather the applicable rules for one chapter |
| PageRank | Arts. 210, 413, 53, 265, 256 | **base offences / definitions** with their own group of dependent articles (aggravations, special cases) |
| In-degree | Arts. 42, 171, 172, 180, 181 | simply "most cited", the same as the top authorities here |

---

## 8. What This Means for Search

- The graph scores are **query-independent**: they describe the structure of the code, not relevance to a query. A high score does not mean an article answers a given question.
- They could be used as a **prior** in ranking (e.g. BM25 + a small bonus for PageRank), as web search engines do. This was **not** done: with this graph PageRank mainly favours base offences, which BM25 already finds by their title, and the evaluation (relevance labels) is needed to show whether a bonus helps.
- Useful in the **search interface**: show "cited by N articles" and links to the citing/cited articles next to a result, so a user reading Art. 42 can jump to every chapter where it applies.

---

## 9. Challenges and Solutions

### 9.1 Brackets lost by the OCR

- **Symptom:** `មាត្រា ១៧៩ កោររឹបអូស …` instead of `មាត្រា ១៧៩ (ការរឹបអូស …)`.
- **Root cause:** no brackets in the OCR alphabet (as for `.` and `-` in [data_pipeline.md §6.9](data_pipeline.md#69-wrong-article-numbers-the-hardest-problem)).
- **Solution:** the title check compares character bigrams, so a damaged first character does not matter. Matches are typically 0.8–1.0.

### 9.2 Bracketed title before the number, or cut short

- **Symptom:** 3 genuine citations did not match at first: in Arts. 187 and 198 the title of Art. 181 appears **before** the number (OCR put the lines in another order); in Art. 644 the bracketed title of Art. 640 is cut after a few words.
- **Solution:** compare with the text before the number as well as after; and measure overlap in both directions (title in context, or context in title when the context is short).
- **Result:** 273 of 273 written citations verified.

### 9.3 Citations of other laws

- **Symptom:** `មាត្រា ២៧ នៃរដ្ឋធម្មនុញ្ញ` (Article 27 of the **Constitution**) would become an edge to Art. 27 of the Criminal Code.
- **Solution:** drop a citation when "Constitution", "of the law …" or "Code of Criminal Procedure" follows the number. The title check would also have rejected them (similarity 0.12–0.24).
- **Result:** 4 dropped, listed in `citations.json` with the reason.

### 9.4 Ranges

- **Symptom:** "Article 200 (…) **to** Article 205 (…)" mentions two numbers but cites six articles.
- **Solution:** when `ដល់` (to) directly precedes `មាត្រា`, add every article between the previous citation and this one.
- **Result:** 4 ranges, 17 extra citations (in Arts. 199, 525, 574, 617).

---

## 10. Tests

`backend/tests/test_citation_graph.py` (17 tests, part of `python -m pytest`):

| Group | What is checked |
|---|---|
| Extraction (toy articles with real Khmer titles) | title similarity, also for a cut-short title; verified citation; OCR digit `៧` corrected to `៣` because the bracketed title is that of Art. 3; range expanded; Constitution citation and non-existent article dropped; citation without title kept as `unverified`; self-citation is not an edge |
| Algorithms | PageRank and HITS equal `networkx` on 3 random graphs; PageRank sums to 1; an exclusive citation is worth more than a shared one; hub and authority on a small hand-checked graph |
| Real corpus | every citation verified or from a range; only Constitution citations dropped; the Art. 199 range; top PageRank = Art. 210, top authorities include Arts. 42 and 181, top hub = Art. 230 |

---

## 11. Limitations

| Limitation | Effect | Possible improvement |
|---|---|---|
| Only explicit numbered citations | references like "the preceding article" or "the offences of this chapter" are not edges | detect "មាត្រាខាងលើ" (article above) etc. |
| Article-level nodes | a citation of "point 8 of Article 265" becomes an edge to Art. 265 | paragraph-level nodes |
| Small, flat graph | PageRank adds little beyond weighted in-degree (§7.3) | adding links to other laws (Procedure Code, Constitution) would make a deeper graph |
| Unweighted edges | citing an article twice counts once | weight by number of mentions |
| Scores not used in ranking | no measured effect on search quality | test PageRank as a prior once relevance labels exist |

---

## 12. How to Run

From `backend/`:

```bash
python -m pipeline graph                 # ~1 s; writes data/processed/citations.json, graph_scores.json
python -m pytest tests/test_citation_graph.py
```

Output of the stage:

```
citations: 290 {'verified': 273, 'corrected': 0, 'range': 17, 'unverified': 0}, dropped: 4 (other law)
graph: 672 articles, 268 edges, 74 citing, 107 cited
top PageRank: 210 (0.0071, in 5, out 0), 413 (0.0060, in 4, out 0), 53 (0.0058, in 3, out 0), …
top authority: 181 (0.0894, in 16, out 0), 42 (0.0894, in 16, out 0), 171 (0.0894, in 16, out 0), …
top hub: 230 (0.0745, in 0, out 14), 277 (0.0739, in 0, out 13), 559 (0.0697, in 0, out 13), …
```
