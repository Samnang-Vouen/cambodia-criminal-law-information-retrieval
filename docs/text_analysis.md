# Text Analysis: From Khmer Text to Index Terms

How article text and user queries are turned into **terms** for the search index: normalization, Khmer word segmentation, stopwords, light stemming and character-cluster n-grams.

_Last updated: 2026-11-23_

---

## Contents

1. [Overview](#1-overview)
2. [Why Khmer Needs Special Treatment](#2-why-khmer-needs-special-treatment)
3. [Analyzer at a Glance](#3-analyzer-at-a-glance)
4. [Step-by-Step Details](#4-step-by-step-details)
5. [Challenges and Solutions](#5-challenges-and-solutions)
6. [Results on the Corpus](#6-results-on-the-corpus)
7. [Python API](#7-python-api)
8. [Tests](#8-tests)
9. [Known Limitations and Future Work](#9-known-limitations-and-future-work)
10. [How to Run](#10-how-to-run)

---

## 1. Overview

| | |
|---|---|
| **Goal** | Turn any Khmer text (an article or a query) into a list of terms that can be matched in an inverted index |
| **Code** | `backend/pipeline/analyzer.py` |
| **Input** | `title` and `text` of each article in `data/processed/articles.jsonl`; user queries at search time |
| **Output** | two term lists per text: **word terms** and **KCC bigrams** (see [3](#3-analyzer-at-a-glance)) |
| **Word segmenter** | [khmer-nltk](https://github.com/VietHoang1512/khmer-nltk) 1.6 (CRF model, runs locally) |
| **Used by** | `build_index.py` (next step) at index time and the search API at query time |
| **Tests** | `backend/tests/test_analyzer.py` (22 tests, all passing) |

**The most important rule:** documents and queries must pass through the **same** analyzer. If an article is indexed as `ការពារ` but a query is turned into `ការពារ៍`, they never match, however good the ranking function is. That is why the analyzer lives in one module that both the indexer and the API import.

---

## 2. Why Khmer Needs Special Treatment

English search engines split text at spaces. That does not work for Khmer:

| Property of Khmer | Consequence for search |
|---|---|
| **No spaces between words.** Spaces separate phrases or clauses, not words: `បុគ្គលណាដែលប្រព្រឹត្តបទល្មើស` is 5 words | Need a **word segmenter** to find word boundaries |
| A written "letter" is a **cluster** of code points: `ព្រ` = ព + ្ + រ (consonant + coeng + subscript consonant) | Character n-grams must be built from **clusters (KCC)**, not from single code points, otherwise grams cut a letter in half |
| **Spelling variants that look identical**: `ក្ដី` / `ក្តី` (subscript DA vs TA), `ឲ្យ` / `ឱ្យ` | Users type either form, so both must map to the same term |
| **Nouns are built with prefixes**: `ការ` + `ការពារ` (to defend) = `ការការពារ` (defence) | The segmenter often keeps the noun as one token, so a query for the verb would miss it |
| No upper/lower case, no plural, no verb conjugation | Heavy stemming (like Porter for English) is not needed |

On top of this, our text comes from **OCR** and still contains misspellings (see [data_pipeline.md §9](data_pipeline.md#9-known-limitations-and-future-work)). The analyzer has to cope with that.

---

## 3. Analyzer at a Glance

```
"ការការពារស្របច្បាប់"
        │ 1. normalize()     NFC, remove invisible chars, ្ដ→្ត, ឲ→ឱ, lowercase Latin
        ▼
"ការការពារស្របច្បាប់"
        │ 2. segment()       split into runs at spaces / punctuation / digits / "\n",
        │                    then khmernltk CRF splits each Khmer run into words
        ▼
ការការពារ[0:9] | ស្រប[9:13] | ច្បាប់[13:19]           (tokens with offsets)
        │
        ├── 3. word_terms()   drop stopwords + OCR fragments, numbers → 0-9,
        │                     add stem of prefixed nouns
        │      ▼
        │   ការការពារ  ការពារ  ស្រប  ច្បាប់
        │
        └── 4. ngram_terms()  overlapping pairs of Khmer Character Clusters per run
               ▼
            ការ រកា ការ រពា ពារ រស្រ ស្រប បច្បា ច្បាប់
```

The two term lists are meant to become **two index fields** in `build_index.py`:

| Field | Precise? | Robust to OCR / segmentation errors? | Role |
|---|---|---|---|
| **word terms** | ✅ high | ❌ one wrong letter = no match | main matching signal |
| **KCC bigrams** | ❌ lower (short grams match unrelated words) | ✅ a damaged word still shares most bigrams | recall safety net |

Which combination works best will be measured in the evaluation step instead of guessed.

---

## 4. Step-by-Step Details

### 4.1 Normalize (`normalize()`)

| Rule | Example | Why |
|---|---|---|
| Unicode **NFC** | | same as `clean.py`, canonical code point order |
| Remove **invisible characters**: U+200B–U+200D (zero-width space/joiners), U+2060, U+FEFF, U+00AD (soft hyphen), U+17B4/U+17B5 (Khmer inherent vowels) | `ការ​ពារ` (with U+200B) → `ការពារ` | often present in text copied from Word or the web, so queries pasted from documents would otherwise not match |
| **Subscript DA → subscript TA** (`្ដ` → `្ត`) | `សេចក្ដី` → `សេចក្តី` | look identical; corpus uses `្ត` 2,459× vs `្ដ` 16× |
| `ឲ` → `ឱ` | `ឲ្យ` → `ឱ្យ` | two accepted spellings of the same word; corpus has `ឱ` 544× vs `ឲ` 6× |
| Lowercase | `ABC` → `abc` | for Latin words in queries (none in the corpus) |

All replacements are **one character → one character** (or same length), and invisible characters were already removed by `clean.py`. So for article text, `normalize(text)` has the same length as `text` and **token offsets point directly into the stored article text**, which is needed later to highlight matches.

### 4.2 Segment (`segment()`)

1. **Split into runs** with one regular expression (`RUN` in `analyzer.py`):
   - **Khmer run:** consonants, vowels and signs (U+1780–U+17D3, U+17DD)
   - **Number:** Khmer or Arabic digits; groups of exactly 3 digits after a separator belong to the same number (`៥០ ០០០ ០០០`)
   - **Latin word:** `a–z`

   Everything else (spaces, `\n`, `។`, `៖`, `ៗ`, `-`, brackets) ends a run and is dropped.
2. **Split each Khmer run into words** with `khmernltk.word_tokenize`, a CRF model that first splits the run into Khmer Character Clusters and then predicts, for each cluster, whether a new word starts there.
3. Keep every token's **start/end offset** (cumulative length inside its run).

Results are **cached** (`lru_cache`) per run: phrases like `ផ្តន្ទាទោសដាក់ពន្ធនាគារ` repeat hundreds of times.

### 4.3 Word terms (`word_terms()`)

For every token from `segment()`:

| Token | Rule | Example |
|---|---|---|
| number | Khmer digits → ASCII, separators and leading zeros removed | `៣៣` → `33`, `៥០ ០០០ ០០០` → `50000000` |
| Latin | kept as is (lowercased) | |
| Khmer, **stopword** | dropped | `ដែល`, `និង`, `ឬ` |
| Khmer, **OCR fragment** | dropped: must start with a consonant or independent vowel (U+1780–U+17B3) and be longer than 1 character | `្នា` (starts with a coeng), `រ` (lone consonant) |
| Khmer, **prefixed noun** | kept, **plus** its stem | `ការការពារ` → `ការការពារ`, `ការពារ` |
| other Khmer word | kept | `ស្រប`, `ច្បាប់` |

Duplicates are kept on purpose: how often a term occurs in an article is its **term frequency**, which BM25 needs.

#### Stopwords

48 function words (`STOPWORDS` in `analyzer.py`), **chosen by hand** from the highest-document-frequency tokens of the corpus (`term_stats.json`, see [§10](#10-how-to-run)): conjunctions (`និង`, `ឬ`), relative/demonstrative words (`ដែល`, `នេះ`, `នោះ`), prepositions (`ក្នុង`, `នៃ`, `ពី`, `ដោយ`), auxiliaries (`ត្រូវ`, `អាច`, `បាន`), etc. The stats script checks that every stopword actually occurs in the corpus (catches typos in the list).

What is **not** a stopword, on purpose:
- **Frequent legal words** like `ផ្តន្ទាទោស` (to punish, 491 of 672 articles), `ពន្ធនាគារ` (prison), `បទល្មើស` (offence). They are frequent *in this corpus*, but a user may well search for them. BM25's IDF already gives them a low weight, so removing them would only lose information.
- **`ខ្លួន` (self)**: it looks like a function word, but `ការពារខ្លួន` = self-defence.

Why not an automatic cutoff ("remove every term in > 50 % of articles")? Because in a legal corpus the most frequent words are mostly legal words, not function words (see the top-terms list in [§6](#6-results-on-the-corpus)).

#### Light stemming: noun prefixes

Khmer makes nouns with prefixes. The segmenter often keeps prefix + word as **one token**:

| Token | Stem added |
|---|---|
| `ការការពារ` (defence) | `ការពារ` (to defend) |
| `ការផ្តន្ទាទោស` (punishment) | `ផ្តន្ទាទោស` (to punish) |
| `ការរឹបអូស` (confiscation) | `រឹបអូស` (to confiscate) |
| `សេចក្តីសម្រេច` (decision) | `សម្រេច` (to decide) |
| `អំពើហិង្សា` (act of violence) | `ហិង្សា` (violence) |
| `អ្នកសមគំនិត` (accomplice) | `សមគំនិត` |
| `ភាពទន់ខ្សោយ` (vulnerability) | `ទន់ខ្សោយ` (weak) |

Prefixes: `ការ`, `សេចក្តី`, `អំពើ`, `អ្នក`, `ភាព`. A corpus check found 175 `ការ…` tokens whose remainder also occurs as a word on its own.

Design decisions:
- The stem is **added**, the full word is **kept**. A query for the noun still prefers articles that contain the noun.
- The stem must itself be a word (start with a consonant/independent vowel): `ការិយាល័យ` (office) must not give `ិយាល័យ`.
- The stem must have **2+ clusters or a vowel**: `ការណ៍` → `ណ៍` is rejected, but one-syllable verbs work (`ការធ្វើ` → `ធ្វើ`, `ការឃុំ` → `ឃុំ` detention).
- **Exceptions** (`NOT_PREFIXED`): `ការពារ` (to defend) and `ការងារ` (work) only *look* prefixed. Without this, `ការពារ` produced the fake stem `ពារ`, and BM25 counted "defend" twice for the query `ការពារខ្លួន` (found through the score explanations, see [indexing_and_ranking.md §8.1](indexing_and_ranking.md#81-the-same-concept-counted-twice-found-through-score-explanations)).
- Only one level: `ការការពារ` → `ការពារ`, not further.
- Every stem the rule produces on the corpus was listed and reviewed once by hand.

### 4.4 KCC n-grams (`ngram_terms()`)

A **Khmer Character Cluster (KCC)** is the smallest unit that a reader sees as one letter: a base consonant with its subscripts, vowels and signs (`ព្រ`, `ហ្ម`, `ណ្ឌ`). We use khmer-nltk's `seg_kcc()` to split each Khmer run into clusters and emit every **pair of neighbouring clusters** (bigram):

```
ព្រហ្មទណ្ឌ → ព្រ | ហ្ម | ទ | ណ្ឌ → ព្រហ្ម, ហ្មទ, ទណ្ឌ
```

- Grams are built **inside a run**, so they never cross a space or punctuation mark.
- A run of ≤ 2 clusters becomes a single gram (`ឬ` → `ឬ`), so short words are still indexed.
- No stopword removal here: bigrams of function words are frequent, so IDF makes them nearly weightless.

**Why bigrams help with OCR errors:**

| | word terms | KCC bigrams |
|---|---|---|
| correct `ព្រហ្មទណ្ឌ` | `ព្រហ្មទណ្ឌ` | `ព្រហ្ម` `ហ្មទ` `ទណ្ឌ` |
| OCR `ព្រហ្មទណ្` (lost `ឌ`) | `ព្រហ្មទ` `ណ្` | `ព្រហ្ម` `ហ្មទ` `ទណ្` |
| overlap | **none** | **2 of 3** |

They also help when the segmenter splits a word differently in the query than in the document, because bigrams don't depend on word boundaries.

---

## 5. Challenges and Solutions

Each challenge lists: **symptom → root cause → solution → result**.

### 5.1 khmer-nltk deletes line breaks

- **Symptom / risk:** Passing a whole article to `word_tokenize()` would merge the last word of one paragraph with the first word of the next into one token.
- **Root cause:** Found by reading the source (`khmernltk/utils/data.py`, `cleanup_str()`), which, like KhmerOCR, has no documented Python API beyond the main function: it does `text.replace("\n", "")`. Article `text` joins paragraphs with `\n`.
- **Solution:** Never pass whole texts to khmer-nltk. `segment()` first splits the text into runs at every space, newline, punctuation mark and digit, and only gives khmer-nltk one Khmer run at a time.
- **Result:** Test `test_line_break_separates_words`: `ច្បាប់\nបុគ្គល` → `ច្បាប់`, `បុគ្គល`.

### 5.2 Exact offsets for highlighting

- **Symptom / risk:** `cleanup_str()` also applies a correction table (`constants.mistokenized_dict`, 195 entries) that can insert U+200B into the text, so tokens might not add up to the input, and offsets would be wrong.
- **Solution:** Remove U+200B from tokens; if the tokens still don't join back to the run, fall back to the whole run as one token.
- **Result:** The fallback never triggers on the corpus: for all 672 articles, every token satisfies `normalize(text)[start:end] == token` (test `test_every_article_analyzes_with_valid_offsets`).

### 5.3 Noun prefix mismatch

- **Symptom:** khmer-nltk segments `ការការពារស្របច្បាប់` as `ការការពារ | ស្រប | ច្បាប់`. A user searching `ការពារ` (defend) would not match the title of Article 33 (self-defence).
- **Root cause:** `ការ` + verb is a regular way to form nouns; the CRF model learned to keep them together.
- **Solution:** Add the stem as an extra term ([§4.3](#light-stemming-noun-prefixes)).
- **Result:** 2,870 stems added over the corpus. Tests `test_verb_query_matches_noun_in_document`, `test_word_that_only_looks_prefixed_is_not_stemmed`, `test_one_syllable_verb_is_a_stem`.

### 5.4 OCR fragments as tokens

- **Symptom:** Among the most frequent tokens: `្នា` (355 articles), `រ` (177×), `គ`, `ព`, `ទ`.
- **Root cause:** OCR splits or damages words; the segmenter then returns pieces. A Khmer word can never start with a coeng (`្`) or a dependent vowel.
- **Solution:** `_is_word()`: a word term must start with a consonant or independent vowel and be longer than 1 character.
- **Result:** 2,865 fragment tokens dropped from the word terms. They are still covered by the KCC bigrams.

### 5.5 Penalty durations badly damaged by OCR (finding)

- **Symptom:** Frequent garbage tokens `ប្រៅ` (255 articles), `មួេយ` (133), `ឆនា` (127), `ពើរ` (113).
- **Root cause:** The code writes durations and amounts as digits **plus words in brackets**, e.g. `៥ (ប្រាំ) ឆ្នាំ` ("5 (five) years"). OCR loses the brackets and misreads the words: `ប្រាំ` → `ប្រៅ`, `ឆ្នាំ` → `្នា` / `ឆនា`, `មួយ` → `មួេយ`, `ពីរ` → `ពើរ`, `ថ្ងៃ` → `ផ្ងៃ`.
  ```
  ...លើសពី ៥ ប្រៅ ្នា និងក្រោម...        (OCR)
  ...លើសពី ៥ (ប្រាំ) ឆ្នាំ និងក្រោម...     (printed)
  ```
- **Solution (now):** Nothing is deleted. The **digits** (`5`) are indexed correctly as numbers, and the bigram field partly covers the damaged words.
- **Recommended fix:** This is a cleaning problem, not an analysis problem. A small replacement table in `clean.py` for the ~20 number words (`ប្រៅ` → `ប្រាំ`, `ឆនា`/`្នា` → `ឆ្នាំ`, …), applied only directly after a number, would fix most of them. See [§9](#9-known-limitations-and-future-work).

### 5.6 Money amounts became the term `0`

- **Symptom:** The term `0` appeared in 237 articles, although `០` alone occurs only twice in the text.
- **Root cause:** Amounts use spaces as thousands separators: `៥០ ០០០ ០០០ រៀល` (50,000,000 riel). Each `០០០` group became its own number, and `"000"` → `"0"`.
- **Solution:** In the `RUN` regex, a number continues over a space, `.` or `,` when **exactly 3 digits** follow. Separators are removed in the term.
- **Result:** `៥០ ០០០ ០០០` → `50000000`. Short lists like `ចំណុច ១ ២០ ៣` stay separate (`1`, `20`, `3`). Test `test_amount_with_thousands_separator_is_one_number`.

### 5.7 Spelling variants

- **Symptom:** `សេចក្ដី` (typed with subscript DA, common on Khmer keyboards) and `សេចក្តី` (subscript TA, used in the corpus) are different strings.
- **Solution:** `normalize()` maps `្ដ` → `្ត` and `ឲ` → `ឱ` ([§4.1](#41-normalize-normalize)).
- **Result:** `word_terms("សេចក្ដីសម្រេច") == word_terms("សេចក្តីសម្រេច")` (test `test_query_with_variant_spelling_matches_document`).

### 5.8 Model loading time

- **Symptom:** The first call takes ~4.2 s; later queries take ~0.5 ms.
- **Root cause:** khmer-nltk loads its CRF model (a pickle) on the first `word_tokenize()` call.
- **Solution (for the API step):** analyze any short Khmer text once at application startup (e.g. `word_terms("ការពារ")`), so the first user does not wait.
- Also: khmer-nltk logs `Loaded model from …` at INFO level; `analyzer.py` sets its logger to WARNING.

---

## 6. Results on the Corpus

`python -m pipeline.analyzer --stats`, over title + text of all 672 articles:

| Metric | Value |
|---|---|
| Time (whole corpus, incl. model load) | 8.5 s |
| Tokens from `segment()` | 50,544 |
| … of which Khmer word tokens | 47,473 |
| Stopword tokens removed | 15,656 (33 % of Khmer tokens) |
| OCR fragment tokens removed | 2,865 |
| Stems added (prefixed nouns) | 2,870 |
| **Word terms** (total / distinct) | **36,441 / 2,572** |
| **KCC bigrams** (total / distinct) | **111,904 / 6,381** |
| Stopwords missing from the corpus | 0 |
| Query analysis time (after model load) | ~0.5 ms |

Top word terms by document frequency (number of articles):

```
ផ្តន្ទាទោស 491 · ដាក់ 380 · ពន្ធនាគារ 367 · បទ 325 · 1 308 · ទោស 288 · 2 283 · អំពើ 281
ប្រាក់ 260 · ពិន័យ 256 · ប្រព្រឹត្ត 256 · ប្រៅ 255 · មួយ 248 · 5 240 · រៀល 236 · បទល្មើស 201
```

This list confirms the stopword decision: the most common remaining terms are **legal vocabulary** ("punish", "prison", "fine", "riel") and penalty numbers, exactly what BM25's IDF is designed to down-weight. `ប្រៅ` is the OCR damage from [§5.5](#55-penalty-durations-badly-damaged-by-ocr-finding).

The full document frequencies are written to `data/interim/analysis/term_stats.json` (regenerable, not in git).

---

## 7. Python API

```python
from pipeline.analyzer import normalize, segment, word_terms, ngram_terms, analyze

normalize("សេចក្ដី ឲ្យ")             # 'សេចក្តី ឱ្យ'
segment("មាត្រា ៣៣")                # [Token(text='មាត្រា', start=0, end=6, kind='khmer'),
                                     #  Token(text='៣៣', start=7, end=9, kind='number')]
word_terms("ការការពារស្របច្បាប់")    # ['ការការពារ', 'ការពារ', 'ស្រប', 'ច្បាប់']
ngram_terms("ព្រហ្មទណ្ឌ")             # ['ព្រហ្ម', 'ហ្មទ', 'ទណ្ឌ']
analyze(text)                        # Analysis(words=[...], ngrams=[...])
```

| Function | Returns | Used for |
|---|---|---|
| `normalize(text)` | normalized string | everything else calls it first |
| `segment(text)` | `list[Token]` (`text`, `start`, `end`, `kind`), nothing filtered | highlighting matched words in results |
| `word_terms(text)` | list of terms, in order, with duplicates | word field of the index; query terms |
| `ngram_terms(text, n=2)` | list of KCC n-grams | n-gram field of the index; query grams |
| `analyze(text)` | both lists | convenience |

Planned use in the next step (`build_index.py`): analyze `title` and `text` **separately**, so the index can give title matches a higher weight.

---

## 8. Tests

`backend/tests/test_analyzer.py`, run with `python -m pytest` from `backend/` (configured by `backend/pytest.ini`). 22 tests, ~9 s (most of it is the full-corpus test):

| Group | What is checked |
|---|---|
| Normalization | `្ដ`/`្ត` and `ឲ`/`ឱ` unified; invisible characters removed; variant query == document terms |
| Segmentation | offsets point into the normalized text; `\n` separates words; punctuation is never a token |
| Word terms | stopwords removed but legal words kept; prefix stems; verb query matches noun; `ការណ៍` gives no stem; one-syllable verb stems (`ការធ្វើ`); `ការពារ`/`ការងារ` not stemmed; `ការិយាល័យ` gives no vowel-initial stem; fragments dropped; numbers → ASCII; amounts with separators |
| KCC n-grams | bigrams of `ព្រហ្មទណ្ឌ`; short runs; OCR-damaged word still matches; no grams across spaces |
| Whole corpus | all 672 articles analyze with valid offsets and non-empty term lists; Article 33 contains the terms of the query `ការពារខ្លួន` (self-defence) |

---

## 9. Known Limitations and Future Work

| Limitation | Impact | Possible improvement |
|---|---|---|
| OCR damage in penalty words (`ប្រៅ`, `ឆនា`, `មួេយ`, `ពើរ`, …) | queries with written-out numbers or "year" (`ឆ្នាំ`) match poorly | fix in `clean.py` with a small replacement table applied after numbers ([§5.5](#55-penalty-durations-badly-damaged-by-ocr-finding)) |
| CRF segmenter trained on general (news) text | legal compounds may be split inconsistently | bigram field compensates; optional: user dictionary of legal terms, merged after segmentation |
| Stopword list is hand-made | subjective | compare retrieval scores with/without stopwords in the evaluation step |
| Prefix stemming is rule-based and one level | misses other derivations (`សេច` + …, reduplication `ៗ`) | extend the prefix list if evaluation shows misses |
| No synonym handling (`ចោរកម្ម` theft vs `លួច` steal) | everyday-language queries may miss legal wording | synonym list for common legal terms, or semantic (embedding) search later |
| Bigrams are imprecise on their own | false matches between unrelated words sharing two clusters | weight the bigram field lower than the word field; tune in evaluation |
| Model load ~4 s | slow first query | warm up at API startup |

---

## 10. How to Run

From `backend/` with the venv active (`PYTHONUTF8=1` on Windows to print Khmer):

```bash
pip install -r requirements.txt              # adds khmer-nltk and pytest

python -m pipeline.analyzer "ការការពារស្របច្បាប់"   # show the analysis of one text
python -m pipeline.analyzer --stats          # corpus statistics → data/interim/analysis/term_stats.json
python -m pytest                             # run the tests
```

Example output:

```
normalized : ការការពារស្របច្បាប់
tokens     : ការការពារ[0:9] | ស្រប[9:13] | ច្បាប់[13:19]
word terms : ការការពារ ការពារ ស្រប ច្បាប់
kcc bigrams: ការ រកា ការ រពា ពារ រស្រ ស្រប បច្បា ច្បាប់
```
