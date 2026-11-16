# Data Pipeline: From PDF to Clean Article Dataset

How the Cambodian Criminal Code (ក្រមព្រហ្មទណ្ឌ) was turned from a scanned PDF into a structured dataset of 672 articles for information retrieval and RAG.

_Last updated: 2026-11-15_

---

## Contents

1. [Overview](#1-overview)
2. [Source Document](#2-source-document)
3. [Pipeline at a Glance](#3-pipeline-at-a-glance)
4. [Folder Structure](#4-folder-structure)
5. [Stage-by-Stage Details](#5-stage-by-stage-details)
6. [Challenges and Solutions](#6-challenges-and-solutions)
7. [Validation and Final Results](#7-validation-and-final-results)
8. [Output Data Format](#8-output-data-format)
9. [Known Limitations and Future Work](#9-known-limitations-and-future-work)
10. [Lessons Learned / Design Principles](#10-lessons-learned--design-principles)
11. [How to Reproduce](#11-how-to-reproduce)

---

## 1. Overview

| | |
|---|---|
| **Goal** | Build a searchable dataset of the Cambodian Criminal Code, one record per article |
| **Input** | `backend/data/raw/criminal_law.pdf` (292 pages, Khmer) |
| **Main output** | `backend/data/processed/articles.jsonl` (672 articles) |
| **Secondary output** | `backend/data/processed/structure.json` (the table of contents as a tree) |
| **OCR engine** | [KhmerOCR](https://github.com/seanghay/KhmerOCR) (ONNX models, runs locally) |
| **Language** | Python 3.11, run as `python -m pipeline <stage>` from `backend/` |

**Final result:** all **672 articles** found, numbered 1–672 in order, each with title, text, page numbers and its position in the code (book → title → chapter → section → sub-section).

---

## 2. Source Document

The PDF is a scanned/printed document. Its pages fall into four parts:

| Pages | Content | Used? |
|---|---|---|
| 1 | Cover page (ក្រមព្រហ្មទណ្ឌ, contact info) | ❌ skipped |
| 2 | Royal decree promulgating the code (ព្រះរាជក្រម) | ❌ skipped |
| **3 – 247** | **The law itself**, up to the signature block on p247 | ✅ used |
| 248 – 292 | Detailed table of contents (មាតិកាពិស្តារ) | ❌ skipped (it repeats every heading and would create duplicate search results) |

### Structure of the code

The code has 5 levels above the article:

```
គន្ថី      (Book)          e.g. គន្ថីទី ១ បទប្បញ្ញត្តិទូទៅ
 └ មាតិកា  (Title)         e.g. មាតិកាទី ២ ការទទួលខុសត្រូវព្រហ្មទណ្ឌ
    └ ជំពូក  (Chapter)
       └ ផ្នែក  (Section)
          └ កថាភាគ (Sub-section)
             └ មាត្រា (Article)   e.g. មាត្រា ៣៣.- ការការពារស្របច្បាប់
```

- Numbering of each level **restarts inside its parent** (every book has its own មាតិកាទី ១).
- Articles are numbered **continuously from 1 to 672** across the whole code.
- `ជំពូកទោល` / `ផ្នែកទោល` means "the only chapter/section" (= number 1).

### Page layout (at 300 DPI, page = 2480 × 3509 px)

| Element | Position | Used to detect |
|---|---|---|
| Body text, continuation lines | left edge x ≈ 420 | continuation of a paragraph |
| First line of a paragraph | indented, x ≈ 560–620 | start of a new paragraph |
| Headings (book, chapter, …) | centered, x > 800 | hierarchy markers and names |
| Article heading `មាត្រា N.-` | left edge x ≈ 413, bold | article start |
| Page number | bottom center, y > 3250 | removed as noise |

These positions are stored as settings in `backend/pipeline/config.py`.

---

## 3. Pipeline at a Glance

```
criminal_law.pdf
      │  render_pages.py     PDF → PNG (300 DPI, grayscale)
      ▼
data/interim/pages/page_001.png … page_292.png
      │  ocr.py              KhmerOCR: detect text boxes → recognise each box
      ▼
data/interim/ocr/page_001.json …   (text + confidence + bounding box per word segment)
      │  clean.py            pick body pages, remove noise, rebuild lines & paragraphs,
      │                      repair list markers and article numbers
      ▼
data/interim/clean/blocks.jsonl     (headings / article headings / paragraphs)
data/interim/clean/body.txt         (same text, human-readable, for checking)
data/interim/clean/article_report.json
      │  segment.py          group blocks into articles, attach hierarchy
      ▼
data/processed/articles.jsonl       ← final dataset (672 articles)
data/processed/structure.json       ← table of contents tree
```

| Stage | Command | Time | Needs previous stage? |
|---|---|---|---|
| Render | `python -m pipeline.render_pages` | ~1–2 min | PDF |
| OCR | `python -m pipeline.ocr` | ~1.2 s/page (≈ 6 min total) | page images |
| Clean | `python -m pipeline.clean` | < 1 s | OCR JSON |
| Segment | `python -m pipeline.segment` | < 1 s | clean blocks |

Each stage reads only the output of the previous one, so a change in cleaning rules only requires re-running `clean` and `segment`, **not** the slow OCR.

---

## 4. Folder Structure

```
backend/
├── app/                      FastAPI application (search API), reads only data/processed
├── pipeline/                 offline data processing, run by hand
│   ├── __init__.py           empty (marks the package)
│   ├── __main__.py           runs stages in order: python -m pipeline [stage ...]
│   ├── config.py             all paths, thresholds and layout positions
│   ├── render_pages.py       stage 1
│   ├── ocr.py                stage 2
│   ├── clean.py              stage 3
│   ├── segment.py            stage 4
│   └── build_index.py        stage 5 (search index, next step)
├── data/
│   ├── raw/                  original PDF, never modified
│   ├── interim/              regenerable intermediate files (gitignored)
│   │   ├── pages/
│   │   ├── ocr/
│   │   └── clean/
│   └── processed/            final dataset (committed)
└── requirements.txt
```

**Rules followed:**
- `raw/` is never edited; everything else can be regenerated from it.
- Module names do not start with digits (`01_ocr.py` cannot be imported in Python).
- All paths are built from `config.py` using `__file__`, so scripts work from any folder.

---

## 5. Stage-by-Stage Details

### 5.1 Render pages (`render_pages.py`)

- Uses **PyMuPDF** (`import pymupdf`) to render each page to PNG.
- **300 DPI, grayscale** (`colorspace=pymupdf.csGRAY`): good OCR quality, ~3× smaller files than colour.
- Skips pages already rendered, so it can be stopped and resumed.
- Can render selected pages for testing: `python -m pipeline.render_pages 1 5 12`.

### 5.2 OCR (`ocr.py`)

KhmerOCR provides two functions (found by reading its source code; the README only documents the CLI):

| Function | Input | Output |
|---|---|---|
| `detect(page_image)` | whole page | boxes `[x1, y1, x2, y2, class, score]`; class 1 = text, 0 = figure |
| `recognize(crop)` | one text box | `{"text", "text_confidence", "font", "font_confidence"}` |

Process per page:
1. Detect boxes; ignore boxes with score < 0.5 (`DET_THRESHOLD`, same as KhmerOCR's CLI) and figures.
2. Recognise each text box separately.
3. Group the boxes into lines with our own `group_into_lines()` (see [Challenge 6.4](#64-two-text-rows-merged-into-one-line)).
4. Save one JSON per page with, for every line: text, min confidence, bounding box, and the individual segments.

Robustness features:
- **Resumable:** pages with an existing JSON are skipped (`--force` to redo).
- **Atomic writes:** write to `.tmp`, then rename, so a crash never leaves a broken file.
- **Fault-tolerant:** a failing page is logged in `_failed.txt` and the run continues.
- **Optional parallelism:** `--workers N` (not needed: one page takes ~1.2 s).

### 5.3 Clean (`clean.py`)

Turns 292 pages of OCR lines into typed text blocks. Steps, in order:

| # | Step | Purpose |
|---|---|---|
| 1 | Load pages 3–247, stop at signature block `ធ្វើនៅ…` | keep only the law |
| 2 | Re-group OCR segments into lines | fix merged rows |
| 3 | Drop lines with y > 3250 | remove page numbers |
| 4 | Normalise text: Unicode NFC, remove zero-width chars, collapse spaces, `។ ។` → `។` | consistent text |
| 5 | Find article headings by sequence (DP) | correct numbers, reject cross-references |
| 6 | Classify each line by position (centered / indented / left / article) | layout → meaning |
| 7 | Join wrapped lines into paragraphs (also across pages) | readable paragraphs |
| 8 | Repair list markers (`៤១ការ…` → `៤- ការ…`) | correct numbering |
| 9 | Flag low-confidence blocks | review hints |

Output block types: `heading`, `article`, `paragraph`. Every corrected block keeps its original text in `raw_text` and a note (`article_no_fixed`, `number_from_sequence`, `list_marker_fixed`, `low_conf`).

### 5.4 Segment (`segment.py`)

Walks through the blocks in order and keeps track of the current position in the code:

- A **marker** (`ជំពូកទី ២`) opens a new hierarchy node and resets all lower levels.
- The **first block after a marker** is its name (plus centered continuation lines).
- An **article heading** starts a new article and copies the current hierarchy path.
- **Paragraphs** are added to the current article.
- A non-marker centered line directly after an article heading continues the **article title**.

Then it validates the result (see [Section 7](#7-validation-and-final-results)).

---

## 6. Challenges and Solutions

Each challenge lists: **symptom → root cause → solution → result**.

### 6.1 Khmer PDFs and OCR choice

- **Symptom / context:** Khmer PDFs often have no usable text layer, or a text layer with broken font encoding (copying text gives garbled characters).
- **Solution:** Render pages to images and run **KhmerOCR**, a Khmer-specific OCR model that runs locally.
- **Note:** a text-layer check (`page.get_text()`) was suggested as a first step but its result was not recorded. If a clean text layer exists, it would be more accurate than OCR → see [Future Work](#9-known-limitations-and-future-work).

### 6.2 Development environment problems

| Symptom | Root cause | Solution |
|---|---|---|
| `bash: ..venvScriptsActivate.ps1: command not found` | PowerShell command typed in Git Bash (`\` is an escape char in bash) | Git Bash: `source .venv/Scripts/activate` |
| `ModuleNotFoundError: No module named 'fitz'` although installed | Terminal used the **system Python** (venv not active) | Activate venv in every new terminal; check `which python`; select the venv interpreter in VS Code |
| Packages installed on drive C: | `pip install` run without venv → system Python | Install only inside the activated venv |
| `FutureWarning: The fitz API is deprecated` | PyMuPDF 2.x renamed the module | Use `import pymupdf` |
| `01_render_pages.py` cannot be imported | Python module names cannot start with a digit | Renamed to `render_pages.py`, etc. |
| `UnicodeEncodeError: 'charmap' codec…` when printing Khmer | Windows console uses cp1252 | `export PYTHONUTF8=1` (files are always written as UTF-8) |
| C: drive filling up | pip download cache (~365 MB) in `AppData\Local\pip` | `pip cache purge`, or move cache: `pip config set global.cache-dir "D:/cache/pip"` |

### 6.3 Undocumented OCR library API

- **Symptom:** The README only shows the CLI (`khmerocr image.jpg`), no Python API.
- **Root cause:** Library is small and mostly CLI-oriented.
- **Solution:** Read the installed source (`.venv/Lib/site-packages/khmerocr/__init__.py` and `cli.py`). Found `detect()` / `recognize()`, the box format, the class meaning (1 = text, 0 = figure) and the CLI's detection threshold (0.5), and copied the CLI's approach (recognise each box separately).
- **Result:** First test page: 27 lines in 1.2 s.

### 6.4 Two text rows merged into one line

- **Symptom:** `មាត្រា ច្បាប់បញ្ញត្តថា ២ និយមន័យ អំពើខ្លះ…` instead of the heading `មាត្រា ២ និយមន័យ និងចំណាត់ថ្នាក់នៃបទល្មើស` followed by the paragraph `ច្បាប់បញ្ញត្តថា អំពើខ្លះ…`.
- **Root cause:** Every word was recognised correctly, but KhmerOCR's line grouping (`get_sorted_lines`, inside `detect()`) compares each box only to the previous one. An article heading and the **indented** first line below it got put into one line, then sorted left-to-right, so their words were **interleaved**:
  ```
  មាត្រា | ច្បាប់បញ្ញត្តថា | ២ | និយមន័យ | អំពើខ្លះ… | និងចំណាត់ថ្នាក់… | ឬនីតិបុគ្គល…
   row 1      row 2       row 1   row 1     row 2         row 1             row 2
  ```
- **Solution:** Our own `group_into_lines()` in `ocr.py`: a segment joins a row only if its vertical centre is within **half a text height** of the row's centre; rows are then sorted left to right. `clean.py` re-applies it to the saved segments, so **no OCR re-run was needed**.
- **Result:** 115 merged lines (one per page on 115 pages) fixed; article headings with merged text went from 38 to 0.

### 6.5 Page-number footers

- **Symptom:** Short garbage lines at the bottom of pages: `០១០១`, `ខម`, `១១ទ`, `៖២`.
- **Root cause:** Page numbers in a decorative font, misread by OCR.
- **Solution:** Drop every line whose top is below y = 3250 (`FOOTER_Y`). All page numbers were found at y ≈ 3300–3410.
- **Result:** 219 footer lines removed from the body pages (264 in the whole PDF).

### 6.6 Non-law pages (cover, decree, table of contents)

- **Symptom:** The table of contents (p248–292) repeats every heading with page numbers, e.g. `មាត្រា ៦៦០១ ទោសបន្ថែម … ២៣៩ ។`.
- **Root cause:** These pages are part of the PDF but not of the law.
- **Solution:** Only pages 3–247 are processed (`BODY_FIRST_PAGE`, `BODY_LAST_PAGE`), stopping at the signature block (`BODY_END_MARKER = "ធ្វើនៅ"`).
- **Result:** No duplicate articles; the TOC is instead rebuilt from the body as `structure.json`.

### 6.7 Rebuilding paragraphs from lines

- **Symptom:** OCR gives lines; a paragraph spans several lines and sometimes two pages.
- **Root cause:** Printed text wraps lines; OCR has no notion of paragraphs.
- **Solution:** Use **indentation**:
  - x ≥ 500 → first line of a new paragraph;
  - x < 500 → continuation of the current paragraph (also at the top of a new page);
  - lines are joined **without a space**, because Khmer does not put spaces between words.
- **Result:** 5,730 lines → 2,034 paragraphs.

### 6.8 Misread list markers

- **Symptom:** Numbered list items `៤- ការ…` came out as `៤១ការ…`, `១- …` as `ខប…`, `១០- …` as `១០ៈ…` or `១០ប…`, `៣- បទល្មើស` as `៣៧បទល្មើស`.
- **Root cause:** The OCR alphabet has **no `-` character**, so the dash is replaced by a lookalike digit (`១`, `០`, `៧`) or letter (`ប`, `ង`, `ព`, `ៈ`, `៖`). Sometimes the number itself is read as a letter (`១` → `ខ`/`ឧ`, `៨` → `ជ`, `៣` → `ព`).
- **Solution:** `repair_list_marker()` in `clean.py`:
  - keeps a **counter of the current list**; the expected next number decides ambiguous cases (`៤១` while expecting 4 → `៤-`; while expecting 41 is impossible since lists have ≤ 30 items);
  - treats a letter as a dash **only if the next character is not a vowel sign/coeng** (so `ប្រសិនបើ` keeps its `ប`) and never `បទ` (a very common word: `បទល្មើស`, `បទឧក្រិដ្ឋ`);
  - when the number is lost entirely and only `៖` remains (`…ដូចខាងក្រោមនេះ ៖` then `៖បទល្មើស…`), it inserts the next number.
- **Bugs found while testing:** `២១ប្រសិនបើ` → `២១- ្រសិនបើ` (letter `ប` wrongly treated as dash) and `៣៧បទល្មើស` → `៣- ទល្មើស`. Both fixed by the vowel-sign and `បទ` rules above, verified with a small table of test cases.
- **Result:** 948 + 17 list markers repaired, 0 broken markers left at paragraph starts.

### 6.9 Wrong article numbers (the hardest problem)

- **Symptom:** `មាត្រា ១៧` instead of `មាត្រា ១`, `មាត្រា ៣៣០` instead of `៣៣`, `១២៤` instead of `១២៨`. Article heading count: 655 instead of 672.
- **Root cause (3 different errors):**
  1. Headings are printed `មាត្រា ១.-`. The OCR alphabet has **no `.` or `-`**, so `.-` becomes an extra digit: `១.-` → `១៧`, `៣៣.-` → `៣៣០`.
  2. The bold heading font makes the OCR **confuse digits**: `៤ ↔ ៨` and `៣ ↔ ៧` (e.g. 124 ↔ 128, 437 ↔ 433).
  3. Some heading lines did not start with `មាត្រា` + number at all (merged lines, see 6.4).
- **Solution, step by step:**
  1. **Use domain knowledge:** articles are numbered 1, 2, 3, …, 672 in order.
  2. **Generate candidate readings** per heading: the number as read, with 1–2 trailing characters dropped, and with one confused digit swapped. `៣៣០` → {330, 33, 37, 73, 3, …}.
  3. **First attempt (failed): decide one heading at a time** ("is it previous + 1?"). One wrong decision pushed every later heading out of order → **79 real headings were lost**.
  4. **Final approach: choose all numbers together** with dynamic programming (a weighted **longest increasing subsequence**): pick at most one reading per heading so that the chosen numbers strictly increase, scoring +1 per heading and +1 extra when it is exactly previous + 1.
  5. **Fill gaps:** remaining candidates that fit exactly between two known numbers get that number (one candidate between 212 and 214 → 213).
  6. **Let the sequence decide what is a heading:** every left-margin line starting with `មាត្រា` is a *candidate*; the layout rule "previous line is full width → it's a sentence" was dropped, because OCR line errors made it unreliable (it caused 17 missing articles).
- **Result:** **672/672 articles**, numbers strictly increasing, no duplicates. 364 heading numbers corrected, 15 taken from the sequence. Every corrected heading keeps the OCR original in `raw_text`.

### 6.10 Cross-references that look like headings

- **Symptom:** Lines like `មាត្រា ១៨១ ការផ្សាយសេចក្តីសម្រេច… នៃក្រមនេះ ។` at the left margin.
- **Root cause:** Articles often refer to other articles ("…as in article 181 of this code"), and a wrapped sentence can start with `មាត្រា`.
- **Solution:** Handled by the sequence method in 6.9: a reference to article 181 inside article 450 cannot fit the increasing sequence, so it stays paragraph text.
- **Result:** 29 such lines correctly kept as text.

### 6.11 Building the hierarchy (book / title / chapter / section)

| Symptom | Root cause | Solution |
|---|---|---|
| Book markers read as `គន្ថីនី`, `គន្ីិីន`, `គន្លឹន`, `គន្ិឹន` | Decorative font | Match any word starting with `គន` + a number |
| Section/chapter has "no name" | A long name is not centered, so `clean.py` typed it as a paragraph | The **first block after a marker is always its name**, whatever its type |
| Names split over two lines | Wrapped centered heading | Following centered lines are appended to the name |
| Stray "headings" like `សេពសន្ថវៈ` | Second line of a **wrapped article title** is indented far right | A non-marker centered line right after an article heading → appended to the article title |
| `ផ្នែកទៀត` treated as a section, article 465 lost its text | It means "another part": the end of article 465's title | A real marker must have a number or `ទោល` |
| Marker number unreadable | OCR | Use previous number + 1 and log a warning |

- **Result:** 185 hierarchy nodes (6 books, 12 titles, 50 chapters, 95 sections, 22 sub-sections); validation passed.

### 6.12 Misleading confidence scores

- **Symptom:** 1,863 blocks flagged as low confidence, including correct text like `ក្រោមក៏ដោយ ៖` (confidence 0.20).
- **Root cause:** Line confidence = minimum over its segments; a tiny segment like `៖` gets a low score even when correct.
- **Solution:** Confidence is used only as a **hint for manual review**, never to delete text. Stored in `ocr.min_conf`.

---

## 7. Validation and Final Results

### Automatic checks (run by `segment.py`)

- Article numbers are exactly `1 … 672`, in order, no gaps, no duplicates.
- Every article has a non-empty title and text.
- Every article belongs to a book.

### Numbers

| Metric | Value |
|---|---|
| PDF pages / pages used | 292 / 245 (p3–247) |
| OCR lines (whole PDF) | 6,996 |
| Body lines after cleaning | 5,730 |
| Blocks (headings / articles / paragraphs) | 399 / 672 / 2,034 |
| Footer lines removed (body pages) | 219 |
| Hierarchy nodes (books / titles / chapters / sections / sub-sections) | 6 / 12 / 50 / 95 / 22 |
| Merged OCR lines split | 115 |
| List markers repaired | 965 |
| Article numbers corrected / from sequence | 364 / 15 |
| `មាត្រា` lines rejected as cross-references | 29 |
| **Articles found** | **672 / 672** |
| Articles per book (1–6) | 182, 170, 80, 235, 3, 2 |
| Article text length (characters) | median ≈ 250, max ≈ 2,300 |

### Manual check

Compare `data/interim/clean/body.txt` with the PDF. Good places to check: articles with `ocr.heading_fixed = true` and blocks with the note `number_from_sequence`.

---

## 8. Output Data Format

### `data/processed/articles.jsonl` (one JSON object per line)

```json
{
  "id": "art_033",
  "article_no": 33,
  "title": "ការការពារស្របច្បាប់",
  "text": "paragraphs joined with \\n",
  "paragraphs": ["បុគ្គលណាដែល…", "…ដូចខាងក្រោមនេះ ៖", "១- …", "២- …", "៣- …"],
  "path": {
    "book":    {"no": 1, "name": "បទប្បញ្ញត្តិទូទៅ"},
    "title":   {"no": 2, "name": "ការទទួលខុសត្រូវព្រហ្មទណ្"},
    "chapter": {"no": 2, "name": "មូលហេតុនៃការមិនទទួលខុសត្រូវ …"},
    "section": null,
    "subsection": null
  },
  "breadcrumb": "គន្ថីទី ១ … › មាតិកាទី ២ … › ជំពូកទី ២ …",
  "pages": [12],
  "ocr": {
    "min_conf": 0.342,
    "heading_fixed": true,
    "raw_heading": "មាត្រា ៣៣០ ការការពារស្របច្បាប់",
    "list_markers_fixed": 3,
    "low_conf_blocks": 2
  }
}
```

| Field | Use |
|---|---|
| `id`, `article_no` | unique key, citation ("Article 33") |
| `title` | separate field so search can weight title matches higher |
| `text`, `paragraphs` | indexing; highlighting the matching paragraph |
| `path`, `breadcrumb` | filters (e.g. only Book 2), display, context header for RAG chunks |
| `pages` | link back to the PDF |
| `ocr` | audit trail only; **not used by search**. `raw_heading` is the uncorrected OCR text, kept on purpose |

### `data/processed/structure.json`

Nested tree: `[{level, no, name, page, children: [...]}, ...]` for book → title → chapter → section → sub-section. Usable as a browse/filter menu.

### `data/interim/clean/article_report.json`

`candidates` (number of `មាត្រា` lines), `not_headings` (rejected cross-references with page), `missing` (should be empty).

---

## 9. Known Limitations and Future Work

| Limitation | Impact | Possible improvement |
|---|---|---|
| OCR spelling errors remain (e.g. `ព្រហ្មទណ្` missing `ឌ`, `ទ្រព្យសម្ប្ិ`, `បេទ` for `បទ`) | some words won't match in keyword search | dictionary-based correction; frequency-based replacement of common misspellings; manual fixes for headings |
| Text-layer check not recorded | OCR may be unnecessary if the PDF has a clean Khmer text layer | run `page.get_text()` on a few pages and compare |
| 1 list item still starts with `៖` | cosmetic | manual fix |
| Hierarchy names come from OCR without correction | small typos in breadcrumbs | correct the 185 names by hand (small, fixed set) |
| Layout thresholds are specific to this PDF | pipeline must be re-tuned for other documents | detect margins/indentation automatically from x-histograms |
| Confidence scores are per-line minimums | many false "low confidence" flags | use mean or length-weighted confidence |

**Next pipeline steps:** `build_index.py` (Khmer tokenization with khmernltk → inverted index + BM25), then a chunking step for RAG (one chunk per article, long articles split at paragraph/list boundaries, breadcrumb prepended).

---

## 10. Lessons Learned / Design Principles

1. **Test on a few pages first.** Pages 1, 5, 12, 40, 100 were used to check OCR quality and speed before the full run.
2. **Keep every stage's output on disk.** Slow steps (OCR) run once; fast steps (clean, segment) can be changed and re-run in seconds.
3. **Keep raw data rich.** Saving each segment's bounding box made it possible to fix line grouping later without re-running OCR.
4. **Fix a problem at the stage where it originates** (line grouping → OCR stage; numbering → cleaning stage).
5. **Use domain knowledge as a constraint.** "Articles are numbered 1…672 in order" fixed hundreds of OCR errors and also told headings apart from cross-references.
6. **Decide globally, not greedily.** One-at-a-time decisions cascade errors; the dynamic-programming approach chose all article numbers together.
7. **Never silently change data.** Every correction keeps the original (`raw_text`, `raw_heading`) and a note, so it can be audited and reverted.
8. **Validate automatically.** A run that does not end in "672/672, strictly increasing" is immediately visible.
9. **Read the library source** when documentation is missing.
10. **Use confidence scores as hints, not filters.**

---

## 11. How to Reproduce

From `backend/` in Git Bash:

```bash
# one-time setup
python -m venv .venv
source .venv/Scripts/activate            # PowerShell: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt          # includes khmer-ocr pinned to a git commit
export PYTHONUTF8=1                      # Windows: print Khmer in the terminal

# full pipeline
python -m pipeline.render_pages          # ~1–2 min
python -m pipeline.ocr                   # ~6 min, resumable
python -m pipeline clean segment         # ~1 s

# useful variants
python -m pipeline.render_pages 1 5 12   # selected pages only
python -m pipeline.ocr 12 --force        # redo one page
python -m pipeline.ocr --workers 3       # parallel OCR
```

Key package versions (see `requirements.txt`): Python 3.11, `pymupdf==2.1`, `onnxruntime==1.32.0`, `pillow==13.0.0`, `khmer-ocr` @ commit `68c2c72`.

**Git:** commit `data/raw/` and `data/processed/`; keep `data/interim/` out of git (add `backend/data/interim/` to `.gitignore`); the page images alone are several hundred MB.
