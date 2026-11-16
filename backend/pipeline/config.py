from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BACKEND_DIR / "data"

RAW_PDF       = DATA_DIR / "raw" / "criminal_law.pdf"
PAGES_DIR     = DATA_DIR / "interim" / "pages"
OCR_DIR       = DATA_DIR / "interim" / "ocr"
CLEAN_DIR     = DATA_DIR / "interim" / "clean"
ARTICLES_FILE = DATA_DIR / "processed" / "articles.jsonl"
INDEX_DIR     = BACKEND_DIR / "index"

DPI = 300
LOW_CONF_THRESHOLD = 0.8   # recognized text below this confidence gets flagged for review
DET_THRESHOLD = 0.5        # ignore detected boxes below this score (same as KhmerOCR's CLI)

# --- Document layout (pixel values for 300 DPI, page = 2480 x 3509) ---
BODY_FIRST_PAGE = 3        # p1 = cover, p2 = royal decree (ព្រះរាជក្រម)
BODY_LAST_PAGE = 247       # p248-292 = table of contents (មាតិកាពិស្តារ), skipped
BODY_END_MARKER = "ធ្វើនៅ"  # signature block after the last article on p247
FOOTER_Y = 3250            # lines starting below this are page numbers
INDENT_X = 500             # x1 >= this -> indented first line of a paragraph
CENTER_X = 800             # x1 >= this -> centered heading (book/chapter/section)
FULL_WIDTH_X2 = 2250       # x2 >= this -> justified line that continues on the next line
LAST_ARTICLE = 672         # the Criminal Code ends with មាត្រា ៦៧២ (p246)