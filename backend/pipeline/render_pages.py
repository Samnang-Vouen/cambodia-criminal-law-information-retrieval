import sys
import pymupdf
from pipeline.config import RAW_PDF, PAGES_DIR, DPI

def run(pages: list[int] | None = None):
    PAGES_DIR.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open(RAW_PDF)
    numbers = pages or range(1, len(doc) + 1)
    for n in numbers:
        out = PAGES_DIR / f"page_{n:03d}.png"
        if out.exists():
            continue
        doc[n - 1].get_pixmap(dpi=DPI, colorspace=pymupdf.csGRAY).save(out)
        print("rendered", out.name, flush=True)
    print(f"done ({len(doc)} pages in PDF)")

if __name__ == "__main__":
    run([int(a) for a in sys.argv[1:]] or None)