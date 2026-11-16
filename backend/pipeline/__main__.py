import sys
from pipeline import render_pages, ocr, clean, segment, build_index

STAGES = {
    "render": render_pages.run,
    "ocr": ocr.run,
    "clean": clean.run,
    "segment": segment.run,
    "index": build_index.run,
}

if __name__ == "__main__":
    selected = sys.argv[1:] or list(STAGES)   # no arguments → run every stage
    for name in selected:
        print(f"== {name} ==")
        STAGES[name]()
