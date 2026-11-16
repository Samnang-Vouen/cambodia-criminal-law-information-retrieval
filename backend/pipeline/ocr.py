"""Stage 2: OCR page images -> one JSON file per page.

Input : data/interim/pages/page_XXX.png
Output: data/interim/ocr/page_XXX.json

    {
      "page": 12,
      "seconds": 4.21,
      "lines": [
        {"text": "មាត្រា ១ ...", "conf": 0.97, "bbox": [x1, y1, x2, y2],
         "segments": [{"text": "...", "conf": 0.98, "bbox": [...], "font": "Regular"}]}
      ],
      "figures": [[x1, y1, x2, y2]]
    }

Usage (from backend/, venv active):
    python -m pipeline.ocr                  # all pages, skips pages already done
    python -m pipeline.ocr 1 5 12           # only these pages
    python -m pipeline.ocr --workers 3      # parallel
    python -m pipeline.ocr 12 --force       # redo a page even if its JSON exists
"""

import argparse
import json
import statistics
import time
from pathlib import Path

from PIL import Image

from pipeline.config import DET_THRESHOLD, LOW_CONF_THRESHOLD, OCR_DIR, PAGES_DIR

TEXT_CLASS = 1  # KhmerOCR detector classes: 1 = text, 0 = figure/image


def page_number(path: Path) -> int:
    return int(path.stem.split("_")[1])


def group_into_lines(segments: list[dict]) -> list[dict]:
    """Group recognised segments into text lines, top to bottom, left to right.

    KhmerOCR's own grouping (inside detect()) sometimes merges two neighbouring rows,
    e.g. an article heading and the indented first line below it, and then interleaves
    their words. Here a segment joins a row only if its vertical centre is within half
    a text height of that row's centre.
    """
    if not segments:
        return []
    height = statistics.median(s["bbox"][3] - s["bbox"][1] for s in segments)
    center = lambda s: (s["bbox"][1] + s["bbox"][3]) / 2

    rows: list[list[dict]] = []
    for seg in sorted(segments, key=center):
        if rows and abs(center(seg) - statistics.mean(map(center, rows[-1]))) < height / 2:
            rows[-1].append(seg)
        else:
            rows.append([seg])

    lines = []
    for row in rows:
        row.sort(key=lambda s: s["bbox"][0])
        lines.append({
            # KhmerOCR outputs no spaces; segments are separate boxes, so join with a space
            "text": " ".join(s["text"] for s in row),
            "conf": round(min(s["conf"] for s in row), 4),
            "bbox": [
                min(s["bbox"][0] for s in row), min(s["bbox"][1] for s in row),
                max(s["bbox"][2] for s in row), max(s["bbox"][3] for s in row),
            ],
            "segments": row,
        })
    return lines


def ocr_page(img_path: Path) -> dict:
    # Imported here: khmerocr loads its ONNX models at import time,
    # so each worker process loads its own copy only when it needs it.
    from khmerocr import detect, recognize

    img = Image.open(img_path).convert("RGB")
    segments, figures = [], []

    for det_line in detect(img):
        for box in det_line:
            x1, y1, x2, y2, cls_id, score = (float(v) for v in box)
            if score < DET_THRESHOLD:
                continue
            bbox = [round(x1), round(y1), round(x2), round(y2)]
            if int(cls_id) != TEXT_CLASS:
                figures.append(bbox)
                continue
            r = recognize(img.crop(bbox))
            if not r["text"]:
                continue
            segments.append({
                "text": r["text"],
                "conf": round(r["text_confidence"], 4),
                "bbox": bbox,
                "font": r["font"],
            })

    return {"page": page_number(img_path), "lines": group_into_lines(segments), "figures": figures}


def process(img_path: Path, force: bool = False) -> tuple[str, str]:
    out = OCR_DIR / f"{img_path.stem}.json"
    if out.exists() and not force:
        return img_path.name, "skipped (already done)"
    start = time.perf_counter()
    try:
        result = ocr_page(img_path)
    except Exception as e:  # one bad page must not stop the whole run
        return img_path.name, f"FAILED {e!r}"
    result["seconds"] = round(time.perf_counter() - start, 2)

    # Write to a temp file, then rename, so a crash never leaves a half-written JSON
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(out)

    low = sum(1 for l in result["lines"] if l["conf"] < LOW_CONF_THRESHOLD)
    return img_path.name, f"ok  {result['seconds']:>6}s  {len(result['lines'])} lines, {low} low-conf"


def run(pages: list[int] | None = None, workers: int = 1, force: bool = False):
    OCR_DIR.mkdir(parents=True, exist_ok=True)
    paths = sorted(PAGES_DIR.glob("page_*.png"))
    if pages:
        wanted = set(pages)
        paths = [p for p in paths if page_number(p) in wanted]
        missing = wanted - {page_number(p) for p in paths}
        if missing:
            print(f"warning: no rendered image for pages {sorted(missing)} "
                  f"(run: python -m pipeline.render_pages {' '.join(map(str, sorted(missing)))})")
    if not paths:
        print(f"no page images found in {PAGES_DIR}")
        return

    print(f"OCR on {len(paths)} pages with {workers} worker(s)")
    start = time.perf_counter()
    failed = []

    if workers == 1:
        results = (process(p, force) for p in paths)
        for i, (name, status) in enumerate(results, 1):
            print(f"[{i}/{len(paths)}] {name}  {status}", flush=True)
            if status.startswith("FAILED"):
                failed.append(name)
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(process, p, force) for p in paths]
            for i, fut in enumerate(as_completed(futures), 1):
                name, status = fut.result()
                print(f"[{i}/{len(paths)}] {name}  {status}", flush=True)
                if status.startswith("FAILED"):
                    failed.append(name)

    print(f"finished in {time.perf_counter() - start:.1f}s")
    failed_file = OCR_DIR / "_failed.txt"
    if failed:
        failed_file.write_text("\n".join(sorted(failed)), encoding="utf-8")
        print(f"{len(failed)} page(s) failed, listed in {failed_file}. Re-run to retry them.")
    elif failed_file.exists():
        failed_file.unlink()


if __name__ == "__main__":  # required on Windows when using multiple workers
    parser = argparse.ArgumentParser(description="OCR rendered pages with KhmerOCR")
    parser.add_argument("pages", nargs="*", type=int, help="page numbers (default: all)")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--force", action="store_true", help="redo pages that already have JSON")
    args = parser.parse_args()
    run(pages=args.pages or None, workers=args.workers, force=args.force)
