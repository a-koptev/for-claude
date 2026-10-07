from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from ocr_pipeline import OCRPipeline


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def collect_images(values):
    paths = []
    for value in values:
        p = Path(value)
        if p.is_dir():
            paths.extend(sorted(x for x in p.rglob("*") if x.suffix.lower() in IMAGE_EXTS))
        elif p.is_file():
            paths.append(p)
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--input", nargs="+", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--force-subtype", choices=["type1", "type1a", "type1b", "type9", "type10", "AM", "BY", "KG", "KZ"], default=None)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    paths = collect_images(args.input)
    if not paths:
        raise SystemExit("No input images found")

    pipe = OCRPipeline(args.checkpoint, device=args.device, batch_size=args.batch_size)
    forced = [args.force_subtype] * len(paths) if args.force_subtype else None
    results = pipe.recognize_paths(paths, force_subtypes=forced)

    for item in results:
        print(json.dumps(item, ensure_ascii=False))

    if args.csv:
        with open(args.csv, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["path", "text", "subtype", "subtype_confidence", "char_confidence", "bucket"],
            )
            writer.writeheader()
            writer.writerows(results)
        print(f"Saved: {args.csv}")


if __name__ == "__main__":
    main()
