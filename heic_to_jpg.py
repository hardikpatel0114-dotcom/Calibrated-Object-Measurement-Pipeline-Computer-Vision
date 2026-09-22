#!/usr/bin/env python3
"""
Convert all HEIC/HEIF images in a folder to JPEG (upright, full quality).

iPhones save photos as HEIC, which OpenCV cannot decode. Run this on any folder
of iPhone originals before feeding them to the pipeline. EXIF orientation is
applied so every image is upright and dimensions are consistent.

Usage:
    python tools/heic_to_jpg.py --input calibration/images
    python tools/heic_to_jpg.py --input dataset/raw --quality 95
"""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageOps
import pillow_heif

pillow_heif.register_heif_opener()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", type=Path, default=None, help="default: same as --input")
    ap.add_argument("--quality", type=int, default=95)
    ap.add_argument("--delete-heic", action="store_true",
                    help="remove each .HEIC after converting")
    args = ap.parse_args()

    out = args.output or args.input
    out.mkdir(parents=True, exist_ok=True)
    heics = [p for p in sorted(args.input.iterdir())
             if p.suffix.lower() in {".heic", ".heif"}]
    if not heics:
        print(f"No HEIC/HEIF files in {args.input}")
        return
    for p in heics:
        img = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
        img.save(out / f"{p.stem}.jpg", "JPEG", quality=args.quality)
        if args.delete_heic:
            p.unlink()
    print(f"Converted {len(heics)} HEIC -> JPEG in {out}"
          + (" (HEIC removed)" if args.delete_heic else ""))


if __name__ == "__main__":
    main()
