#!/usr/bin/env python3
"""
Automatic instance-mask labelling of the phone cover (dark-object cue + GrabCut).

On a plain mid-tone background the cover is the dominant DARK object; the white
reference card and the background are excluded by construction. Steps:
  1. Threshold dark pixels -> take the largest dark blob as the cover's location.
  2. Refine the exact silhouette with GrabCut (rectangle-initialised).
  3. Export COCO instance-segmentation polygons + a QA overlay per image.

No YOLO, no Roboflow — classical CV only.

Usage:
    python dataset/grabcut_label.py --images dataset/undistorted \
        --out-json dataset/exports/annotations.json \
        --overlays dataset/exports/overlays --category phone_cover
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def cover_box(img, dark_thr=None, min_frac=0.02):
    """Bounding box of the largest INTERIOR dark blob (the cover).

    Border-touching dark regions (edge vignetting / shadows) are de-prioritised so
    the compact, centrally-placed cover is chosen instead of a dark frame.
    """
    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # threshold relative to the (dominant) background brightness — robust to the
    # image being dark/vignetted, unlike a fixed value or a global percentile
    thr = dark_thr if dark_thr else int(np.clip(0.6 * np.median(gray), 45, 105))
    dark = (gray < thr).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((45, 45), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(dark)
    interior, fallback, ia, fa = None, None, 0, 0
    for i in range(1, n):
        x, y, w, h, a = stats[i][:5]
        if a < min_frac * H * W:
            continue
        if a > fa:
            fa, fallback = a, (int(x), int(y), int(w), int(h))
        touch = x <= 2 or y <= 2 or x + w >= W - 2 or y + h >= H - 2
        if not touch and a > ia:
            ia, interior = a, (int(x), int(y), int(w), int(h))
    return interior if interior else fallback


def grabcut_contour(img, box, pad=20, iters=5):
    """Rectangle-initialised GrabCut -> largest refined foreground contour."""
    H, W = img.shape[:2]
    x, y, w, h = box
    x, y = max(0, x - pad), max(0, y - pad)
    w, h = min(W - x, w + 2 * pad), min(H - y, h + 2 * pad)
    # GrabCut needs background pixels OUTSIDE the rect; keep a ~1% border margin
    mx, my = max(2, W // 100), max(2, H // 100)
    x, y = max(mx, x), max(my, y)
    w, h = min(w, W - x - mx), min(h, H - y - my)
    if w < 10 or h < 10:
        return None
    mask = np.zeros((H, W), np.uint8)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(img, mask, (x, y, w, h), bgd, fgd, iters, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return None
    m = np.where((mask == 2) | (mask == 0), 0, 1).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return max(cnts, key=cv2.contourArea) if cnts else None


def contour_to_polygon(contour, eps_frac=0.0015):
    eps = eps_frac * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, eps, True)
    if len(approx) < 3:
        approx = contour
    return approx.reshape(-1).astype(float).tolist()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True, type=Path)
    ap.add_argument("--out-json", default="dataset/exports/annotations.json", type=Path)
    ap.add_argument("--overlays", default="dataset/exports/overlays", type=Path)
    ap.add_argument("--category", default="phone_cover")
    ap.add_argument("--dark-thr", type=int, default=None, help="fixed dark threshold (0-255)")
    ap.add_argument("--min-area-frac", type=float, default=0.02)
    ap.add_argument("--limit", type=int, default=0, help="only process first N (testing)")
    ap.add_argument("--work-size", type=int, default=1200,
                    help="downscale long side to this for GrabCut speed; mask scaled back up")
    args = ap.parse_args()

    files = sorted(p for p in args.images.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if args.limit:
        files = files[:args.limit]
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.overlays.mkdir(parents=True, exist_ok=True)

    coco = {"info": {"description": f"Auto-labelled {args.category} (GrabCut)"},
            "images": [], "annotations": [],
            "categories": [{"id": 1, "name": args.category, "supercategory": "object"}]}
    ann_id, ok, miss = 1, 0, []
    for img_id, p in enumerate(files, start=1):
        img = cv2.imread(str(p))
        if img is None:
            miss.append(p.name); continue
        h, w = img.shape[:2]
        coco["images"].append({"id": img_id, "file_name": p.name, "width": w, "height": h})

        # Work at reduced resolution for speed; scale the mask back to full res.
        scale = args.work_size / max(h, w) if max(h, w) > args.work_size else 1.0
        small = cv2.resize(img, (round(w * scale), round(h * scale))) if scale < 1.0 else img
        box = cover_box(small, args.dark_thr, args.min_area_frac)
        cs = grabcut_contour(small, box) if box else None
        c = (cs.astype(np.float32) / scale).astype(np.int32) if cs is not None else None

        ov = small.copy()
        if c is None or cv2.contourArea(c) < args.min_area_frac * h * w:
            miss.append(p.name)
        else:
            x, y, bw, bh = cv2.boundingRect(c)
            coco["annotations"].append({
                "id": ann_id, "image_id": img_id, "category_id": 1,
                "segmentation": [contour_to_polygon(c)],
                "bbox": [int(x), int(y), int(bw), int(bh)],
                "area": float(cv2.contourArea(c)), "iscrowd": 0})
            ann_id += 1; ok += 1
            xs, ys, bws, bhs = cv2.boundingRect(cs)
            cv2.drawContours(ov, [cs], -1, (0, 255, 0), 3)
            cv2.rectangle(ov, (xs, ys), (xs + bws, ys + bhs), (0, 128, 255), 2)
        cv2.imwrite(str(args.overlays / p.name), ov)

    with open(args.out_json, "w") as f:
        json.dump(coco, f, indent=2)
    print(f"Labelled {ok}/{len(files)} -> {args.out_json}")
    print(f"QA overlays -> {args.overlays}")
    if miss:
        print(f"MISSED {len(miss)}: {', '.join(miss[:12])}" + (" ..." if len(miss) > 12 else ""))


if __name__ == "__main__":
    main()
