#!/usr/bin/env python3
"""
Automatic instance-mask labelling for a single flat object on a plain background.

Strategy (classical CV, no YOLO / no Roboflow models):
  1. Estimate the background colour from the image border.
  2. Segment foreground by colour distance from the background (Otsu threshold).
  3. Clean the mask (morphology), find contours.
  4. Drop the reference card (a blob whose min-area rectangle is card-shaped,
     aspect ratio ~1.585) so only the target object is labelled.
  5. Keep the largest remaining blob as the object, emit its polygon.

Outputs:
  - a COCO instance-segmentation JSON (one category, one instance/image)
  - a QA overlay per image so labels can be visually verified before training

Usage:
    python dataset/auto_label.py --images dataset/undistorted \
        --out-json dataset/exports/annotations.json \
        --overlays dataset/exports/overlays \
        --category phone_cover

Tune with --min-area-frac / --card-aspect / --exclude-card as needed, then
re-check the overlays. A SAM-based labeller (center-point prompt) can be swapped
in for tricky contrast; see docs/SETUP.md.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
CARD_ASPECT = 85.60 / 53.98  # ISO 7810 ID-1 long/short ~= 1.585


def background_color(img: np.ndarray, border: int = 12) -> np.ndarray:
    """Median BGR of a frame around the image edge (assumed background)."""
    h, w = img.shape[:2]
    edges = np.concatenate([
        img[:border].reshape(-1, 3),
        img[-border:].reshape(-1, 3),
        img[:, :border].reshape(-1, 3),
        img[:, -border:].reshape(-1, 3),
    ])
    return np.median(edges, axis=0)


def foreground_mask(img: np.ndarray) -> np.ndarray:
    """Binary mask of everything that differs from the background colour."""
    bg = background_color(img)
    diff = np.linalg.norm(img.astype(np.float32) - bg, axis=2)
    diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    return mask


def is_cardlike(contour, card_aspect: float, tol: float = 0.12) -> bool:
    """True if the blob looks like the ISO reference card (rectangular, ~1.585)."""
    (_, _), (bw, bh), _ = cv2.minAreaRect(contour)
    if bw < 1 or bh < 1:
        return False
    aspect = max(bw, bh) / min(bw, bh)
    rect_fill = cv2.contourArea(contour) / (bw * bh)  # ~1.0 for a filled rectangle
    return abs(aspect - card_aspect) < tol and rect_fill > 0.85


def object_contour(img: np.ndarray, min_area_frac: float,
                   exclude_card: bool, card_aspect: float):
    """Return the target object's contour (largest non-card blob), or None."""
    mask = foreground_mask(img)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = min_area_frac * img.shape[0] * img.shape[1]
    cands = [c for c in contours if cv2.contourArea(c) >= min_area]
    if exclude_card:
        cands = [c for c in cands if not is_cardlike(c, card_aspect)] or cands
    if not cands:
        return None
    return max(cands, key=cv2.contourArea)


def contour_to_polygon(contour, epsilon_frac: float = 0.002):
    """Simplify a contour and flatten to a COCO polygon [x1,y1,x2,y2,...]."""
    eps = epsilon_frac * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, eps, True)
    if len(approx) < 3:
        approx = contour
    return approx.reshape(-1).astype(float).tolist()


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True, type=Path)
    ap.add_argument("--out-json", default="dataset/exports/annotations.json", type=Path)
    ap.add_argument("--overlays", default="dataset/exports/overlays", type=Path)
    ap.add_argument("--category", default="object")
    ap.add_argument("--min-area-frac", type=float, default=0.003,
                    help="ignore blobs smaller than this fraction of the image")
    ap.add_argument("--exclude-card", action="store_true", default=True)
    ap.add_argument("--no-exclude-card", dest="exclude_card", action="store_false")
    ap.add_argument("--card-aspect", type=float, default=CARD_ASPECT)
    args = ap.parse_args()

    files = sorted(p for p in args.images.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not files:
        raise SystemExit(f"No images in {args.images}")
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.overlays.mkdir(parents=True, exist_ok=True)

    coco = {
        "info": {"description": f"Auto-labelled {args.category} dataset"},
        "images": [], "annotations": [],
        "categories": [{"id": 1, "name": args.category, "supercategory": "object"}],
    }
    ann_id, ok, miss = 1, 0, []
    for img_id, p in enumerate(files, start=1):
        img = cv2.imread(str(p))
        if img is None:
            miss.append(p.name); continue
        h, w = img.shape[:2]
        coco["images"].append({"id": img_id, "file_name": p.name,
                               "width": w, "height": h})
        c = object_contour(img, args.min_area_frac, args.exclude_card, args.card_aspect)
        overlay = img.copy()
        if c is None:
            miss.append(p.name)
        else:
            x, y, bw, bh = cv2.boundingRect(c)
            coco["annotations"].append({
                "id": ann_id, "image_id": img_id, "category_id": 1,
                "segmentation": [contour_to_polygon(c)],
                "bbox": [int(x), int(y), int(bw), int(bh)],
                "area": float(cv2.contourArea(c)), "iscrowd": 0,
            })
            ann_id += 1; ok += 1
            cv2.drawContours(overlay, [c], -1, (0, 255, 0), 2)
            cv2.rectangle(overlay, (x, y), (x + bw, y + bh), (0, 128, 255), 2)
        cv2.imwrite(str(args.overlays / p.name), overlay)

    with open(args.out_json, "w") as f:
        json.dump(coco, f, indent=2)

    print(f"Labelled {ok}/{len(files)} images -> {args.out_json}")
    print(f"QA overlays -> {args.overlays}  (review these before training!)")
    if miss:
        print(f"NO object found in {len(miss)}: {', '.join(miss[:12])}"
              + (" ..." if len(miss) > 12 else ""))


if __name__ == "__main__":
    main()
