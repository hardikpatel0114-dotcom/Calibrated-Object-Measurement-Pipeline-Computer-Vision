#!/usr/bin/env python3
"""
Auto-label the phone cover with MobileSAM (point-prompted) — clean object masks.

Classical thresholding cannot reliably separate a black cover from a grey towel
with dark pile shadows, so we use a promptable segmentation model:
  1. Seed the cover with its darkest region's centroid (a robust foreground point).
  2. Add a negative point on the bright reference card so it is excluded.
  3. Take MobileSAM's best mask, export COCO polygons + QA overlays.

MobileSAM (TinyViT-distilled SAM) is used only for LABELLING — not the trained
segmentation model — so it does not fall under the YOLO/Roboflow restriction.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from mobile_sam import SamPredictor, sam_model_registry

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def largest_component_centroid(binary):
    n, _, stats, cent = cv2.connectedComponentsWithStats(binary)
    if n < 2:
        return None, 0
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return cent[idx], int(stats[idx, cv2.CC_STAT_AREA])


def object_box_and_negative(img):
    """Bounding box of the largest INTERIOR dark blob (cover) + a negative point on the card.

    A BOX prompt (rather than a point) keeps SAM from latching onto a sub-part such
    as the circular MagSafe ring; the interior constraint avoids the dark vignette
    corners.
    """
    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thr = int(np.clip(0.6 * np.median(gray), 45, 105))
    dark = (gray < thr).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(dark)
    interior, fb, ia, fa = None, None, 0, 0
    for i in range(1, n):
        x, y, w, h, a = stats[i][:5]
        if a < 0.015 * H * W:
            continue
        if a > fa:
            fa, fb = a, i
        if not (x <= 2 or y <= 2 or x + w >= W - 2 or y + h >= H - 2) and a > ia:
            ia, interior = a, i
    idx = interior if interior else fb
    if idx is None:
        return None
    x, y, w, h = stats[idx][:4]
    pad = int(0.03 * max(H, W))
    box = np.array([max(0, x - pad), max(0, y - pad),
                    min(W, x + w + pad), min(H, y + h + pad)], np.float32)
    neg = None
    bright = cv2.morphologyEx((gray > np.percentile(gray, 96)).astype(np.uint8),
                              cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    nb, _, sb, cb = cv2.connectedComponentsWithStats(bright)
    if nb >= 2:
        j = 1 + int(np.argmax(sb[1:, cv2.CC_STAT_AREA]))
        if sb[j, cv2.CC_STAT_AREA] > 0.005 * H * W:
            neg = cb[j]
    return box, neg


def pick_mask(masks, scores, img_area, neg_pt=None):
    """Largest cover-plausible mask (2–45 % area) that does NOT cover the card point.

    Preferring the largest (over highest-scoring) avoids sub-part masks like the ring.
    """
    def covers_card(m):
        return neg_pt is not None and m[int(neg_pt[1]), int(neg_pt[0])] > 0

    cands = [m for m, s in zip(masks, scores)
             if 0.02 <= m.sum() / img_area <= 0.45 and not covers_card(m)]
    if cands:
        return max(cands, key=lambda m: m.sum()).astype(np.uint8)
    cands = [m for m, s in zip(masks, scores) if not covers_card(m)]
    if cands:
        return max(cands, key=lambda m: m.sum()).astype(np.uint8)
    return min(masks, key=lambda m: m.sum()).astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True, type=Path)
    ap.add_argument("--out-json", default="dataset/exports/annotations.json", type=Path)
    ap.add_argument("--overlays", default="dataset/exports/overlays", type=Path)
    ap.add_argument("--category", default="phone_cover")
    ap.add_argument("--checkpoint", default="models/weights/mobile_sam.pt")
    ap.add_argument("--work-size", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    sam = sam_model_registry["vit_t"](checkpoint=args.checkpoint)
    sam.to("cpu").eval()
    predictor = SamPredictor(sam)

    files = sorted(p for p in args.images.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if args.limit:
        files = files[:args.limit]
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.overlays.mkdir(parents=True, exist_ok=True)

    coco = {"info": {"description": f"MobileSAM-labelled {args.category}"},
            "images": [], "annotations": [],
            "categories": [{"id": 1, "name": args.category, "supercategory": "object"}]}
    ann_id, ok, miss = 1, 0, []
    for img_id, p in enumerate(files, start=1):
        img = cv2.imread(str(p))
        if img is None:
            miss.append(p.name); continue
        h, w = img.shape[:2]
        coco["images"].append({"id": img_id, "file_name": p.name, "width": w, "height": h})
        scale = args.work_size / max(h, w) if max(h, w) > args.work_size else 1.0
        small = cv2.resize(img, (round(w * scale), round(h * scale))) if scale < 1.0 else img
        prompt = object_box_and_negative(small)
        if prompt is None:
            miss.append(p.name); cv2.imwrite(str(args.overlays / p.name), small); continue
        box, neg = prompt
        predictor.set_image(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        pc = np.array([neg], np.float32) if neg is not None else None
        pl = np.array([0], np.int32) if neg is not None else None
        with torch.no_grad():
            masks, scores, _ = predictor.predict(point_coords=pc, point_labels=pl,
                                                 box=box, multimask_output=True)
        m = pick_mask(masks, scores, small.shape[0] * small.shape[1], neg)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        ov = small.copy()
        if not cnts:
            miss.append(p.name)
        else:
            cs = max(cnts, key=cv2.contourArea)
            c = (cs.astype(np.float32) / scale).astype(np.int32)
            x, y, bw, bh = cv2.boundingRect(c)
            eps = 0.0012 * cv2.arcLength(cs, True)
            poly = (cv2.approxPolyDP(cs, eps, True).reshape(-1).astype(float) / scale).tolist()
            coco["annotations"].append({
                "id": ann_id, "image_id": img_id, "category_id": 1,
                "segmentation": [poly], "bbox": [int(x), int(y), int(bw), int(bh)],
                "area": float(cv2.contourArea(c)), "iscrowd": 0})
            ann_id += 1; ok += 1
            cv2.drawContours(ov, [cs], -1, (0, 255, 0), 3)
            cv2.rectangle(ov, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 128, 255), 2)
            if neg is not None:
                cv2.circle(ov, (int(neg[0]), int(neg[1])), 8, (0, 0, 255), -1)
        cv2.imwrite(str(args.overlays / p.name), ov)

    with open(args.out_json, "w") as f:
        json.dump(coco, f, indent=2)
    print(f"Labelled {ok}/{len(files)} -> {args.out_json}")
    if miss:
        print(f"MISSED {len(miss)}: {', '.join(miss[:12])}")


if __name__ == "__main__":
    main()
