#!/usr/bin/env python3
"""
Inference pipeline: undistort -> Mask R-CNN segmentation -> annotated output.

Accepts a single image or a folder, applies the stored camera calibration to
undistort, runs the trained model, and writes an annotated image (mask overlay,
box, confidence) plus the raw binary mask for each input.

Usage:
    python inference/infer.py --input path.jpg \
        --weights models/weights/maskrcnn_best.pth --out inference/demo_outputs
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from calibration.camera_utils import load_calibration, undistort  # noqa: E402
from inference.model import load_model, predict  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def annotate(img, dets):
    vis = img.copy()
    for d in dets:
        color = (0, 200, 0)
        overlay = vis.copy()
        overlay[d["mask"].astype(bool)] = color
        vis = cv2.addWeighted(overlay, 0.35, vis, 0.65, 0)
        x1, y1, x2, y2 = [int(v) for v in d["box"]]
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(vis, f"{d['score']:.2f}", (x1, max(12, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return vis


def process(path, model, K, dist, device, out_dir, do_undistort=True):
    img = cv2.imread(str(path))
    if img is None:
        return None
    if do_undistort:
        img = undistort(img, K, dist)
    dets = predict(model, img, device=device, score_thresh=0.5)
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / f"{path.stem}_pred.jpg"), annotate(img, dets))
    if dets:
        cv2.imwrite(str(out_dir / f"{path.stem}_mask.png"), dets[0]["mask"] * 255)
    return [{"score": d["score"], "box": d["box"]} for d in dets]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="image file or folder")
    ap.add_argument("--calib", default="calibration/calibration.json")
    ap.add_argument("--weights", default="models/weights/maskrcnn_best.pth")
    ap.add_argument("--out", default="inference/demo_outputs", type=Path)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-undistort", dest="undistort", action="store_false")
    args = ap.parse_args()

    K = dist = None
    if args.undistort:
        K, dist, _ = load_calibration(args.calib)
    model = load_model(args.weights, num_classes=2, device=args.device)

    inp = Path(args.input)
    files = ([inp] if inp.is_file()
             else sorted(p for p in inp.iterdir() if p.suffix.lower() in IMAGE_EXTS))
    summary = {}
    for p in files:
        res = process(p, model, K, dist, args.device, args.out, args.undistort)
        summary[p.name] = res
        print(f"{p.name}: {0 if res is None else len(res)} detection(s)")
    with open(args.out / "predictions.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Annotated outputs -> {args.out}")


if __name__ == "__main__":
    main()
