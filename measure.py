#!/usr/bin/env python3
"""
Pixel-to-millimetre measurement of a segmented object using a reference card.

Pipeline (per the spec's conceptual flow):
  1. Undistort the image with the stored intrinsics (mandatory before measuring).
  2. Detect the reference card (ISO 7810 ID-1, 85.60 x 53.98 mm) -> pixels_per_mm.
  3. Obtain the object mask (trained Mask R-CNN, or classical CV for validation).
  4. Fit a min-area rectangle to the mask -> object pixel dimensions.
  5. Convert to millimetres via pixels_per_mm -> width_mm, height_mm.
  6. Annotate the image with the mask overlay and metric labels.

Usage:
    python measurement/measure.py --image path.jpg --method cv
    python measurement/measure.py --image path.jpg --method model \
        --weights models/weights/maskrcnn_best.pth --out out.jpg

Why undistortion matters: lens distortion bends straight edges and changes the
apparent pixel size non-uniformly across the frame, so pixels_per_mm measured
from the card would not apply correctly to the object. Run with --no-undistort
to reproduce the (incorrect) distorted measurement for the report.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# make repo-root packages importable when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from calibration.camera_utils import load_calibration, undistort  # noqa: E402

CARD_LONG_MM = 85.60
CARD_SHORT_MM = 53.98
CARD_ASPECT = CARD_LONG_MM / CARD_SHORT_MM  # ~1.585


# ----------------------------- segmentation (CV) ----------------------------- #
def _foreground_mask(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    b = 12
    edges = np.concatenate([img[:b].reshape(-1, 3), img[-b:].reshape(-1, 3),
                            img[:, :b].reshape(-1, 3), img[:, -b:].reshape(-1, 3)])
    bg = np.median(edges, axis=0)
    diff = np.linalg.norm(img.astype(np.float32) - bg, axis=2)
    diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=2)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)


def _rect_metrics(contour):
    (cx, cy), (w, h), ang = cv2.minAreaRect(contour)
    if w < 1 or h < 1:
        return None
    long_px, short_px = max(w, h), min(w, h)
    aspect = long_px / short_px
    fill = cv2.contourArea(contour) / (w * h)
    box = cv2.boxPoints(((cx, cy), (w, h), ang))
    return dict(long_px=long_px, short_px=short_px, aspect=aspect, fill=fill,
               box=box.astype(int), area=cv2.contourArea(contour))


# ----------------------------- reference card ------------------------------- #
def _card_box(img, min_frac=0.008):
    """Bounding box of the brightest large foreground blob (the white card)."""
    mask = _foreground_mask(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    A = img.shape[0] * img.shape[1]
    best, best_bright = None, -1.0
    for c in contours:
        if cv2.contourArea(c) < min_frac * A:
            continue
        m = np.zeros(gray.shape, np.uint8)
        cv2.drawContours(m, [c], -1, 255, -1)
        bright = cv2.mean(gray, mask=m)[0]
        if bright > best_bright:
            best_bright, best = bright, c
    return cv2.boundingRect(best) if best is not None else None


def _grabcut(img, box, pad=15, iters=5):
    """Rectangle-initialised GrabCut -> largest refined foreground contour."""
    H, W = img.shape[:2]
    x, y, w, h = box
    x, y = max(0, x - pad), max(0, y - pad)
    w, h = min(W - x, w + 2 * pad), min(H - y, h + 2 * pad)
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
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return max(cnts, key=cv2.contourArea) if cnts else None


def detect_reference_card(img: np.ndarray, card_long=CARD_LONG_MM,
                          card_short=CARD_SHORT_MM, work_size=1200):
    """Detect the card with GrabCut and derive pixels_per_mm. Returns dict or None.

    The card is the brightest large object; GrabCut extracts a clean silhouette
    whose min-area rectangle gives an accurate pixel size (robust to background
    texture, unlike raw thresholding).
    """
    H, W = img.shape[:2]
    scale = work_size / max(H, W) if max(H, W) > work_size else 1.0
    small = cv2.resize(img, (round(W * scale), round(H * scale))) if scale < 1.0 else img
    box = _card_box(small)
    if box is None:
        return None
    contour = _grabcut(small, box)
    if contour is None:
        return None
    (cw, ch) = cv2.minAreaRect(contour)[1]
    long_px, short_px = max(cw, ch) / scale, min(cw, ch) / scale
    if short_px < 1:
        return None
    ppm = (long_px / card_long + short_px / card_short) / 2.0
    box_pts = (cv2.boxPoints(cv2.minAreaRect(contour)) / scale).astype(int)
    return {"pixels_per_mm": float(ppm), "box": box_pts,
            "long_px": long_px, "short_px": short_px}


# ------------------------------ object dims --------------------------------- #
def object_mask_cv(img: np.ndarray, card_box=None):
    """Largest foreground blob that is not the reference card (classical CV)."""
    mask = _foreground_mask(img)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    card_c = None
    if card_box is not None:
        cx, cy = card_box.mean(axis=0)
        card_c = (cx, cy)
    def is_card(c):
        if card_c is None:
            return False
        M = cv2.moments(c)
        if M["m00"] == 0:
            return False
        return abs(M["m10"] / M["m00"] - card_c[0]) < 20 and abs(M["m01"] / M["m00"] - card_c[1]) < 20
    cands = [c for c in contours if not is_card(c)] or contours
    biggest = max(cands, key=cv2.contourArea)
    out = np.zeros(img.shape[:2], np.uint8)
    cv2.drawContours(out, [biggest], -1, 255, cv2.FILLED)
    return out


def measure_object(object_mask: np.ndarray, ppm: float):
    """Min-area-rectangle dimensions of a binary mask, converted to mm."""
    contours, _ = cv2.findContours(object_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    m = _rect_metrics(max(contours, key=cv2.contourArea))
    short_mm = m["short_px"] / ppm
    long_mm = m["long_px"] / ppm
    return {"width_mm": round(short_mm, 2), "height_mm": round(long_mm, 2),
            "short_mm": round(short_mm, 2), "long_mm": round(long_mm, 2),
            "long_px": round(m["long_px"], 1), "short_px": round(m["short_px"], 1),
            "box": m["box"]}


# ------------------------------- annotation --------------------------------- #
def annotate(img, card, obj_mask, dims, score=None):
    vis = img.copy()
    overlay = vis.copy()
    overlay[obj_mask > 0] = (0, 200, 0)
    vis = cv2.addWeighted(overlay, 0.35, vis, 0.65, 0)
    if card is not None:
        cv2.drawContours(vis, [card["box"]], -1, (0, 165, 255), 3)
        cv2.putText(vis, "REF 85.6x54.0mm", tuple(card["box"][0]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
    if dims is not None:
        cv2.drawContours(vis, [dims["box"]], -1, (0, 255, 255), 3)
        label = f"W: {dims['width_mm']:.1f} mm  H: {dims['height_mm']:.1f} mm"
        if score is not None:
            label += f"  conf: {score:.2f}"
        org = (int(dims["box"][:, 0].min()), int(dims["box"][:, 1].min()) - 12)
        cv2.putText(vis, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
        cv2.putText(vis, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    return vis


# ------------------------------- driver ------------------------------------- #
def get_object_mask(img, method, weights, category, device, card_box, model=None):
    if method == "cv":
        return object_mask_cv(img, card_box), None
    # model method (lazy import so CV path needs no torch)
    from inference.model import load_model, predict
    if model is None:
        model = load_model(weights, num_classes=2, device=device)
    dets = predict(model, img, device=device, score_thresh=0.5)
    if not dets:
        return None, None
    top = dets[0]
    return (top["mask"] * 255).astype(np.uint8), top["score"]


def run(image_path, calib="calibration/calibration.json", method="cv",
        weights=None, category="object", do_undistort=True, device="cpu", model=None):
    img = cv2.imread(str(image_path))
    if img is None:
        raise SystemExit(f"Cannot read {image_path}")
    if do_undistort:
        K, dist, _ = load_calibration(calib)
        img = undistort(img, K, dist)
    card = detect_reference_card(img)
    if card is None:
        raise SystemExit("Reference card not detected — check contrast/placement.")
    obj_mask, score = get_object_mask(img, method, weights, category, device,
                                      card["box"], model=model)
    if obj_mask is None:
        raise SystemExit("Object not detected.")
    dims = measure_object(obj_mask, card["pixels_per_mm"])
    vis = annotate(img, card, obj_mask, dims, score)
    result = {
        "image": str(image_path), "undistorted": do_undistort, "method": method,
        "pixels_per_mm": round(card["pixels_per_mm"], 4),
        "width_mm": dims["width_mm"], "height_mm": dims["height_mm"],
        "confidence": round(score, 3) if score is not None else None,
    }
    return result, vis


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True)
    ap.add_argument("--calib", default="calibration/calibration.json")
    ap.add_argument("--method", choices=["cv", "model"], default="cv")
    ap.add_argument("--weights", default="models/weights/maskrcnn_best.pth")
    ap.add_argument("--category", default="object")
    ap.add_argument("--no-undistort", dest="undistort", action="store_false")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    result, vis = run(args.image, args.calib, args.method, args.weights,
                      args.category, args.undistort, args.device)
    print(json.dumps(result, indent=2))
    out = args.out or str(Path("inference/demo_outputs") /
                          (Path(args.image).stem + "_measured.jpg"))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(out, vis)
    print(f"Annotated image -> {out}")


if __name__ == "__main__":
    main()
