#!/usr/bin/env python3
"""
Accuracy validation: system width/height vs physical (calliper) ground truth.

Runs the measurement pipeline over >=10 images of the object and compares each
result to the ground-truth dimensions, reporting mean absolute error (MAE, mm)
and mean percentage error (MPE, %). Predicted and ground-truth dimensions are
sorted (short, long) before comparison, so results are robust to the object's
in-plane rotation.

Ground truth can be a single object measured once (--gt-width/--gt-height) or
per-image (--gt-csv with columns: image,gt_width_mm,gt_height_mm).

Usage:
    python measurement/validate_accuracy.py --images measurement/eval_images \
        --method model --weights models/weights/maskrcnn_best.pth \
        --gt-width 60.1 --gt-height 122.4
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from measurement.measure import run  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def load_gt_csv(path: str) -> dict:
    gt = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            gt[row["image"]] = (float(row["gt_width_mm"]), float(row["gt_height_mm"]))
    return gt


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True, type=Path)
    ap.add_argument("--calib", default="calibration/calibration.json")
    ap.add_argument("--method", choices=["cv", "model"], default="cv")
    ap.add_argument("--weights", default="models/weights/maskrcnn_best.pth")
    ap.add_argument("--gt-width", type=float, help="constant GT width (mm)")
    ap.add_argument("--gt-height", type=float, help="constant GT height (mm)")
    ap.add_argument("--gt-csv", help="per-image GT csv (image,gt_width_mm,gt_height_mm)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-undistort", dest="undistort", action="store_false",
                    help="skip undistortion (to demonstrate the calibration dependency)")
    ap.add_argument("--out", default="measurement/accuracy_results.csv")
    args = ap.parse_args()

    gt_map = load_gt_csv(args.gt_csv) if args.gt_csv else None
    model = None
    if args.method == "model":
        from inference.model import load_model
        model = load_model(args.weights, num_classes=2, device=args.device)

    files = sorted(p for p in args.images.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    rows, acc = [], {"s_ae": 0.0, "l_ae": 0.0, "s_pe": 0.0, "l_pe": 0.0, "n": 0}
    for p in files:
        try:
            res, _ = run(p, args.calib, args.method, args.weights,
                         do_undistort=args.undistort, device=args.device, model=model)
        except SystemExit as e:
            print(f"  [skip] {p.name}: {e}")
            continue
        gw, gh = (gt_map.get(p.name, (None, None)) if gt_map
                  else (args.gt_width, args.gt_height))
        if gw is None or gh is None:
            print(f"  [skip] {p.name}: no ground truth")
            continue
        pred = sorted([res["width_mm"], res["height_mm"]])
        gt = sorted([gw, gh])
        s_ae, l_ae = abs(pred[0] - gt[0]), abs(pred[1] - gt[1])
        s_pe, l_pe = 100 * s_ae / gt[0], 100 * l_ae / gt[1]
        rows.append({
            "image": p.name, "pred_short_mm": pred[0], "pred_long_mm": pred[1],
            "gt_short_mm": gt[0], "gt_long_mm": gt[1],
            "short_abs_err_mm": round(s_ae, 2), "long_abs_err_mm": round(l_ae, 2),
            "short_pct_err": round(s_pe, 2), "long_pct_err": round(l_pe, 2),
        })
        for k, v in zip(("s_ae", "l_ae", "s_pe", "l_pe"), (s_ae, l_ae, s_pe, l_pe)):
            acc[k] += v
        acc["n"] += 1

    n = acc["n"]
    if n == 0:
        raise SystemExit("No measurements produced — check images / ground truth.")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    mae = (acc["s_ae"] + acc["l_ae"]) / (2 * n)
    mpe = (acc["s_pe"] + acc["l_pe"]) / (2 * n)
    print(f"\n===== ACCURACY ({n} instances, method={args.method}) =====")
    print(f"Short side   MAE: {acc['s_ae']/n:6.2f} mm    MPE: {acc['s_pe']/n:5.2f} %")
    print(f"Long side    MAE: {acc['l_ae']/n:6.2f} mm    MPE: {acc['l_pe']/n:5.2f} %")
    print(f"OVERALL      MAE: {mae:6.2f} mm    MPE: {mpe:5.2f} %")
    print(f"Per-image table -> {args.out}")


if __name__ == "__main__":
    main()
