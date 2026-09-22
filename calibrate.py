#!/usr/bin/env python3
"""
Intrinsic camera calibration from checkerboard images.

Detects the inner-corner grid on a set of checkerboard photos, runs OpenCV
intrinsic calibration, reports the reprojection error, and saves the camera
matrix + distortion coefficients for the inference and measurement pipelines.

Target board: calibration/checkerboard.html — 8x10 squares => 7x9 inner corners.

Usage:
    python calibration/calibrate.py \
        --images calibration/images \
        --cols 7 --rows 9 --square-size 20 \
        --output calibration/calibration.json --save-detections

Note on --square-size: it is in millimetres but does NOT affect the intrinsic
matrix or distortion coefficients used for undistortion (it only scales the
unused extrinsic translation). An approximate value is therefore fine — the
real-world scale for measurement comes from the reference card, not the board.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def find_images(folder: Path):
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def detect_corners(gray: np.ndarray, pattern: tuple[int, int]):
    """Detect inner corners. Prefer the robust SB detector; fall back to classic."""
    flags_sb = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
    found, corners = cv2.findChessboardCornersSB(gray, pattern, flags_sb)
    if found:
        return True, corners
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, pattern, flags)
    if found:
        term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), term)
    return found, corners


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--images", default="calibration/images", type=Path)
    ap.add_argument("--cols", type=int, default=7, help="inner corners per row")
    ap.add_argument("--rows", type=int, default=9, help="inner corners per column")
    ap.add_argument("--square-size", type=float, default=20.0, help="mm (approx; not critical)")
    ap.add_argument("--output", default="calibration/calibration.json", type=Path)
    ap.add_argument("--save-detections", action="store_true",
                    help="save corner-overlay images to calibration/detected/")
    ap.add_argument("--drop-worst", type=int, default=0,
                    help="re-calibrate after removing the N highest-error frames")
    ap.add_argument("--fix-k3", action="store_true",
                    help="constrain k3=0 for a stable, physically gentle distortion fit "
                         "(recommended for phone cameras / screen-based calibration)")
    args = ap.parse_args()

    pattern = (args.cols, args.rows)
    images = find_images(args.images)
    if not images:
        sys.exit(f"No images found in {args.images}")

    # 3D object points for one board (z = 0 plane), scaled to mm.
    objp = np.zeros((args.rows * args.cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:args.cols, 0:args.rows].T.reshape(-1, 2)
    objp *= args.square_size

    objpoints, imgpoints, used = [], [], []
    image_size = None
    det_dir = Path("calibration/detected")
    if args.save_detections:
        det_dir.mkdir(parents=True, exist_ok=True)

    for p in images:
        img = cv2.imread(str(p))
        if img is None:
            print(f"  [skip] unreadable: {p.name}")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = (gray.shape[1], gray.shape[0])
        found, corners = detect_corners(gray, pattern)
        print(f"  [{'OK ' if found else 'MISS'}] {p.name}")
        if found:
            objpoints.append(objp)
            imgpoints.append(corners)
            used.append(p.name)
            if args.save_detections:
                vis = img.copy()
                cv2.drawChessboardCorners(vis, pattern, corners, found)
                cv2.imwrite(str(det_dir / p.name), vis)

    n = len(objpoints)
    print(f"\nDetected board in {n}/{len(images)} images.")
    if n < 10:
        sys.exit(f"Only {n} usable images — need >= 10 (spec: 20+). "
                 f"Check that --cols/--rows match the board's inner corners.")

    calib_flags = cv2.CALIB_FIX_K3 if args.fix_k3 else 0

    def calibrate_set(objp_l, imgp_l, used_l):
        """Calibrate on one set of views; return rms, K, dist, per-image error, RMS."""
        rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
            objp_l, imgp_l, image_size, None, None, flags=calib_flags)
        per_image, tse, tp = [], 0.0, 0
        for i in range(len(objp_l)):
            proj, _ = cv2.projectPoints(objp_l[i], rvecs[i], tvecs[i], K, dist)
            e = cv2.norm(imgp_l[i], proj, cv2.NORM_L2)
            per_image.append({"image": used_l[i], "rmse_px": float(e / np.sqrt(len(proj)))})
            tse += e ** 2
            tp += len(proj)
        return rms, K, dist, per_image, float(np.sqrt(tse / tp))

    rms, K, dist, per_image, mean_reproj = calibrate_set(objpoints, imgpoints, used)

    dropped_names = []
    if args.drop_worst > 0 and (n - args.drop_worst) >= 10:
        order = sorted(range(n), key=lambda i: per_image[i]["rmse_px"], reverse=True)
        drop = set(order[:args.drop_worst])
        dropped_names = [used[i] for i in drop]
        objpoints = [o for i, o in enumerate(objpoints) if i not in drop]
        imgpoints = [o for i, o in enumerate(imgpoints) if i not in drop]
        used = [o for i, o in enumerate(used) if i not in drop]
        n = len(used)
        rms, K, dist, per_image, mean_reproj = calibrate_set(objpoints, imgpoints, used)
        print(f"Dropped {len(drop)} worst frame(s): {', '.join(dropped_names)}")

    result = {
        "image_size": list(image_size),
        "pattern_inner_corners": [args.cols, args.rows],
        "square_size_mm": args.square_size,
        "num_images_total": len(images),
        "num_images_used": n,
        "num_images_dropped": len(dropped_names),
        "dropped_frames": dropped_names,
        "rms_reprojection_error_px": float(rms),
        "mean_reprojection_error_px": mean_reproj,
        "camera_matrix": K.tolist(),
        "distortion_coefficients": dist.ravel().tolist(),
        "per_image_error": per_image,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    np.savez(args.output.with_suffix(".npz"), K=K, dist=dist,
             image_size=np.array(image_size))

    print("\n===== CALIBRATION SUMMARY =====")
    print(f"Images used          : {n}/{len(images)}")
    print(f"RMS reproj error     : {rms:.4f} px")
    print(f"Mean reproj error    : {mean_reproj:.4f} px  (spec: <0.5 acceptable, <0.3 excellent)")
    print(f"fx, fy               : {K[0, 0]:.2f}, {K[1, 1]:.2f}")
    print(f"cx, cy               : {K[0, 2]:.2f}, {K[1, 2]:.2f}")
    print(f"dist (k1 k2 p1 p2 k3): {np.round(dist.ravel(), 5)}")
    print(f"Saved                : {args.output} and {args.output.with_suffix('.npz')}")


if __name__ == "__main__":
    main()
