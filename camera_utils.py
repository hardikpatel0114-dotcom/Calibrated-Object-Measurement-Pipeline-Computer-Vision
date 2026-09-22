"""
Camera-calibration I/O and undistortion helpers, shared across the pipeline.

Loaded by the inference and measurement scripts so that undistortion is applied
identically everywhere. Undistortion uses the stored intrinsics exactly as the
spec requires (cv2.undistort with the original camera matrix).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def load_calibration(path: str | Path = "calibration/calibration.json"):
    """Return (K, dist, image_size) from the calibration JSON produced by calibrate.py."""
    path = Path(path)
    with open(path) as f:
        c = json.load(f)
    K = np.array(c["camera_matrix"], dtype=np.float64)
    dist = np.array(c["distortion_coefficients"], dtype=np.float64)
    image_size = tuple(c.get("image_size", ()))
    return K, dist, image_size


def undistort(img: np.ndarray, K: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """Undistort an image with the stored intrinsics (radial + tangential removal)."""
    return cv2.undistort(img, K, dist, None, K)


def _batch(args) -> None:
    K, dist, _ = load_calibration(args.calib)
    inp, out = Path(args.input), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    files = [p for p in sorted(inp.iterdir()) if p.suffix.lower() in IMAGE_EXTS]
    for p in files:
        img = cv2.imread(str(p))
        if img is None:
            print(f"  [skip] unreadable: {p.name}")
            continue
        cv2.imwrite(str(out / p.name), undistort(img, K, dist))
    print(f"Undistorted {len(files)} image(s) -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Batch-undistort a folder of images.")
    ap.add_argument("--calib", default="calibration/calibration.json")
    ap.add_argument("--input", required=True, help="folder of distorted images")
    ap.add_argument("--output", required=True, help="folder for undistorted output")
    _batch(ap.parse_args())
