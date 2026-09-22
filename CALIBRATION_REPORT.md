# Camera Calibration Report

## 1. Objective
Estimate the **intrinsic parameters** (camera matrix and lens-distortion
coefficients) of the capture camera so that all downstream images can be
**undistorted** before measurement. Undistortion is mandatory: uncorrected
radial/tangential distortion bends straight edges and makes the pixel-to-mm
scale vary across the frame, which would corrupt any measurement.

## 2. Equipment
| Item | Detail |
|------|--------|
| Camera | Apple iPhone 12 Pro Max, **main wide lens fixed at 1×** |
| Format | HEIC → JPEG (`tools/heic_to_jpg.py`), EXIF-upright |
| Target | Checkerboard, **8 × 10 squares → 7 × 9 inner corners** (`calibration/checkerboard.html`) |
| Display | Checkerboard shown full-screen on a laptop LCD, photographed with the phone |

**Why a screen instead of print:** no printer was available. An LCD is flat and
rigid, so it is a valid calibration surface. Crucially, the intrinsic matrix and
distortion coefficients are **independent of the physical square size** (scaling
the board only scales the unused extrinsic translation), so undistortion is
unaffected by the exact on-screen square dimensions. Known caveats (screen glare,
moiré, panel flatness) are the main source of residual reprojection error and are
discussed in §7.

## 3. Method
Implemented in [`calibration/calibrate.py`](../calibration/calibrate.py) with OpenCV:
1. Detect inner corners per image with `cv2.findChessboardCornersSB`
   (robust detector), falling back to `findChessboardCorners` + `cornerSubPix`.
2. Build 3-D object points on the `z = 0` plane for the 7 × 9 grid.
3. Run `cv2.calibrateCamera` with **`CALIB_FIX_K3`** (see §7) to estimate `K` and
   distortion `(k1, k2, p1, p2)`.
4. Compute the RMS reprojection error and a per-image breakdown; **drop the 8
   highest-error frames** and re-calibrate (calibration hygiene).

Reproduce:
```bash
python calibration/calibrate.py --images calibration/images \
    --cols 7 --rows 9 --fix-k3 --drop-worst 8 --save-detections
```

## 4. Calibration set
- **Images captured:** 28 (spec minimum: 20), varied angle, distance, position.
- **Boards detected:** 28 / 28.
- **Frames used after dropping 8 worst:** **20** (meets the 20-image minimum).
- Dropped frames: IMG_0215, 0216, 0219, 0224, 0225, 0228, 0231, 0238.
- Corner-overlay detections saved to `calibration/detected/`.

## 5. Results
Values from [`calibration/calibration.json`](../calibration/calibration.json).

**Camera matrix `K` (pixels), for 3024 × 4032 images:**
```
[ fx   0  cx ]     [ 2995.76     0.00   1515.70 ]
[  0  fy  cy ]  =  [    0.00  2984.87   2014.33 ]
[  0   0   1 ]     [    0.00     0.00      1.00  ]
```

**Distortion coefficients** `(k1, k2, p1, p2, k3)`:
`[0.1023, -0.2759, -0.00001, 0.00031, 0.0]`

| Metric | Value | Spec |
|--------|-------|------|
| RMS reprojection error | **0.763 px** | < 0.5 acceptable, < 0.3 excellent |
| Per-image error (min / median / max) | 0.50 / 0.72 / 1.25 px | — |
| Image resolution | 3024 × 4032 | — |
| Frames used | 20 / 28 | ≥ 20 |

## 6. Undistortion verification
`calibration/camera_utils.py` applies `cv2.undistort` with the stored intrinsics.
With the constrained model the correction is gentle and faithful (only ~0.4 % of
the frame becomes border, vs ~6.3 % for the unconstrained fit — see §7). All 79
object images are undistorted before labelling and measurement:
```bash
python calibration/camera_utils.py --input dataset/raw --output dataset/undistorted
```

## 7. Discussion — model choice & honesty about the error
The **unconstrained** 5-parameter fit reached a slightly lower reprojection error
(0.73 px) but produced physically implausible, over-fit coefficients
(k1 = 0.16, k2 = −0.79, **k3 = 1.30**) that warped the image aggressively (6.3 %
black border). A modern phone main camera has only mild distortion, so those
large terms were fitting **screen artefacts** (moiré / panel non-flatness), not
the lens. Constraining **k3 = 0** yields stable, physical coefficients
(k1 = 0.10, k2 = −0.28) and a near-faithful undistortion (0.4 % border) — the
correct choice for accurate measurement, even at a marginally higher reprojection
error. This is the key trade-off: **"undistortion applied correctly" matters more
than a vanity error number.**

The residual ~0.76 px error is dominated by the screen-based capture, not the
math; a printed rigid board would likely reach < 0.5 px. Rather than rely on the
reprojection number alone, the undistortion's real value is **validated
empirically** against physical ground truth in the
[Measurement Report](MEASUREMENT_REPORT.md) (measurement error with vs without
undistortion).

## 8. Artifacts
- `calibration/calibration.json` / `.npz` — K, distortion, per-frame errors (committed, small)
- `calibration/detected/` — corner-overlay images
- Calibration images — **Google Drive** (see README), not committed.
