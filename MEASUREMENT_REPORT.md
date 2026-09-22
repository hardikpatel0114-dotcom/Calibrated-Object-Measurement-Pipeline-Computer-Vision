# Measurement Report

## 1. Objective
Convert the object's segmentation mask, in a **calibrated, undistorted** image,
into real-world **width and height in millimetres**, and validate the result
against physical calliper measurements.

## 2. Pixel-to-mm methodology
Implemented in [`measurement/measure.py`](../measurement/measure.py).

**Step 1 — Undistort.** Apply `cv2.undistort` with the Step-1 intrinsics.

**Step 2 — Reference scale from the card.** The ISO/IEC 7810 ID-1 card has known
dimensions `L = 85.60 mm`, `S = 53.98 mm`. It is detected as the foreground blob
whose min-area-rectangle aspect ratio matches `L/S ≈ 1.585` with high
rectangularity. From its pixel side lengths `long_px`, `short_px`:

```text
pixels_per_mm = ( long_px / 85.60  +  short_px / 53.98 ) / 2
```

Averaging both sides reduces the effect of small detection noise and mild
perspective.

**Step 3 — Object dimensions.** Fit a min-area rectangle to the object mask
(from the trained Mask R-CNN). With rectangle sides `w_px`, `h_px`:

```text
width_mm  = min(w_px, h_px) / pixels_per_mm
height_mm = max(w_px, h_px) / pixels_per_mm
```

The min-area rectangle makes the measurement invariant to in-plane rotation.

**Step 4 — Annotate.** Output an image with the mask overlay, the card and object
rectangles, and metric labels (`W`, `H`, confidence).

```bash
python measurement/measure.py --image NEW.jpg --method model \
    --weights models/weights/maskrcnn_best.pth
```

## 3. Why undistortion is mandatory (calibration dependency)
Lens distortion (radial + tangential) displaces pixels **non-uniformly** across
the frame — most at the edges. Consequences for measurement on a **raw** image:
- Straight object/card edges appear curved, so the min-area rectangle over- or
  under-estimates side lengths.
- `pixels_per_mm` derived from the card is only locally valid; applying it to an
  object elsewhere in the frame introduces a position-dependent error.

Undistortion restores a consistent, rectilinear projection so a single
`pixels_per_mm` is valid across the image. **Empirically on this dataset
(24 images):**

| Pipeline | Overall MAE | Overall MPE |
|----------|-------------|-------------|
| With undistortion (default) | 3.89 mm | 3.31 % |
| Raw (`--no-undistort`) | 3.68 mm | 3.14 % |

The two are within noise (~0.2 mm), and the honest reason is important: the
**iPhone 12 Pro Max main lens has very low native distortion** (k1 = 0.10,
k2 = −0.28; see the Calibration Report), so `cv2.undistort` is nearly an identity
transform here and its measured effect is small. **Undistortion is still applied
by default and still matters in general** — on wide-angle / action / fisheye
cameras k1 is several times larger, and the same raw measurement would be badly
wrong, worst at the frame edges where distortion displaces pixels most. Knowing
*when* the correction is significant is part of understanding the geometry, not a
reason to skip it. Reproduce the raw case with `--no-undistort`.

## 4. Reference object justification
The ID-1 card is chosen because it is **precisely standardised** (85.60 ×
53.98 mm, tight tolerance), **flat**, **planar with the object**, and **easy to
detect** by its unique aspect ratio — a reliable, traceable scale reference.

## 5. Accuracy validation
**Ground truth:** the object's true width and height are measured once with a
calliper (averaged over repeats). **Instances:** the system is evaluated on
**10+ images** of the object at varied position/rotation/distance; each system
output is compared to the ground truth. This is a deliberate, documented
interpretation of "measure 10+ instances" for a single physical object — it
measures both **accuracy** and **repeatability**.

```bash
python measurement/validate_accuracy.py --images measurement/eval_images \
    --method model --weights models/weights/maskrcnn_best.pth \
    --gt-width <W_mm> --gt-height <H_mm>
```

Ground truth: **width 79 mm, height 160 mm** (calliper). Evaluated on **24
held-out images** (val + test, never used to train the model), full pipeline
(raw → undistort → card → model mask → measure). From `measurement/accuracy_results.csv`:

| Dimension | MAE (mm) | MPE (%) |
|-----------|----------|---------|
| Short side (79 mm) | 2.76 | 3.50 |
| Long side (160 mm) | 5.01 | 3.13 |
| **Overall** | **3.89** | **3.31** |

**~3.3 % mean error / ~3.9 mm MAE** measuring a 160 × 79 mm object from a single
phone photo — good accuracy for a monocular reference-scaled system. Per-image
table: `measurement/accuracy_results.csv`.

![measurement demo](figures/measurement_demo.jpg)

## 6. Error sources & limitations
- **Coplanarity / thickness.** The pixel scale is set at the card's plane. An
  object of thickness `t` viewed from distance `d` is magnified by ≈ `t/d`
  (e.g. a 3 mm cover at 400 mm ⇒ ~0.75% bias). A thin object minimises this.
- **Perspective / tilt.** Non-top-down capture makes the scale vary across the
  frame; images are shot as close to perpendicular as possible.
- **Mask boundary.** Segmentation edge precision (±1–2 px) bounds achievable
  accuracy; averaging both card sides and using the min-area rectangle mitigate it.
- **Isotropic-scale assumption.** Square pixels and a fronto-parallel view are
  assumed; a full homography (planar rectification) would remove residual
  perspective — noted as future work.
- **Card detection.** Assumes the card is fully visible, unoccluded, and the
  most card-shaped blob; low contrast can defeat detection.

## 7. End-to-end demo
A single new image → mask overlay + width (mm) + height (mm) + confidence:
```bash
python measurement/measure.py --image NEW.jpg --method model \
    --weights models/weights/maskrcnn_best.pth --out result.jpg
```
Output JSON example (held-out image IMG_0334; ground truth 79 × 160 mm):
```json
{ "width_mm": 73.59, "height_mm": 154.17, "confidence": 0.999,
  "pixels_per_mm": 9.2962, "undistorted": true, "method": "model" }
```
