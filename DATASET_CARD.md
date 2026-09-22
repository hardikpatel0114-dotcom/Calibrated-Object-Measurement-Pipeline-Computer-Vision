# Dataset Card

## 1. Object selection
| Property | Detail |
|----------|--------|
| **Object** | Flat, rigid **phone cover** (silicone, black with copper accents) |
| **Real dimensions (ground truth)** | **160 mm (height) × 79 mm (width)**, ruler-measured |
| **Reference object** | ISO/IEC 7810 **ID-1 card** (85.60 × 53.98 mm) in-frame |

**Why this object (justification):**
- **Availability** — a single consistent item, always on hand; the ID card is a
  precise, internationally standardised scale reference.
- **Geometry** — thin and flat, so its top surface is ~coplanar with the
  reference card, keeping a single pixels-per-mm scale valid. Its **2.03 aspect
  ratio is clearly distinct from the card's 1.585**, so the two never get
  confused. Distinct width ≠ height gives a clean two-dimension measurement demo.
- **Labelling ease** — a solid dark object; on a plain background it is the
  dominant dark region, enabling reliable automatic mask labelling.

## 2. Collection strategy
- **Camera:** iPhone 12 Pro Max, main lens fixed at **1×** (same camera as
  calibration), JPEG (converted from HEIC), full resolution **3024 × 4032**.
- **View:** top-down (camera parallel to the surface), ~30–50 cm.
- **Background:** plain **mid-grey towel** — contrasts with both the black cover
  and the white card, and (unlike the initial wood-table attempt) carries no
  colour that confuses the segmenter. Only the cover + card in frame, small gap.
- **Diversity:** varied object position, rotation, distance, and lighting.
- **Undistortion:** every image is undistorted with the Step-1 calibration
  **before** labelling and training.
- **Count:** **79** images (spec minimum: 70).

## 3. Labelling
| Property | Detail |
|----------|--------|
| **Method** | Automatic instance-mask labelling with **MobileSAM** (promptable Segment Anything) — [`dataset/sam_label.py`](../dataset/sam_label.py). SAM is used only for *labelling*; it is **not** the trained segmentation model, so it does not fall under the YOLO/Roboflow restriction. |
| **How** | The cover is located as the largest **interior dark** blob; its bounding box, plus a negative point on the bright reference card, prompts SAM. The largest card-free mask in the plausible size range is kept as the cover silhouette. |
| **QA** | An overlay is written for every image (`dataset/exports/overlays/`); label aspect ratios were checked against the object's true 2.03 — **mean 1.99, median 2.02**, with 75/79 in the 1.7–2.4 band. |
| **Export** | COCO instance-segmentation JSON (`dataset/exports/annotations.json`). |

> **Approach history (documented for transparency)** — three iterations, each
> fixing a real, measured failure:
> 1. A cluttered **wood** table defeated colour/brightness cues (the wood's orange
>    hue collided with the cover's copper accents; desk clutter added distractors).
> 2. Re-shooting on a plain **grey towel** fixed localization, but a classical
>    **GrabCut** labeller over-expanded into the towel/card on **62/79** images
>    (mask aspect collapsed toward 1.0). A first model faithfully learned these
>    bad masks — measurement then failed on the short side (~77 % error).
> 3. A **box-prompted MobileSAM** labeller produced clean cover masks
>    (**75/79** with aspect 1.7–2.4), which the final model learned correctly.
>
> This is concrete evidence that **labelling quality — not just model choice —
> drives measurement accuracy**, and that the pipeline was validated end-to-end
> rather than trusted on proxy metrics.

## 4. Statistics
Seeded split (`--seed 42`) via [`dataset/split_dataset.py`](../dataset/split_dataset.py).

| Split | Images | Instances |
|-------|--------|-----------|
| Train (70%) | 55 | 55 |
| Val (20%)   | 16 | 16 |
| Test (10%)  | 8  | 8  |
| **Total**   | 79 | 79 |

- **Classes:** 1 (`phone_cover`) + background. **Class balance:** single-class,
  one instance per image (balanced by construction).

## 5. Hosting
Raw images, undistorted images, and the COCO export are on **Google Drive**
(links in README). Only code and this card are committed to GitHub.
