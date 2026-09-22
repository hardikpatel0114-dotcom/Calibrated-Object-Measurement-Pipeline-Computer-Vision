# Model Training Report

## 1. Model selection & rationale
**Architecture:** **Mask R-CNN with a ResNet-50-FPN backbone** (torchvision),
COCO-pretrained, with the box and mask heads replaced for **2 classes**
(background + object). Implemented in [`inference/model.py`](../inference/model.py).

**Why this model (and not the excluded ones):**
- The assessment **excludes Ultralytics YOLO and Roboflow models**; Mask R-CNN is
  a well-established, independent alternative.
- It is a **two-stage instance segmenter** producing high-quality per-instance
  masks — ideal for precise dimensional measurement from mask contours.
- A **COCO-pretrained** backbone transfer-learns effectively on a small (~70
  image) custom dataset, which is critical given the limited data.

## 2. Training setup
| Hyperparameter | Value |
|----------------|-------|
| Split | 70% train / 20% val / 10% test (seeded) |
| Optimizer | SGD, momentum 0.9, weight decay 5e-4 |
| Learning rate | 0.005, StepLR (step 8, γ 0.1) |
| Epochs | 20 |
| Batch size | 2 |
| Augmentation | random horizontal flip, brightness jitter (0.8–1.2×) |
| Input | full-resolution undistorted images |
| Hardware | Google Colab GPU (T4) / local CPU |
| Framework | PyTorch 2.3.1 + torchvision 0.18.1 |

Reproduce (or use `models/Train_MaskRCNN_Colab.ipynb`):
```bash
python models/train.py --data-root dataset --epochs 25 --batch-size 2 --lr 0.005
```

## 3. Metrics
From `models/metrics.json` (validation set, 16 images). Segmentation (mask) metrics.

| Metric | Value |
|--------|-------|
| mAP@0.5 | **0.871** |
| mAP@0.5:0.95 | **0.840** |
| Mean mask IoU | **0.840** |
| Precision | 0.875 |
| Recall | 0.875 |
| F1 | 0.875 |

- **Loss curves (train/val):** ![loss curves](figures/loss_curves.png)
- **Per-epoch log:** `models/training_log.csv` (train/val loss + metrics per epoch)
- **Metric definitions:** mAP via COCO evaluation (`pycocotools`, segmentation);
  IoU/precision/recall/F1 computed on the mask at score ≥ 0.5, IoU ≥ 0.5.
- **Convergence:** train loss 1.17 → 0.11; validation mAP@0.5 climbs from 0.48
  (epoch 1) and plateaus at 0.87 by ~epoch 12. These are **genuine** cover-
  segmentation metrics — an earlier model trained on over-loose auto-labels showed
  an inflated mAP@0.5 = 1.0 that did **not** translate to accurate measurement,
  which is exactly why the labelling was rebuilt with SAM (see DATASET_CARD).

## 4. Qualitative results
Predictions on the held-out **test** split (mask overlay + confidence):

![test predictions](figures/test_predictions.jpg)

Full outputs are in `inference/demo_outputs/`. Generate with:
```bash
python inference/infer.py --input dataset/test \
    --weights models/weights/maskrcnn_best.pth
```

## 5. Inference usage
```bash
# Segmentation only (annotated mask + confidence)
python inference/infer.py --input NEW.jpg --weights models/weights/maskrcnn_best.pth
# End-to-end measurement (mask + width/height in mm)
python measurement/measure.py --image NEW.jpg --method model \
    --weights models/weights/maskrcnn_best.pth
```
Both scripts **undistort** the input first via the stored calibration.

## 6. Discussion
Training logs, loss curves and the best checkpoint (selected by validation
mAP@0.5) are saved automatically. With a small single-class dataset and a
pretrained backbone, high mask IoU is expected; the held-out test set guards
against overfitting. The best weights (`maskrcnn_best.pth`) are hosted on
**Google Drive** (link in README) and are not committed.
