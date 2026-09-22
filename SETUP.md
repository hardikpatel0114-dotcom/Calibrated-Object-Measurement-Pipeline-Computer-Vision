# SETUP — Installation, Environment & Run Guide

## 1. Prerequisites
- **Python 3.12** (tested on 3.12.10)
- **git**
- Optional: an **NVIDIA GPU** for training. If you don't have one, use the
  **Google Colab** notebook (`models/Train_MaskRCNN_Colab.ipynb`) for a free GPU.

## 2. Environment setup

### Windows (PowerShell)
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
# CPU-only torch (skip if you have CUDA and want a GPU build):
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### Linux / macOS
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

> **CUDA note:** `requirements.txt` pins CPU-friendly versions. For a local GPU,
> install the matching CUDA build of torch/torchvision from
> <https://pytorch.org/get-started/locally/>. On Colab, torch/torchvision are
> pre-installed — only `pycocotools` and `opencv-python-headless` are added.

> **Auto-labelling with MobileSAM** (`dataset/sam_label.py`, used only to create
> training labels — not the trained model):
> ```
> pip install timm git+https://github.com/ChaoningZhang/MobileSAM.git
> ```
> Checkpoint `models/weights/mobile_sam.pt` (~37 MB) is on Drive (README) or via
> `huggingface_hub.hf_hub_download("dhkim2810/MobileSAM", "mobile_sam.pt")`.

## 3. Data layout
Large files live on Google Drive (Section 2.2). Download them from the links in
[../README.md](../README.md) and place them as:
```
calibration/images/     <- 25 checkerboard photos
dataset/raw/            <- 70+ phone-cover photos
models/weights/         <- maskrcnn_best.pth (or train it yourself)
```

## 4. Run the pipeline end-to-end

| Step | Command | Output |
|------|---------|--------|
| 1. Calibrate | `python calibration/calibrate.py --images calibration/images --save-detections` | `calibration/calibration.json` (K, dist, reprojection error) |
| 2. Undistort | `python calibration/camera_utils.py --input dataset/raw --output dataset/undistorted` | undistorted images |
| 3. Auto-label | `python dataset/sam_label.py --images dataset/undistorted --category phone_cover` | `dataset/exports/annotations.json` + QA overlays |
| 4. Split | `python dataset/split_dataset.py --images dataset/undistorted` | `dataset/{train,val,test}` + split JSONs |
| 5. Train | `python models/train.py --data-root dataset --epochs 25` *(or the Colab notebook)* | `models/weights/maskrcnn_best.pth`, `docs/figures/loss_curves.png`, `models/metrics.json` |
| 6. Infer | `python inference/infer.py --input dataset/test --weights models/weights/maskrcnn_best.pth` | annotated masks in `inference/demo_outputs/` |
| 7. Measure | `python measurement/measure.py --image NEW.jpg --method model --weights models/weights/maskrcnn_best.pth` | `{width_mm, height_mm, confidence}` + annotated image |
| 8. Validate | `python measurement/validate_accuracy.py --images measurement/eval_images --method model --weights models/weights/maskrcnn_best.pth --gt-width <W> --gt-height <H>` | `measurement/accuracy_results.csv` + MAE/MPE |

**Verify calibration is required:** compare undistorted vs raw measurement with
`--no-undistort` on step 7 to reproduce the (incorrect) distorted result.

## 5. Optional — Docker
A minimal image (CPU inference/measurement):
```dockerfile
FROM python:3.12-slim
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENTRYPOINT ["python"]
```
Build & run:
```bash
docker build -t xis-measure .
docker run --rm -v ${PWD}:/app xis-measure measurement/measure.py --image NEW.jpg --method model
```

## 6. Reproducibility
- Dataset split is seeded (`--seed 42`).
- Training seeds Python/torch; the config is saved to `models/metrics.json`.
- All commands above are deterministic given the same inputs and seed.
