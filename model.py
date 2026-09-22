"""
Mask R-CNN (ResNet-50-FPN) model definition + inference helpers.

Architecture choice: Mask R-CNN via torchvision — a two-stage instance
segmentation network. Chosen deliberately over Ultralytics YOLO and Roboflow
models (excluded by the assessment). Rationale: mature, COCO-pretrained
backbone that transfer-learns well on a small (~70 image) custom dataset, and
produces high-quality per-instance masks suited to precise measurement.

Shared by the training notebook and the inference / measurement scripts so the
model is defined in exactly one place.
"""
from __future__ import annotations

import numpy as np
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


def build_model(num_classes: int, pretrained: bool = True):
    """Mask R-CNN with COCO-pretrained backbone and fresh heads.

    num_classes includes background (e.g. 2 = background + object).
    """
    weights = "DEFAULT" if pretrained else None
    model = maskrcnn_resnet50_fpn(weights=weights)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)
    return model


def load_model(weights_path, num_classes: int = 2, device: str = "cpu"):
    """Load trained weights for inference."""
    model = build_model(num_classes, pretrained=False)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return model


@torch.no_grad()
def predict(model, image_bgr: np.ndarray, device: str = "cpu",
            score_thresh: float = 0.5, mask_thresh: float = 0.5):
    """Run inference on a BGR uint8 image.

    Returns a list of detections sorted by score (desc), each a dict with:
        {"box": [x1,y1,x2,y2], "score": float, "mask": HxW uint8}
    """
    import cv2
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0).to(device)
    out = model([tensor])[0]

    results = []
    for box, score, mask in zip(out["boxes"], out["scores"], out["masks"]):
        if float(score) < score_thresh:
            continue
        binary = (mask[0].cpu().numpy() >= mask_thresh).astype(np.uint8)
        results.append({
            "box": box.cpu().numpy().tolist(),
            "score": float(score),
            "mask": binary,
        })
    results.sort(key=lambda d: d["score"], reverse=True)
    return results
