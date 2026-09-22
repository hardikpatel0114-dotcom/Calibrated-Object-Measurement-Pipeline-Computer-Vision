#!/usr/bin/env python3
"""
Train Mask R-CNN (ResNet-50-FPN) on the custom COCO-format dataset.

Logs train/val loss per epoch, evaluates segmentation mAP@0.5, mAP@0.5:0.95,
mean mask IoU, precision, recall and F1, saves the best weights, a loss-curve
figure, a metrics JSON and a CSV training log — everything the TRAINING_REPORT
needs. Runs on GPU (Colab) or CPU.

Usage:
    python models/train.py --data-root dataset --epochs 25 --batch-size 2 \
        --lr 0.005 --out models/weights
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from pycocotools.coco import COCO
from torch.utils.data import DataLoader, Dataset
from torchvision.ops import masks_to_boxes

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from inference.model import build_model  # noqa: E402


# --------------------------------- dataset ---------------------------------- #
class CocoInstanceDataset(Dataset):
    def __init__(self, images_dir, ann_json, train=False):
        self.images_dir = Path(images_dir)
        self.coco = COCO(ann_json)
        self.ids = [i for i in self.coco.getImgIds()
                    if self.coco.getAnnIds(imgIds=i)]
        self.train = train

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        info = self.coco.loadImgs(img_id)[0]
        img = cv2.cvtColor(cv2.imread(str(self.images_dir / info["file_name"])),
                           cv2.COLOR_BGR2RGB)
        anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
        masks = np.stack([self.coco.annToMask(a) for a in anns]).astype(np.uint8)

        if self.train:
            if random.random() < 0.5:                       # horizontal flip
                img = img[:, ::-1, :].copy()
                masks = masks[:, :, ::-1].copy()
            if random.random() < 0.5:                       # brightness jitter
                img = np.clip(img.astype(np.float32) *
                              random.uniform(0.8, 1.2), 0, 255).astype(np.uint8)

        masks_t = torch.as_tensor(masks, dtype=torch.uint8)
        boxes = masks_to_boxes(masks_t)
        keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        boxes, masks_t = boxes[keep], masks_t[keep]
        target = {
            "boxes": boxes.float(),
            "labels": torch.ones((len(boxes),), dtype=torch.int64),   # 1 = object
            "masks": masks_t,
            "image_id": torch.tensor([img_id]),
            "area": (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]),
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
        }
        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return img_t, target


def collate(batch):
    return tuple(zip(*batch))


# ------------------------------- evaluation --------------------------------- #
@torch.no_grad()
def evaluate(model, dataset, device):
    from pycocotools import mask as mask_utils
    from pycocotools.cocoeval import COCOeval

    model.eval()
    results, ious = [], []
    tp = fp = fn = 0
    for idx in range(len(dataset)):
        img_t, target = dataset[idx]
        img_id = int(target["image_id"])
        out = model([img_t.to(device)])[0]
        scores = out["scores"].cpu().numpy()
        keep = scores >= 0.5
        pmasks = (out["masks"][keep, 0].cpu().numpy() >= 0.5).astype(np.uint8)
        pboxes = out["boxes"][keep].cpu().numpy()
        for m, s, b in zip(pmasks, scores[keep], pboxes):
            rle = mask_utils.encode(np.asfortranarray(m))
            rle["counts"] = rle["counts"].decode()
            results.append({
                "image_id": img_id, "category_id": 1, "segmentation": rle,
                "score": float(s),
                "bbox": [float(b[0]), float(b[1]),
                         float(b[2] - b[0]), float(b[3] - b[1])],
            })
        gt = target["masks"].numpy()
        gt_union = gt.any(0).astype(np.uint8) if len(gt) else None
        if gt_union is not None:
            if len(pmasks):
                inter = np.logical_and(pmasks[0], gt_union).sum()
                union = np.logical_or(pmasks[0], gt_union).sum()
                iou = inter / union if union else 0.0
                ious.append(iou)
                tp, fp, fn = (tp + 1, fp, fn) if iou >= 0.5 else (tp, fp + 1, fn + 1)
            else:
                ious.append(0.0)
                fn += 1

    metrics = {"mAP_0.5:0.95": 0.0, "mAP_0.5": 0.0}
    if results:
        coco_dt = dataset.coco.loadRes(results)
        ev = COCOeval(dataset.coco, coco_dt, "segm")
        ev.evaluate(); ev.accumulate(); ev.summarize()
        metrics["mAP_0.5:0.95"], metrics["mAP_0.5"] = float(ev.stats[0]), float(ev.stats[1])
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    metrics.update({
        "mean_mask_IoU": float(np.mean(ious)) if ious else 0.0,
        "precision": prec, "recall": rec,
        "f1": 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0,
    })
    return metrics


@torch.no_grad()
def validation_loss(model, loader, device):
    model.train()                       # loss dict is only returned in train mode
    total, nb = 0.0, 0
    for images, targets in loader:
        images = [i.to(device) for i in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        losses = model(images, targets)
        total += sum(l.item() for l in losses.values())
        nb += 1
    return total / max(nb, 1)


# --------------------------------- training --------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default="dataset", type=Path)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--num-classes", type=int, default=2)   # bg + object
    ap.add_argument("--pretrained", action="store_true", default=True,
                    help="start from COCO-pretrained weights (default)")
    ap.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    ap.add_argument("--out", default="models/weights", type=Path)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = args.device
    exports = args.data_root / "exports"

    train_ds = CocoInstanceDataset(args.data_root / "train", exports / "train.json", train=True)
    val_ds = CocoInstanceDataset(args.data_root / "val", exports / "val.json", train=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate)
    print(f"train={len(train_ds)}  val={len(val_ds)}  device={device}")

    model = build_model(args.num_classes, pretrained=args.pretrained).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)

    args.out.mkdir(parents=True, exist_ok=True)
    fig_dir = Path("docs/figures"); fig_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path("models/training_log.csv")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = Path("models/metrics.json")
    history, best_map = [], -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running, nb = 0.0, 0
        for images, targets in train_loader:
            images = [i.to(device) for i in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            losses = model(images, targets)
            loss = sum(losses.values())
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            running += loss.item(); nb += 1
        scheduler.step()

        train_loss = running / max(nb, 1)
        val_loss = validation_loss(model, val_loader, device)
        metrics = evaluate(model, val_ds, device)
        row = {"epoch": epoch, "train_loss": round(train_loss, 4),
               "val_loss": round(val_loss, 4),
               **{k: round(v, 4) for k, v in metrics.items()}}
        history.append(row)
        print(f"epoch {epoch:02d}  train={train_loss:.3f}  val={val_loss:.3f}  "
              f"mAP@.5={metrics['mAP_0.5']:.3f}  IoU={metrics['mean_mask_IoU']:.3f}")

        torch.save(model.state_dict(), args.out / "maskrcnn_last.pth")
        if metrics["mAP_0.5"] >= best_map:
            best_map = metrics["mAP_0.5"]
            torch.save(model.state_dict(), args.out / "maskrcnn_best.pth")

    # ---- persist logs, metrics, curves ----
    with open(log_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        w.writeheader(); w.writerows(history)
    with open(metrics_path, "w") as f:
        json.dump({"best_mAP_0.5": best_map, "final": history[-1],
                   "config": vars(args) | {"out": str(args.out),
                                           "data_root": str(args.data_root)}},
                  f, indent=2, default=str)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ep = [h["epoch"] for h in history]
    plt.figure(figsize=(7, 4))
    plt.plot(ep, [h["train_loss"] for h in history], label="train loss")
    plt.plot(ep, [h["val_loss"] for h in history], label="val loss")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend()
    plt.title("Mask R-CNN training / validation loss")
    plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(fig_dir / "loss_curves.png", dpi=130)

    print(f"\nBest mAP@0.5 = {best_map:.3f}")
    print(f"Weights  -> {args.out}/maskrcnn_best.pth")
    print(f"Log      -> {log_path}")
    print(f"Curves   -> {fig_dir}/loss_curves.png")


if __name__ == "__main__":
    main()
