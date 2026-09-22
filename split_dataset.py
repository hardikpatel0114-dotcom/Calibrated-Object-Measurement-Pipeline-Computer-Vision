#!/usr/bin/env python3
"""
Split a COCO-labelled image set into train / val / test (default 70/20/10).

Copies image files into dataset/{train,val,test}/ and writes a COCO JSON per
split to dataset/exports/{train,val,test}.json. The split is seeded for
reproducibility (spec: reproducible training pipeline).

Usage:
    python dataset/split_dataset.py --images dataset/undistorted \
        --ann dataset/exports/annotations.json --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path


def load_coco(p: Path) -> dict:
    with open(p) as f:
        return json.load(f)


def subset(coco: dict, image_ids) -> dict:
    ids = set(image_ids)
    return {
        "info": coco.get("info", {}),
        "images": [im for im in coco["images"] if im["id"] in ids],
        "annotations": [a for a in coco["annotations"] if a["image_id"] in ids],
        "categories": coco["categories"],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True, type=Path, help="folder with the labelled images")
    ap.add_argument("--ann", default="dataset/exports/annotations.json", type=Path)
    ap.add_argument("--out-root", default="dataset", type=Path)
    ap.add_argument("--ratios", nargs=3, type=float, default=[0.7, 0.2, 0.1],
                    metavar=("TRAIN", "VAL", "TEST"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    coco = load_coco(args.ann)
    annotated = {a["image_id"] for a in coco["annotations"]}
    images = [im for im in coco["images"] if im["id"] in annotated]
    dropped = [im for im in coco["images"] if im["id"] not in annotated]
    if dropped:
        print(f"WARNING: {len(dropped)} unlabelled image(s) excluded from splits.")

    rng = random.Random(args.seed)
    rng.shuffle(images)
    n = len(images)
    n_tr = int(round(args.ratios[0] * n))
    n_va = int(round(args.ratios[1] * n))
    splits = {
        "train": images[:n_tr],
        "val": images[n_tr:n_tr + n_va],
        "test": images[n_tr + n_va:],
    }

    exports = args.out_root / "exports"
    exports.mkdir(parents=True, exist_ok=True)
    for name, ims in splits.items():
        dst = args.out_root / name
        dst.mkdir(parents=True, exist_ok=True)
        for old in dst.glob("*"):          # clear previous split copies, keep .gitkeep
            if old.is_file() and old.name != ".gitkeep":
                old.unlink()
        for im in ims:
            shutil.copy2(args.images / im["file_name"], dst / im["file_name"])
        with open(exports / f"{name}.json", "w") as f:
            json.dump(subset(coco, [im["id"] for im in ims]), f, indent=2)
        print(f"{name:5s}: {len(ims):3d} images -> {dst}")

    print(f"Total: {n} images  (seed={args.seed}, ratios={args.ratios})")
    print(f"Split annotations written to {exports}/")


if __name__ == "__main__":
    main()
