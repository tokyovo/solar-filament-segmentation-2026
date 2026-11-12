"""Seed refiner: re-segment a crop around each candidate filament at full resolution.

The first stage (the 1024 + 2048 U-Nets and ``postprocess``) gives candidate regions. Many of
them cover only part of a filament (NOTES.md section 13). The refiner looks at a square crop
around one candidate and predicts the whole filament that the candidate belongs to.

Input channels of a crop (all resized to ``CROP`` x ``CROP``):

1. the grayscale image;
2. the seed: the candidate mask;
3. the first-stage probability map.

Target: for each annotator of the image, the union of the GT filaments that touch the
candidate; the mean over the annotators (a soft label, like the first stage). A candidate
that touches no GT filament has an empty target, so the refiner can also reject it.

Build the crops (CPU), train one refiner per fold (GPU), then score::

    python -m src.refiner build
    python -m src.refiner train --fold 0 --out data/models/refiner_f0
    python -m src.refiner validate --fold 0 --model data/models/refiner_f0
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from pycocotools import mask as mask_util
from torch.utils.data import DataLoader, Dataset

from src import data, metric, postprocess
from src.predict import DEVICE, instances_frame
from src.train import loss_fn

CROP = 256
MARGIN = 1.5     # crop side = MARGIN x the longer side of the candidate's bounding box
MIN_SIDE = 128   # in 2048-pixel units
PP = {"threshold": 0.5, "min_area": 300, "merge_dist": 15}
MODELS = data.DATA_DIR.parent / "models"
CACHE = data.DATA_DIR.parent / "cache" / "refiner"


def first_stage_map(image_name: str, fold: int) -> np.ndarray:
    """Out-of-fold probability of the 1024 + 2048 pair, (2048, 2048) float in [0, 1]."""
    maps = [cv2.imread(str(MODELS / m / f"probs_fold{fold}" / f"{image_name}.png"), cv2.IMREAD_GRAYSCALE)
            for m in (f"unet_r34_f{fold}", f"unet_r34_r2048_f{fold}")]
    return np.mean(maps, axis=0) / 255


def crop_box(mask: np.ndarray) -> tuple[int, int, int]:
    """Square box (y0, x0, side) around a mask, clipped to the image."""
    ys, xs = np.nonzero(mask)
    side = int(max(MIN_SIDE, MARGIN * max(ys.max() - ys.min() + 1, xs.max() - xs.min() + 1)))
    side = min(side, data.SIZE)
    cy, cx = (ys.min() + ys.max()) // 2, (xs.min() + xs.max()) // 2
    y0 = int(np.clip(cy - side // 2, 0, data.SIZE - side))
    x0 = int(np.clip(cx - side // 2, 0, data.SIZE - side))
    return y0, x0, side


def crop(array: np.ndarray, box: tuple[int, int, int], interpolation: int) -> np.ndarray:
    y0, x0, side = box
    return cv2.resize(array[y0:y0 + side, x0:x0 + side], (CROP, CROP), interpolation=interpolation)


def candidate_inputs(image: np.ndarray, prob: np.ndarray, seeds: list[np.ndarray]):
    """Stack the 3 input channels for every candidate. Returns (inputs, boxes)."""
    inputs, boxes = [], []
    for seed in seeds:
        box = crop_box(seed)
        inputs.append(np.stack([crop(image, box, cv2.INTER_AREA),
                                crop(seed * 255, box, cv2.INTER_AREA),
                                crop((prob * 255).astype(np.uint8), box, cv2.INTER_AREA)]))
        boxes.append(box)
    return np.array(inputs, np.uint8).reshape(-1, 3, CROP, CROP), boxes


def build(fold: int, annotations: pd.DataFrame) -> None:
    """Write the crops and targets of all candidates of one fold to ``CACHE/fold<k>.npz``."""
    folds = data.assign_folds(annotations["image_name"].tolist())
    names = sorted(n for n in annotations["image_name"].unique() if folds[n] == fold)
    all_inputs, all_targets, meta = [], [], []
    for name in names:
        image = data.load_image(data.TRAIN_IMAGES / f"{name}.jpeg")
        prob = first_stage_map(name, fold)
        seeds = postprocess.probability_to_instances(prob, **PP)
        if not seeds:
            continue
        inputs, boxes = candidate_inputs(image, prob, seeds)
        entries = [[mask_util.decode(data.polygon_to_rle(p)) for p in g["polygon"]]
                   for _, g in annotations[annotations["image_name"] == name].groupby("entry_id")]
        for seed, box, x in zip(seeds, boxes, inputs):
            target = np.zeros(seed.shape, np.float32)
            for gt_masks in entries:
                union = np.zeros(seed.shape, np.uint8)
                for m in gt_masks:
                    if (m & seed).any():
                        union |= m
                target += union
            target /= len(entries)
            all_inputs.append(x)
            all_targets.append(np.round(crop(target, box, cv2.INTER_AREA) * 255).astype(np.uint8))
            meta.append({"image_name": name, "y0": box[0], "x0": box[1], "side": box[2]})
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE / f"fold{fold}.npz", inputs=np.array(all_inputs), targets=np.array(all_targets))
    pd.DataFrame(meta).to_csv(CACHE / f"fold{fold}.csv", index=False)
    print(f"fold {fold}: {len(names)} images, {len(meta)} candidates")


class CropSet(Dataset):
    """Cached candidate crops with flip and 90-degree rotation augmentation."""

    def __init__(self, folds: list[int]):
        parts = [np.load(CACHE / f"fold{f}.npz") for f in folds]
        self.inputs = np.concatenate([p["inputs"] for p in parts])
        self.targets = np.concatenate([p["targets"] for p in parts])

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, i: int):
        x, y = self.inputs[i], self.targets[i][None]
        rng = np.random.default_rng()
        if rng.random() < 0.5:
            x, y = x[..., ::-1], y[..., ::-1]
        k = int(rng.integers(4))
        x, y = np.rot90(x, k, (1, 2)), np.rot90(y, k, (1, 2))
        return (torch.from_numpy(x.astype(np.float32) / 255),
                torch.from_numpy(y.astype(np.float32) / 255))


def build_model(weights: str | None = "imagenet") -> torch.nn.Module:
    return smp.Unet("resnet34", encoder_weights=weights, in_channels=3, classes=1)


def train(fold: int, out: Path, epochs: int, batch: int, lr: float, seed: int) -> None:
    """Train on the candidates of every fold except ``fold``."""
    out.mkdir(parents=True, exist_ok=True)
    train_folds = [f for f in range(5) if f != fold and (CACHE / f"fold{f}.npz").exists()]
    (out / "config.json").write_text(json.dumps(
        {"fold": fold, "train_folds": train_folds, "epochs": epochs, "batch": batch, "lr": lr,
         "seed": seed, "crop": CROP, "margin": MARGIN, "min_side": MIN_SIDE, "pp": PP}, indent=2))
    loader = DataLoader(CropSet(train_folds), batch_size=batch, shuffle=True, num_workers=4,
                        drop_last=True, persistent_workers=True)
    torch.manual_seed(seed)
    model = build_model().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=epochs * len(loader), pct_start=0.05)
    scaler = torch.amp.GradScaler()
    for epoch in range(epochs):
        model.train()
        start, total = time.time(), 0.0
        for x, y in loader:
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(x)
            loss = loss_fn(logits.float(), y)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total += loss.item()
        print(f"epoch {epoch + 1}/{epochs} loss {total / len(loader):.4f} {time.time() - start:.0f}s "
              f"train_folds {train_folds} n {len(loader.dataset)}", flush=True)
        torch.save(model.state_dict(), out / "model.pt")


@torch.no_grad()
def refine(models: list[torch.nn.Module], image: np.ndarray, prob: np.ndarray,
           seeds: list[np.ndarray], threshold: float = 0.5, min_area: int = 300) -> list[np.ndarray]:
    """Replace each seed by the refiner's mask. Contested pixels go to the higher probability.

    The refiner output of each candidate is averaged over ``models`` and over a horizontal
    flip, resized back into its box, and pasted into a full-size map. A pixel joins the
    candidate with the highest refined probability, if that probability is above
    ``threshold``. Candidates smaller than ``min_area`` afterwards are dropped.
    """
    if not seeds:
        return []
    inputs, boxes = candidate_inputs(image, prob, seeds)
    x = torch.from_numpy(inputs.astype(np.float32) / 255).to(DEVICE)
    parts = []
    with torch.autocast(DEVICE, dtype=torch.float16, enabled=DEVICE == "cuda"):
        for chunk in torch.split(x, 32):
            p = 0
            for model in models:
                p = p + torch.sigmoid(model(chunk).float())
                p = p + torch.sigmoid(model(chunk.flip(-1)).float()).flip(-1)
            parts.append(p / (2 * len(models)))
    crops = torch.cat(parts)[:, 0].cpu().numpy()
    best = np.zeros((data.SIZE, data.SIZE), np.float32)
    owner = np.zeros((data.SIZE, data.SIZE), np.int32)
    for k, (c, (y0, x0, side)) in enumerate(zip(crops, boxes), start=1):
        full = cv2.resize(c, (side, side), interpolation=cv2.INTER_LINEAR)
        region = best[y0:y0 + side, x0:x0 + side]
        win = full > region
        region[win] = full[win]
        owner[y0:y0 + side, x0:x0 + side][win] = k
    owner[best <= threshold] = 0
    masks = [(owner == k).astype(np.uint8) for k in range(1, len(seeds) + 1)]
    return [m for m in masks if m.sum() >= min_area]


def load_refiners(model_dirs: list[Path]) -> list[torch.nn.Module]:
    models = []
    for d in model_dirs:
        m = build_model(weights=None)
        m.load_state_dict(torch.load(d / "model.pt", map_location="cpu"))
        models.append(m.to(DEVICE).eval())
    return models


def validate(fold: int, model_dirs: list[Path], thresholds: list[float]) -> None:
    """PQ of the first stage alone and with the refiner, on one fold's images."""
    annotations = data.load_annotations()
    folds = data.assign_folds(annotations["image_name"].tolist())
    val = annotations[annotations["image_name"].map(folds) == fold]
    gt_df = data.ground_truth_frame(val)
    models = load_refiners(model_dirs)
    base, refined = [], {t: [] for t in thresholds}
    for name in sorted(val["image_name"].unique()):
        image = data.load_image(data.TRAIN_IMAGES / f"{name}.jpeg")
        prob = first_stage_map(name, fold)
        seeds = postprocess.probability_to_instances(prob, **PP)
        base.append(instances_frame(name, seeds))
        for t in thresholds:
            refined[t].append(instances_frame(name, refine(models, image, prob, seeds, threshold=t)))
    r = metric.panoptic_quality(gt_df, pd.concat(base, ignore_index=True))
    print(f"fold {fold} first stage: pq {r.pq:.4f} sq {r.sq:.4f} rq {r.rq:.4f} tp {r.tp} fp {r.fp} fn {r.fn}")
    for t in thresholds:
        r = metric.panoptic_quality(gt_df, pd.concat(refined[t], ignore_index=True))
        print(f"fold {fold} refined t={t}: pq {r.pq:.4f} sq {r.sq:.4f} rq {r.rq:.4f} tp {r.tp} fp {r.fp} fn {r.fn}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["build", "train", "validate"])
    parser.add_argument("--fold", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--out", type=Path)
    parser.add_argument("--model", type=Path, nargs="+")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.4, 0.5])
    args = parser.parse_args()
    if args.mode == "build":
        annotations = data.load_annotations()
        for f in args.fold:
            build(f, annotations)
    elif args.mode == "train":
        train(args.fold[0], args.out, args.epochs, args.batch, args.lr, args.seed)
    else:
        validate(args.fold[0], args.model, args.thresholds)


if __name__ == "__main__":
    main()
