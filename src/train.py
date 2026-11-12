"""Train a U-Net on the consensus masks of the downscaled training images.

Usage (take the GPU lock first, see ../GPU-SCHEDULE.md)::

    python -m src.train --fold 0 --out data/models/unet_r34_f0

``--fold -1`` trains on every image, for the final model.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src import data

CACHE = data.DATA_DIR.parent / "cache"


class CropDataset(Dataset):
    """Random square crops of cached images. Most crops are centred near a filament.

    ``target`` chooses the label:

    - ``soft``: the fraction of annotators that marked the pixel.
    - ``majority``: 1 where at least half of the annotators marked the pixel.
    - ``random``: the mask of one annotator, drawn again for every sample.

    With ``spine=True`` the label gets a second channel: the fraction of annotators whose
    filament spine passes through the pixel (``build_cache`` stores it as ``spine``).
    """

    def __init__(self, names: list[str], res: int, crop: int, samples_per_image: int,
                 target: str = "soft", filament_bias: float = 0.7, spine: bool = False, clean: bool = False):
        folder = CACHE / (f"r{res}" + ("c" if clean else ""))
        if (folder / f"{names[0]}.image.npy").exists():
            # Memory-mapped cache (build_cache --npy): crops are read from disk on demand.
            assert target == "soft", "the .npy cache holds only the soft target"
            self.items = [(np.load(folder / f"{n}.image.npy", mmap_mode="r"),
                           np.load(folder / f"{n}.target.npy", mmap_mode="r")) for n in names]
            self.entries = None
            self.spines = None
        else:
            cached = [np.load(folder / f"{n}.npz") for n in names]
            self.items = [(it["image"], it["target"]) for it in cached]
            self.entries = [it["entries"] * 255 for it in cached] if target == "random" else None
            self.spines = [it["spine"] for it in cached] if spine else None
        if target == "majority":
            self.items = [(image, (soft >= 128).astype(np.uint8) * 255) for image, soft in self.items]
        # A sample of filament pixels per image is enough to centre the crops.
        rng = np.random.default_rng(0)
        self.filament_pixels = []
        for _, t in self.items:
            pixels = np.argwhere(np.asarray(t) > 0).astype(np.int32)
            if len(pixels) > 4000:
                pixels = pixels[rng.choice(len(pixels), 4000, replace=False)]
            self.filament_pixels.append(pixels)
        self.crop, self.samples, self.bias = crop, samples_per_image, filament_bias

    def __len__(self) -> int:
        return len(self.items) * self.samples

    def __getitem__(self, index: int):
        image, target = self.items[index % len(self.items)]
        size, crop, rng = image.shape[0], self.crop, np.random.default_rng()
        if self.entries is not None:
            entries = self.entries[index % len(self.items)]
            target = entries[rng.integers(len(entries))]
        pixels = self.filament_pixels[index % len(self.items)]
        if len(pixels) and rng.random() < self.bias:
            cy, cx = pixels[rng.integers(len(pixels))]
            y0 = int(np.clip(cy - crop // 2 + rng.integers(-crop // 4, crop // 4), 0, size - crop))
            x0 = int(np.clip(cx - crop // 2 + rng.integers(-crop // 4, crop // 4), 0, size - crop))
        else:
            y0, x0 = rng.integers(0, size - crop + 1, size=2)
        image = image[y0:y0 + crop, x0:x0 + crop]
        target = target[y0:y0 + crop, x0:x0 + crop]
        if self.spines is not None:
            spine = self.spines[index % len(self.items)][y0:y0 + crop, x0:x0 + crop]
            target = np.stack([target, spine], axis=-1)
        if rng.random() < 0.5:
            image, target = image[:, ::-1], target[:, ::-1]
        k = int(rng.integers(4))
        image, target = np.rot90(image, k), np.rot90(target, k)
        image = image.astype(np.float32) / 255.0
        image = image * rng.uniform(0.8, 1.2) + rng.uniform(-0.1, 0.1)  # brightness and contrast
        target = torch.from_numpy(target.astype(np.float32) / 255.0)
        target = target.permute(2, 0, 1) if target.ndim == 3 else target[None]
        return torch.from_numpy(image.copy())[None], target


def build_model(encoder: str = "resnet34", weights: str | None = "imagenet", classes: int = 1) -> torch.nn.Module:
    """A one-channel-in U-Net. Channel 0 is the filament logit; channel 1, if any, the spine."""
    return smp.Unet(encoder, encoder_weights=weights, in_channels=1, classes=classes)


def loss_fn(logits: torch.Tensor, target: torch.Tensor, kind: str = "bce_dice") -> torch.Tensor:
    """BCE (or focal BCE) on the soft target plus a soft Dice term over the whole batch.

    ``focal_dice`` weights each pixel's BCE by |target - probability|^2, the focal weight for
    soft labels; for 0/1 labels it equals the usual (1 - p_t)^2.
    """
    prob = torch.sigmoid(logits)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    if kind == "focal_dice":
        bce = bce * (target - prob).abs().pow(2)
    dice = 1 - (2 * (prob * target).sum() + 1) / (prob.sum() + target.sum() + 1)
    return bce.mean() + dice


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--encoder", default="resnet34")
    parser.add_argument("--res", type=int, default=1024)
    parser.add_argument("--crop", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--samples-per-image", type=int, default=2)
    parser.add_argument("--target", choices=["soft", "majority", "random"], default="soft")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--loss", choices=["bce_dice", "focal_dice"], default="bce_dice")
    parser.add_argument("--spine", action="store_true", help="also predict the filament spine (weight 0.5)")
    parser.add_argument("--clean", action="store_true", help="train on cleaned input (src/preprocess.py)")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "config.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2))

    names = sorted(data.load_annotations()["image_name"].unique())
    folds = data.assign_folds(names)
    train_names = [n for n in names if folds[n] != args.fold]
    loader = DataLoader(CropDataset(train_names, args.res, args.crop, args.samples_per_image, args.target,
                                    spine=args.spine, clean=args.clean),
                        batch_size=args.batch, shuffle=True, num_workers=4, drop_last=True,
                        persistent_workers=True)

    torch.manual_seed(args.seed)
    model = build_model(args.encoder, classes=2 if args.spine else 1).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=args.epochs * len(loader), pct_start=0.05)
    scaler = torch.amp.GradScaler()
    for epoch in range(args.epochs):
        model.train()
        start, total = time.time(), 0.0
        for image, target in loader:
            image, target = image.cuda(non_blocking=True), target.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(image)
            loss = loss_fn(logits[:, :1].float(), target[:, :1], args.loss)
            if args.spine:
                loss = loss + 0.5 * loss_fn(logits[:, 1:].float(), target[:, 1:], args.loss)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total += loss.item()
        print(f"epoch {epoch + 1}/{args.epochs} loss {total / len(loader):.4f} "
              f"{time.time() - start:.0f}s max_mem {torch.cuda.max_memory_allocated() / 2**30:.2f}GiB",
              flush=True)
        torch.save(model.state_dict(), args.out / "model.pt")


if __name__ == "__main__":
    main()
