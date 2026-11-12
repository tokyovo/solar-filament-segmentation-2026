"""Downscale every training image and its consensus mask once, for fast training.

Writes ``data/cache/r<res>/<image_name>.npz`` with:

- ``image``: (res, res) uint8 grayscale image.
- ``target``: (res, res) uint8, 255 x the fraction of annotators that marked the pixel.
- ``entries``: (n_annotators, res, res) uint8 in {0, 1}, one mask per annotator.
- ``spine``: (res, res) uint8, 255 x the fraction of annotators whose filament spine (drawn
  7 pixels wide at 2048) passes through the pixel.

With ``--npy`` it writes ``<image_name>.image.npy`` and ``<image_name>.target.npy`` instead,
without ``entries``. Training opens these files memory-mapped, so large resolutions do not
need the whole training set in RAM.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_util
from tqdm import tqdm

from src import data
from src.preprocess import clean_image

parser = argparse.ArgumentParser()
parser.add_argument("--res", type=int, default=1024)
parser.add_argument("--npy", action="store_true")
parser.add_argument("--clean", action="store_true", help="clean the input first (src/preprocess.py); folder r<res>c")
args = parser.parse_args()

out_dir = data.DATA_DIR.parent / "cache" / (f"r{args.res}" + ("c" if args.clean else ""))
out_dir.mkdir(parents=True, exist_ok=True)
annotations = data.load_annotations()
for image_name, group in tqdm(annotations.groupby("image_name")):
    image = data.load_image(data.TRAIN_IMAGES / f"{image_name}.jpeg")
    if args.clean:
        image = clean_image(image)
    small = lambda a: cv2.resize(a, (args.res, args.res), interpolation=cv2.INTER_AREA)
    votes = np.zeros((data.SIZE, data.SIZE), np.float32)
    spines = np.zeros((data.SIZE, data.SIZE), np.float32)
    entries = []
    for _, entry in group.groupby("entry_id"):
        union = np.zeros((data.SIZE, data.SIZE), np.uint8)
        for polygon in entry["polygon"]:
            union |= mask_util.decode(data.polygon_to_rle(polygon))
        votes += union
        lines = np.zeros((data.SIZE, data.SIZE), np.uint8)
        for spine in entry["spine"]:
            points = np.round(np.array(spine, np.float32).reshape(-1, 2)).astype(np.int32)
            cv2.polylines(lines, [points], isClosed=False, color=1, thickness=7)
        spines += lines
        entries.append((small(union.astype(np.float32)) >= 0.5).astype(np.uint8))
    votes /= group["entry_id"].nunique()
    spines /= group["entry_id"].nunique()
    if args.npy:
        np.save(out_dir / f"{image_name}.image.npy", small(image))
        np.save(out_dir / f"{image_name}.target.npy", np.round(small(votes) * 255).astype(np.uint8))
        continue
    np.savez(out_dir / f"{image_name}.npz", image=small(image),
             target=np.round(small(votes) * 255).astype(np.uint8), entries=np.stack(entries),
             spine=np.round(small(spines) * 255).astype(np.uint8))
