"""Features and labels of every out-of-fold refined candidate, for the learned candidate filter.

Reads ``data/oof/refined.csv`` (``scripts/oof_predictions.py``) and the out-of-fold first-stage
maps. Writes ``data/oof/candidates.csv`` with one row per candidate:

- ``filament_id``, ``image_name``, ``fold``;
- features from the mask, the first-stage probability and the image (see ``features``);
- ``match_frac``: the fraction of the image's annotators that have a GT filament matching the
  candidate at IoU > 0.5, which is what PQ counts.

Only the test image and the model outputs are used, so the same features exist at inference::

    python -m scripts.candidate_features
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from src import data, metric, refiner
from src.candidate_filter import disk_geometry, features

OOF = data.DATA_DIR.parent / "oof"


def image_rows(args: tuple) -> list[dict]:
    name, fold, preds, gts = args
    image = data.load_image(data.TRAIN_IMAGES / f"{name}.jpeg")
    prob = refiner.first_stage_map(name, fold)
    disk = disk_geometry(image)
    masks = [data.rle_string_to_mask(c) for c in preds["segmentation_rle"]]
    union = np.zeros(image.shape, np.uint8)
    for m in masks:
        union |= m
    matched = np.zeros(len(masks))
    for gt_counts in gts:
        iou = metric.iou_matrix(gt_counts, list(preds["segmentation_rle"]))
        matched += (iou > metric.IOU_THRESHOLD).any(axis=0)
    rows = []
    for k, (fid, m) in enumerate(zip(preds["filament_id"], masks)):
        row = {"filament_id": fid, "image_name": name, "fold": fold, "n_candidates": len(masks),
               **features(m, prob, image, disk, union - m), "match_frac": matched[k] / len(gts)}
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="", help="suffix of the input and output files, e.g. _lo")
    tag = parser.parse_args().tag
    refined = pd.read_csv(OOF / f"refined{tag}.csv")
    gt = pd.read_csv(OOF / "gt.csv")
    gt["entry"] = gt["filament_id"].str.split("_", n=1).str[0]
    gt["image"] = gt["entry"].str.split("-", n=1).str[1]
    refined["image"] = refined["filament_id"].str.rsplit("_", n=1).str[0]
    folds = data.assign_folds(gt["image"].tolist())
    gts = {img: [list(g["segmentation_rle"]) for _, g in grp.groupby("entry")] for img, grp in gt.groupby("image")}
    jobs = [(name, folds[name], grp, gts[name]) for name, grp in refined.groupby("image")]
    with ProcessPoolExecutor(4) as pool:
        rows = [r for rs in pool.map(image_rows, jobs, chunksize=4) for r in rs]
    table = pd.DataFrame(rows)
    table.to_csv(OOF / f"candidates{tag}.csv", index=False)
    print(f"{len(table)} candidates from {table['image_name'].nunique()} images; "
          f"match_frac mean {table['match_frac'].mean():.3f}, share 0: {(table['match_frac'] == 0).mean():.3f}")


if __name__ == "__main__":
    main()
