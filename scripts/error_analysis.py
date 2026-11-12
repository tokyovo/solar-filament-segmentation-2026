"""Split the FP and FN of one prediction set into merges, fragments and plain errors.

For each annotator entry (IoU > 0.5 is a match, as in the metric):

- merge: an unmatched prediction that overlaps 2 or more GT filaments;
- fragment: an unmatched GT filament that overlaps 2 or more predictions;
- plain FP / FN: everything else that is unmatched.

This bounds what better instance separation (for example a detector) could gain::

    python -m scripts.error_analysis data/models/ens_A_s0_R2048/probs_fold0
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src import data, metric, postprocess
from src.predict import instances_frame

parser = argparse.ArgumentParser()
parser.add_argument("probs", type=Path)
parser.add_argument("--settings", default='{"threshold": 0.5, "min_area": 300, "merge_dist": 15}')
args = parser.parse_args()
settings = json.loads(args.settings)

annotations = data.load_annotations()
names = sorted(p.stem for p in args.probs.glob("*.png"))
gt_df = data.ground_truth_frame(annotations[annotations["image_name"].isin(names)])
gt_entry = gt_df["filament_id"].str.split("_", n=1).str[0]
pred = {n: instances_frame(n, postprocess.probability_to_instances(
            cv2.imread(str(args.probs / f"{n}.png"), cv2.IMREAD_GRAYSCALE) / 255, **settings))
        for n in names}

c = Counter()
for entry_id, gt_counts in gt_df.groupby(gt_entry, sort=False)["segmentation_rle"]:
    pred_counts = list(pred[entry_id.split("-", 1)[1]]["segmentation_rle"])
    iou = metric.iou_matrix(list(gt_counts), pred_counts)
    hit, touch = iou > metric.IOU_THRESHOLD, iou > 0
    c["tp"] += int(hit.sum())
    for j in np.flatnonzero(hit.sum(axis=0) == 0):
        c["fp_merge" if touch[:, j].sum() >= 2 else "fp_touch1" if touch[:, j].sum() == 1 else "fp_plain"] += 1
    for i in np.flatnonzero(hit.sum(axis=1) == 0):
        c["fn_fragment" if touch[i].sum() >= 2 else "fn_touch1" if touch[i].sum() == 1 else "fn_plain"] += 1
for k in ["tp", "fp_merge", "fp_touch1", "fp_plain", "fn_fragment", "fn_touch1", "fn_plain"]:
    print(f"{k:12s} {c[k]}")
print("fp_touch1 = overlaps one GT but IoU <= 0.5 (shape or extent error)")
print("fn_touch1 = overlapped by one prediction but IoU <= 0.5")
