"""Score saved validation maps of several models with fixed post-processing settings.

Each line is ``<name>=<probs dir>``. Every setting is applied to every model::

    python -m scripts.compare_models A=data/models/unet_r34_f0/probs_fold0 \\
        --settings '{"threshold": 0.5, "min_area": 300, "merge_dist": 15}'
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from src import data
from scripts.sweep_postprocess import score

parser = argparse.ArgumentParser()
parser.add_argument("models", nargs="+")
parser.add_argument("--settings", type=json.loads, nargs="+", required=True)
args = parser.parse_args()

annotations = data.load_annotations()
rows = []
for item in args.models:
    name, prob_dir = item.split("=", 1)
    prob_dir = Path(prob_dir)
    names = {p.stem for p in prob_dir.glob("*.png")}
    gt_df = data.ground_truth_frame(annotations[annotations["image_name"].isin(names)])
    for setting in args.settings:
        rows.append({"model": name, **score(({"seed_threshold": None, **setting}, [prob_dir], gt_df))})
print(pd.DataFrame(rows).drop(columns="seed_threshold").to_string(index=False))
