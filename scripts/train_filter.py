"""Train the final candidate filter on all out-of-fold candidates and save it.

Threshold 0.35 is the median of the thresholds chosen by the inner cross-validation in
``scripts/filter_cv.py`` (0.40, 0.40, 0.35, 0.35, 0.30)::

    python -m scripts.train_filter --out data/models/filter_v1
"""

import argparse
from pathlib import Path

import pandas as pd

from scripts.filter_cv import fit
from src import candidate_filter, data

parser = argparse.ArgumentParser()
parser.add_argument("--out", type=Path, default=Path("data/models/filter_v1"))
parser.add_argument("--threshold", type=float, default=0.35)
args = parser.parse_args()
cand = pd.read_csv(data.DATA_DIR.parent / "oof" / "candidates.csv")
candidate_filter.save(fit(cand, candidate_filter.FEATURES), args.threshold, args.out)
print(f"trained on {len(cand)} candidates, threshold {args.threshold} -> {args.out}")
