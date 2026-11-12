"""Score post-processing settings on saved validation maps of one or more folds.

Every setting is applied to all given folds, and the PQ is pooled over them, so the chosen
setting does not fit one fold only. Example::

    python -m scripts.sweep_postprocess --probs data/models/unet_r34_f0/probs_fold0 \\
        data/models/unet_r34_f1/probs_fold1 --out research/sweeps/pp_v1.csv
"""

import argparse
import itertools
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import pandas as pd

from src import data, metric, postprocess
from src.predict import instances_frame

GRID = {
    "threshold": [0.3, 0.4, 0.5],
    "min_area": [300, 600, 1000],
    "merge_dist": [0, 15, 31, 61],
    "seed_threshold": [None, 0.6, 0.8],
}


def predictions(prob_dir: Path, setting: dict) -> pd.DataFrame:
    frames = []
    for path in sorted(prob_dir.glob("*.png")):
        prob = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) / 255
        frames.append(instances_frame(path.stem, postprocess.probability_to_instances(prob, **setting)))
    return pd.concat(frames, ignore_index=True)


def score(args: tuple[dict, list[Path], pd.DataFrame]) -> dict:
    setting, prob_dirs, gt_df = args
    pred_df = pd.concat([predictions(d, setting) for d in prob_dirs], ignore_index=True)
    r = metric.panoptic_quality(gt_df, pred_df)
    return {**setting, "pq": r.pq, "sq": r.sq, "rq": r.rq, "tp": r.tp, "fp": r.fp, "fn": r.fn}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probs", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--grid", type=json.loads, help="replaces GRID, e.g. '{\"threshold\": [0.5]}'")
    args = parser.parse_args()

    names = {p.stem for d in args.probs for p in d.glob("*.png")}
    annotations = data.load_annotations()
    gt_df = data.ground_truth_frame(annotations[annotations["image_name"].isin(names)])
    grid = args.grid or GRID
    settings = [dict(zip(grid, values)) for values in itertools.product(*grid.values())]
    settings = [s for s in settings
                if s.get("seed_threshold") is None or s["seed_threshold"] > s["threshold"]]
    with ProcessPoolExecutor(args.workers) as pool:
        rows = list(pool.map(score, [(s, args.probs, gt_df) for s in settings]))
    table = pd.DataFrame(rows).sort_values("pq", ascending=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    print(f"images={len(names)} settings={len(settings)}")
    print(table.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
