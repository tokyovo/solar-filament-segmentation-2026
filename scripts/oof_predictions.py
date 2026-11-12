"""Write out-of-fold predictions of the selected pipeline for every training image.

For each fold, the first stage is the mean map of that fold's 1024 and 2048 models (saved by
``src.predict probs``), and the refiner is that fold's ``refiner_v2``. No model ever sees the
labels of the images it predicts. Output (in ``data/oof/``, git-ignored):

- ``gt.csv``: the ground truth in the official format (``<entry_id>_<k>``);
- ``first_stage.csv``: candidates of the 1024 + 2048 mean map, before the refiner;
- ``refined.csv``: the same candidates after the refiner.

The report figures (``scripts/report_figures.py``) read these files::

    python -m scripts.oof_predictions
"""

import argparse
from pathlib import Path

import pandas as pd

from src import data, postprocess, refiner
from src.predict import instances_frame

OUT = data.DATA_DIR.parent / "oof"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--threshold", type=float, default=refiner.PP["threshold"])
    parser.add_argument("--min-area", type=int, default=refiner.PP["min_area"])
    parser.add_argument("--tag", default="", help="suffix of the output files, e.g. _lo")
    args = parser.parse_args()
    annotations = data.load_annotations()
    folds = data.assign_folds(annotations["image_name"].tolist())
    OUT.mkdir(parents=True, exist_ok=True)
    data.ground_truth_frame(annotations).to_csv(OUT / "gt.csv", index=False)
    first, refined = [], []
    for fold in args.folds:
        models = refiner.load_refiners([refiner.MODELS / f"refiner_v2_f{fold}"])
        names = sorted(n for n in annotations["image_name"].unique() if folds[n] == fold)
        for name in names:
            image = data.load_image(data.TRAIN_IMAGES / f"{name}.jpeg")
            prob = refiner.first_stage_map(name, fold)
            seeds = postprocess.probability_to_instances(prob, args.threshold, args.min_area,
                                                         refiner.PP["merge_dist"])
            first.append(instances_frame(name, seeds))
            refined.append(instances_frame(name, refiner.refine(models, image, prob, seeds, min_area=args.min_area)))
        print(f"fold {fold}: {len(names)} images", flush=True)
    pd.concat(first, ignore_index=True).to_csv(OUT / f"first_stage{args.tag}.csv", index=False)
    pd.concat(refined, ignore_index=True).to_csv(OUT / f"refined{args.tag}.csv", index=False)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
