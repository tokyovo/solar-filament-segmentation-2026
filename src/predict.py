"""Run a trained model on full images and score or write the predictions.

Validation, with a sweep of the post-processing values::

    python -m src.predict validate --model data/models/unet_r34_f0 --fold 0

Validation maps saved as PNG, for post-processing sweeps (scripts/sweep_postprocess.py)::

    python -m src.predict probs --model data/models/unet_r34_f0 --fold 0

Test submission. With several ``--model`` folders, their probability maps are averaged::

    python -m src.predict submit --model data/models/unet_r34_f0 data/models/unet_r34_r2048_f0 \
        --out submission/x.csv
"""

import argparse
import itertools
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from src import data, metric, postprocess
from src.preprocess import clean_image
from src.train import build_model

# Inference runs on the GPU when there is one; otherwise on the CPU in float32.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(model_dir: Path) -> tuple[torch.nn.Module, int]:
    config = json.loads((model_dir / "config.json").read_text())
    model = build_model(config["encoder"], weights=None, classes=2 if config.get("spine") == "True" else 1)
    model.load_state_dict(torch.load(model_dir / "model.pt", map_location="cpu"))
    model.clean = config.get("clean") == "True"  # trained on cleaned input (src/preprocess.py)
    return model.to(DEVICE).eval(), int(config["res"])


@torch.no_grad()
def predict_probability(model: torch.nn.Module, image: np.ndarray, res: int,
                        tta: str = "hflip") -> np.ndarray:
    """Full-size (2048, 2048) filament probability, with logits averaged over TTA views.

    ``tta`` is ``hflip`` (identity and horizontal flip) or ``d4`` (the 8 flips and 90-degree
    rotations of the square image).
    """
    if getattr(model, "clean", False):
        image = clean_image(image)
    small = cv2.resize(image, (res, res), interpolation=cv2.INTER_AREA)
    x = torch.from_numpy(small.astype(np.float32) / 255.0)[None, None].to(DEVICE)
    views = [(k, flip) for k in range(4) for flip in (False, True)] if tta == "d4" else [(0, False), (0, True)]
    logits = 0
    with torch.autocast(DEVICE, dtype=torch.float16, enabled=DEVICE == "cuda"):
        for k, flip in views:
            v = torch.rot90(x.flip(-1) if flip else x, k, (-2, -1))
            out = torch.rot90(model(v)[:, :1], -k, (-2, -1))  # channel 0: filament
            logits = logits + (out.flip(-1) if flip else out)
    prob = torch.sigmoid(logits.float() / len(views))[0, 0].cpu().numpy()
    return cv2.resize(prob, (data.SIZE, data.SIZE), interpolation=cv2.INTER_LINEAR)


def instances_frame(image_name: str, masks: list[np.ndarray]) -> pd.DataFrame:
    """Submission rows for one image: ``<image_name>_<k>`` and the RLE string."""
    return pd.DataFrame({
        "filament_id": [f"{image_name}_{k}" for k in range(1, len(masks) + 1)],
        "segmentation_rle": [data.mask_to_rle_string(m) for m in masks],
    })


def validate(model_dir: Path, fold: int) -> None:
    annotations = data.load_annotations()
    folds = data.assign_folds(annotations["image_name"].tolist())
    val = annotations[annotations["image_name"].map(folds) == fold]
    gt_df = data.ground_truth_frame(val)
    model, res = load_model(model_dir)
    # Kept as uint8 (0 to 255) so the fold's maps fit in about 0.6 GB of RAM.
    probs = {n: np.round(predict_probability(model, data.load_image(data.TRAIN_IMAGES / f"{n}.jpeg"), res)
                         * 255).astype(np.uint8)
             for n in sorted(val["image_name"].unique())}

    rows = []
    for threshold, min_area in itertools.product([0.3, 0.4, 0.5, 0.6], [0, 100, 300, 600, 1000]):
        pred_df = pd.concat([instances_frame(n, postprocess.probability_to_instances(p / 255, threshold, min_area))
                             for n, p in probs.items()], ignore_index=True)
        r = metric.panoptic_quality(gt_df, pred_df)
        rows.append({"threshold": threshold, "min_area": min_area, "pq": r.pq, "sq": r.sq,
                     "rq": r.rq, "tp": r.tp, "fp": r.fp, "fn": r.fn})
        print(rows[-1], flush=True)
    table = pd.DataFrame(rows).sort_values("pq", ascending=False)
    table.to_csv(model_dir / f"sweep_fold{fold}.csv", index=False)
    print(f"images={len(probs)} entries={gt_df['filament_id'].str.split('_', n=1).str[0].nunique()}")
    print(table.head(5).to_string(index=False))


def save_probabilities(model_dir: Path, fold: int, tta: str = "hflip") -> None:
    """Write the fold's validation maps as uint8 PNGs to ``<model_dir>/probs_fold<k>/``.

    With ``tta="d4"`` the folder is ``probs_fold<k>_d4``, so the default maps stay.
    """
    annotations = data.load_annotations()
    folds = data.assign_folds(annotations["image_name"].tolist())
    names = sorted(n for n in annotations["image_name"].unique() if folds[n] == fold)
    out = model_dir / (f"probs_fold{fold}" + ("" if tta == "hflip" else f"_{tta}"))
    out.mkdir(exist_ok=True)
    model, res = load_model(model_dir)
    for n in names:
        prob = predict_probability(model, data.load_image(data.TRAIN_IMAGES / f"{n}.jpeg"), res, tta)
        cv2.imwrite(str(out / f"{n}.png"), np.round(prob * 255).astype(np.uint8))
    print(f"wrote {len(names)} maps to {out}")


def predict_frame(image_paths: list[Path], model_dirs: list[Path], refiner_dirs: list[Path] | None = None,
                  threshold: float = 0.5, min_area: int = 300, merge_dist: int = 15,
                  tta: str = "hflip", filter_dir: Path | None = None,
                  filter_log: Path | None = None) -> pd.DataFrame:
    """Submission rows for the given images.

    The probability maps of ``model_dirs`` are averaged; the map is split into candidate
    filaments by ``postprocess``; with ``refiner_dirs``, each candidate is re-segmented by the
    mean of those seed refiners (``src/refiner.py``); with ``filter_dir``, the learned
    candidate filter (``src/candidate_filter.py``) removes candidates that annotators are
    unlikely to mark. The filter scores each candidate with the mean map of each fold's models
    and averages the scores; ``filter_log`` receives one row per candidate (score, features).
    """
    models = [load_model(d) for d in model_dirs]
    folds = [json.loads((d / "config.json").read_text())["fold"] for d in model_dirs]
    if refiner_dirs:
        from src import refiner  # refiner imports this module, so import it here
        refiners = refiner.load_refiners(refiner_dirs)
    if filter_dir:
        from src import candidate_filter
        booster, keep_threshold = candidate_filter.load(filter_dir)
    frames, logs = [], []
    for path in image_paths:
        image = data.load_image(path)
        maps = [predict_probability(model, image, res, tta) for model, res in models]
        prob = np.mean(maps, axis=0)
        masks = postprocess.probability_to_instances(prob, threshold, min_area, merge_dist)
        if refiner_dirs:
            masks = refiner.refine(refiners, image, prob, masks, min_area=min_area)
        if filter_dir and masks:
            pair_maps = [np.mean([m for m, f in zip(maps, folds) if f == k], axis=0) for k in sorted(set(folds))]
            scores, table = candidate_filter.score_masks(masks, image, pair_maps, booster)
            logs.append(table.assign(image_name=path.stem, score=scores, kept=scores >= keep_threshold))
            masks = [m for m, s in zip(masks, scores) if s >= keep_threshold]
        frames.append(instances_frame(path.stem, masks))
    if filter_log and logs:
        pd.concat(logs, ignore_index=True).to_csv(filter_log, index=False)
    return pd.concat(frames, ignore_index=True)


def submit(model_dirs: list[Path], out: Path, threshold: float, min_area: int,
           merge_dist: int, tta: str = "hflip", refiner_dirs: list[Path] | None = None,
           filter_dir: Path | None = None) -> None:
    """Write the CSV for all test images (and, with a filter, ``<out>.filter_log.csv``)."""
    paths = sorted(data.TEST_IMAGES.glob("*.jpeg"))
    log = out.with_suffix(".filter_log.csv") if filter_dir else None
    predict_frame(paths, model_dirs, refiner_dirs, threshold, min_area, merge_dist, tta, filter_dir,
                  log).to_csv(out, index=False)
    print(f"wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["validate", "probs", "submit"])
    parser.add_argument("--model", type=Path, nargs="+", required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--threshold", type=float, default=0.5)  # defaults: pp_v1 sweep
    parser.add_argument("--min-area", type=int, default=300)
    parser.add_argument("--merge-dist", type=int, default=15)
    parser.add_argument("--tta", choices=["hflip", "d4"], default="hflip")
    parser.add_argument("--refiner", type=Path, nargs="+", help="seed refiner folders (submit only)")
    parser.add_argument("--filter", type=Path, help="candidate filter folder (submit only)")
    args = parser.parse_args()
    if args.mode != "submit" and len(args.model) > 1:
        parser.error(f"{args.mode} takes one --model")
    if args.mode == "validate":
        validate(args.model[0], args.fold)
    elif args.mode == "probs":
        save_probabilities(args.model[0], args.fold, args.tta)
    else:
        submit(args.model, args.out, args.threshold, args.min_area, args.merge_dist, args.tta,
               args.refiner, args.filter)


if __name__ == "__main__":
    main()
