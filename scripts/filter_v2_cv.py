"""Test D: filter v2 with six more candidate features, against filter v1 (NOTES.md section 24).

Extra features, from the out-of-fold maps of the 1024 and 2048 models and the image:
``p1024_mean``, ``p2048_mean``, ``p_absdiff`` (mean |1024 - 2048| inside), ``edge`` (Sobel
magnitude on the border / disk median), ``ring_std`` (intensity spread in the ring / disk
median), ``i_q10`` (10th percentile of intensity inside / disk median)::

    python -m scripts.filter_v2_cv
"""

from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
import pandas as pd

from scripts.filter_cv import OOF, load_candidates, nested, pq
from src import candidate_filter, data
from src.refiner import MODELS

EXTRA = ["p1024_mean", "p2048_mean", "p_absdiff", "edge", "ring_std", "i_q10"]


def extra_features(mask, p1024, p2048, image, gradient, disk_median, others) -> dict:
    inside = mask > 0
    border = inside & (cv2.erode(mask, np.ones((3, 3), np.uint8)) == 0)
    ring = (cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))) > 0) & ~inside & (others == 0)
    ring_i = image[ring].astype(np.float32)
    return {"p1024_mean": p1024[inside].mean(), "p2048_mean": p2048[inside].mean(),
            "p_absdiff": np.abs(p1024[inside] - p2048[inside]).mean(),
            "edge": gradient[border].mean() / disk_median,
            "ring_std": (ring_i.std() if ring_i.size else 0.0) / disk_median,
            "i_q10": np.quantile(image[inside], 0.1) / disk_median}


def image_rows(args: tuple) -> list[dict]:
    name, fold, ids, counts = args
    image = data.load_image(data.TRAIN_IMAGES / f"{name}.jpeg")
    read = lambda m: cv2.imread(str(MODELS / m / f"probs_fold{fold}" / f"{name}.png"), cv2.IMREAD_GRAYSCALE) / 255
    p1024, p2048 = read(f"unet_r34_f{fold}"), read(f"unet_r34_r2048_f{fold}")
    blurred = cv2.GaussianBlur(image.astype(np.float32), (0, 0), 1)
    gradient = np.hypot(cv2.Sobel(blurred, cv2.CV_32F, 1, 0), cv2.Sobel(blurred, cv2.CV_32F, 0, 1))
    disk_median = candidate_filter.disk_geometry(image)[3]
    masks = [data.rle_string_to_mask(c) for c in counts]
    union = np.zeros(image.shape, np.uint8)
    for m in masks:
        union |= m
    return [{"filament_id": i, **extra_features(m, p1024, p2048, image, gradient, disk_median, union - m)}
            for i, m in zip(ids, masks)]


def main() -> None:
    cand, gt = load_candidates()
    refined = pd.read_csv(OOF / "refined.csv")
    refined["image"] = refined["filament_id"].str.rsplit("_", n=1).str[0]
    folds = data.assign_folds(gt["image"].tolist())
    jobs = [(img, folds[img], list(g["filament_id"]), list(g["segmentation_rle"])) for img, g in refined.groupby("image")]
    with ProcessPoolExecutor(4) as pool:
        extra = pd.DataFrame([r for rs in pool.map(image_rows, jobs, chunksize=4) for r in rs])
    extra.to_csv(OOF / "candidates_v2_extra.csv", index=False)
    cand = cand.merge(extra, on="filament_id")

    results = {}
    for name, features in [("v1", candidate_filter.FEATURES), ("v2", candidate_filter.FEATURES + EXTRA)]:
        table, _, keep = nested(cand, gt, features)
        results[name] = (table, pq(cand, keep, len(gt)))
    compare = results["v1"][0][["fold", "refined"]].assign(
        v1=results["v1"][0]["filtered"], v2=results["v2"][0]["filtered"],
        v2_threshold=results["v2"][0]["threshold"])
    compare["v2_minus_v1"] = compare["v2"] - compare["v1"]
    print(compare.round(4).to_string(index=False))
    print(f"pooled: v1 {results['v1'][1]:.4f}, v2 {results['v2'][1]:.4f}, "
          f"gain {results['v2'][1] - results['v1'][1]:+.4f}; folds better: {(compare['v2_minus_v1'] > 0).sum()} of 5")


if __name__ == "__main__":
    main()
