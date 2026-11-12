"""Uncertainty of the out-of-fold scores for the report.

1. Bootstrap over images: the out-of-fold predictions (``data/oof/``) are split into TP, FP,
   FN and the sum of TP IoU per image (official counting rules). Images are resampled with
   replacement 2,000 times; the pooled PQ of each sample gives a 95% interval for the first
   stage, the refined pipeline, the final pipeline (refined + filter) and their paired gains.
2. Spread over folds: the per-fold PQ of the four pipelines (``research/results.jsonl``) as
   mean +/- standard deviation, with one point per fold.

Output: ``report/tables/bootstrap.csv`` and ``report/figures/folds_errorbars.png``::

    python -m scripts.report_uncertainty
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import data, metric  # noqa: E402

OOF = data.DATA_DIR.parent / "oof"
RNG = np.random.default_rng(0)
N_BOOT = 2000


def per_image_counts(gt_df: pd.DataFrame, pred_df: pd.DataFrame) -> pd.DataFrame:
    """TP, FP, FN and the sum of TP IoU per image, pooled over its annotator entries."""
    gt_entry = gt_df["filament_id"].str.split("_", n=1).str[0]
    pred_image = pred_df["filament_id"].str.rsplit("_", n=1).str[0]
    pred_by_image = pred_df.groupby(pred_image)["segmentation_rle"].apply(list).to_dict()
    rows = []
    for entry_id, gt_counts in gt_df.groupby(gt_entry, sort=False)["segmentation_rle"]:
        image = entry_id.split("-", 1)[1]
        iou = metric.iou_matrix(list(gt_counts), pred_by_image.get(image, []))
        hit = iou > metric.IOU_THRESHOLD
        rows.append({"image": image, "tp": int(hit.sum()), "sum_iou": float(iou[hit].sum()),
                     "fp": int((hit.sum(axis=0) == 0).sum()), "fn": int((hit.sum(axis=1) == 0).sum())})
    return pd.DataFrame(rows).groupby("image").sum()


def pq(c: np.ndarray) -> np.ndarray:
    """Pooled PQ of count arrays with columns (tp, sum_iou, fp, fn); works on a batch."""
    tp, s, fp, fn = c[..., 0], c[..., 1], c[..., 2], c[..., 3]
    return s / (tp + 0.5 * fp + 0.5 * fn)


def bootstrap_table() -> pd.DataFrame:
    gt_df = pd.read_csv(OOF / "gt.csv")
    counts = {name: per_image_counts(gt_df, pd.read_csv(OOF / f"{file}.csv"))[["tp", "sum_iou", "fp", "fn"]]
              for name, file in [("first stage", "first_stage"), ("refined", "refined"), ("final", "final")]}
    images = counts["first stage"].index.union(counts["refined"].index)
    arrays = {k: v.reindex(images, fill_value=0).to_numpy(float) for k, v in counts.items()}
    idx = RNG.integers(len(images), size=(N_BOOT, len(images)))
    samples = {k: pq(a[idx].sum(axis=1)) for k, a in arrays.items()}
    point = {k: float(pq(a.sum(axis=0))) for k, a in arrays.items()}
    for a, b in [("refined", "first stage"), ("final", "refined"), ("final", "first stage")]:
        samples[f"{a} - {b}"] = samples[a] - samples[b]
        point[f"{a} - {b}"] = point[a] - point[b]
    rows = [{"quantity": k, "pq": round(point[k], 4), "ci95_low": round(float(np.percentile(v, 2.5)), 4),
             "ci95_high": round(float(np.percentile(v, 97.5)), 4), "share_above_0": round(float((v > 0).mean()), 4)}
            for k, v in samples.items()]
    return pd.DataFrame(rows)


def folds_figure() -> pd.DataFrame:
    rows = [json.loads(line) for line in Path("research/results.jsonl").read_text().splitlines() if line.strip()]
    oof = next(r for r in rows if r["id"] == "oof_5fold_summary")
    refined = next(r for r in rows if r["id"] == "refiner_v2_5fold")
    filtered = next(r for r in rows if r["id"] == "learned_filter_nested")
    folds = {"1024": oof["A"], "2048": oof["R2048"], "1024 + 2048": oof["pair"], "+ refiner": refined["refined"],
             "+ filter": filtered["per_fold"]["filtered"]}
    table = pd.DataFrame({k: v for k, v in folds.items()})
    plt.figure(figsize=(4.6, 2.8))
    x = np.arange(len(folds))
    means, sds = table.mean().to_numpy(), table.std(ddof=1).to_numpy()
    # Points, not bars: the axis does not start at zero, so bar lengths would mislead.
    for k, column in enumerate(table):
        plt.scatter(np.full(5, k) + np.linspace(-0.18, 0.18, 5), table[column], s=10, color="#8c8c8c", zorder=2)
    plt.errorbar(x, means, yerr=sds, fmt="o", color="#2b5d9a", markersize=6, capsize=5, linewidth=1.5, zorder=3)
    plt.xticks(x, list(folds), fontsize=8)
    plt.ylim(0.39, 0.465)
    plt.ylabel("out-of-fold PQ")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("report/figures/folds_errorbars.png", dpi=250)
    plt.close()
    return pd.DataFrame({"pipeline": list(folds), "mean": means.round(4), "sd": sds.round(4)})


def main() -> None:
    boot = bootstrap_table()
    boot.to_csv("report/tables/bootstrap.csv", index=False)
    print(boot.to_string(index=False))
    print(folds_figure().to_string(index=False))


if __name__ == "__main__":
    main()
