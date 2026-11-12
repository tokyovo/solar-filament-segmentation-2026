"""Report figures and tables from the out-of-fold predictions (``scripts/oof_predictions.py``).

The distribution and matching plots use the organisers' own plotting code
(``scripts/official_plots.py``, copied verbatim), so the figures show what the jury measures.
Output in ``report/``::

    python -m scripts.report_figures
"""

from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from scripts import official_plots  # noqa: E402
from src import data, metric  # noqa: E402
from tests import official_selfeval  # noqa: E402

OOF = data.DATA_DIR.parent / "oof"
FIGURES = Path("report/figures")
TABLES = Path("report/tables")
COLORS = [(0, 200, 255), (255, 200, 0), (200, 0, 255)]  # annotators (RGB)
PRED_COLOR = (255, 40, 40)


def overlap_frame(gt_df: pd.DataFrame, pred_df: pd.DataFrame) -> pd.DataFrame:
    """The ``overlap_df`` of the official notebook, from RLE IoU (Dice = 2 IoU / (1 + IoU))."""
    gt_entry = gt_df["filament_id"].str.split("_", n=1).str[0]
    pred_image = pred_df["filament_id"].str.rsplit("_", n=1).str[0]
    pred_by_image = pred_df.groupby(pred_image)["segmentation_rle"].apply(list).to_dict()
    rows = []
    for entry_id, gt_counts in gt_df.groupby(gt_entry, sort=False)["segmentation_rle"]:
        pred_counts = pred_by_image.get(entry_id.split("-", 1)[1], [])
        iou = torch.from_numpy(metric.iou_matrix(list(gt_counts), pred_counts)).float()
        rows.append({"annotator_image": entry_id, "iou_matrix": iou, "dice_matrix": 2 * iou / (1 + iou),
                     "n_gt": len(gt_counts), "n_pred": len(pred_counts)})
    return pd.DataFrame(rows)


def summary(name: str, overlap: pd.DataFrame, gt_df: pd.DataFrame, pred_df: pd.DataFrame) -> dict:
    r = metric.panoptic_quality(gt_df, pred_df)
    official = official_selfeval.get_pq_score(overlap)
    assert abs(official - r.pq) < 1e-6, (official, r.pq)
    one_to_many = sum(int((m > 0).sum(dim=1).gt(1).sum()) for m in overlap["iou_matrix"] if m.numel())
    many_to_one = sum(int((m > 0).sum(dim=0).gt(1).sum()) for m in overlap["iou_matrix"] if m.numel())
    ious = np.concatenate([m[m > 0].numpy() for m in overlap["iou_matrix"] if m.numel()])
    return {"pipeline": name, "pq": round(r.pq, 4), "sq": round(r.sq, 4), "rq": round(r.rq, 4), "tp": r.tp,
            "fp": r.fp, "fn": r.fn, "gt_with_2plus_preds": one_to_many, "pred_with_2plus_gt": many_to_one,
            "mean_iou_overlapping_pairs": round(float(ious.mean()), 4),
            "mean_dice_overlapping_pairs": round(float((2 * ious / (1 + ious)).mean()), 4)}


def official_panels(overlaps: dict[str, pd.DataFrame]) -> None:
    for column, label, stem in [("iou_matrix", "IoU", "iou"), ("dice_matrix", "Dice", "dice")]:
        plt.figure(figsize=(5.0 * len(overlaps), 3.6))
        for k, (name, overlap) in enumerate(overlaps.items(), start=1):
            plt.subplot(1, len(overlaps), k)
            official_plots.plot_distribution(overlap[column], f"{label}: {name}", label)
        plt.tight_layout()
        plt.savefig(FIGURES / f"{stem}_distribution.png", dpi=200)
        plt.close()
    plt.figure(figsize=(6.0 * len(overlaps), 4.0))
    for k, (name, overlap) in enumerate(overlaps.items(), start=1):
        plt.subplot(1, len(overlaps), k)
        official_plots.plot_m2n_counts(overlap, f"Matching: {name}")
    plt.tight_layout()
    plt.savefig(FIGURES / "matching_counts.png", dpi=200)
    plt.close()


def per_image_pq(overlap: pd.DataFrame) -> pd.Series:
    """PQ of each image, pooled over its annotators, with the official counting rules."""
    overlap = overlap.assign(image=overlap["annotator_image"].str.split("-", n=1).str[1])
    return pd.Series({image: official_selfeval.get_pq_score(group) for image, group in overlap.groupby("image")})


def overlay(image_name: str, gt_df: pd.DataFrame, pred_df: pd.DataFrame, size: int = 900,
            margin: int = 120) -> np.ndarray:
    """Image with each annotator's outlines (cyan, yellow, magenta) and our outlines (red).

    The result is the square crop around all drawn masks (plus ``margin`` pixels), so small
    filaments stay visible in a printed figure.
    """
    image = data.load_image(data.TRAIN_IMAGES / f"{image_name}.jpeg")
    canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    covered = np.zeros(image.shape, np.uint8)
    masks = []
    entries = gt_df[gt_df["filament_id"].str.split("_", n=1).str[0].str.endswith("-" + image_name)]
    for color, (_, group) in zip(COLORS, entries.groupby(entries["filament_id"].str.split("_", n=1).str[0])):
        for counts in group["segmentation_rle"]:
            masks.append((data.rle_string_to_mask(counts), color))
    for counts in pred_df.loc[pred_df["filament_id"].str.rsplit("_", n=1).str[0] == image_name, "segmentation_rle"]:
        masks.append((data.rle_string_to_mask(counts), PRED_COLOR))
    for mask, _ in masks:
        covered |= mask
    ys, xs = np.nonzero(covered)
    side = min(data.SIZE, max(ys.max() - ys.min(), xs.max() - xs.min()) + 2 * margin)
    thickness = max(1, round(1.6 * side / size))  # about 1.6 px wide after resizing
    for mask, color in masks:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(canvas, contours, -1, color, thickness)
    y0 = int(np.clip((ys.min() + ys.max() - side) // 2, 0, data.SIZE - side))
    x0 = int(np.clip((xs.min() + xs.max() - side) // 2, 0, data.SIZE - side))
    return cv2.resize(canvas[y0:y0 + side, x0:x0 + side], (size, size), interpolation=cv2.INTER_AREA)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    gt_df = pd.read_csv(OOF / "gt.csv")
    # "final" = refined candidates kept by the nested filter (scripts/filter_cv.py, final.csv).
    preds = {"first stage": pd.read_csv(OOF / "first_stage.csv"), "refined": pd.read_csv(OOF / "refined.csv"),
             "final": pd.read_csv(OOF / "final.csv")}
    overlaps = {name: overlap_frame(gt_df, p) for name, p in preds.items()}
    table = pd.DataFrame([summary(name, overlaps[name], gt_df, p) for name, p in preds.items()])
    table.to_csv(TABLES / "oof_summary.csv", index=False)
    print(table.to_string(index=False))
    official_panels({name: overlaps[name] for name in ("first stage", "final")})

    scores = per_image_pq(overlaps["final"])
    entry = gt_df["filament_id"].str.split("_", n=1).str[0]
    per_entry = entry.value_counts().groupby(lambda e: e.split("-", 1)[1]).mean()
    scores = scores[scores.index.isin(per_entry[per_entry >= 3].index)].sort_values()
    picks = {"low PQ": scores.index[0], "median PQ": scores.index[len(scores) // 2], "high PQ": scores.index[-1]}
    plt.figure(figsize=(15, 5.4))
    for k, (label, name) in enumerate(picks.items(), start=1):
        plt.subplot(1, 3, k)
        plt.imshow(overlay(name, gt_df, preds["final"]))
        plt.title(f"{label}: {name} (PQ {scores[name]:.2f})", fontsize=10)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(FIGURES / "examples.png", dpi=150)
    plt.close()
    scores.to_csv(TABLES / "per_image_pq_final.csv", header=["pq"])
    print("examples:", {k: (v, round(float(scores[v]), 3)) for k, v in picks.items()})


if __name__ == "__main__":
    main()
