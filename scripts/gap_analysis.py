"""Where the final pipeline loses PQ, out of fold (official pooled counting), for the report.

Reads ``data/oof/gt.csv`` and ``data/oof/final.csv`` (``scripts/oof_predictions.py`` and
``scripts/filter_cv.py``) and the out-of-fold first-stage maps. Writes:

- ``report/tables/gap.csv``: every error type with its count and the PQ if it were fixed;
- ``report/figures/disagreement.png``: three zoomed out-of-fold cases (annotators disagree, a
  filament that one annotator marks and the others do not, a near miss).

It also prints our PQ against the same targets as the annotator agreement
(``scripts/annotator_agreement.py``)::

    python -m scripts.gap_analysis
"""

from concurrent.futures import ProcessPoolExecutor

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import data, metric, refiner  # noqa: E402

OOF = data.DATA_DIR.parent / "oof"
COLORS = [(0, 190, 255), (255, 200, 0), (220, 0, 220)]  # annotators: cyan, yellow, magenta
PRED_COLOR = (255, 30, 30)  # our masks: red


def load() -> tuple[pd.DataFrame, dict, dict]:
    gt = pd.read_csv(OOF / "gt.csv")
    gt["entry"] = gt["filament_id"].str.split("_", n=1).str[0]
    gt["image"] = gt["entry"].str.split("-", n=1).str[1]
    pred = pd.read_csv(OOF / "final.csv")
    preds = pred.groupby(pred["filament_id"].str.rsplit("_", n=1).str[0])["segmentation_rle"].apply(list).to_dict()
    return gt, preds, data.assign_folds(gt["image"].tolist())


def image_rows(args: tuple) -> tuple[list[dict], list[dict], np.ndarray]:
    """Per GT filament and per prediction of one image: best IoU and the context of each error."""
    image, entries, preds, fold = args
    prob = refiner.first_stage_map(image, fold)
    pred_masks = [data.rle_string_to_mask(c) for c in preds]
    gt_rows, pred_rows, model_counts = [], [], np.zeros(4)
    for entry, counts in entries:
        iou = metric.iou_matrix(counts, preds) if preds else np.zeros((len(counts), 0))
        others = [c for e, c in entries if e != entry]
        if others:  # our masks scored against this entry, as the annotators are scored
            hit = iou > metric.IOU_THRESHOLD
            model_counts += [hit.sum(), iou[hit].sum(), (hit.sum(0) == 0).sum(), (hit.sum(1) == 0).sum()]
        for i, c in enumerate(counts):
            mask = data.rle_string_to_mask(c)
            best = int(iou[i].argmax()) if preds else -1
            inter = (mask & pred_masks[best]).sum() if preds else 0
            gt_rows.append({
                "image": image, "entry": entry, "k": i, "n_entries": len(entries), "area": int(mask.sum()),
                "best": float(iou[i].max()) if preds else 0.0, "n_touch": int((iou[i] > 0.05).sum()) if preds else 0,
                "recall": inter / mask.sum(), "precision": inter / max(pred_masks[best].sum(), 1) if preds else 0.0,
                "spans": int((iou[:, best] > 0.05).sum()) if preds else 0,
                "high_px": int(((prob >= 0.5) & (mask > 0)).sum()),
                "agree": max((metric.iou_matrix([c], o).max() for o in others if o), default=np.nan)})
        for j in range(len(preds)):
            pred_rows.append({"image": image, "entry": entry, "best": float(iou[:, j].max()) if len(counts) else 0.0})
    return gt_rows, pred_rows, model_counts


def pq(tp: float, s: float, fp: float, fn: float) -> float:
    return s / (tp + 0.5 * fp + 0.5 * fn)


def gap_table(g: pd.DataFrame, p: pd.DataFrame, refined_tp_lost: int) -> pd.DataFrame:
    tp_mask = g["best"] > metric.IOU_THRESHOLD
    tp, s = int(tp_mask.sum()), g.loc[tp_mask, "best"].sum()
    fn, fp = int((~tp_mask).sum()), int((p["best"] <= metric.IOU_THRESHOLD).sum())
    base = pq(tp, s, fp, fn)
    miss = g[~tp_mask]
    never = miss[miss["best"] < 0.05]
    near = miss[miss["best"] >= 0.3]
    near_fp = int(((p["best"] >= 0.3) & (p["best"] <= metric.IOU_THRESHOLD)).sum())
    multi = miss[miss["n_entries"] > 1]
    rows = [
        ("final pipeline", tp, base),
        ("FN never detected (IoU < 0.05)", len(never), pq(tp + len(never), s + 0.6 * len(never), fp, fn - len(never))),
        ("  no first-stage pixel >= 0.5", int((never["high_px"] == 0).sum()), np.nan),
        ("  1-299 such pixels", int(((never["high_px"] > 0) & (never["high_px"] < 300)).sum()), np.nan),
        ("  300 or more such pixels", int((never["high_px"] >= 300).sum()), np.nan),
        ("  refined TP removed by the filter", refined_tp_lost, np.nan),
        ("FN weak overlap (0.05-0.3)", int(((miss["best"] >= 0.05) & (miss["best"] < 0.3)).sum()), np.nan),
        ("near miss pairs (IoU 0.3-0.5)", len(near), pq(tp + len(near), s + 0.55 * len(near), fp - near_fp, fn - len(near))),
        ("  prediction smaller than GT", int((near["recall"] < near["precision"] - 0.15).sum()), np.nan),
        ("  prediction larger than GT", int((near["precision"] < near["recall"] - 0.15).sum()), np.nan),
        ("  GT split over 2+ predictions", int((near["n_touch"] >= 2).sum()), np.nan),
        ("  prediction spans 2+ GT", int((near["spans"] >= 2).sum()), np.nan),
        ("FP with no GT nearby (IoU < 0.05)", int((p["best"] < 0.05).sum()), pq(tp, s, fp - int((p["best"] < 0.05).sum()), fn)),
        ("missed GT on multi-annotator images", len(multi), np.nan),
        ("  also marked by another annotator", int((multi["agree"] > 0.3).sum()), np.nan),
    ]
    matched_multi = g[tp_mask & (g["n_entries"] > 1)]
    rows.append(("matched GT also marked by another", int((matched_multi["agree"] > 0.3).sum()), len(matched_multi)))
    return pd.DataFrame(rows, columns=["quantity", "count", "pq_if_fixed"])


def crop_overlay(image: str, gt: pd.DataFrame, preds: dict, center: tuple, side: int) -> np.ndarray:
    """A square crop around ``center`` with each annotator's outlines and ours."""
    canvas = cv2.cvtColor(data.load_image(data.TRAIN_IMAGES / f"{image}.jpeg"), cv2.COLOR_GRAY2RGB)
    for color, (_, group) in zip(COLORS, gt[gt["image"] == image].groupby("entry")):
        for c in group["segmentation_rle"]:
            contours, _ = cv2.findContours(data.rle_string_to_mask(c), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(canvas, contours, -1, color, 2)
    for c in preds.get(image, []):
        contours, _ = cv2.findContours(data.rle_string_to_mask(c), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(canvas, contours, -1, PRED_COLOR, 2)
    y0 = int(np.clip(center[0] - side // 2, 0, data.SIZE - side))
    x0 = int(np.clip(center[1] - side // 2, 0, data.SIZE - side))
    return cv2.resize(canvas[y0:y0 + side, x0:x0 + side], (700, 700), interpolation=cv2.INTER_AREA)


def gt_center(gt: pd.DataFrame, row: pd.Series) -> tuple[tuple, int]:
    counts = gt[(gt["image"] == row["image"]) & (gt["entry"] == row["entry"])]["segmentation_rle"].iloc[int(row["k"])]
    ys, xs = np.nonzero(data.rle_string_to_mask(counts))
    side = int(max(320, 1.6 * max(np.ptp(ys), np.ptp(xs))))
    return ((ys.min() + ys.max()) // 2, (xs.min() + xs.max()) // 2), min(side, data.SIZE)


def main() -> None:
    gt, preds, folds = load()
    jobs = [(img, [(e, list(g["segmentation_rle"])) for e, g in grp.groupby("entry")], preds.get(img, []), folds[img])
            for img, grp in gt.groupby("image")]
    with ProcessPoolExecutor(4) as pool:
        out = list(pool.map(image_rows, jobs, chunksize=4))
    g = pd.DataFrame([r for rows, _, _ in out for r in rows])
    p = pd.DataFrame([r for _, rows, _ in out for r in rows])
    model = np.sum([c for _, _, c in out], axis=0)
    print(f"our masks against each annotator of the {sum(1 for j in jobs if len(j[1]) > 1)} multi-annotator images: "
          f"PQ {pq(*model):.4f}, SQ {model[1] / model[0]:.4f}")

    refined = pd.read_csv(OOF / "refined.csv")
    refined_preds = refined.groupby(refined["filament_id"].str.rsplit("_", n=1).str[0])["segmentation_rle"].apply(list)
    lost = 0
    for (img, entry), grp in gt.groupby(["image", "entry"]):
        r, f = refined_preds.get(img, []), preds.get(img, [])
        counts = list(grp["segmentation_rle"])
        in_refined = (metric.iou_matrix(counts, r) > metric.IOU_THRESHOLD).any(1) if r else np.zeros(len(counts), bool)
        in_final = (metric.iou_matrix(counts, f) > metric.IOU_THRESHOLD).any(1) if f else np.zeros(len(counts), bool)
        lost += int((in_refined & ~in_final).sum())
    table = gap_table(g, p, lost)
    table.to_csv("report/tables/gap.csv", index=False)
    print(table.round(4).to_string(index=False))

    # Figure: one case per row of the analysis, chosen by a fixed rule (largest qualifying GT).
    multi = g[g["n_entries"] == 3]
    picks = [
        ("annotators disagree", multi[(multi["agree"] < 0.5) & (multi["agree"] > 0.2) & (multi["best"] > 0.5)]),
        ("one annotator only, missed", multi[(multi["agree"] < 0.05) & (multi["best"] < 0.05)]),
        ("near miss, annotators agree", multi[(multi["agree"] > 0.6) & (multi["best"].between(0.3, 0.5))]),
    ]
    plt.figure(figsize=(15, 5.4))
    for k, (label, cands) in enumerate(picks, start=1):
        row = cands.sort_values("area").iloc[-1]
        center, side = gt_center(gt, row)
        plt.subplot(1, 3, k)
        plt.imshow(crop_overlay(row["image"], gt, preds, center, side))
        plt.title(f"{label}: {row['image']} (best IoU {row['best']:.2f})", fontsize=10)
        plt.axis("off")
        print(label, row["image"], row["entry"], round(row["best"], 3), round(row["agree"], 3), int(row["area"]))
    plt.tight_layout()
    plt.savefig("report/figures/disagreement.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
