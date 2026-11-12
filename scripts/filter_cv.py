"""Nested cross-validation of the learned candidate filter (NOTES.md section 23).

For each outer fold, a LightGBM regressor learns ``match_frac`` on the other four folds, the
keep threshold is chosen by an inner 4-fold cross-validation on those folds, and both are
applied to the outer fold. PQ is computed exactly from per-candidate match statistics: the
candidates do not overlap, so each candidate matches at most one GT filament per annotator::

    python -m scripts.filter_cv
"""

import argparse
from concurrent.futures import ProcessPoolExecutor

import lightgbm as lgb
import numpy as np
import pandas as pd

from src import data, metric

OOF = data.DATA_DIR.parent / "oof"
THRESHOLDS = np.round(np.arange(0.0, 0.61, 0.05), 2)
NOT_FEATURES = {"filament_id", "image_name", "fold", "match_frac"}


def match_stats(args: tuple) -> list[tuple]:
    """Per candidate of one image: (filament_id, annotators, matches, sum of matched IoU)."""
    ids, counts, gts = args
    n_match, s_iou = np.zeros(len(ids)), np.zeros(len(ids))
    for gt_counts in gts:
        iou = metric.iou_matrix(gt_counts, counts)
        hit = iou > metric.IOU_THRESHOLD
        n_match += hit.any(axis=0)
        s_iou += (iou * hit).sum(axis=0)
    return [(i, len(gts), m, s) for i, m, s in zip(ids, n_match, s_iou)]


def pq(c: pd.DataFrame, keep: np.ndarray, gt_total: int) -> float:
    """Pooled PQ of the kept candidates; ``gt_total`` counts all GT filaments of these images."""
    k = c[keep]
    tp = k["n_match"].sum()
    fp = (k["n_entries"] - k["n_match"]).sum()
    return k["sum_iou"].sum() / (tp + 0.5 * fp + 0.5 * (gt_total - tp))


def fit(train: pd.DataFrame, features: list[str]) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=20,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, n_jobs=4, verbose=-1,
                              random_state=0)
    return model.fit(train[features], train["match_frac"])


def best_threshold(c: pd.DataFrame, scores: np.ndarray, gt_total: int) -> float:
    return float(max(THRESHOLDS, key=lambda t: pq(c, scores >= t, gt_total)))


def load_candidates(tag: str = "") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Out-of-fold candidates with their match statistics, and the GT table."""
    cand = pd.read_csv(OOF / f"candidates{tag}.csv")
    refined = pd.read_csv(OOF / f"refined{tag}.csv")
    gt = pd.read_csv(OOF / "gt.csv")
    gt["entry"] = gt["filament_id"].str.split("_", n=1).str[0]
    gt["image"] = gt["entry"].str.split("-", n=1).str[1]
    refined["image"] = refined["filament_id"].str.rsplit("_", n=1).str[0]
    gts = {img: [list(g["segmentation_rle"]) for _, g in grp.groupby("entry")] for img, grp in gt.groupby("image")}
    jobs = [(list(g["filament_id"]), list(g["segmentation_rle"]), gts[img]) for img, g in refined.groupby("image")]
    with ProcessPoolExecutor(4) as pool:
        stats = [r for rs in pool.map(match_stats, jobs, chunksize=4) for r in rs]
    cand = cand.merge(pd.DataFrame(stats, columns=["filament_id", "n_entries", "n_match", "sum_iou"]), on="filament_id")
    return cand, gt


def nested(cand: pd.DataFrame, gt: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Nested CV: per outer fold, inner-CV threshold, then model and threshold on the fold.

    Returns the per-fold table, the out-of-fold scores and the keep decisions.
    """
    folds = data.assign_folds(gt["image"].tolist())
    gt_by_fold = gt["image"].map(folds).value_counts().to_dict()
    scores = np.zeros(len(cand))
    rows = []
    for k in range(5):
        train_mask = (cand["fold"] != k).to_numpy()
        train = cand[train_mask]
        inner = np.zeros(len(train))
        for j in sorted(train["fold"].unique()):
            m = (train["fold"] == j).to_numpy()
            inner[m] = fit(train[~m], features).predict(train[m][features])
        t = best_threshold(train, inner, sum(gt_by_fold[f] for f in range(5) if f != k))
        test = cand[~train_mask]
        scores[~train_mask] = fit(train, features).predict(test[features])
        keep = scores[~train_mask] >= t
        rows.append({"fold": k, "threshold": t, "refined": pq(test, np.ones(len(test), bool), gt_by_fold[k]),
                     "filtered": pq(test, keep, gt_by_fold[k]), "removed": int((~keep).sum()), "candidates": len(test)})
    table = pd.DataFrame(rows)
    table["gain"] = table["filtered"] - table["refined"]
    keep = scores >= cand["fold"].map(table.set_index("fold")["threshold"]).to_numpy()
    return table, scores, keep


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="", help="suffix of the input and output files, e.g. _lo")
    tag = parser.parse_args().tag
    cand, gt = load_candidates(tag)
    features = [c for c in cand.columns if c not in NOT_FEATURES | {"n_entries", "n_match", "sum_iou"}]
    all_keep = np.ones(len(cand), bool)
    print(f"check: all candidates kept, pooled PQ {pq(cand, all_keep, len(gt)):.4f}")
    table, scores, keep = nested(cand, gt, features)
    print(table.round(4).to_string(index=False))
    print(f"pooled: refined {pq(cand, all_keep, len(gt)):.4f} -> filtered {pq(cand, keep, len(gt)):.4f} "
          f"(removed {int((~keep).sum())} of {len(cand)})")
    importance = pd.Series(fit(cand, features).feature_importances_, index=features).sort_values(ascending=False)
    print("feature importance (splits):", importance.head(8).to_dict())
    cand.assign(score=scores, kept=keep).to_csv(OOF / f"candidates_scored{tag}.csv", index=False)


if __name__ == "__main__":
    main()
