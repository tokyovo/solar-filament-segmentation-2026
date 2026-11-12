"""Learned candidate filter: keep a refined candidate only if annotators are likely to mark it.

A LightGBM regressor predicts, from 17 features of a candidate, the fraction of annotators
whose filament it matches at IoU > 0.5 (the quantity PQ counts). Candidates with a predicted
fraction below the threshold are removed. Nested cross-validation: NOTES.md section 23.

The features use one first-stage map of a 1024 + 2048 pair, as in training. At test time each
candidate is scored once per fold's pair map, and the scores are averaged.
"""

import json
from pathlib import Path

import cv2
import lightgbm as lgb
import numpy as np
import pandas as pd

FEATURES = ["n_candidates", "area", "log_area", "bbox_h", "bbox_w", "major_axis", "elongation", "solidity",
            "p_mean", "p_max", "p_q90", "p_ring", "i_inside", "i_ring", "contrast", "i_inside_std", "radial"]


def disk_geometry(image: np.ndarray) -> tuple[float, float, float, float]:
    """Centre (y, x), radius and median intensity of the solar disk (Otsu threshold)."""
    _, mask = cv2.threshold(cv2.GaussianBlur(image, (0, 0), 5), 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = stats[k, cv2.CC_STAT_AREA]
    return float(centroids[k, 1]), float(centroids[k, 0]), float(np.sqrt(area / np.pi)), float(np.median(image[labels == k]))


def features(mask: np.ndarray, prob: np.ndarray, image: np.ndarray, disk: tuple, others: np.ndarray) -> dict:
    """Shape, probability, darkness and position of one candidate mask."""
    cy, cx, radius, disk_median = disk
    ys, xs = np.nonzero(mask)
    area = len(ys)
    cov = np.cov(np.stack([ys, xs])) if area > 2 else np.eye(2)
    eig = np.sort(np.linalg.eigvalsh(cov))[::-1].clip(1e-6)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = cv2.convexHull(np.concatenate(contours))
    ring = (cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))) > 0) & (mask == 0) & (others == 0)
    inside_p, ring_p = prob[mask > 0], prob[ring]
    inside_i, ring_i = image[mask > 0].astype(np.float32), image[ring].astype(np.float32)
    return {
        "area": area, "log_area": np.log(area),
        "bbox_h": int(ys.max() - ys.min() + 1), "bbox_w": int(xs.max() - xs.min() + 1),
        "major_axis": 4 * np.sqrt(eig[0]), "elongation": np.sqrt(eig[0] / eig[1]),
        "solidity": area / max(cv2.contourArea(hull), 1.0),
        "p_mean": inside_p.mean(), "p_max": inside_p.max(), "p_q90": np.quantile(inside_p, 0.9),
        "p_ring": ring_p.mean() if ring_p.size else 0.0,
        "i_inside": inside_i.mean() / disk_median, "i_ring": (ring_i.mean() if ring_i.size else disk_median) / disk_median,
        "contrast": ((ring_i.mean() if ring_i.size else disk_median) - inside_i.mean()) / disk_median,
        "i_inside_std": inside_i.std() / disk_median,
        "radial": np.hypot(ys.mean() - cy, xs.mean() - cx) / radius,
    }


def feature_table(masks: list[np.ndarray], prob: np.ndarray, image: np.ndarray) -> pd.DataFrame:
    """One row of ``FEATURES`` per candidate of one image."""
    disk = disk_geometry(image)
    union = np.zeros(image.shape, np.uint8)
    for m in masks:
        union |= m
    rows = [{"n_candidates": len(masks), **features(m, prob, image, disk, union - m)} for m in masks]
    return pd.DataFrame(rows, columns=FEATURES)


def save(model: lgb.LGBMRegressor, threshold: float, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(out / "model.txt"))
    (out / "config.json").write_text(json.dumps({"kind": "candidate_filter", "features": FEATURES,
                                                 "threshold": threshold}, indent=2))


def load(folder: Path) -> tuple[lgb.Booster, float]:
    config = json.loads((folder / "config.json").read_text())
    assert config["features"] == FEATURES, "feature list changed since training"
    return lgb.Booster(model_file=str(folder / "model.txt")), float(config["threshold"])


def score_masks(masks: list[np.ndarray], image: np.ndarray, pair_maps: list[np.ndarray],
                booster: lgb.Booster) -> tuple[np.ndarray, pd.DataFrame]:
    """Score of each candidate, averaged over the pair maps, and its mean features (for logs)."""
    tables = [feature_table(masks, prob, image) for prob in pair_maps]
    scores = np.mean([booster.predict(t) for t in tables], axis=0)
    return scores, sum(tables) / len(tables)
