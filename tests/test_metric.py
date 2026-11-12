"""Check src.metric against the official notebook code, and src.data against the JSON."""

import cv2
import numpy as np
import pandas as pd
import pytest

from src import data, metric
from tests import official_selfeval

N_ENTRIES = 12


@pytest.fixture(scope="module")
def annotations() -> pd.DataFrame:
    return data.load_annotations()


@pytest.fixture(scope="module")
def gt_df(annotations) -> pd.DataFrame:
    # Take every entry of a few multi-annotator images, so the reuse of one prediction
    # set across annotators is tested too.
    counts = annotations.groupby("image_name")["entry_id"].nunique()
    images = counts[counts >= 2].index[:N_ENTRIES // 2]
    return data.ground_truth_frame(annotations[annotations["image_name"].isin(images)])


def _perturbed_predictions(gt_df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Turn one annotator's GT into a noisy prediction: shift, dilate, drop, duplicate."""
    rng = np.random.default_rng(seed)
    entry = gt_df["filament_id"].str.split("_", n=1).str[0]
    first_entry_per_image = entry.groupby(entry.str.split("-", n=1).str[1]).transform("first")
    source = gt_df[entry == first_entry_per_image]
    rows = []
    for filament_id, counts in zip(source["filament_id"], source["segmentation_rle"]):
        image_name = filament_id.split("_", 1)[0].split("-", 1)[1]
        mask = data.rle_string_to_mask(counts)
        action = rng.integers(5)
        if action == 0:
            continue
        if action == 1:
            mask = np.roll(mask, rng.integers(-6, 7), axis=1)
        if action == 2:
            mask = cv2.dilate(mask, np.ones((5, 5), np.uint8))
        copies = 2 if action == 3 else 1
        for _ in range(copies):
            rows.append((image_name, data.mask_to_rle_string(mask)))
    rows.append((rows[0][0], data.mask_to_rle_string(np.pad(np.ones((40, 40), np.uint8), ((0, 2008), (0, 2008))))))
    return pd.DataFrame({
        "filament_id": [f"{name}_{k}" for k, (name, _) in enumerate(rows, start=1)],
        "segmentation_rle": [counts for _, counts in rows],
    })


def test_polygon_area_matches_json(annotations):
    sample = annotations.sample(200, random_state=0)
    areas = [mask_util_area(data.polygon_to_rle(p)) for p in sample["polygon"]]
    ratio = np.array(areas) / sample["area"].to_numpy()
    assert np.median(np.abs(ratio - 1)) < 0.05


def mask_util_area(rle: dict) -> float:
    from pycocotools import mask as mask_util
    return float(mask_util.area(rle))


def test_rle_round_trip(gt_df):
    counts = gt_df["segmentation_rle"].iloc[0]
    assert data.mask_to_rle_string(data.rle_string_to_mask(counts)) == counts


def test_ground_truth_against_itself_is_one(gt_df):
    entry = gt_df["filament_id"].str.split("_", n=1).str[0]
    one_entry = gt_df[entry == entry.iloc[0]]
    preds = one_entry.assign(filament_id=one_entry["filament_id"].str.split("-", n=1).str[1])
    result = metric.panoptic_quality(one_entry, preds)
    assert result.pq == pytest.approx(1.0)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_official_notebook(gt_df, seed):
    preds = _perturbed_predictions(gt_df, seed)
    official = official_selfeval.get_pq_score(official_selfeval.get_overlap_df(gt_df, preds))
    ours = metric.panoptic_quality(gt_df, preds)
    assert ours.pq == pytest.approx(official, abs=1e-6)
