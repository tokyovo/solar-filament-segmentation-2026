"""Load the MAGFiLO COCO annotations and turn them into masks and RLE strings.

Vocabulary used across the repo:

- ``image_name``: the GONG file stem, e.g. ``20140609195854Bh``. Predictions use it.
- ``entry_id``: one annotator's labelling of one image, ``<annotator>-<image_name>``,
  e.g. ``040301-20140609195854Bh``. This is the COCO image ``id``. The metric scores each
  entry separately, so an image with three annotators counts three times.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from pycocotools import mask as mask_util

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "MAGFiLO_1.0_Kaggle_2026"
TRAIN_JSON = DATA_DIR / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
TRAIN_IMAGES = DATA_DIR / "train" / "train_images"
TEST_IMAGES = DATA_DIR / "test" / "test_images"
SIZE = 2048


def load_annotations(path: Path = TRAIN_JSON) -> pd.DataFrame:
    """Return one row per filament with its entry, image, annotator and polygon."""
    coco = json.loads(Path(path).read_text())
    entries = {im["id"]: im for im in coco["images"]}
    rows = []
    for ann in coco["annotations"]:
        entry = entries[ann["image_id"]]
        rows.append({
            "filament_uid": ann["id"],
            "entry_id": entry["id"],
            "annotator": entry["id"].split("-", 1)[0],
            "image_name": Path(entry["file_name"]).stem,
            "polygon": ann["segmentation"],
            "spine": ann["spine"],
            "area": ann["area"],
            "category_id": ann["category_id"],
        })
    return pd.DataFrame(rows)


def polygon_to_rle(polygon: list[list[float]], size: int = SIZE) -> dict:
    """Rasterise a COCO polygon into one COCO RLE dict with pycocotools."""
    return mask_util.merge(mask_util.frPyObjects(polygon, size, size))


def mask_to_rle_string(mask: np.ndarray) -> str:
    """Encode a binary (H, W) mask as the RLE counts string the submission expects."""
    rle = mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))
    return rle["counts"].decode("ascii")


def rle_string_to_mask(counts: str, size: int = SIZE) -> np.ndarray:
    """Decode one RLE counts string back into a (H, W) uint8 mask."""
    return mask_util.decode({"size": [size, size], "counts": counts.encode("ascii")})


def ground_truth_frame(annotations: pd.DataFrame) -> pd.DataFrame:
    """Build the official GT format: ``filament_id`` = ``<entry_id>_<k>``, plus the RLE."""
    rows = []
    for entry_id, group in annotations.groupby("entry_id", sort=False):
        for k, polygon in enumerate(group["polygon"], start=1):
            counts = polygon_to_rle(polygon)["counts"].decode("ascii")
            rows.append({"filament_id": f"{entry_id}_{k}", "segmentation_rle": counts})
    return pd.DataFrame(rows)


def load_image(path: Path) -> np.ndarray:
    """Read a GONG JPEG as a (H, W) uint8 grayscale array."""
    import cv2
    return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)


def assign_folds(image_names: list[str], n_folds: int = 5, seed: int = 0) -> dict[str, int]:
    """Give every image one fold. All annotators of an image share it, so no image leaks."""
    names = sorted(set(image_names))
    order = np.random.default_rng(seed).permutation(len(names))
    return {names[i]: rank % n_folds for rank, i in enumerate(order)}
