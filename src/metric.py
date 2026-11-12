"""Panoptic Quality exactly as the organisers' self-evaluation notebook computes it.

The rules, copied from that notebook:

- Each GT entry (``<annotator>-<image_name>``) is scored against all predictions whose
  ``filament_id`` starts with ``<image_name>_``. One prediction set is reused for every
  annotator of an image.
- Every (GT, prediction) pair with IoU > 0.5 is a true positive. There is no one-to-one
  assignment, so overlapping predictions can give one GT several TPs.
- A prediction with no hit is a false positive. A GT with no hit is a false negative.
- Predictions for an image that has no GT entry are never counted.
- PQ = sum(TP IoU) / (TP + FP / 2 + FN / 2), pooled over all entries.

This module computes IoU on the RLEs with pycocotools instead of dense 2048 x 2048 tensors.
``tests/test_metric.py`` checks that the result equals the official code.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from pycocotools import mask as mask_util

SIZE = 2048
IOU_THRESHOLD = 0.5


@dataclass
class PQResult:
    pq: float
    tp: int
    fp: int
    fn: int
    sum_iou: float
    tp_ious: list[float] = field(repr=False)

    @property
    def sq(self) -> float:
        """Segmentation quality: the mean IoU of the TP pairs."""
        return self.sum_iou / self.tp if self.tp else 0.0

    @property
    def rq(self) -> float:
        """Recognition quality: TP / (TP + FP/2 + FN/2). PQ = SQ * RQ."""
        denominator = self.tp + 0.5 * self.fp + 0.5 * self.fn
        return self.tp / denominator if denominator else 0.0


def _rles(counts: list[str]) -> list[dict]:
    return [{"size": [SIZE, SIZE], "counts": c.encode("ascii")} for c in counts]


def iou_matrix(gt_counts: list[str], pred_counts: list[str]) -> np.ndarray:
    """IoU of every (GT, prediction) pair, shape (n_gt, n_pred)."""
    if not gt_counts or not pred_counts:
        return np.zeros((len(gt_counts), len(pred_counts)))
    # pycocotools.iou(dt, gt, iscrowd) returns shape (len(dt), len(gt)).
    ious = mask_util.iou(_rles(pred_counts), _rles(gt_counts), [0] * len(gt_counts))
    return np.asarray(ious).T


def panoptic_quality(gt_df: pd.DataFrame, pred_df: pd.DataFrame) -> PQResult:
    """Score a prediction frame against a GT frame. Both have the submission columns."""
    gt_entry = gt_df["filament_id"].str.split("_", n=1).str[0]
    pred_image = pred_df["filament_id"].str.split("_", n=1).str[0]
    pred_by_image = pred_df.groupby(pred_image)["segmentation_rle"].apply(list).to_dict()

    tp_ious: list[float] = []
    fp = fn = 0
    for entry_id, gt_counts in gt_df.groupby(gt_entry, sort=False)["segmentation_rle"]:
        image_name = entry_id.split("-", 1)[1]
        pred_counts = pred_by_image.get(image_name, [])
        hits = iou_matrix(list(gt_counts), pred_counts)
        hit = hits > IOU_THRESHOLD
        tp_ious.extend(hits[hit].tolist())
        fp += int((hit.sum(axis=0) == 0).sum())
        fn += int((hit.sum(axis=1) == 0).sum())

    tp = len(tp_ious)
    sum_iou = float(sum(tp_ious))
    denominator = tp + 0.5 * fp + 0.5 * fn
    return PQResult(sum_iou / denominator if denominator else 0.0, tp, fp, fn, sum_iou, tp_ious)
