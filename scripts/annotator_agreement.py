"""Score each annotator against the other annotators of the same image (a human ceiling)."""

import pandas as pd

from src import data, metric

annotations = data.load_annotations()
gt = data.ground_truth_frame(annotations)
gt["entry_id"] = gt["filament_id"].str.split("_", n=1).str[0]
gt["image_name"] = gt["entry_id"].str.split("-", n=1).str[1]
multi = gt.groupby("image_name")["entry_id"].nunique()
gt = gt[gt["image_name"].isin(multi[multi >= 2].index)]

total = metric.PQResult(0, 0, 0, 0, 0.0, [])
for entry_id, as_pred in gt.groupby("entry_id"):
    others = gt[(gt["image_name"] == as_pred["image_name"].iloc[0]) & (gt["entry_id"] != entry_id)]
    preds = pd.DataFrame({
        "filament_id": as_pred["filament_id"].str.split("-", n=1).str[1],
        "segmentation_rle": as_pred["segmentation_rle"],
    })
    r = metric.panoptic_quality(others[["filament_id", "segmentation_rle"]], preds)
    total = metric.PQResult(0, total.tp + r.tp, total.fp + r.fp, total.fn + r.fn,
                            total.sum_iou + r.sum_iou, [])
denominator = total.tp + 0.5 * total.fp + 0.5 * total.fn
print(f"images={len(multi[multi >= 2])} PQ={total.sum_iou / denominator:.4f} "
      f"SQ={total.sq:.4f} RQ={total.rq:.4f} TP={total.tp} FP={total.fp} FN={total.fn}")
