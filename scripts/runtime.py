"""Time each stage of the selected pipeline per test image, on the current device.

Writes one row per stage to ``report/tables/runtime_<device>.csv``. The first image warms up
the models and is not counted::

    python -m scripts.runtime --images 11                          # GPU
    CUDA_VISIBLE_DEVICES="" python -m scripts.runtime --images 3   # CPU
"""

import argparse
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src import data, postprocess, refiner
from src.predict import DEVICE, instances_frame, load_model, predict_probability

WEIGHTS = Path("release/weights")


def sync() -> float:
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=int, default=11)
    args = parser.parse_args()
    m1024 = [load_model(WEIGHTS / f"unet_r34_f{f}") for f in range(5)]
    m2048 = [load_model(WEIGHTS / f"unet_r34_r2048_f{f}") for f in range(5)]
    refiners = refiner.load_refiners([WEIGHTS / f"refiner_v2_f{f}" for f in range(5)])
    times = defaultdict(list)
    for k, path in enumerate(sorted(data.TEST_IMAGES.glob("*.jpeg"))[: args.images]):
        t = {"start": sync()}
        image = data.load_image(path)
        t["load image"] = sync()
        maps = [predict_probability(m, image, res) for m, res in m1024]
        t["5 U-Nets at 1024"] = sync()
        maps += [predict_probability(m, image, res) for m, res in m2048]
        t["5 U-Nets at 2048"] = sync()
        prob = np.mean(maps, axis=0)
        seeds = postprocess.probability_to_instances(prob, **refiner.PP)
        t["post-processing"] = sync()
        masks = refiner.refine(refiners, image, prob, seeds)
        t["5 seed refiners"] = sync()
        instances_frame(path.stem, masks)
        t["RLE encoding"] = sync()
        if k == 0:
            continue  # warm-up
        stamps = list(t.values())
        for name, a, b in zip(list(t)[1:], stamps[:-1], stamps[1:]):
            times[name].append(b - a)
        times["total"].append(stamps[-1] - stamps[0])
    rows = [{"device": DEVICE, "stage": name, "seconds_per_image": round(float(np.mean(v)), 3),
             "images": len(v)} for name, v in times.items()]
    table = pd.DataFrame(rows)
    out = Path(f"report/tables/runtime_{DEVICE}.csv")
    table.to_csv(out, index=False)
    print(table.to_string(index=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
