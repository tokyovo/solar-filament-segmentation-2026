"""Compare the probability maps of one 1024 + 2048 pair with the 10-model mean on test images.

The seed refiners learn on out-of-fold maps of one pair; on the test images they receive the
10-model mean. This prints both distributions on the same images::

    python -m scripts.test_map_stats --images 20
"""

import argparse

import numpy as np

from src import data, postprocess
from src.predict import load_model, predict_probability

parser = argparse.ArgumentParser()
parser.add_argument("--images", type=int, default=20)
args = parser.parse_args()

paths = sorted(data.TEST_IMAGES.glob("*.jpeg"))[:: max(1, 180 // args.images)][: args.images]
models = {f: [load_model(data.DATA_DIR.parent / "models" / f"unet_r34{r}_f{f}") for r in ("", "_r2048")]
          for f in range(5)}
stats = {"pair_f0": [], "mean10": []}
for path in paths:
    image = data.load_image(path)
    pair_maps = {f: np.mean([predict_probability(m, image, res) for m, res in ms], axis=0) for f, ms in models.items()}
    for name, prob in (("pair_f0", pair_maps[0]), ("mean10", np.mean(list(pair_maps.values()), axis=0))):
        masks = postprocess.probability_to_instances(prob, 0.5, 300, 15)
        inside = np.concatenate([prob[m > 0] for m in masks]) if masks else np.zeros(0)
        edge = prob[(prob > 0.1) & (prob < 0.9)]
        stats[name].append((len(masks), inside.mean() if len(inside) else np.nan,
                            (inside > 0.9).mean() if len(inside) else np.nan, edge.size))
for name, rows in stats.items():
    r = np.array(rows, dtype=float)
    print(f"{name:8s} cand/img {r[:, 0].mean():5.2f}  prob inside mean {np.nanmean(r[:, 1]):.3f}  "
          f"share inside > 0.9 {np.nanmean(r[:, 2]):.3f}  pixels in (0.1, 0.9) per img {r[:, 3].mean():8.0f}")
