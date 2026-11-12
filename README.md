# Solar filament instance segmentation (IEEE Big Data Cup 2026)

Code for the Kaggle competition
[Solar Filament Segmentation Challenge 2026](https://www.kaggle.com/competitions/filament-segmentation-2026).
The task: find every solar filament in a GONG H-alpha full-disk image (2048 x 2048) and
output one mask per filament. The metric is Panoptic Quality (PQ).

> The trained weights are a release asset of this repository:
> [`filament-seg-weights.tar`](https://github.com/tokyovo/solar-filament-segmentation-2026/releases/download/v1.0/filament-seg-weights.tar) (702 MiB). The demo notebook downloads them.

## Method

```
GONG H-alpha image (2048 x 2048, grayscale)
        |
        +--> U-Net r34 at 1024 (5 folds) --+
        |                                   +--> mean probability map (10 models, h-flip TTA)
        +--> U-Net r34 at 2048 (5 folds) --+                |
                                                           v
                     threshold 0.5, connected components, join regions closer than 15 px,
                     drop regions under 300 px  ->  candidate filaments
                                                           |
                                                           v
                     seed refiner (U-Net r34 on a 256 crop around each candidate:
                     image + candidate mask + probability), mean of 5 folds
                                                           |
                                                           v
                     learned filter (LightGBM on 17 candidate features): keep a
                     candidate if the predicted share of annotators who mark it >= 0.35
                                                           |
                                                           v
                     one mask per filament, no overlaps  ->  submission CSV (COCO RLE)
```

- **Target.** Several annotators label many images, and they agree with each other at only
  PQ 0.34. The first-stage models learn the per-pixel mean of all annotators' masks.
- **Two resolutions.** The 2048 model draws finer outlines; the 1024 model sees more context.
  Their mean beats either model alone on all 5 folds.
- **Seed refiner.** Most errors are candidates on the wrong place or with too little overlap.
  The refiner re-segments each candidate at full resolution and learns to reject candidates
  that no annotator marked.
- **Learned filter.** A gradient-boosting model predicts, from the shape, probability,
  darkness and disk position of each candidate, the share of annotators who mark it. It is
  trained on out-of-fold candidates and judged with nested cross-validation.

## Results (out of fold, 5 folds, our exact copy of the official PQ)

| Pipeline | mean PQ | public LB |
|---|---|---|
| U-Net r34 at 1024 | 0.413 | 0.35 (one fold) |
| 1024 + 2048 mean | 0.426 | 0.36 |
| 1024 + 2048 mean + seed refiner | 0.432 | 0.37 |
| + learned candidate filter | 0.445 (nested CV, pooled) | **0.38** |

`NOTES.md` records every experiment, including the ones that failed, and
`research/results.jsonl` holds the numbers.

## Repository layout

| Path | Content |
|---|---|
| `src/data.py` | annotation loading, masks, RLE, folds |
| `src/metric.py` | PQ, identical to the organisers' self-evaluation notebook (`tests/test_metric.py`) |
| `src/train.py` | first-stage U-Net training |
| `src/postprocess.py` | probability map to filament masks |
| `src/refiner.py` | seed refiner: crops, training, inference |
| `src/candidate_filter.py` | learned candidate filter: features, scoring |
| `src/weights.py` | download and check of the release weights |
| `src/predict.py` | inference, validation maps and the submission CSV |
| `scripts/` | cache building, sweeps, analyses, provenance sidecars |
| `submission/*.json` | provenance of each Kaggle submission (commit, config, sha256, score) |

## Reproduce the predictions

`notebooks/demo.ipynb` rebuilds the selected submission from the stored weights, without
training. It needs the test images in `data/MAGFiLO_1.0_Kaggle_2026/test/test_images/`. On the
first run it downloads the 16 weight folders (10 U-Nets, 5 refiners and the filter; one
702 MiB archive, a release asset of this repository) into `release/weights/` and checks their
SHA-256 (`src/weights.py`).

`USE_FILTER = True` rebuilds our first selected submission (`mix10_refiner5_filter1.csv`,
public 0.38); `USE_FILTER = False` rebuilds the second (`mix10_refiner5.csv`, public 0.37).
float16 results differ slightly between GPU generations, so the CSV is byte-identical only on
the GPU type that made the submitted file:

| Selected file | Made on | sha256 |
|---|---|---|
| `mix10_refiner5_filter1.csv` | NVIDIA Quadro RTX 8000 | `9fac0e35011809beeac63a4a9045ce0e73b8a603a30d7dab64745c195a862f72` |
| `mix10_refiner5.csv` | NVIDIA GeForce RTX 3070 | `e66c87e65679f14bd1eb8790a98f23a5809fb02a7cf011281329eeb1ee0a082e` |

On another GPU type the masks agree with a mean IoU of 0.9998 (checked: RTX 3070 file rebuilt
on the Quadro). On a CPU (float32) they agree with a mean IoU of 0.997.

| Hardware | Time for 180 images |
|---|---|
| GPU (RTX 3070 or Quadro RTX 8000) | 5 to 10 minutes (1.5 s per image) |
| CPU (4 threads) | about 4 hours (80 s per image) |

The same run from the command line:

```bash
python -m src.predict submit \
  --model release/weights/unet_r34_f{0,1,2,3,4} release/weights/unet_r34_r2048_f{0,1,2,3,4} \
  --refiner release/weights/refiner_v2_f{0,1,2,3,4} \
  --filter release/weights/filter_v1 \
  --out submission.csv
```

## Train from scratch

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -r requirements.txt
python -m scripts.build_cache --res 1024
python -m scripts.build_cache --res 2048 --npy
for f in 0 1 2 3 4; do
  python -m src.train --fold $f --out data/models/unet_r34_f$f
  python -m src.train --fold $f --res 2048 --crop 1024 --batch 4 --out data/models/unet_r34_r2048_f$f
  python -m src.predict probs --model data/models/unet_r34_f$f --fold $f
  python -m src.predict probs --model data/models/unet_r34_r2048_f$f --fold $f
done
python -m src.refiner build
for f in 0 1 2 3 4; do python -m src.refiner train --fold $f --out data/models/refiner_v2_f$f; done
python -m scripts.oof_predictions        # out-of-fold candidates of the refined pipeline
python -m scripts.candidate_features     # their features and labels
python -m scripts.filter_cv              # nested cross-validation of the filter
python -m scripts.train_filter --out data/models/filter_v1
```

One RTX 3070 (8 GiB) trains a 1024 model in about 20 minutes, a 2048 model in about
70 minutes and a refiner in about 10 minutes.

## Data

The competition data (GONG H-alpha images and MAGFiLO annotations) is **not included**. It is
available on the competition page and may be used **for non-commercial purposes only**
(competition, academic research and education), under the competition rules. Put it in
`data/MAGFiLO_1.0_Kaggle_2026/`.

At inference the pipeline uses only the pixels of each test image. The file name is used
only to build the output `filament_id`.

## Acknowledgements

`tests/official_selfeval.py` and `scripts/official_plots.py` are verbatim copies of the
organisers' [self-evaluation notebook](https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook)
by Azim Ahmadzadeh. We use them to check that our PQ equals the official one and to draw the
report figures. They keep that notebook's Apache 2.0 licence (`third_party/NOTICE.md`,
`third_party/LICENSE-Apache-2.0.txt`) and are not covered by our MIT licence.

## Licence

The code and the trained weights are released under the MIT licence (`LICENSE`). The
competition data is not included; it is licensed for non-commercial use only. The encoders start from ImageNet weights through
[segmentation-models-pytorch](https://github.com/qubvel-org/segmentation_models.pytorch).
