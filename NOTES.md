# IEEE Big Data Cup 2026: Solar Filament Segmentation Challenge (Kaggle)

Everything known about this competition, in one place. Read this first when you resume work.

https://www.kaggle.com/competitions/filament-segmentation-2026
Self-evaluation notebook: https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook

**Kaggle closes 2026-11-15 06:00 UTC.** Set up 2026-10-28, which leaves 18 days.

## 1. Key dates (UTC)

| Date | Event | Source |
|---|---|---|
| 2026-07-10 | Launch | Important Dates page |
| **2026-11-15 06:00** | **Kaggle submission deadline** | `kaggle competitions list` |
| 2026-11-15 "11:59 pm CDT" | Final reports and solutions (Google form) | Important Dates page |
| 2026-11-30 | Winners announced | Important Dates page |
| 2026-12-14 to 17 | IEEE BigData, Phoenix (optional attendance) | Important Dates page |

"11:59 pm CDT" on Nov 15 is 04:59 UTC on Nov 16. CDT is not in use in November, so the
page may mean CST (05:59 UTC Nov 16). Both are later than the Kaggle close. **Treat
2026-11-15 06:00 UTC as the deadline for everything.** No entry deadline is published.

## 2. Facts from the Kaggle CLI and pages (read 2026-10-28)

| Item | Value |
|---|---|
| Sponsor | U.S. NSF National Solar Observatory (NSO). Organisers: Ahmadzadeh (UMSL), Kempton (GSU), Li (NJIT), Pevtsov (NSO). |
| Prize | **Up to $3,000** for the top 3, split in proportion to (rubric score)². `compute_prizes(..., temperature=2.0)` on the Prizes page. |
| Payment | AURA/NSO bank transfer. The winner must be able to receive payments from U.S. banks and complete vendor setup, or the prize is lost. |
| Field (2026-10-28) | 1,226 teams. Top 0.70, rank 10 0.60 (Kaggle shows 2 decimals). Higher is better. |
| Submissions | 5 per day. Select up to 2 final submissions. |
| Team size | Up to 5 |
| Submission type | CSV upload: `filament_id, segmentation_rle`. No notebook requirement. |
| Final package | Google form: a **4-page technical report** (PDF, Overleaf template) and a **public Git repo**. |
| Data licence | Competition use, non-commercial and academic research only. |
| Winner licence | OSI open-source licence that allows commercial use. Code, scripts and configs must reproduce the results. |
| Eligibility | Anyone. People supervised or mentored by an organiser cannot win prizes. |
| AI tools | Allowed, but winners must understand, explain and reproduce every major part. |

## 3. The task

- **Input:** GONG H-alpha full-disk images, 2048 × 2048, 8-bit grayscale JPEG. Treat them
  as grayscale, not RGB. The file name encodes time and observatory (`YYYYMMDDHHMMSSII`).
- **Training labels:** MAGFiLO, COCO-style JSON. One polygon per filament, plus bbox, spine
  and chirality class (Left, Right, Unidentifiable, Ambiguous). Several annotators may label
  the same image; treat each annotation set as a separate image. The Cup page says MAGFiLO
  has 10,244 filaments in 1,593 images.
- **Output:** one CSV row per predicted filament. `filament_id` is the image id plus a
  unique suffix (`20150125172714Mh_1`). `segmentation_rle` is the COCO RLE counts string
  from `pycocotools`, without quotes. Size is fixed at 2048 × 2048.
- **Metric:** **Panoptic Quality (PQ)**. Higher is better. The leaderboard shows about 50%
  of the test images. PQ replaced Dice on 2026-08-07.
- **Final judging rubric:**
  - Quantitative 70%: PQ, the distributions of Dice and IoU, and the counts of one-to-many
    and many-to-one matches.
  - Qualitative 30%: the pipeline description, the look of the predicted shapes, and code
    quality (modularity and documentation).
  - The Goal section also mentions end-to-end computational efficiency.
- Organiser notes: on 2026-08-20 they said some teams "hack the system" and any PQ above
  0.35 has value. On 2026-10-23 they wrote "a PQ greater than 3.5"; this is probably 0.35.

## 4. Rules that limit the method

- At inference, use only the provided H-alpha test images.
- Models may not use other ground-truth metadata for training (Evaluation page). The exact
  meaning is unclear (unverified). Ask before using external filament labels.
- External data is allowed if public and of minimal cost.
- **The Git repo must be public at the close and stay public until the winners are
  announced.** It needs `requirements.txt`, a notebook that runs the whole pipeline, and the
  **model weights in the repo**. The committee must reproduce the predictions without
  retraining and without extra files.
- Without the Google form, a team is not considered for the final evaluation.

## 5. The paper outlet

| Item | Value |
|---|---|
| Report | **4-page technical report**, PDF, from the Overleaf template https://www.overleaf.com/read/vjztpvrnbrdh#43fbbe . Sent with the Google form. |
| Deadline | The close (treat as 2026-11-15 06:00 UTC). |
| Who may submit | Every team that wants final evaluation. |
| Proceedings | **Unverified.** The Cup page offers a report slot and invitations to extend winning work into conference proceedings. |
| SABID workshop | The organisers suggested the SABID 2026 workshop at IEEE BigData. **Its submission deadline (2026-10-06) has passed.** Camera-ready was 2026-11-14. Up to 10 pages (6 for short papers), IEEE 2-column, via CyberChair. |
| Cup-wide | Camera-ready deadline about 2026-11-20 (Cup page). |

## 6. What this means for us

- **The leaderboard is not the whole score.** 30% is report, shapes and code quality. A
  clean, documented repo and good figures count.
- **PQ punishes fragments and merges.** A filament split into two parts gives a false
  positive. Post-processing that joins nearby pieces and removes small islands matters.
- The public board may hold scores from "hacking". Do not chase the top 0.70 blindly.
- **GPU need: 8 GiB is fine.** A U-Net or instance model on crops or downscaled images fits
  the RTX 3070. Follow `../GPU-SCHEDULE.md`.
- We must publish a public GitHub repo with weights at the close. The user creates it.
  This local repo has no remote.

## 7. Status

- 2026-10-28: repo created. Not registered.

### What the user must do to register

1. Open https://www.kaggle.com/competitions/filament-segmentation-2026 as `nhatvo`, click
   **Join Competition** and accept the rules.
2. Before the close: create a public GitHub repo for the solution (with weights), and
   complete the Google form linked on the "Final Submission" section of the overview page:
   https://docs.google.com/forms/d/e/1FAIpQLSfyrM4dhc8QzxR90PtF6fb018KPf0S0Cbf5hPqMuUognmzQsA/viewform
3. A winner must later provide bank details to AURA.

- 2026-10-28: joined on Kaggle as `nhatvo` (rules accepted by the user).
- 2026-10-28: data downloaded. Environment, data loader and local PQ metric done (section 8).

## 8. Data and metric findings (2026-10-28)

### Data

- Train: 707 unique images, 1,154 entries (one entry = one annotator on one image),
  8,199 filaments, 37 annotators. 411 images have 1 entry, 145 have 2, 151 have 3.
- Test: 180 images. No file name is shared with train.
- The JSON fields are typed (lists, floats, ints). The COCO image `id` is the entry id
  `<annotator>-<image_name>`, for example `040301-20140609195854Bh`.
- Filaments are small. Area percentiles in pixels: 10% 409, median 1,228, 90% 4,687,
  max 37,739. The image has 4.2 M pixels.

### How the official PQ really works (`research/ref/selfeval/`, `src/metric.py`)

- Each GT entry is scored separately against the prediction set of its image. An image with
  3 annotators is scored 3 times with the same predictions.
- **There is no one-to-one matching.** Every (GT, prediction) pair with IoU > 0.5 counts
  as a TP. So overlapping copies of a good prediction each count as another TP.
- Predictions for images that have no GT entry are ignored.
- `src/metric.py` computes IoU on RLEs with pycocotools. `tests/test_metric.py` checks it
  against a verbatim copy of the notebook code: same PQ to 1e-6, duplicates included.

### Human ceiling

`scripts/annotator_agreement.py` scores each annotator against the other annotators of the
same image (296 images):

| PQ | SQ | RQ | TP | FP | FN |
|---|---|---|---|---|---|
| 0.340 | 0.634 | 0.536 | 4,292 | 3,709 | 3,709 |

The organisers said any PQ above 0.35 has value. **The public top (0.70) and rank 10 (0.60)
are about twice the agreement between humans.** The likely cause is the TP rule above.
**Decision: do not send duplicate or overlapping masks.** The final rubric counts
one-to-many matches, and the organisers know about the "hacks". Expect our honest score to
be far below the public top, and judge progress on local CV, not on rank.

## 9. Public leaderboard and public notebooks (2026-10-29)

Leaderboard: 1,257 teams. Top 0.70, rank 10 0.60, ranks 25 to 100 all at 0.55, rank 200
0.39, rank 400 0.37. 104 teams are above 0.5.

| Notebook | What it really is |
|---|---|
| a public "0.55" notebook | Does not use its model. It writes a test CSV that is embedded as an lzma/base85 payload. The CSV has 1,342 masks on 178 images, no overlaps and no duplicates, median area 1,294 px. Its origin is not stated. This file probably explains the plateau at 0.55. |
| `phuongncn/lb-0-70-...` | Mask R-CNN + U-Net refiner. Its 0.70 is from 2026-08-05, before the switch to PQ, so it was probably a Dice score. Its docstring mentions "public MAGFiLO/test-overlap labels", so labels for some test images may be public (unverified). |
| `artkomissar/please-fix-the-leaderboard` | Dice era. A partial empty submission scored 1.00. The PQ code in the self-evaluation notebook would score that as 0. |

Consequence: the public board mixes copied files, possible label leaks and maybe old
Dice-era scores. Our only reliable guide is local CV with the exact metric.

**Decision (supervisor, 2026-10-29): the public plateau at 0.55 comes from the copied file
above. Do not chase it.** Never submit copied files, decoded CSVs or leaked labels. The
decoded CSV in `research/cache/public055/` exists only for the analysis above.

Baseline (`src/train.py`, `src/predict.py`): fold 0 PQ 0.407 (SQ 0.674, RQ 0.604,
TP 932, FP 555, FN 669), threshold 0.4, min area 600.

Calibration submission 57763107 (fold 0 model, same settings): **public 0.34** against local
fold 0 PQ 0.407. The public board shows 2 decimals and holds about half of the 180 test
images, so this gap is only a rough guide.

## 10. Plan and GPU limits (2026-10-29, supervisor)

- **urbantwin needs the GPU from 2026-11-01.** Save GPU hours before that date.
- Order: post-processing sweep pooled over folds 0 and 1, then a training-target test, then
  2048 crops (check memory first), then a larger encoder with 5 folds and a fold ensemble.
- Training-target test: A = soft mean of annotators (the baseline), B = hard majority vote,
  C = one random annotator per sample. Run B and C on fold 0 first. Go to fold 1 only if
  one of them beats A.
- No new submission until the local PQ improves on both folds.
- Tell the supervisor 2 days before the close (by 2026-11-13) that the repo must be public.

## 11. Results of 2026-10-29

Post-processing sweep (`scripts/sweep_postprocess.py`, folds 0 and 1 pooled, 284 images):
**threshold 0.5, min area 300, merge_dist 15** is the new default. It gives fold 0 0.4104
(was 0.4066) and fold 1 0.4212 (was 0.4020). Joining fragments within 15 px helps. A seed
threshold does not help.

Training target, fold 0, same post-processing for all (`scripts/compare_models.py`):

| Target | PQ (new pp) | PQ (old pp) |
|---|---|---|
| A soft mean (baseline) | **0.4104** | 0.4066 |
| B majority vote | 0.4078 | 0.4099 |
| C one random annotator | 0.4051 | 0.4067 |

All differences are 0.005 or less with one seed each. **Keep A.** B and C were not run on
fold 1, as the supervisor decided.

Calibration 2 (submission 57767993): fold 1 model with pp_v1, **public 0.35**, local fold 1
0.421. Calibration 1 was public 0.34 with local fold 0 0.407. Both points show a gap of
about 0.07 between local CV and the public board. The public board shows 2 decimals, so it
cannot resolve gains below about 0.01.

**Seed noise (fold 0, model A):** seed 0 and seed 1 differ by 0.0017 (new pp) and 0.0036
(old pp). **A change must beat A by more than 0.007** to count. Target B's +0.003 was noise.

GPU memory (`scripts/bench_gpu.py`, `research/bench_gpu_2026-10-29.txt`): resnet34 crop 1024
batch 4 needs 3.88 GiB, resnet50 crop 1024 batch 2 needs 3.43 GiB, efficientnet-b3 crop 768
batch 4 needs 5.18 GiB. The 2048 model uses resnet34, crop 1024, batch 4: the same view of the
Sun as the 512 crops at 1024, with 4 times the detail.

**2048 model (fold 0, pp_v1):** PQ 0.4154 against the A mean 0.4112 (+0.004, below the bar).
SQ rises from 0.672 to 0.680; RQ does not change. Better outlines, not more filaments found.
71 s/epoch, about 4 times the 1024 cost.

**Ensembles of saved fold 0 maps (pp_v1):** A s0 + A s1 0.4142; **A s0 + 2048 0.4199**
(+0.0087 over the A mean, above the bar); all three 0.4180. The mix of resolutions helps
more than a second seed. This is the best of 3 combinations on one fold.

**FP filter (pp_v2 sweep, folds 0+1):** a mean-probability filter gives at most +0.003
(below the bar). It removes only 40 of 1,350 FP, so most FP are confident regions. Many are
probably real filaments that some annotators skipped: each annotator is scored alone, so one
annotator's omission makes a FP for that entry. **Keep pp_v1.**

**Fold 1 check of the mix (pre-chosen pair only):** A 0.4212, 2048 alone 0.4312,
**A + 2048 0.4336 (+0.012)**. With fold 0 (+0.009), the mix passes the bar on both folds.
**Decision: the final model is a 5-fold ensemble of 1024 and 2048 U-Nets (resnet34).**

Calibration 3 (submission 57773048): 4-model mix (1024 + 2048, folds 0 and 1), pp_v1,
**public 0.36**. Local pair scores were 0.420 (fold 0) and 0.434 (fold 1). The three points
(0.407 -> 0.34, 0.421 -> 0.35, about 0.427 -> 0.36) all move in the same direction as local CV.

## 12. Public notebooks of the PQ era (read 2026-10-31)

Rank on 2026-10-31: **522 of 1,291** (public 0.36). A subagent read 7 notebooks run between
2026-10-20 and 2026-10-30 (copies in `research/ref/nb/`). None uses copied CSVs or labels.

| Notebook | Method | Score |
|---|---|---|
| josngelmarrero/filament-segmentation-lb-0-34 | U-Net r34, 512 crops, full-image inference, random annotator, disk z-score, CC, min area 400 | public 0.34 (close to our own method) |
| jianansun/filament-yolo-seed-refiner | YOLO11s boxes at 1280+640, then a r34 U-Net refiner on 640 crops with a box "seed" channel and CLAHE, 7-view TTA per box, no overlaps by confidence | credits "LB ~0.409" to Tim Gluhkikh (secondhand, not checked) |
| muhammadshoaibaltaf/magfilo-v11 | YOLO26x at 1536, then U-Net++ effb4 per box crop at 512, no overlaps | local PQ 0.435 on a 15% split (no public score) |
| mengxinlab/...-reproducible-release | dark threshold + elongation filter | public 0.15 |
| prvsiyan (2 notebooks) | reviews; no own model | quoted 0.90-0.93 are from the Dice era |

What this means:
- **The best legitimate public score we know of is about 0.41**, from a detector plus a crop
  refiner. The detector can separate touching filaments, which connected components cannot.
- Ideas still open for us: a detector stage; 8-view flip/rotation TTA (we use only a
  horizontal flip); folds grouped by ISO week (photos of the same filament on nearby days
  may make our random folds optimistic).

## 13. Error analysis and validation check (2026-10-31)

**Fold 2:** A 0.4060, 2048 0.4222, A + 2048 0.4192. The pair beats A on folds 0, 1 and 2
(+0.009, +0.012, +0.013).

**Where the errors are** (`scripts/error_analysis.py`, fold 0, A + 2048, pp_v1):

| | merge / fragment | overlaps one, IoU <= 0.5 | no overlap |
|---|---|---|---|
| FP (746) | 42 | 341 | 363 |
| FN (552) | 65 | 284 | 203 |

- Merges and fragments are small (about 107 of 1,298 errors). A detector mainly fixes those.
- The 324 near-miss pairs (one prediction, one GT, IoU <= 0.5) have median IoU 0.35, and
  116 are between 0.4 and 0.5. **82% of these predictions are smaller than the GT**, with a
  median area ratio of 0.46 (TP: 1.04). The main error is partial coverage of a filament.

**Folds by time:** 28% of validation images have a train image within 1 day and 55% within
2 days. **For the test images it is 36% and 60%.** So the random folds match the test
situation. Folds by ISO week would be harder than the test. **Keep the random folds.**

## 14. Plan: seed refiner (instead of a new detector)

Reason: our errors are mostly partial coverage, not merges. A refiner that re-segments a crop
around each candidate at full resolution targets that error. It reuses our U-Nets as the
detector, so it needs no new package (YOLO/ultralytics is AGPL).

1. Candidates: connected components of the out-of-fold A + 2048 maps (pp_v1, maybe a lower
   threshold for recall). Out-of-fold maps give realistic seeds for every training image.
2. Training sample: a square crop (bounding box + 50% margin, resized to 384) at 2048
   resolution. Input = image crop, seed mask of the candidate, CLAHE crop. Target = the
   union of the GT filaments that the candidate overlaps, for one annotator drawn at random.
3. Model: U-Net r34, 3 channels in. 5 folds, trained on the candidates of the other folds.
4. Inference: run the refiner on every candidate, paste the masks back, give contested
   pixels to the larger mask, then apply min area.
5. Judge on folds 0 and 1 against A + 2048 with the 0.007 bar. Cost: about 1 day of code,
   about 2-3 GPU hours.

**Free version of the refiner fails (pp_v3, pair maps, folds 0+1, baseline 0.4265):** growing
each mask by 2-12 px gives at most +0.001 with a floor of 0.3 and loses up to 0.14 without a
floor. The missing parts of near-miss filaments have probability below 0.3.

**Near misses by annotator:** 16% of all GT filaments end as near misses. The rate is 6-30%
per annotator and does not rise with the annotator's outline size (correlation -0.26 over 21
annotators). So near misses look like a model weakness, not a drawing style. **Build the seed
refiner** (section 14).

## 15. Forum findings (supervisor check 2026-10-31 02:25 UTC, plus CLI reads)

- The leaderboard ranks by PQ. **The top 3 are chosen by the jury with the rubric**, not by PQ
  alone (thread 753665). The rubric (section 3): PQ, the Dice and IoU distributions, the
  counts of one-to-many and many-to-one matches, the look of the shapes, pipeline
  description, code quality and efficiency.
- The host wants fine details such as barbs when they are visible (753665).
- **At inference, only the test H-alpha image may be used**: no file-name metadata, no
  neighbouring frames, no external labels such as MAGFiLO (753665). Our pipeline uses the
  file name only to build `filament_id`. Checked 2026-10-31.
- An ensemble is fine. The whole code, configs and weights go in one repo, and the demo must
  run completely on a different system (753665).
- Unlabeled H-alpha from KSO, Meudon or BBSO may be used for self-supervised pretraining,
  if it excludes the test images and observations within +/- 1 hour of them (748498).
- **The test PQ has "some subtle differences" from the self-evaluation notebook**, which the
  host will not reveal (753816). This explains part of our steady local-to-public gap of
  about 0.07. The host also will not say how multi-annotator test images are scored (753043).

Repo plan from this:
- Weights: 10 first-stage models are 747 MB in fp32 (each file under 100 MB). Store them in
  fp16 for the release (about 375 MB), plus the refiners if they pass.
- The demo must run without our GPU: make inference choose CUDA when available, else CPU.

## 16. Five-fold result of the 1024 + 2048 pair (2026-10-31)

| Fold | A (1024) | 2048 alone | A + 2048 |
|---|---|---|---|
| 0 | 0.4104 | 0.4154 | 0.4199 |
| 1 | 0.4212 | 0.4312 | 0.4336 |
| 2 | 0.4060 | 0.4222 | 0.4192 |
| 3 | 0.4062 | 0.4225 | 0.4229 |
| 4 | 0.4222 | 0.4310 | 0.4363 |
| **mean** | **0.4132** | **0.4245** | **0.4264** |

The pair beats A on every fold (mean +0.013). All 10 models (5 folds x 2 resolutions) are
trained. The 10-model mean of maps is the next submission.

Submission 57833984 (10-model mix, pp_v1): **public 0.36**, the same as the 4-model mix. The
public board cannot resolve a gain of about 0.01.

**Seed refiner, fold 0** (trained on the candidates of folds 1-3, `src/refiner.py`):
first stage 0.4201 -> **refined 0.4312 (+0.011)** at threshold 0.5 (set before the test).
TP 1049 -> 1039, FP 745 -> 628, FN 552 -> 562. The gain comes from rejecting false
candidates, not from recovering near misses. Fold 1 is next; accept only if it passes too.

**Seed refiner, fold 1** (trained on folds 0, 2, 3): 0.4334 -> **0.4417 (+0.008)** at
threshold 0.5; FP 561 -> 476. **The refiner passes the bar on folds 0 and 1.** Next: 5 final
refiners, each trained on the 4 other folds, with an out-of-fold score per fold, then a
refined 10-model submission.

**Input shift check for the refiner:** on 20 test images, the 10-model mean is only a little
smoother than one pair (probability inside candidates 0.925 against 0.931; uncertain band
+9.5%; same candidate count). Out-of-fold pair candidates and the 10-model test CSV match
(7.64 against 7.79 per image, median area 1,161 against 1,134). The shift is small, so the
refined file is submitted as planned.

## 17. Final seed refiners, five folds (2026-10-31)

| Fold | first stage | refined | gain |
|---|---|---|---|
| 0 | 0.4201 | 0.4274 | +0.0073 |
| 1 | 0.4334 | 0.4401 | +0.0067 |
| 2 | 0.4196 | 0.4192 | -0.0004 |
| 3 | 0.4233 | 0.4283 | +0.0050 |
| 4 | 0.4336 | 0.4425 | +0.0089 |
| **mean** | **0.4260** | **0.4315** | **+0.0055** |

The mean gain is below the single-fold bar (0.007) but positive on 4 of 5 folds (standard
error about 0.0016). The test file `mix10_refiner5.csv` (10-model map + mean of 5 refiners)
has 1,264 masks against 1,403 without the refiner. Submitted 2026-10-31 as information.
The unrefined 10-model blend stays the safe pick until the final selection.

**Provisional final selection (supervisor for Nhat, 2026-10-31):** (1) `mix10_refiner5.csv`
(refined 10-model file), (2) `mix_r34_1024_2048_f01234.csv` (unrefined 10-model blend, safe
pick). Revisit only if a stronger candidate appears before 2026-11-15.

## 18. Plan for 2026-11-01 to 2026-11-15

Close: 2026-11-15 06:00 UTC. The jury picks the top 3 with the rubric, so the repo, demo and
report matter as much as small PQ gains. GPU is shared with urbantwin (final phase from
2026-11-01, close 2026-11-15 00:00), so each GPU test must be small and pass on folds 0-1.

| Dates | Work | Check |
|---|---|---|
| 11-01 to 11-02 | d4 TTA on folds 0-1 (4 first-stage models, about 30 min GPU) | pair OOF gain > 0.007 on both folds |
| 11-01 to 11-02 | refiner: average 2 seeds per fold, folds 0-1 (about 20 min GPU) | gain over one refiner on both folds |
| 11-02 to 11-03 | resnet50 at 1024, fold 0 (about 35 min GPU), as a third ensemble member | pair or trio gain > 0.007 |
| 11-03 to 11-05 | apply what passed to all 5 folds; one submission per change | five-fold OOF and public score |
| 11-04 to 11-07 | repo: `filament_seg` package, CPU fallback, fp16 weights, one demo notebook that writes the CSV from the weights, README, licence | the notebook runs in a fresh venv on CPU and gives the same CSV (sha256) |
| 11-06 to 11-09 | figures with the official plots: IoU and Dice distributions, 1:n and n:1 counts, overlays of good and bad cases; runtime per image (GPU and CPU) | figures in `report/figures/` |
| 11-07 to 11-10 | 4-page report draft in the Overleaf template (method, validation, ablations, errors, efficiency) | draft PDF for Nhat |
| 11-11 to 11-12 | full dry run of the demo on a clean machine or container | identical CSV |
| **by 11-13** | **tell the supervisor: Nhat must make the repo public and send the Google form** | message sent |
| 11-14 | final selection confirmed; NOTES.md complete | |

Repo size: 10 first-stage models (fp16 about 375 MB) + 5 refiners (fp16 about 245 MB) =
about 620 MB, every file under 100 MB. If that is too large for the remote, keep the
refiners only in the selected file's configuration and document it.

## 19. Licences (exact wording, read 2026-10-31 with `kaggle competitions pages list --content`)

Copy of all pages: `research/ref/pages/pages.json`.

- **Rules 5a, Winner License:** "You hereby license and will license your winning Submission
  and the source code used to generate the Submission under an Open Source Initiative-approved
  license ... that in no event limits commercial use of such code or model containing or
  depending on such code." Exception: "In the event that input data or pretrained models with
  an incompatible license are used to generate your winning solution, you do not need to grant
  an open source license in the preceding Section for that data and/or model(s)."
- **Rules 5b:** the sponsor may ask for a detailed description that lets someone reproduce the
  approach (architecture, preprocessing, loss, training details, hyper-parameters) and a link
  to a code repository with complete instructions.
- **Data Access and Use:** "You may access and use the Competition Data for non-commercial
  purposes only, including for participating in the Competition and on Kaggle.com forums, and
  for academic research and education."
- **Open-Access Policy:** the repo "must be made publicly accessible immediately upon the close
  of the competition and must remain publicly accessible at least until when the winners are
  announced." Code quality (modularity, reproducibility, documentation) is judged.

Decision (supervisor for Nhat, 2026-10-31): **code under MIT.**
Proposal for Nhat, to send with the 11-13 message: **weights under MIT too.** Reason: the
exception covers input data and outside pretrained models, not the weights we trained; a
non-commercial licence on our weights would break 5a. The README must say that the
competition data is not included and is licensed for non-commercial use only, and that the
encoders start from ImageNet weights (via segmentation-models-pytorch). Nhat decides the
weights licence when he makes the repo public.

Submission 57843779 (`mix10_refiner5.csv`): **public 0.37**, up from 0.36 for the unrefined
10-model blend. This agrees with the out-of-fold gain of the refiner (+0.0055).

**Refiner seed averaging (GPU test 1 of 3) fails:** fold 0 s0 0.4274, s1 0.4306, average
0.4291; fold 1 s0 0.4401, s1 0.4383, average 0.4403. No gain over the mean of single seeds.
Refiner seed noise is about 0.002-0.003. Keep one refiner per fold.

**d4 TTA (GPU test 2 of 3) fails:** pair maps fold 0 0.4199 -> 0.4237, fold 1 0.4336 ->
0.4305. Keep the horizontal flip, which is 4 times cheaper.

**resnet50 at 1024 (GPU test 3 of 3) fails the bar, fold 0:** r50 alone 0.4167 (r34 alone
0.4104); pair A + 2048 0.4199; trio A + 2048 + r50 0.4238 (+0.004); pair r50 + 2048 0.4259
(+0.006, best of 3 combinations). Not run on fold 1.

**All three approved GPU tests failed** (refiner seeds, d4 TTA, resnet50). The selected
pipeline stays: 10-model 1024 + 2048 map + 5 seed refiners.

**Pipeline frozen (supervisor for Nhat, 2026-10-31, option A):** 10-model 1024 + 2048 map +
5 seed refiners. No more PQ tests. The three failed tests go into the report's ablation
section. Remaining work: demo, weights, figures, report.

**CPU fallback (2026-10-31):** `src/predict.py` and `src/refiner.py` run on CUDA when it is
available, else on the CPU in float32. One fold 0 image on CPU (4 threads): 1024 model
1.8 s, 2048 model 8.8 s, refiner 1.8 s; the masks match the GPU maps within 4 pixels
(largest probability difference 0.10, from float16 on the GPU). The full pipeline on CPU
needs about 1 minute per image (about 3 hours for 180). So a CPU run gives nearly the same
CSV, not the same sha256; only a GPU run can match the submitted file exactly.

**Release weights reproduce the selected file exactly (2026-10-31):** `scripts/export_weights.py`
writes 15 models (10 first-stage + 5 refiners) to `release/weights/` (702 MiB, each file
under 50 MiB; convolution weights float16, BatchNorm float32). A GPU rebuild from these
weights gives `mix10_refiner5.csv` with the same sha256 (`e66c87e6...`). GPU inference is
deterministic. **Open for Nhat:** how the weights are published (CLAUDE.md forbids committing
weights; the competition requires them in the repo).

## 20. Report figures (2026-11-01)

`scripts/oof_predictions.py` (systemd unit, 12 min GPU) writes out-of-fold predictions of the
selected pipeline to `data/oof/`. `scripts/report_figures.py` draws the organisers' own plots
(`scripts/official_plots.py`, verbatim) into `report/figures/` and checks our PQ against the
official `get_pq_score`. Pooled over all 707 images:

| | PQ | SQ | RQ | TP | FP | FN | GT with 2+ preds | pred with 2+ GT |
|---|---|---|---|---|---|---|---|---|
| first stage | 0.4258 | 0.678 | 0.628 | 5,329 | 3,437 | 2,870 | 394 | 257 |
| refined | 0.4312 | 0.676 | 0.638 | 5,250 | 3,022 | 2,949 | 371 | 281 |

The refiner removes 415 FP and reduces one-to-many cases, **but raises many-to-one cases**
(257 -> 281): a refined mask more often covers two annotated filaments. The report must say
this, because the rubric counts both. Mean IoU of overlapping pairs is unchanged (0.570 ->
0.571). The median example image shows one annotator marking filaments the others skip.

## 21. Release status (2026-11-01)

Done:
- `notebooks/demo.ipynb`: rebuilds the selected CSV from `release/weights/` (GPU: same sha256;
  CPU: mean IoU 0.997 on 2 images).
- Runtime (`scripts/runtime.py`, `report/tables/runtime_*.csv`): 1.5 s per image on an RTX
  3070 (2048 models 0.83 s), 79 s on 4 CPU threads.
- `report/report.tex` (IEEEtran conference, 3 pages, tectonic): pipeline diagram, per-fold
  table, official IoU/Dice/matching plots, zoomed examples, ablations, error analysis,
  runtime, reproducibility.
- The verbatim copies of the organisers' notebook name their source; README credits them.

Open:
- The official Overleaf template (read-only link) holds content instructions; it needs a
  browser. Asked the supervisor (2026-11-01).
- The licence of the organisers' notebook (Kaggle page; usually Apache 2.0). Asked the
  supervisor.
- For Nhat with the public release: weights licence (proposal: MIT) and how weights are
  published (CLAUDE.md forbids committing weights; the competition requires them).

**Uncertainty (2026-11-01, `scripts/report_uncertainty.py`):** image bootstrap of the pooled
out-of-fold PQ: first stage 0.4258 [0.4163, 0.4352], refined 0.4312 [0.4218, 0.4402],
**paired gain +0.0054 [0.0032, 0.0078], positive in all 2,000 samples.** The interval covers
image sampling only, not refiner seed noise (about 0.002-0.003). Fold standard deviations
(0.007-0.010) are larger than the gains, because folds differ in difficulty.

**Template (supervisor, 2026-11-01):** the official template is `acmart` sigconf, not
IEEEtran, with fixed black text, a fixed subtitle, a fixed abstract opening and Introduction
opening, required Methodology and Evaluation content, and a fixed Acknowledgment. Limit 4
pages without Acknowledgment and References. Waiting for the exact footnote, `main.bib` and
`preamble.tex`.

## 22. Forum check and new tests (2026-11-08)

Forum (read with `kaggle competitions topic-messages`, copies in `research/ref/forum/`):
- **755907:** the host does not need checkpoints, only the final model, and "you can upload your
  model anywhere you want. As long as it is accessible through your demo notebook without any
  challenge". The committee will not download models by hand.
- **755241:** the spine, bbox and area fields may be used as training signals; the model must
  not need them at inference.
- **755906:** the duplicate-mask hack works only in the self-evaluation notebook; "the
  leaderboard handles that".
- **749526:** efficiency is a qualitative filter, not a competing factor.
- **747960:** both selected submissions must be reproducible from the repository.
- **748045, 749225:** copying training labels to GONG frames a few minutes away is allowed for
  training; the GONG archive blocks fast downloads.
- **744765, 745880:** honest teams report local PQ 0.46-0.49 and public 0.40-0.41. Their posted
  counts show far fewer FP than ours (for example TP 863, FP 293, FN 395). Focal + Dice loss
  is named as useful.

**Local GPU down since 2026-11-06 17:20** (RTX 3070 driver crash, NVRM GSP RPC errors). Needs a
reboot by Nhat. Work moved to senai-0 GPU 1: folder
a folder on the remote server, conda env `filament-seg`, `run.sh` with a shared lock,
owner line and a 4 GB cap. GPU 1 is shared by about 10 jobs: 77 s per 1024 epoch.

Tests (decided with the supervisor 2026-11-08):
- **A:** focal + Dice loss (`--loss focal_dice`), 1024, fold 0, against A seeds 0.4104 / 0.4121.
- **B:** spine supervision (`--spine`, second output channel, weight 0.5), same comparison.
- Bar: +0.007 on folds 0 and 1. **Cutoff: if neither passes by 2026-11-12 00:00 UTC, stop and
  keep the frozen pipeline.** A retrain (5 x about 80 min) and the demo check must end by
  2026-11-14.

**Weights plan (for the 755907 rule):**
- One archive of the 15 model folders in `release/weights/` (about 700 MB), attached as a
  release asset to the public GitHub repository (GitHub allows up to 2 GB per asset).
- The demo notebook downloads the archive from a fixed URL when `release/weights/` is missing,
  checks its sha256, and unpacks it. No manual step.
- Nhat creates the repository and the release (his account); then the URL is fixed in the
  notebook and README. Nhat still decides the weights licence (proposal: MIT).

**Local GPU (2026-11-08, Nhat):** "no need gpu on this box". Do not ask for a reboot. All GPU work runs on senai-0 GPU 1.

**Results of the 2026-11-08 tests:**
- **A, focal + Dice: fails.** Fold 0 0.4058 (threshold 0.5), best 0.4063 at 0.6, against A seeds
  0.4104 / 0.4121.
- **B, spine supervision: fails.** Alone: fold 0 0.4166, fold 1 0.4244 (+0.005, +0.003 over A).
  Inside the pair with the 2048 model: +0.0004 and +0.001. The 2048 model already gives what the
  spine adds.
- senai-0 GPU 0 was idle; Nhat allowed short jobs there (2026-11-08). On GPU 0 a 1024 epoch
  takes 16-18 s (GPU 1, shared: 75 s). `run0.sh` uses its own lock file and an owner line.

**Decision (Nhat, 2026-11-08): stop the score work and keep the frozen pipeline** (10-model map
+ 5 seed refiners, public 0.37). The refiner-threshold check (expected gain under 0.003) was
not run. The report now lists focal loss and spine supervision under "What did not help".
senai-0: our `data/` copies were deleted after the results were copied back; the code,
`run.sh`, `run0.sh` and the conda env `filament-seg` stay in
a folder on the remote server (3.6 MB) in case GPU work is needed again.

Remaining steps (all Nhat's): author city/country/institution in the report; read
`report/acm/main.pdf`; create the public GitHub repository; upload
`release/filament-seg-weights.tar` as a release asset (then the URL goes into the demo and
README); weights licence (proposal MIT); confirm the 2 final selections on Kaggle; send the
Google form before 2026-11-15 06:00 UTC.

## 23. Test C: learned candidate filter (design fixed before the first run, 2026-11-08)

Nhat (2026-11-08): "we want to improve the score first before doing anything". Score work
continues; the supervisor's cutoff stays (nothing passes by 2026-11-12 00:00 UTC -> freeze;
final files ready by 2026-11-14).

Idea: a second model decides which refined candidates to keep (a forum team with our design
reports "a learned filter"). Our FP per image is about 5.3 against about 2.8 in posted honest
results.

Design:
- Data: the 5,081 out-of-fold refined candidates (`data/oof/refined.csv`) of 694 images.
  Features (`scripts/candidate_features.py`): shape (area, bbox, elongation, solidity),
  first-stage probability (mean, max, 90th percentile, ring), darkness (inside, ring, contrast,
  all relative to the disk median), radial position on the disk, candidates per image.
  Label: fraction of the image's annotators matched at IoU > 0.5.
- No leakage: candidates and maps come from models that did not train on the image (first-stage
  fold models and `refiner_v2_f<k>`). Features use only the image and these outputs.
- Model: gradient boosting regression on the label, LightGBM, fixed default-like settings.
- **Nested evaluation:** for each outer fold k, train on the other 4 folds; choose the keep
  threshold by an inner 4-fold out-of-fold PQ on those 4 folds; apply model and threshold to
  fold k. Score = pooled PQ of the filtered candidates over the 5 outer folds.
- **Bar:** pooled PQ must beat the refined pipeline (0.4312) by more than 0.007, and the gain
  must be positive on at least 4 of 5 folds.

**Test C result: passes.** Nested CV, pooled PQ 0.4312 -> **0.4449 (+0.014)**; per fold +0.018,
+0.006, +0.016, +0.013, +0.013 (all positive). Inner thresholds 0.30-0.40. 778 of 5,081
candidates removed. Top features: probability in the ring around the candidate, mean
probability, contrast, radial position.

Plan for the final filter (to keep the training feature distribution at test time): the
candidate features use one 1024 + 2048 pair map. At test there are 5 such pair maps (one per
fold). Each test candidate is scored with the features of each pair map; the 5 scores are
averaged. One LightGBM model is trained on all 5,081 candidates; keep threshold 0.35 (the
median of the 5 inner choices).

**Filtered submission (2026-11-08):** `mix10_refiner5_filter1.csv`, made on senai-0 GPU 0 (Quadro RTX
8000) from `release/weights` and `data/models/filter_v1` at commit 925f858. Supervisor check passed:
the filter removed 15.2% of test candidates (15.3% out of fold) and the feature quantiles match.
1,072 masks on 177 images. Uploaded with its sidecar.

**Reproduction across GPU types:** the same pipeline on the Quadro RTX 8000 does not give a
byte-identical CSV to the RTX 3070 file (float16 differs between chip generations), but the same
1,264 masks with mean IoU 0.99984. The demo's sha256 check therefore holds only on the GPU type that
made the file. The new file was made on the Quadro.

Submission 58113312 (`mix10_refiner5_filter1.csv`): **public 0.38** (was 0.37). Rank 383 of 1,421
(was 477). The filter is now part of the selected pipeline.

## 24. Test D: filter v2 with more features (design fixed before the run, 2026-11-08)

Extra features (all available at test time, out of fold in training): mean probability of the
1024 model and of the 2048 model separately, mean absolute difference between them inside the
candidate, edge strength along the border (Sobel magnitude / disk median), intensity spread in
the ring, 10th percentile of intensity inside. Same LightGBM settings and the same nested CV as
Test C. **Bar against filter v1: pooled gain > 0.005 and better on at least 4 of 5 folds**
(stricter than 0.007 against no filter, because this is a second variant on the same folds).

**Test D result: fails.** Filter v2 pooled 0.4445 against v1 0.4449 (-0.0004), better on 1 of 5
folds. Keep filter v1. The supervisor allowed one feature round only, so the filter is final.

**Final selection (supervisor for Nhat, 2026-11-08):** (1) `mix10_refiner5_filter1.csv`
(public 0.38, out-of-fold nested 0.4449), (2) `mix10_refiner5.csv` (public 0.37) as the safe
pick. Nhat confirms both on the Kaggle page. Both come from the same code; the filter is one
optional stage (`--filter`).

**Release check (2026-11-08):** the demo notebook (`USE_FILTER = True`, all 180 images) on the
Quadro RTX 8000 rebuilt `mix10_refiner5_filter1.csv` with the same sha256 (`9fac0e35...`), from
the release weights (archive `ec3d8df8...`, 16 folders including `filter_v1`). 619 s on a shared
GPU. README, demo and report describe the filter and the per-GPU sha256. senai-0: large files
deleted again; code stays.

## 25. Test E: more candidates for the filter (design fixed before the run, 2026-11-11)

Nhat (2026-11-11): "keep improving the score before the deadline". Plan: tests until about
2026-11-13 12:00 UTC, then freeze and update test file, demo, weights and report by 2026-11-14.

Reason: after the filter, FN (3,220) exceed FP (2,026). Many missed filaments never become
candidates, because the first stage keeps only regions above probability 0.5 and 300 px.
With a learned filter that rejects bad candidates, a lower bar may now pay off.

Design (one variant): first-stage threshold **0.4**, min area **150**, merge_dist 15; the same
`refiner_v2_f<k>` (refined masks also kept at >= 150 px); features and labels as in Test C; a
new filter trained on these candidates; the same nested CV (inner threshold choice).
**Bar: pooled PQ > 0.4449 + 0.005, and better than the current final pipeline on at least
4 of 5 folds.**

**Result (2026-11-11, 19:23 UTC): FAIL.** 5,689 candidates (current: fewer, at 0.5/300).
Refined pooled PQ drops to 0.4144; after the new filter, pooled PQ is **0.4449**, the same as the
current final pipeline (gain +0.0000, bar +0.005). Per fold (new vs current): 0.4470 vs 0.4455,
0.4458 vs 0.4464, 0.4387 vs 0.4354, 0.4375 vs 0.4417, 0.4560 vs 0.4558. Better on 3 of 5
folds, two of them by less than 0.002. The filter removes the extra candidates, but the extra
TPs do not pay for the extra FPs it keeps. **Decision: freeze.** The final selection stays
mix10_refiner5_filter1 and mix10_refiner5. No second try (supervisor condition).

## 26. Why we are far from the top (investigation 2026-11-11, 19:40-20:30 UTC)

Nhat (2026-11-11): "our distance is too far from the top, spend time to investigate".

Leaderboard (19:44 UTC, 1,494 teams; copy in the job tmp folder):
- Top 0.70; 10 teams at 0.60-0.64; 107 teams at exactly 0.55; 135 teams at 0.42 or more.
- Since 2026-11-08, 14 teams jumped to 0.54-0.55, several from 0.28-0.39 in a few days,
  for example 0.30 -> 0.55. That jump matches the copied 0.55 CSV (section 9).
- Below the plateau: one team at 0.42, 22 at 0.41, 71 at 0.40, 144 at 0.39. We are at 0.38.

Forum (new threads 757628, 755770, 753816, 753665, 755155; copies in `research/ref/forum/`):
nothing new on method. 753816: overlapping predictions are rejected by the scorer.

New public notebooks (15 read, copies in `research/ref/nb/`): the best honest one is
huendao (10 public U-Nets at 2048, d4 TTA): **OOF 0.4523, public 0.3945**. That is our level
(OOF 0.445, public 0.38). The 0.70 notebooks are copies of the Dice-era phuongncn code. Other
honest scores: YOLO26s-seg 0.36, HDJoJo YOLO26x-seg 0.38, a CSV vote 0.40.

Checks on our own out-of-fold output (final pipeline):
- **Rasterisation:** the JSON `area` equals the pycocotools area on 400 of 400 polygons, so the
  host uses pycocotools. cv2.fillPoly gives masks 10% larger. This does not explain the gap.
- **Mask size:** growing or shrinking every mask by 1 px drops PQ from 0.4448 to 0.418 and
  0.417. Our mask size is already right.
- **Metric variants:** pooled 0.4448, first annotator only 0.4374, mean per entry 0.4359,
  mean per image 0.4302. No variant explains the gap of 0.065 to public.
- **Misses:** 3,220 FN. 1,867 have no overlap at all with any prediction; 1,173 overlap at
  IoU 0.2-0.5. Recall is 0.32 for GT under 521 px and about 0.69 above 1,900 px.
- **Human agreement:** on the 296 images with 2 or 3 annotators, one annotator scored against
  the others gives **PQ 0.340** (SQ 0.634, RQ 0.536). Our model against the same targets
  gives **0.443**. The model already agrees with the annotators better than they agree with
  each other. Label noise sets the ceiling.

Conclusion: the distance to the top is not a real model gap. Scores of 0.55 and higher come
from the copied CSV, Dice-era code or scorer defects (duplicate masks, empty files). The
honest frontier is about 0.40-0.42 public, which is +0.02 to +0.04 above us. The best honest
public notebook is at our level. d4 TTA (2026-10-31), focal loss, spine head, filter v2 and
test E all failed. One untested idea is left: normalising each image over the solar disk
(we only divide by 255 and use brightness/contrast augmentation).

### 26.1 Evidence on the hidden 0.55 file (2026-11-11)

Notebook a public "0.55" notebook (49 votes, last run 2026-09-03):
- Cell 4 trains a small U-Net for 5 epochs. No later cell calls the model.
- Cell 5 holds `CHAMPION_PAYLOAD` (1,690 lines of base85 text, comment "verified 0.55 on
  Leaderboard"). `lzma.decompress(base64.b85decode(...))` turns it into a CSV of 1,342 masks.
- Cell 6 writes those masks to `submission.csv` by test file name. Images that are missing get a
  48 x 48 square.
- Shape check (600 masks each): hand polygons (train GT) have 0% multi-part masks and 13.8
  vertices per 100 px of boundary. Our predictions have 14% and 11.6. The hidden file has 9% and
  11.2. **The hidden file looks like model output, not like hand labels.**
- Against our test submission: 60% of its masks match one of ours at IoU > 0.5; PQ of ours with
  the hidden file as GT is 0.497. It has 7.5 masks per image (ours 6.1).
- Its origin stays unknown: the notebook does not show the model that made it.

### 26.2 Papers that compare models on MAGFiLO (search 2026-11-11)

Nhat: use published MAGFiLO comparisons to choose an architecture (no MAGFiLO labels used by us).
- **EdgeAttNet (arXiv 2509.02964, 2025)** is the only model comparison on MAGFiLO found. It is
  semantic segmentation (mIoU), not PQ. Split 1,295 / 45 / 99 of all MAGFiLO images, so its
  weights have very likely seen our test images: never use its weights or data.
  Table II (mIoU pairwise / multiscale): U-Net 0.5724 / 0.5848; U-Net + MHSA bottleneck
  0.5856 / 0.6000; + positional encoding 0.6200 / 0.6601; EdgeAttNet (edge map added to Q and K
  of the bottleneck attention) 0.6451 / 0.7032. Baselines are trained from scratch, without
  augmentation or pretrained weights, so the gain over our pretrained, augmented r34 U-Net is
  unknown. Preprocessing: disk mask, radial flattening, 3x3 Gaussian blur, CLAHE.
  Code: https://github.com/dasjar/EdgeAttNet (paper claim, not checked).
- **MORDEN (arXiv 2607.24525)** and **Flat U-Net (arXiv 2502.07259)** use CHASE and BBSO data,
  not MAGFiLO. MORDEN ablation: 5x5 kernels +0.033 IoU, Instance Norm +0.018; focal loss.
- **Diercke et al. 2024 (A&A 686)**: YOLOv5 boxes, then a semi-supervised U-Net; no MAGFiLO
  comparison.
Transferable and untested by us: input preprocessing (disk mask, limb-darkening flattening,
CLAHE), which matches the open disk-normalisation idea.

## 27. Test F: input cleaning (design fixed before the run, 2026-11-11)

Nhat (2026-11-11): "yes, start the input cleaning test". Idea from EdgeAttNet (section 26.2).
`src/preprocess.clean_image`: disk mask, radial flattening by 64 ring medians, 3 x 3 Gaussian
blur (sigma 0.7), CLAHE (clip 2, 8 x 8). Applied at 2048 before the resize, in training
(`build_cache --clean`, `train --clean`) and inference (`config.json` "clean").
Run: model A setup (r34, 1024, 512 crops, soft target, 60 epochs, seed 0), folds 0 and 1, on
senai-0 GPU 0 (idle; short jobs allowed by Nhat on 2026-11-08). Scored with pp_v1.
**Bar: alone, beat A by more than 0.007 on both folds (A: 0.4104 and 0.4212). If it passes,
also check the pair with the 2048 model (pair: 0.4199, 0.4336): it must gain on both folds,
since the spine test gained alone and not in the pair.** No second variant.

**Result (2026-11-11, 20:57 UTC): FAIL.** Alone (pp_v1): fold 0 0.4067 vs A 0.4104 (-0.004),
fold 1 0.4197 vs 0.4212 (-0.002). In the pair with the 2048 model: 0.4196 vs 0.4199 and 0.4324
vs 0.4336. The cleaning removes no error that the U-Net does not already handle; our
brightness/contrast augmentation and the pretrained encoder probably cover it. Training took
about 32 minutes for both folds side by side on senai-0 GPU 0. The code stays (option off by
default); the final pipeline does not change.

### 26.3 Why the top is at 0.70 (2026-11-11)

- The top score of 0.70 has been on the board since 2026-10-28 or earlier (section 2). Kaggle shows
  each team's best score, not when they made it.
- The public "LB 0.70" notebook (Fususu/phuongncn) says its CSV scored 0.70 on 2026-08-05, in
  the Dice era. Forum users who ran it scored about 0.35 PQ (745880). Its text names "public
  MAGFiLO/test-overlap labels", "contamination mirrors" and "test-label hashes". The author
  knew that labels for test images are public in MAGFiLO.
- **Leak test on train:** submitting the true labels of one annotator per image, scored against
  all annotators, gives **PQ 0.758**. A partial leak or a mix with a model gives 0.60 to 0.70,
  which is the range of the top 10. The best honest model is at about 0.45 locally.
Conclusion: the 0.60-0.70 scores fit leaked MAGFiLO test labels or Dice-era scores; no honest
method in the forum, the notebooks or the papers comes near them. Not proven for any team.
- **Correction (board lookup, 2026-11-11):** team Fususu (user phuongncn, author of the "LB 0.70"
  notebook) now shows **0.36**, last submission 2026-08-05. So the board was re-scored with PQ
  after the switch, and Dice-era scores do not stand. The current 0.60-0.70 are PQ scores; the
  leak explanation remains, the Dice explanation is withdrawn.

### 26.4 Forum threads on leaks and the top scores (read 2026-11-11; copies in `research/ref/forum/`)

- **728062 (2026-07-21, 8 votes):** 139 of the 180 test images are byte-identical to images in
  the public MAGFiLO 1.0 release, which has full labels. Another user: all 180 test file names
  appear with labels in MAGFiLO, so the public/private split does not separate leaks. The poster
  replaced model output with MAGFiLO labels and gained 0.73 -> 0.82 (Dice era). Host: the
  leaderboard is "only a preliminary way of filtering"; "Submissions without a decent strategy
  will NOT be rewarded in any ways."
- **729396:** host: the ground truth "should be limited to what is provided through this
  competition. Otherwise, the submissions will be disregarded."
- **736543:** host: no information from external sources "leaking the ground-truth data of the
  test set into your final submission".
- **737177:** a user reports that test labels leaked at the start, some teams briefly had 1.00;
  host: "Any score higher than 0.35 is of great value"; aim is quality "without any data leakage".
- **743958:** a user names a public "0.55" notebook as "doing exactly
  that" (leaking) at exactly 0.55. Host: "we take a score of 0.41 very seriously".
- **742143:** host: "we are well equipped to distinguish a few disingenuous scores from the rest".
- **724503 (host, 2026-10-23):** review of the top 250 submissions; "We can easily identify
  submissions which used 'shortcuts'"; **"We will clean the leaderboard after the deadline, so
  that credit is given to genuine submissions."** 2026-10-26: "We decided to clear the board after
  the deadline." 2026-07-16: AI tools are allowed, but prize-eligible participants "must fully
  understand every component" and be able to explain, justify and reproduce each step.
- 733912: the board was re-scored with PQ in August (matches the Fususu 0.36 finding).
