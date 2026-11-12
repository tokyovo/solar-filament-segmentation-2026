"""Check the grouping options of src.postprocess on small synthetic maps."""

import numpy as np

from src import postprocess


def two_blobs(gap: int) -> np.ndarray:
    prob = np.zeros((200, 200))
    prob[50:60, 20:80] = 0.7
    prob[50:60, 80 + gap:140] = 0.7
    return prob


def test_defaults_give_connected_components():
    masks = postprocess.probability_to_instances(two_blobs(10), threshold=0.5, min_area=0)
    assert sorted(m.sum() for m in masks) == [500, 600]


def test_merge_dist_joins_close_fragments_without_changing_pixels():
    prob = two_blobs(10)
    masks = postprocess.probability_to_instances(prob, threshold=0.5, min_area=0, merge_dist=15)
    assert len(masks) == 1
    assert masks[0].sum() == (prob > 0.5).sum()


def test_merge_dist_keeps_far_fragments_apart():
    masks = postprocess.probability_to_instances(two_blobs(40), threshold=0.5, min_area=0, merge_dist=15)
    assert len(masks) == 2


def test_min_area_and_seed_threshold():
    prob = two_blobs(40)
    prob[52, 30] = 0.9
    assert len(postprocess.probability_to_instances(prob, 0.5, min_area=550)) == 1
    masks = postprocess.probability_to_instances(prob, 0.5, min_area=0, seed_threshold=0.8)
    assert len(masks) == 1 and masks[0][52, 30] == 1


def test_min_mean_prob_drops_weak_regions():
    prob = two_blobs(40)
    prob[50:60, 120:140] = 0.55
    masks = postprocess.probability_to_instances(prob, 0.5, min_area=0, min_mean_prob=0.65)
    assert len(masks) == 1 and masks[0][55, 30] == 1


def test_grow_enlarges_masks_only_above_the_floor():
    prob = np.zeros((100, 100))
    prob[40:60, 40:60] = 0.9
    prob[40:60, 60:70] = 0.3
    base = postprocess.probability_to_instances(prob, 0.5, min_area=0)[0]
    grown = postprocess.probability_to_instances(prob, 0.5, min_area=0, grow=3, grow_floor=0.2)[0]
    assert base.sum() == 400
    assert grown[50, 62] == 1 and grown[50, 36] == 0
    assert grown.sum() == 400 + 20 * 3
