"""Turn a full-size probability map into one binary mask per filament."""

import cv2
import numpy as np


def probability_to_instances(prob: np.ndarray, threshold: float = 0.5, min_area: int = 200,
                             merge_dist: int = 0,
                             seed_threshold: float | None = None,
                             min_mean_prob: float | None = None, grow: int = 0,
                             grow_floor: float = 0.0) -> list[np.ndarray]:
    """Threshold the map, group the pixels into filaments and drop the small ones.

    Args:
        prob: (2048, 2048) float map with values in [0, 1].
        threshold: pixels above this value are filament.
        min_area: filaments with fewer pixels than this are removed.
        merge_dist: regions closer than this many pixels become one filament. The pixels
            of the mask do not change; only the grouping does. 0 turns it off.
        seed_threshold: if set, keep only the regions that hold at least one pixel above
            this value (hysteresis).
        min_mean_prob: if set, drop filaments whose mean probability is below this value.
        grow: enlarge each kept filament by this many pixels, only into pixels that belong
            to no filament and whose probability is above ``grow_floor``. 0 turns it off.

    Returns:
        A list of (2048, 2048) uint8 masks that do not overlap.
    """
    binary = (prob > threshold).astype(np.uint8)
    grouping = binary
    if merge_dist > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (merge_dist, merge_dist))
        grouping = cv2.dilate(binary, kernel)
    n, labels = cv2.connectedComponents(grouping, connectivity=8)
    labels = labels * binary
    areas = np.bincount(labels.ravel(), minlength=n)
    keep = areas >= min_area
    keep[0] = False
    if seed_threshold is not None:
        seeded = np.zeros(n, bool)
        seeded[np.unique(labels[prob > seed_threshold])] = True
        keep &= seeded
    if min_mean_prob is not None:
        sums = np.bincount(labels.ravel(), weights=prob.ravel(), minlength=n)
        keep &= sums >= min_mean_prob * np.maximum(areas, 1)
    kept = np.flatnonzero(keep)
    if grow > 0 and len(kept):
        # Relabel the kept filaments 1..m, then grow them with a maximum filter.
        relabel = np.zeros(n, np.uint16)
        relabel[kept] = np.arange(1, len(kept) + 1)
        final = relabel[labels]
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1))
        grown = cv2.dilate(final, kernel)
        free = (final == 0) & (prob > grow_floor)
        final[free] = grown[free]
        return [(final == k).astype(np.uint8) for k in range(1, len(kept) + 1)]
    return [(labels == k).astype(np.uint8) for k in kept]
