"""Input cleaning of a GONG H-alpha image, after EdgeAttNet (arXiv 2509.02964), NOTES.md 26.2.

1. Disk mask: pixels outside the solar disk become 0.
2. Radial flattening: each pixel is divided by the median intensity of its ring (64 rings
   from the centre to the limb), so the darkening towards the limb is removed.
3. A 3 x 3 Gaussian blur (sigma 0.7).
4. CLAHE (clip 2.0, 8 x 8 tiles) for local contrast.

The output is uint8 like the input, so the cache, training and inference code do not change.
"""

import cv2
import numpy as np

from src.candidate_filter import disk_geometry

RINGS = 64


def clean_image(image: np.ndarray) -> np.ndarray:
    cy, cx, radius, disk_median = disk_geometry(image)
    yy, xx = np.indices(image.shape, dtype=np.float32)
    ring = np.minimum((np.hypot(yy - cy, xx - cx) / radius * RINGS).astype(np.int32), RINGS)
    inside = ring < RINGS
    # Ring medians from every 4th pixel are exact enough and 16 times faster.
    sub_ring, sub_image = ring[::4, ::4].ravel(), image[::4, ::4].ravel()
    profile = np.array([np.median(sub_image[sub_ring == k]) if (sub_ring == k).any() else disk_median
                        for k in range(RINGS)], np.float32)
    flat = image.astype(np.float32) / np.maximum(profile[np.minimum(ring, RINGS - 1)], 1.0) * disk_median
    flat = np.where(inside, np.clip(flat, 0, 255), 0).astype(np.uint8)
    flat = cv2.GaussianBlur(flat, (3, 3), 0.7)
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(flat)
