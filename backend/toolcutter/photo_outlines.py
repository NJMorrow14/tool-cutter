"""Filter image proposals into whole, non-duplicate tool silhouettes."""
import cv2
import numpy as np


def select_silhouettes(proposals, mm_per_px, marker_mask=None, min_area_mm2=200):
    selected = []
    for proposal in sorted(proposals, key=lambda p: p['area'], reverse=True):
        mask = np.asarray(proposal['segmentation'], dtype=bool)
        area = int(mask.sum())
        if area * mm_per_px ** 2 < min_area_mm2 or area > mask.size * .65:
            continue
        # Reject calibration paper, not just its black printed squares.
        if marker_mask is not None and np.count_nonzero(mask & marker_mask) > .25 * area:
            continue
        # Fill internal holes for a foam pocket before checking nested parts:
        # a tape's centre, logo, and buttons must not become extra pockets.
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        filled = np.zeros(mask.shape, np.uint8)
        cv2.drawContours(filled, [contour], -1, 1, cv2.FILLED)
        filled = filled > 0
        if any(np.count_nonzero(filled & existing) / max(1, filled.sum()) > .8 for existing in selected):
            continue
        selected.append(filled)
        if len(selected) == 40:
            break
    return selected
