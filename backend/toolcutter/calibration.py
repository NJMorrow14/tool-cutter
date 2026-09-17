"""Perspective rectification against a rectangle of known size (the mat / drawer / sheet)."""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import cv2
import numpy as np


def order_corners(points: Sequence[Sequence[float]]) -> np.ndarray:
    """Order 4 arbitrary corner points as TL, TR, BR, BL (image coordinates, y down)."""
    pts = np.asarray(points, dtype=np.float64).reshape(4, 2)
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    order = np.argsort(angles)  # counter-clockwise in math coords == clockwise on screen (y down)
    ring = pts[order]
    # rotate ring so the first point is the top-left (smallest x+y)
    start = int(np.argmin(ring.sum(axis=1)))
    ring = np.roll(ring, -start, axis=0)
    return ring


def edge_lengths_px(corners: np.ndarray) -> Tuple[float, float]:
    """Average horizontal and vertical edge lengths in pixels for TL,TR,BR,BL corners."""
    tl, tr, br, bl = corners
    horiz = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2.0
    vert = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2.0
    return float(horiz), float(vert)


def choose_px_per_mm(corners: np.ndarray, width_mm: float, height_mm: float, max_side: int = 2200,
                     min_ppm: float = 2.0) -> float:
    """Pick an output resolution: never far above the source resolution, capped by max_side."""
    horiz, vert = edge_lengths_px(corners)
    src_ppm = min(horiz / max(width_mm, 1e-6), vert / max(height_mm, 1e-6))
    cap_ppm = max_side / max(width_mm, height_mm)
    ppm = min(cap_ppm, max(src_ppm * 1.15, min_ppm))
    return float(max(ppm, min_ppm))


def rectify(img: np.ndarray, corners: Sequence[Sequence[float]], width_mm: float, height_mm: float,
            extra: Optional[Sequence[np.ndarray]] = None, max_side: int = 2200, already_ordered: bool = False):
    """Warp the quadrilateral `corners` to a top-down rectangle of width_mm x height_mm.

    Returns (warped, mm_per_px, homography, warped_extras). `extra` arrays (e.g. a height map)
    are warped with the same homography.
    """
    ordered = np.asarray(corners, dtype=np.float64).reshape(4, 2) if already_ordered else order_corners(corners)
    ppm = choose_px_per_mm(ordered, width_mm, height_mm, max_side=max_side)
    out_w = max(8, int(round(width_mm * ppm)))
    out_h = max(8, int(round(height_mm * ppm)))
    # Exact mm-per-px after rounding (uniform in x and y by construction of ppm; tiny residual ignored)
    mm_per_px = width_mm / out_w
    dst = np.array([[0, 0], [out_w, 0], [out_w, out_h], [0, out_h]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(ordered.astype(np.float32), dst)
    warped = cv2.warpPerspective(img, H, (out_w, out_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    extras_out = []
    for arr in extra or []:
        if arr is None:
            extras_out.append(None)
            continue
        extras_out.append(cv2.warpPerspective(arr, H, (out_w, out_h), flags=cv2.INTER_LINEAR,
                                              borderMode=cv2.BORDER_REPLICATE))
    return warped, float(mm_per_px), H, extras_out
