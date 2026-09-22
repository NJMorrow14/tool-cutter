"""Conservative visual registration on the drawer floor, excluding tool parallax."""
from __future__ import annotations

import cv2
import numpy as np


def warp_translation(array, dx, dy, border=0, interpolation=cv2.INTER_LINEAR):
    """Translate without numpy.roll's wraparound across the opposite drawer edge."""
    return cv2.warpAffine(array, np.array([[1, 0, dx], [0, 1, dy]], np.float32),
                          (array.shape[1], array.shape[0]), flags=interpolation,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=border)


def floor_alignment(source: dict, reference: dict, mm_per_px: float, max_shift_mm: float = 8.0):
    """Return a verified rigid source->reference correction, or None.

    Only floor features vote. Two-way descriptor matches, RANSAC, spatial spread,
    bounded motion, and improved image residual must all agree. No scaling/shear
    is applied to the metric scan and no speculative correction is propagated.
    """
    h, w = source['valid'].shape
    scale = min(1.0, 1000.0 / max(h, w))
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    if min(size) < 32:
        return None
    pitch = mm_per_px / scale

    def prep(frame):
        mask = frame['valid'].astype(np.uint8)
        height = frame.get('height')
        if height is not None:
            # Dilate the exclusion to include displaced tops as well as their footprints.
            raised = (np.nan_to_num(height, nan=0) > 2).astype(np.uint8)
            radius = max(2, round(8 / mm_per_px))
            mask &= 1 - cv2.dilate(raised, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2))
        gray = cv2.cvtColor(frame['color'], cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
        mask = cv2.erode(mask, np.ones((5, 5), np.uint8))
        return gray, mask

    a, am = prep(source)
    b, bm = prep(reference)
    overlap = am & bm
    if overlap.sum() < 1500:
        return None
    detector = cv2.SIFT_create(nfeatures=2000, contrastThreshold=0.025)
    ka, da = detector.detectAndCompute(a, am * 255)
    kb, db = detector.detectAndCompute(b, bm * 255)
    if da is None or db is None or min(len(ka), len(kb)) < 20:
        return None
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    forward = matcher.knnMatch(da, db, k=2)
    backward = matcher.knnMatch(db, da, k=2)
    reverse = {m.queryIdx: m.trainIdx for pair in backward if len(pair) == 2 for m, n in [pair] if m.distance < .7 * n.distance}
    pairs = [(m.queryIdx, m.trainIdx) for pair in forward if len(pair) == 2 for m, n in [pair]
             if m.distance < .7 * n.distance and reverse.get(m.trainIdx) == m.queryIdx]
    if len(pairs) < 20:
        return None
    src = np.float32([ka[i].pt for i, _ in pairs])
    dst = np.float32([kb[j].pt for _, j in pairs])
    affine, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                                ransacReprojThreshold=max(1.2, 0.7 / pitch), maxIters=2000)
    if affine is None or inliers is None:
        return None
    keep = inliers.ravel().astype(bool)
    if keep.sum() < 20 or keep.mean() < .6:
        return None
    src, dst = src[keep], dst[keep]
    if np.ptp(src, axis=0).min() * pitch < 20 or cv2.contourArea(cv2.convexHull(src)) * pitch**2 < 400:
        return None
    factor = np.hypot(affine[0, 0], affine[1, 0])
    angle = abs(np.degrees(np.arctan2(affine[1, 0], affine[0, 0])))
    if abs(factor - 1) > .005 or angle > 2:
        return None
    affine[:, :2] /= factor  # rigid only: metric dimensions remain unchanged
    affine[:, 2] = dst.mean(axis=0) - affine[:, :2] @ src.mean(axis=0)
    corners = np.float32([[0, 0], [size[0], 0], [size[0], size[1]], [0, size[1]]])
    movement = np.linalg.norm(cv2.transform(corners[None], affine)[0] - corners, axis=1).max() * pitch
    if movement > max_shift_mm or movement < .15:
        return None
    warped_mask = cv2.warpAffine(am, affine, size, flags=cv2.INTER_NEAREST)
    common = (warped_mask & am & bm) > 0
    if common.sum() < 1500:
        return None
    # Compare high-pass texture to remove exposure changes from the decision.
    ah = a.astype(np.float32) - cv2.GaussianBlur(a.astype(np.float32), (0, 0), 3)
    bh = b.astype(np.float32) - cv2.GaussianBlur(b.astype(np.float32), (0, 0), 3)
    aw = cv2.warpAffine(ah, affine, size)
    before = np.mean(np.abs(ah[common] - bh[common]))
    after = np.mean(np.abs(aw[common] - bh[common]))
    if before < 1 or after >= before * .85:
        return None
    result = np.eye(3)
    result[:2] = affine
    result[:2, 2] /= scale
    return result
