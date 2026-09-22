"""Metric AVFoundation RGB-D decoding helpers (native, unmirrored sensor coordinates)."""
from __future__ import annotations

import cv2
import numpy as np


def rectify_lens(image: np.ndarray, depth: np.ndarray, calibration: dict):
    """Undistort synchronized RGB and depth with Apple's inverse radial lookup table.

    remap needs output-to-input coordinates: the inverse table maps a rectilinear
    radius back into the distorted camera image. Depth uses nearest sampling so a
    tool/floor discontinuity never becomes an invented intermediate surface.
    Intrinsics remain in the same image coordinate system.
    """
    lut = np.asarray(calibration.get("inverse_lookup", []), dtype=np.float64)
    center = np.asarray(calibration.get("center", []), dtype=np.float64)
    reference = np.asarray(calibration.get("reference", []), dtype=np.float64)
    if (center.shape != (2,) or reference.shape != (2,) or not np.isfinite(center).all()
            or not np.isfinite(reference).all() or (reference <= 0).any()
            or lut.ndim != 1 or len(lut) > 4096 or not np.isfinite(lut).all()
            or (np.abs(lut) >= 1).any()):
        raise ValueError("Invalid lens calibration")
    if len(lut) == 0:  # no distortion table on this camera/format
        return image, depth
    if len(lut) < 2:
        raise ValueError("Lens calibration requires at least two radial samples")
    radius_max = np.hypot(max(center[0], reference[0] - center[0]),
                          max(center[1], reference[1] - center[1]))

    def maps(shape):
        h, w = shape[:2]
        yy, xx = np.mgrid[:h, :w].astype(np.float64)
        sx, sy = w / reference[0], h / reference[1]
        x, y = xx / sx - center[0], yy / sy - center[1]
        radius = np.hypot(x, y)
        magnification = np.interp(radius / radius_max * (len(lut) - 1), np.arange(len(lut)), lut)
        return ((center[0] + x * (1 + magnification)) * sx).astype(np.float32), \
               ((center[1] + y * (1 + magnification)) * sy).astype(np.float32)

    mx, my = maps(image.shape)
    rgb = cv2.remap(image, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    mx, my = maps(depth.shape)
    z = cv2.remap(depth, mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=float("nan"))
    return rgb, z
