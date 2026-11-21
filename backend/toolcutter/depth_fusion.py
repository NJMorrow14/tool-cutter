"""Memory-bounded fusion of aligned, partially observed height maps."""
import warnings
import numpy as np


def fuse_heights(heights, tile_rows=64):
    """Median of actual observations, without stacking an entire drawer sweep.

    A 130-frame high-resolution drawer otherwise needs several gigabytes just
    for the input stack, plus larger median temporaries. Tiles preserve exactly
    the same pixel medians and NaNs with bounded working memory.
    """
    if not heights:
        return None
    shape = heights[0].shape
    if any(h.shape != shape for h in heights):
        raise ValueError('Aligned height maps must have matching dimensions')
    fused = np.full(shape, np.nan, np.float32)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='All-NaN slice encountered', category=RuntimeWarning)
        for y in range(0, shape[0], tile_rows):
            stack = np.stack([h[y:y + tile_rows] for h in heights])
            fused[y:y + tile_rows] = np.nanmedian(stack, axis=0, overwrite_input=True)
    return fused


def fuse_heights_with_support(heights, tol_mm=3.0, tile_rows=64):
    """`fuse_heights` plus a SUPPORT raster: how many frames measured each cell within `tol_mm` of the fused value.

    Why it exists (Nolan's desk drawer a237eba87dba, 2026-10-03, "the shapes/outlines still look ugly"): 34 of 66
    frames were placed by pose chaining, and a few of them landed 15-30 mm off. A tall object seen by ONE such frame
    leaves a tall ghost on bare floor (a 44 mm sliver beside the power bank, a 28 mm fragment beside the pin box);
    where exactly two frames disagree the median is their midpoint, which is a height nobody measured (a 12 mm
    wedge next to a 25 mm case). The median alone cannot tell those from a real tool — this raster can: real tools
    had 5-7 agreeing frames per cell, every ghost had one frame or none agreeing. Returns (fused, support uint8);
    support is 0 where fused is NaN.
    """
    fused = fuse_heights(heights, tile_rows=tile_rows)
    if fused is None:
        return None, None
    support = np.zeros(fused.shape, np.uint8)
    with np.errstate(invalid='ignore'):
        for h in heights:
            agree = np.isfinite(h) & (np.abs(h - fused) <= tol_mm)
            support += agree.astype(np.uint8)
    return fused, support
