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
