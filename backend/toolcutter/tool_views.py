"""Coverage and frame selection in the common, calibrated drawer raster."""
import cv2
import numpy as np


def coverage_report(frames, shape):
    count = np.zeros(shape, np.uint16)
    for frame in frames:
        count += frame['valid'].astype(np.uint16)
    rows = 6
    cols = max(6, min(24, round(rows * shape[1] / shape[0])))
    covered = cv2.resize((count > 0).astype(np.float32), (cols, rows), interpolation=cv2.INTER_AREA)
    overlap = cv2.resize((count > 1).astype(np.float32), (cols, rows), interpolation=cv2.INTER_AREA)
    return {'covered_fraction': float(np.mean(count > 0)), 'overlap_fraction': float(np.mean(count > 1)),
            'rows': rows, 'cols': cols, 'covered': covered.round(3).tolist(), 'overlap': overlap.round(3).tolist(),
            'photo_count': len(frames)}


def choose_tool_frame(frames, mask, mm_per_px):
    """A complete, sharp view beats a closer frame that clips the tool.

    Distances are divided by camera height to compare viewing angles across
    mixed camera distances. Quality is measured on this tool, not the whole shot.
    """
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    cx, cy = float(xs.mean()), float(ys.mean())
    x0, x1, y0, y1 = xs.min(), xs.max()+1, ys.min(), ys.max()+1
    local = mask[y0:y1, x0:x1]
    candidates = []
    for index, frame in enumerate(frames):
        if frame.get('use', 'both') == 'depth' or np.mean(frame['valid'][mask]) < .995:
            continue
        crop = frame['color'][y0:y1, x0:x1]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_32F)
        sharp = float(np.mean(np.abs(lap[local])))
        distance = np.hypot(cx-frame['nadir_px'][0], cy-frame['nadir_px'][1]) * mm_per_px
        angle = (distance + float(frame.get('quality_penalty_px', 0)) * mm_per_px) / max(100., float(frame.get('camera_height_mm') or 450.))
        clipped = float(np.mean((gray[local] < 4) | (gray[local] > 251)))
        candidates.append((index, angle, sharp, clipped, frame.get('rank', 0)))
    if not candidates:
        return None
    best_sharp = max(1., max(c[2] for c in candidates))
    return min(candidates, key=lambda c: c[1] + .6 * (1-c[2]/best_sharp) + .2*c[3] + .2*c[4])[0]


def discovery_windows(width, height, size=1100, overlap=300):
    """Overlapping local views plus a whole-drawer pass preserve large tools."""
    def starts(length):
        if length <= size:
            return [0]
        return sorted(set(list(range(0, length-size+1, size-overlap)) + [length-size]))
    return [(x, y, min(width, x+size), min(height, y+size)) for y in starts(height) for x in starts(width)]
