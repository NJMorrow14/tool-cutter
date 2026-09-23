"""Mask -> polygon extraction and mm-space outline processing (offset, smoothing, notches)."""
from __future__ import annotations

import math
import os
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.ops import unary_union
from shapely import affinity

Ring = List[List[float]]


# ----------------------------------------------------------------------------- masks -> polygons

def mask_to_polygon(mask: np.ndarray, eps_px: float = 0.6) -> Optional[np.ndarray]:
    """Largest external contour of a boolean mask as an Nx2 float array (px)."""
    u8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 3:
        return None
    ap = cv2.approxPolyDP(cnt, eps_px, True)
    pts = ap.reshape(-1, 2).astype(np.float64) + 0.5  # pixel centers
    return pts


def detect_blobs_color(img_bgr: np.ndarray, mm_per_px: float, min_area_mm2: float = 200.0,
                       border_frac: float = 0.04) -> Tuple[np.ndarray, List[Dict]]:
    """Foreground blobs by color distance from the mat (sampled along the image border)."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    h, w = lab.shape[:2]
    b = max(2, int(round(min(h, w) * border_frac)))
    border = np.concatenate([lab[:b].reshape(-1, 3), lab[-b:].reshape(-1, 3),
                             lab[:, :b].reshape(-1, 3), lab[:, -b:].reshape(-1, 3)])
    bg = np.median(border, axis=0)
    dist = np.linalg.norm(lab - bg, axis=2)
    d8 = np.clip(dist * (255.0 / max(1.0, np.percentile(dist, 99.5))), 0, 255).astype(np.uint8)
    d8 = cv2.GaussianBlur(d8, (5, 5), 0)
    thr, fg = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    fg = _clean_fg(fg, mm_per_px)
    return fg > 0, _components(fg, mm_per_px, min_area_mm2)


def detect_blobs_height(height_mm: np.ndarray, mm_per_px: float, threshold_mm: float = 2.0,
                        min_area_mm2: float = 200.0, above_frac: Optional[np.ndarray] = None,
                        edge_rule: str = "half_height") -> Tuple[np.ndarray, List[Dict]]:
    """Blobs where the scan rises above the mat.

    A fixed threshold biases edges outward (interpolation blurs the height step), so each blob is
    re-thresholded at half of its own typical height, which puts the edge at the true wall position.
    """
    h = np.nan_to_num(height_mm, nan=0.0).astype(np.float32)
    fg = (h > threshold_mm).astype(np.uint8) * 255
    fg = _clean_fg(fg, mm_per_px)
    coarse = _components(fg, mm_per_px, min_area_mm2)
    refined_fg = np.zeros_like(fg)
    out: List[Dict] = []
    pad = max(2, int(round(3.0 / mm_per_px)))
    H, W = h.shape
    for c in coarse:
        x0, y0, x1, y1 = c["box"]
        x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
        x1, y1 = min(W, x1 + pad), min(H, y1 + pad)
        region = c["mask"][y0:y1, x0:x1]
        vals = h[y0:y1, x0:x1][region]
        if vals.size == 0:
            continue
        if above_frac is not None:
            local = (above_frac[y0:y1, x0:x1] > 0.5).astype(np.uint8) * 255
        elif edge_rule == "fixed":
            # depth-cleaned captures: a plain threshold keeps low parts of compound tools (a screwdriver
            # shaft next to its grip) that the half-height rule would cut off
            local = (h[y0:y1, x0:x1] > threshold_mm).astype(np.uint8) * 255
        else:
            top = float(np.percentile(vals, 90))
            local_thr = max(threshold_mm * 0.5, 0.5 * top)
            local = (h[y0:y1, x0:x1] > local_thr).astype(np.uint8) * 255
        local = _clean_fg(local, mm_per_px)
        num, labels = cv2.connectedComponents((local > 0).astype(np.uint8), connectivity=8)
        sx, sy = int(round(c["seed"][0])) - x0, int(round(c["seed"][1])) - y0
        lab = labels[min(max(sy, 0), labels.shape[0] - 1), min(max(sx, 0), labels.shape[1] - 1)]
        if lab == 0:
            # seed fell outside after re-threshold; keep the largest piece overlapping the coarse blob
            overlap = np.bincount(labels[region].ravel(), minlength=num)
            overlap[0] = 0
            if overlap.max() == 0:
                continue
            lab = int(np.argmax(overlap))
        piece = labels == lab
        full = np.zeros_like(fg, dtype=bool)
        full[y0:y1, x0:x1] = piece
        refined_fg[full] = 255
    return refined_fg > 0, _components(refined_fg, mm_per_px, min_area_mm2)


def _clean_fg(fg: np.ndarray, mm_per_px: float) -> np.ndarray:
    k_open = max(3, int(round(1.5 / mm_per_px)) | 1)
    k_close = max(3, int(round(2.5 / mm_per_px)) | 1)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open, k_open)))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close)))
    return fg


def _components(fg: np.ndarray, mm_per_px: float, min_area_mm2: float) -> List[Dict]:
    num, labels, stats, cents = cv2.connectedComponentsWithStats((fg > 0).astype(np.uint8), connectivity=8)
    min_px = min_area_mm2 / (mm_per_px ** 2)
    out = []
    h, w = fg.shape
    for i in range(1, num):
        area = float(stats[i, cv2.CC_STAT_AREA])
        if area < min_px:
            continue
        x, y, bw, bh = [int(stats[i, j]) for j in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT)]
        # skip blobs that hug the whole border (shadow bands, drawer walls)
        if bw >= 0.95 * w and bh >= 0.95 * h:
            continue
        comp = labels == i
        # centroid may fall outside a concave shape: pick the interior point farthest from the edge
        dt = cv2.distanceTransform(comp.astype(np.uint8), cv2.DIST_L2, 3)
        yy, xx = np.unravel_index(int(np.argmax(dt)), dt.shape)
        out.append({
            "box": [x, y, x + bw, y + bh],
            "seed": [float(xx), float(yy)],
            "area_px": area,
            "mask": comp,
        })
    out.sort(key=lambda c: -c["area_px"])
    return out


# ----------------------------------------------------------------------------- contour smoothing

def smooth_ring(coords: np.ndarray, sigma: float, tol: float, step: Optional[float] = None) -> np.ndarray:
    """Smooth a closed ring: resample uniformly by arc length, Gaussian-filter along the contour
    (circular), then Douglas-Peucker with `tol` so straight stretches collapse to single segments.
    Units are whatever `coords` is in (mm or px); sigma/tol/step in the same units."""
    pts = np.asarray(coords, dtype=np.float64)
    if len(pts) >= 2 and np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    if len(pts) < 4 or sigma <= 0:
        return pts
    step = step or max(sigma / 4.0, 1e-3)
    seg = np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1)
    total = float(seg.sum())
    if total < 4 * step:
        return pts
    n = max(16, int(total / step))
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    t = np.linspace(0.0, total, n, endpoint=False)
    closed = np.vstack([pts, pts[:1]])
    xs = np.interp(t, cum, closed[:, 0])
    ys = np.interp(t, cum, closed[:, 1])
    # circular Gaussian
    k_half = int(np.ceil(3 * sigma / step))
    kx = np.arange(-k_half, k_half + 1) * step
    kern = np.exp(-0.5 * (kx / sigma) ** 2)
    kern /= kern.sum()
    def conv(v):
        pad = np.concatenate([v[-k_half:], v, v[:k_half]]) if k_half < len(v) else np.tile(v, 3)
        out = np.convolve(pad, kern, mode="same")
        return out[k_half:k_half + len(v)] if k_half < len(v) else out[len(v):2 * len(v)]
    sm = np.column_stack([conv(xs), conv(ys)])
    if tol > 0:
        approx = cv2.approxPolyDP(sm.astype(np.float32).reshape(-1, 1, 2), float(tol), True).reshape(-1, 2)
        if len(approx) >= 3:
            sm = approx.astype(np.float64)
    return sm


def _resample_closed(pts: np.ndarray, step: float) -> np.ndarray:
    """Closed ring re-sampled at a uniform arc-length step (mm in, mm out)."""
    pts = np.asarray(pts, dtype=np.float64)
    seg = np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1)
    total = float(seg.sum())
    if total < 3 * step:
        return pts
    n = max(12, int(round(total / step)))
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    t = np.linspace(0.0, total, n, endpoint=False)
    closed = np.vstack([pts, pts[:1]])
    return np.column_stack([np.interp(t, cum, closed[:, 0]), np.interp(t, cum, closed[:, 1])])


def _fit_circle(pts: np.ndarray):
    """Algebraic (Kasa) circle fit -> (centre, radius, max |radial residual|)."""
    x, y = pts[:, 0], pts[:, 1]
    A = np.column_stack([x, y, np.ones_like(x)])
    b = x * x + y * y
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    cx, cy = sol[0] / 2, sol[1] / 2
    r2 = sol[2] + cx * cx + cy * cy
    if not np.isfinite(r2) or r2 <= 0:
        return None
    r = float(np.sqrt(r2))
    sres = np.hypot(x - cx, y - cy) - r
    if len(sres) > 6:
        from scipy.ndimage import gaussian_filter1d
        lp = gaussian_filter1d(sres, 3.0, mode="nearest")          # samples are ~0.5 mm apart: a ~1.5 mm low-pass
        # systematic misfit (not a circle) shows in the low-passed residual; wobble up to 3x tol is fine
        return np.array([cx, cy]), r, float(max(np.abs(lp).max(), np.percentile(np.abs(sres), 95) / 3.0))
    return np.array([cx, cy]), r, float(np.abs(sres).max())


def clean_ring(pts_mm: np.ndarray, tol: float = 0.4, corner_deg: float = 38.0, step: float = 0.5) -> np.ndarray:
    """Re-express a dense traced outline as the shape a person would draw: sharp corners, straight edges as
    two points, round parts as arcs, and anything else as a smoothed curve — every piece within `tol` mm of
    the trace. Nolan (2026-09-23): outlines were "not smooth, too many vertexes, and not understanding the
    overall shape". The trace itself is fine (0.1-0.3 mm rms wobble); it was being kept at a 0.15 mm tolerance,
    finer than the sensor's own ~2 mm edge response, so every pixel-scale wiggle survived as a vertex.

    Corners first: the turning angle summed over a +-2 mm window, peaks >= corner_deg, at least 3 mm apart. Splitting
    there is what keeps a hammer head square and a ruler's ends crisp while its long sides collapse to 2 points.
    """
    pts = np.asarray(pts_mm, dtype=np.float64)
    if len(pts) < 8:
        return pts
    r = _resample_closed(pts, step)
    n = len(r)
    if n < 12:
        return pts
    # judge corners and fits on a lightly smoothed ring (1 mm, cyclic) so pixel noise cannot fake a corner or
    # fail a straight edge; the pieces are still checked against the trace within tol
    from scipy.ndimage import gaussian_filter1d
    # fitting smoothness scales with the tolerance: a fixed 1 mm ate the tips of 5 mm shafts (hex key -0.06 IoU)
    fit_sig = max(1.0, 0.75 * tol / step)
    r = np.column_stack([gaussian_filter1d(r[:, j], fit_sig, mode="wrap") for j in range(2)])
    # corners are found on a 2 mm-smoothed copy: a real corner still turns ~90 deg over the window, while noise
    # (random in sign) largely cancels in the signed sum below
    rc = np.column_stack([gaussian_filter1d(r[:, j], max(1.0, 2.0 / step), mode="wrap") for j in range(2)])
    d1 = rc - np.roll(rc, 1, axis=0)
    d2 = np.roll(rc, -1, axis=0) - rc
    ang = np.arctan2(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0], (d1 * d2).sum(axis=1))   # signed turn per vertex
    w = max(1, int(round(2.0 / step)))
    kernel = np.ones(2 * w + 1)
    turn = np.abs(np.convolve(np.concatenate([ang[-w:], ang, ang[:w]]), kernel, mode="valid"))   # cyclic window sum
    thresh = np.deg2rad(corner_deg)
    gap = max(1, int(round(3.0 / step)))
    corners = []
    order = np.argsort(-turn)
    taken = np.zeros(n, bool)
    for i in order:
        if turn[i] < thresh:
            break
        if taken[i]:
            continue
        corners.append(int(i))
        lo = np.arange(i - gap, i + gap + 1) % n
        taken[lo] = True
    corners = sorted(corners)
    if len(corners) < 2:
        # no corners: treat the whole ring as one cyclic free curve
        segs = [np.arange(n)]
        cyclic = True
    else:
        segs = [np.arange(a, b + 1) % n for a, b in zip(corners, corners[1:] + [corners[0] + n])]
        cyclic = False
    out: List[np.ndarray] = []
    kinds: List[str] = []
    lines: List[Optional[tuple]] = []          # (centre, direction) of the least-squares line for line pieces
    for idx in segs:
        seg = r[idx]
        kinds.append("other"); lines.append(None)
        if len(seg) < 3:
            out.append(seg[:-1] if not cyclic else seg)
            continue
        a, b = seg[0], seg[-1]
        chord = b - a
        L = float(np.linalg.norm(chord))
        # straight?
        if not cyclic and L > 1e-6:
            nrm = np.array([-chord[1], chord[0]]) / L
            sdev = (seg - a) @ nrm
            dev = np.abs(sdev)
            # A wall that wobbles is still a wall: its deviation from the chord is zero-mean noise, while an arc's is
            # a systematic bow (its sagitta). Low-pass the signed deviation over ~3 mm — noise cancels, a bow does not —
            # and accept a straight run when the bow is within tol even if the raw wobble is up to 3x that. This is
            # what lets a 330 mm steel rule whose edge wanders 1 mm read as two straight lines instead of 100 points.
            lp = gaussian_filter1d(sdev, max(1.0, 3.0 / step), mode="nearest") if len(sdev) > 6 else sdev
            if np.abs(lp).max() <= tol and np.percentile(dev, 95) <= 3 * tol:
                trim = max(1, len(seg) // 6)                 # ends carry the rounded corner; fit the wall itself
                core = seg[trim:-trim] if len(seg) > 3 * trim else seg
                c0 = core.mean(axis=0)
                _, _, vt = np.linalg.svd(core - c0, full_matrices=False)
                kinds[-1] = "line"; lines[-1] = (c0, vt[0])
                out.append(seg[:1])
                continue
        # one circular arc?
        fit = _fit_circle(seg) if len(seg) >= 6 else None
        if fit is not None and 1.5 <= fit[1] <= 400.0 and fit[2] <= tol:
            c, rad, _ = fit
            t0 = np.arctan2(*(seg[0] - c)[::-1]); t1 = np.arctan2(*(seg[-1] - c)[::-1])
            tm = np.arctan2(*(seg[len(seg) // 2] - c)[::-1])
            sweep = (t1 - t0 + np.pi) % (2 * np.pi) - np.pi
            # take the direction that passes through the middle sample
            if ((tm - t0 + np.pi) % (2 * np.pi) - np.pi) * sweep < 0 or cyclic:
                sweep = sweep - np.sign(sweep) * 2 * np.pi if not cyclic else 2 * np.pi
            e = max(0.05, 0.5 * tol)
            dth = 2 * np.arccos(max(-1.0, min(1.0, 1 - e / rad)))
            k = max(2, int(np.ceil(abs(sweep) / dth)))
            th = t0 + sweep * np.arange(0, k, 1) / k if cyclic else t0 + sweep * np.arange(0, k) / k
            arc = np.column_stack([c[0] + rad * np.cos(th), c[1] + rad * np.sin(th)])
            out.append(arc)
            continue
        # free curve: smooth along the run (ends pinned), then keep only what tol requires
        sig = fit_sig
        if len(seg) > 6:
            from scipy.ndimage import gaussian_filter1d
            mode = "wrap" if cyclic else "nearest"
            sm = np.column_stack([gaussian_filter1d(seg[:, j], sig, mode=mode) for j in range(2)])
            if not cyclic:
                sm[0], sm[-1] = seg[0], seg[-1]
        else:
            sm = seg
        dp = cv2.approxPolyDP(sm.astype(np.float32).reshape(-1, 1, 2), float(tol), bool(cyclic)).reshape(-1, 2)
        if not cyclic:
            dp = np.vstack([seg[:1], dp[1:-1], seg[-1:]]) if len(dp) >= 2 else seg[[0, -1]]
            out.append(dp[:-1])
        else:
            out.append(dp.astype(np.float64))
    # Two straight edges meeting at a corner: the true corner is where the fitted walls CROSS. The smoothed apex
    # sits ~0.3 mm inside it, and on a rectangle that is a 0.3 mm inset of every side (IoU 0.984 -> 0.998).
    if not cyclic:
        m = len(out)
        for k in range(m):
            j = (k - 1) % m
            if kinds[k] == "line" and kinds[j] == "line" and lines[k] and lines[j] and len(out[k]):
                (c1, d1_), (c2, d2_) = lines[j], lines[k]
                den = d1_[0] * d2_[1] - d1_[1] * d2_[0]
                if abs(den) > 1e-6:
                    t = ((c2[0] - c1[0]) * d2_[1] - (c2[1] - c1[1]) * d2_[0]) / den
                    x = c1 + t * d1_
                    if np.linalg.norm(x - out[k][0]) <= 3.0:     # never let a near-parallel pair fly off
                        out[k] = out[k].copy(); out[k][0] = x
    res = np.vstack(out) if out else pts
    # drop duplicate consecutive points
    keep = np.linalg.norm(res - np.roll(res, 1, axis=0), axis=1) > 1e-6
    res = res[keep]
    return res if len(res) >= 3 else pts


def straighten_ring(pts: np.ndarray, coarse_tol: float, min_len: float) -> np.ndarray:
    """Replace long nearly-straight stretches of a closed ring by their least-squares line.

    A coarse Douglas-Peucker pass (tolerance `coarse_tol`) finds the stretches; only those longer than
    `min_len` are straightened, so arcs (whose chords at that tolerance are short) keep their points.
    """
    pts = np.asarray(pts, dtype=np.float64)
    n = len(pts)
    if n < 8:
        return pts
    coarse = cv2.approxPolyDP(pts.astype(np.float32).reshape(-1, 1, 2), float(coarse_tol), True).reshape(-1, 2)
    if len(coarse) < 3:
        return pts
    # index of each coarse vertex in the fine ring
    idx = []
    for c in coarse:
        d = np.linalg.norm(pts - c, axis=1)
        idx.append(int(np.argmin(d)))
    idx = sorted(set(idx))
    out = pts.copy()
    lines = {}
    for a, b in zip(idx, idx[1:] + [idx[0] + n]):
        run = np.arange(a, b + 1) % n
        seg = pts[run]
        length = float(np.linalg.norm(seg[-1] - seg[0]))
        if length < min_len or len(run) < 4:
            continue
        # Fit away from the rounded ends: those contain samples from the adjacent
        # sides and would pull the fitted wall inward. A curved run is not a wall.
        trim = max(1, len(seg) // 5)
        core = seg[trim:-trim]
        c = core.mean(axis=0)
        _, _, vt = np.linalg.svd(core - c, full_matrices=False)
        dvec = vt[0]
        normal = np.array([-dvec[1], dvec[0]])
        if np.percentile(np.abs((core - c) @ normal), 95) > 0.12 * coarse_tol:
            continue
        proj = c + np.outer((seg - c) @ dvec, dvec)
        out[run[1:-1]] = proj[1:-1]
        lines[a] = (c, dvec, b % n)
    # Restore corners where two fitted straight sides meet. Keeping the Gaussian-rounded
    # endpoint between projected runs leaves a notch; simplification cannot recover the
    # original intersection. Bound extrapolation so shallow angles cannot create spikes.
    fitted = sorted(lines)
    for j, a in enumerate(fitted):
        previous = fitted[j - 1]
        c0, d0, end = lines[previous]
        c1, d1, _ = lines[a]
        bridge = np.arange(end, a + (n if a < end else 0) + 1) % n
        if np.linalg.norm(np.diff(pts[bridge], axis=0), axis=1).sum() > 4.0 * coarse_tol:
            continue
        cross = d0[0] * d1[1] - d0[1] * d1[0]
        if abs(cross) < 0.25:
            continue
        delta = c1 - c0
        corner = c0 + d0 * ((delta[0] * d1[1] - delta[1] * d1[0]) / cross)
        if np.linalg.norm(pts[bridge] - corner, axis=1).min() <= 2.0 * coarse_tol:
            out[bridge] = corner
    return out


def smooth_polygon(poly: Polygon, sigma_mm: float, tol_mm: float, clean_tol_mm: Optional[float] = None) -> Polygon:
    """Gaussian smoothing + straightening of long straight runs, on the exterior and every interior."""
    if poly.is_empty or sigma_mm <= 0:
        return poly
    def _one(coords):
        r = smooth_ring(np.asarray(coords), sigma_mm, 0.0)          # keep dense for straightening
        r = straighten_ring(r, coarse_tol=max(1.5, sigma_mm), min_len=20.0)   # arcs of r<~30 mm keep their points
        if len(r) >= 3 and clean_tol_mm and clean_tol_mm > 0:
            r = clean_ring(r, tol=float(clean_tol_mm))                           # lines / arcs / smooth curves
        elif len(r) >= 3 and tol_mm > 0:
            r = cv2.approxPolyDP(r.astype(np.float32).reshape(-1, 1, 2), float(tol_mm), True).reshape(-1, 2).astype(np.float64)
        return r
    ext = _one(poly.exterior.coords)
    if len(ext) < 3:
        return poly
    holes = []
    for ring in poly.interiors:
        r = _one(ring.coords)
        if len(r) >= 3 and Polygon(r).area >= 4.0:
            holes.append(r)
    out = Polygon(ext, holes)
    if not out.is_valid:
        out = out.buffer(0)
        if isinstance(out, MultiPolygon):
            out = max(out.geoms, key=lambda g: g.area)
    return out if not out.is_empty else poly


# ----------------------------------------------------------------------------- mm-space outlines

def _to_shapely(poly_px: Sequence[Sequence[float]], mm_per_px: float) -> Polygon:
    pts = np.asarray(poly_px, dtype=np.float64) * mm_per_px
    if len(pts) < 3:
        return Polygon()
    p = Polygon(pts)
    if not p.is_valid:
        p = p.buffer(0)
    if isinstance(p, MultiPolygon):
        p = max(p.geoms, key=lambda g: g.area)
    return p


def _largest_parts(geom, min_area_mm2: float = 4.0) -> List[Polygon]:
    if geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom] if geom.area >= min_area_mm2 else []
    if isinstance(geom, MultiPolygon):
        return [g for g in geom.geoms if g.area >= min_area_mm2]
    try:
        return [g for g in geom.geoms if isinstance(g, Polygon) and g.area >= min_area_mm2]
    except AttributeError:
        return []


def process_outline(poly_px: Sequence[Sequence[float]], mm_per_px: float, *, rotation_deg: float = 0.0,
                    offset_mm: Tuple[float, float] = (0.0, 0.0), clearance_mm: float = 1.0,
                    smoothing_mm: float = 0.6, notch: Optional[Dict] = None,
                    simplify_mm: float = 0.15) -> Dict:
    """Return processed rings (mm) for one tool plus metadata.

    Steps: px->mm, rotate about centroid, translate, morphological smoothing (closing then opening
    with radius smoothing_mm), offset by clearance_mm, union a finger notch circle, simplify.
    """
    p = _to_shapely(poly_px, mm_per_px)
    if p.is_empty:
        return {"rings": [], "area_mm2": 0.0, "bbox_mm": None, "centroid_mm": None}
    centroid = (p.centroid.x, p.centroid.y)
    source_centroid = centroid
    if rotation_deg:
        p = affinity.rotate(p, rotation_deg, origin=centroid)
    if offset_mm and (offset_mm[0] or offset_mm[1]):
        p = affinity.translate(p, xoff=offset_mm[0], yoff=offset_mm[1])
    if smoothing_mm > 0:
        r = smoothing_mm
        p = p.buffer(r, join_style=1).buffer(-2 * r, join_style=1).buffer(r, join_style=1)
        parts = _largest_parts(p)
        p = max(parts, key=lambda g: g.area) if parts else Polygon()
    if p.is_empty:
        return {"rings": [], "area_mm2": 0.0, "bbox_mm": None, "centroid_mm": None}
    if clearance_mm:
        p = p.buffer(clearance_mm, join_style=1)
    if smoothing_mm > 0 and isinstance(p, Polygon):
        # along-contour Gaussian (removes sensor/pixel wobble, rounds corners by ~sigma) + straighten
        p = smooth_polygon(p, sigma_mm=smoothing_mm, tol_mm=max(0.15, 0.25 * smoothing_mm))
    notch_used = None
    if notch and notch.get("diameter_mm", 0) > 0:
        d = float(notch["diameter_mm"])
        click = Point(float(notch["x_mm"]), float(notch["y_mm"]))
        boundary = p.exterior if isinstance(p, Polygon) else unary_union([g.exterior for g in p.geoms])
        anchor = boundary.interpolate(boundary.project(click))
        # keep the circle half inside the pocket so it reads as a finger pull
        p = unary_union([p, Point(anchor.x, anchor.y).buffer(d / 2.0)])
        notch_used = {"x_mm": anchor.x, "y_mm": anchor.y, "diameter_mm": d}
    p = p.simplify(simplify_mm, preserve_topology=True)
    parts = _largest_parts(p)
    rings: List[Ring] = []
    for part in parts:
        rings.append([[float(x), float(y)] for x, y in part.exterior.coords])
        for interior in part.interiors:
            if Polygon(interior).area >= 4.0:
                rings.append([[float(x), float(y)] for x, y in interior.coords])
    merged = unary_union(parts) if parts else Polygon()
    minx, miny, maxx, maxy = merged.bounds if not merged.is_empty else (0, 0, 0, 0)
    return {
        "rings": rings,
        "area_mm2": float(merged.area),
        "bbox_mm": [float(minx), float(miny), float(maxx), float(maxy)],
        "centroid_mm": [float(merged.centroid.x), float(merged.centroid.y)] if not merged.is_empty else None,
        "notch": notch_used,
        "shapely": merged,
        "source_centroid_mm": [float(source_centroid[0]), float(source_centroid[1])],
    }


def mirror_rings(rings: List[Ring], width_mm: float) -> List[Ring]:
    return [[[width_mm - x, y] for x, y in ring] for ring in rings]


def level_height_raster(height: np.ndarray, iters: int = 4, resid_mm: float = 3.0) -> Tuple[np.ndarray, float]:
    """Re-reference a height raster to the floor visible in it (photogrammetry floors are bowed / offset).

    Fits z = a*x + b*y + c to the low cells (robustly, re-weighting to cells within resid_mm of the fit),
    subtracts it and clamps at 0. Returns (levelled raster, max |plane offset| in mm across the raster).
    """
    h = np.asarray(height, dtype=np.float64)
    H, W = h.shape
    ys, xs = np.mgrid[0:H, 0:W]
    step = max(1, int(np.sqrt(H * W / 60000)))
    xs_s, ys_s, hs = xs[::step, ::step].ravel(), ys[::step, ::step].ravel(), h[::step, ::step].ravel()
    valid = np.isfinite(hs)
    xs_s, ys_s, hs = xs_s[valid], ys_s[valid], hs[valid]
    if hs.size < 50:
        return np.maximum(h, 0).astype(np.float32), 0.0
    sel = hs <= np.percentile(hs, 60)          # start from the lower 60 %
    A = np.column_stack([xs_s, ys_s, np.ones_like(xs_s)]).astype(np.float64)
    coef = np.zeros(3)
    for _ in range(iters):
        if sel.sum() < 50:
            break
        coef, *_ = np.linalg.lstsq(A[sel], hs[sel], rcond=None)
        resid = hs - A @ coef
        sel = np.abs(resid) < resid_mm
    plane = coef[0] * xs + coef[1] * ys + coef[2]
    out = (h - plane).astype(np.float32)
    out[out < 0] = 0.0
    return out, float(np.abs(plane).max())


# ----------------------------------------------------------------------------- topographic footprints

LOW_TOOL_FRAC = float(os.environ.get("TC_LOW_FRAC", 0.35))   # threshold as a fraction of a low tool's own height


def topo_footprint(height_mm: np.ndarray, seed_mask: np.ndarray, mm_per_px: float, cell_mm: float = 3.0,
                   threshold_mm: float = 2.0, frac: float = 0.5, slope_max: float = 0.8,
                   restrict: Optional[np.ndarray] = None, field_out: Optional[Dict] = None) -> np.ndarray:
    """Footprint of the tool(s) under `seed_mask` from the scan topography alone (no photo).

    The depth sensor smears every wall into a ramp about one depth cell wide, so a fixed threshold lands the
    edge outside the wall and a fraction of the tool's overall height cuts off low parts. Instead each candidate
    pixel is judged against the height of the nearest *flat* pixel (its own ridge or plateau, gradient below
    `slope_max` mm/mm): it belongs to the tool when it is at least `frac` of that local top. For a vertical wall
    blurred by the sensor that is exactly the wall position; for a lying cylinder or a sphere it is the widest
    point (height = radius), i.e. the true footprint. A 7 mm shaft next to a 28 mm grip is judged against the
    shaft's own ridge, not the grip, so compound tools stay whole; neighbours 4 mm apart separate because the
    dip between them falls below half of either top.
    """
    from scipy import ndimage as ndi
    import os
    frac = float(os.environ.get("TC_TOPO_FRAC", frac))
    slope_max = float(os.environ.get("TC_TOPO_SLOPE", slope_max))

    h = np.nan_to_num(height_mm, nan=0.0).astype(np.float32)
    cell_px = max(1.0, cell_mm / mm_per_px)
    kd = int(round(2.0 * cell_px)) | 1
    region = cv2.dilate(seed_mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kd, kd))) > 0
    if restrict is not None:
        region &= restrict
    # A FIXED threshold is wrong for a low tool. A 3 mm steel rule cut at 2 mm keeps only its top third, and
    # what survives is a ragged band whose outline wanders by a centimetre across a dead-straight object — while
    # the height map itself resolves that rule to 26 mm +/- 1.5. So scale the threshold to the blob's own
    # height, with a floor just above the mat's noise (bare mat here: p90 0.08 mm) and a ceiling at the
    # caller's value, which keeps tall tools exactly as they were.
    noise_mm = float(os.environ.get("TC_NOISE_FLOOR", 0.8))
    probe = region & (h > noise_mm)
    if not probe.any():
        return np.zeros_like(seed_mask, dtype=bool)
    thr_eff = float(np.clip(LOW_TOOL_FRAC * float(np.percentile(h[probe], 90)), noise_mm, threshold_mm))
    threshold_mm = thr_eff
    cand = region & (h > threshold_mm)
    if not cand.any():
        return np.zeros_like(seed_mask, dtype=bool)
    # work on the bounding box only
    ys, xs = np.nonzero(cand)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    hb = h[y0:y1, x0:x1]
    cb = cand[y0:y1, x0:x1]
    gy, gx = np.gradient(ndi.gaussian_filter(hb, 0.35 * cell_px))
    slope = np.hypot(gx, gy) / mm_per_px                         # mm of height per mm of travel
    flat = cb & (slope < slope_max) & (hb > threshold_mm)
    if not flat.any():
        flat = cb & (hb >= np.percentile(hb[cb], 70))
    # each candidate looks up the top of its nearest flat pixel
    dist, (iy, ix) = ndi.distance_transform_edt(~flat, return_indices=True)
    top = hb[iy, ix]
    # no flat pixel within ~1.5 cells (small round tops, noisy tops): judge against the blob's overall top instead
    blob_top = float(np.percentile(hb[cb], 90))
    far = dist > 1.5 * cell_px
    # features narrower than ~2.5 depth cells (shafts, hex keys) never reach their true height on the grid and
    # their ramps are all flank: judge them at a lower fraction of their (already low) ridge
    dtc = ndi.distance_transform_edt(cb)
    thin = dtc[iy, ix] < 1.25 * cell_px
    frac_px = np.where(thin & ~far, frac * 0.7, frac)
    # a blob that is thin everywhere (hex key, pin, blade) never reaches its true height: same allowance for its
    # far pixels, so the ends of its legs are not cut off
    frac_far = frac * 0.7 if float(dtc.max()) < 1.25 * cell_px else frac
    level = np.where(far, frac_far * blob_top, frac_px * top)
    level = np.maximum(threshold_mm, level)
    keep = cb & (hb >= level)
    if field_out is not None:
        # how far above/below its own keep level each pixel sits. The mask is the sign of this; the edge is
        # its zero crossing, which topo_polygon uses to place the outline between pixels instead of on them.
        sg = np.full(height_mm.shape, -1e3, np.float32)
        sg[y0:y1, x0:x1] = np.where(cb, hb - level, -1e3)
        field_out["signed"] = sg
    # the level rule may cut a tool at a low joint (pliers pivot, a thin neck): the seed blob was connected, so
    # bridge pieces that are within one cell of each other with the low-threshold pixels between them. Whether a
    # blob is really two tools is decided by split_at_saddles, not here.
    num, lab = cv2.connectedComponents(keep.astype(np.uint8), connectivity=8)
    if num > 2:
        kb = int(round(cell_px)) | 1
        ker_b = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kb, kb))
        count = np.zeros(keep.shape, np.uint8)
        for i in range(1, num):
            count += cv2.dilate((lab == i).astype(np.uint8), ker_b)
        keep |= cb & (count >= 2)
    # tidy: ≤1 mm speckle / pinholes
    k1 = max(1, int(round(0.7 / mm_per_px)))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k1 + 1, 2 * k1 + 1))
    k8 = cv2.morphologyEx(keep.astype(np.uint8), cv2.MORPH_OPEN, ker)
    k8 = cv2.morphologyEx(k8, cv2.MORPH_CLOSE, ker)
    cs, _ = cv2.findContours(k8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    k8 = np.zeros_like(k8)
    cv2.drawContours(k8, cs, -1, 1, thickness=cv2.FILLED)
    out = np.zeros_like(seed_mask, dtype=bool)
    out[y0:y1, x0:x1] = k8 > 0
    return out


def split_components(mask: np.ndarray, min_area_px: float) -> List[np.ndarray]:
    """Connected components of a mask, largest first, dropping slivers."""
    num, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    comps = [(float(st[i, cv2.CC_STAT_AREA]), i) for i in range(1, num) if st[i, cv2.CC_STAT_AREA] >= min_area_px]
    comps.sort(reverse=True)
    return [lab == i for _, i in comps]


def subpixel_ring(pts: np.ndarray, signed: np.ndarray, max_shift_px: float = 2.5) -> np.ndarray:
    """Slide every vertex along the outline's local normal onto the zero crossing of `signed`.

    Tracing a binary mask can only put a vertex on a whole pixel, so the outline comes out as a staircase
    whose steps are the raster pitch — the "rough" look. The height field underneath is smooth, though, and
    the mask is just `signed >= 0`, so the true edge lies between two pixels and can be interpolated. Vertices
    with no crossing within `max_shift_px` (a corner, a hole) stay where they were.
    """
    n = len(pts)
    if n < 8 or signed is None:
        return pts
    f = np.where(np.isfinite(signed), signed, -1e3).astype(np.float32)
    tang = np.roll(pts, -1, axis=0) - np.roll(pts, 1, axis=0)
    ln = np.hypot(tang[:, 0], tang[:, 1])
    ok = ln > 1e-6
    nrm = np.zeros_like(tang)
    nrm[ok] = np.column_stack([tang[ok, 1], -tang[ok, 0]]) / ln[ok, None]
    ts = np.linspace(-max_shift_px, max_shift_px, 21, dtype=np.float32)
    # Rings use pixel-centre coordinates (index + 0.5); remap uses array indices.
    xs = (pts[:, 0:1] - 0.5 + nrm[:, 0:1] * ts[None, :]).astype(np.float32)
    ys = (pts[:, 1:2] - 0.5 + nrm[:, 1:2] * ts[None, :]).astype(np.float32)
    s = cv2.remap(f, xs, ys, interpolation=cv2.INTER_LINEAR,
                  borderMode=cv2.BORDER_CONSTANT, borderValue=-1e3)
    out = pts.copy()
    pos = s > 0
    change = pos[:, :-1] != pos[:, 1:]
    for i in np.nonzero(change.any(axis=1))[0]:
        j = np.nonzero(change[i])[0]
        a, b = s[i, j], s[i, j + 1]
        with np.errstate(all="ignore"):
            t = ts[j] + (ts[j + 1] - ts[j]) * (-a) / (b - a)
        t = t[np.isfinite(t)]
        if len(t):
            out[i] = pts[i] + nrm[i] * t[np.argmin(np.abs(t))]
    return out


def topo_polygon(mask: np.ndarray, mm_per_px: float, signed: Optional[np.ndarray] = None,
                 sigma_mm: float = 1.0) -> Optional[np.ndarray]:
    """Outline of a topographic footprint: largest contour, put on the sub-pixel edge of the height field
    (`signed` = height - the local keep level) and then smoothed of what noise is left."""
    u8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    if len(cnt) < 8:
        return None
    pts = cnt.reshape(-1, 2).astype(np.float64) + 0.5
    if signed is not None and os.environ.get("TC_SUBPIXEL", "0") == "1":
        pts = subpixel_ring(pts, signed)
    # the grid carries ~1 mm of noise along the edge: Gaussian along the contour, then straighten runs that are
    # straight within 1.5 mm (geometry.smooth_polygon does both; it works in mm)
    try:
        poly = Polygon(pts * mm_per_px)
        if poly.is_valid and poly.area > 1.0:
            clean = float(os.environ.get("TC_CLEAN_TOL", "0") or 0) or min(0.5, max(0.35, 0.6 * sigma_mm))
            out = smooth_polygon(poly.buffer(0), sigma_mm=sigma_mm, tol_mm=0.15, clean_tol_mm=clean)
            if out is not None and not out.is_empty:
                ring = max(out.geoms, key=lambda g: g.area) if out.geom_type == "MultiPolygon" else out
                arr = np.asarray(ring.exterior.coords[:-1], dtype=np.float64) / mm_per_px
                if len(arr) >= 3:
                    return arr
    except Exception:  # noqa: BLE001
        pass
    sm = smooth_ring(pts, sigma=sigma_mm / mm_per_px, tol=0.3 / mm_per_px)
    if len(sm) >= 8:
        clean = float(os.environ.get("TC_CLEAN_TOL", "0") or 0) or min(0.5, max(0.35, 0.6 * sigma_mm))
        if clean > 0:
            cl = clean_ring(sm * mm_per_px, tol=float(clean)) / mm_per_px
            if len(cl) >= 3 and cv2.contourArea(cl.astype(np.float32)) > 0:
                sm = cl
    return sm if len(sm) >= 3 else None


def split_at_saddles(mask: np.ndarray, height_mm: np.ndarray, mm_per_px: float, cell_mm: float = 3.0,
                     marker_frac: float = 0.75, saddle_frac: float = 0.55, min_area_mm2: float = 150.0) -> List[np.ndarray]:
    """Split a footprint that merged neighbouring tools: seeds are the high plateaus (>= marker_frac of the blob top),
    the mask is flooded from them down the height map (watershed), and two regions stay separate only when the
    saddle between them is below saddle_frac of the lower of their tops — a compound tool (grip + shaft, hammer
    head + handle) has no such saddle and comes back whole. Returns the pieces (largest first)."""
    from scipy import ndimage as ndi

    h = np.nan_to_num(height_mm, nan=0.0).astype(np.float32)
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return []
    pad = 2
    y0, y1, x0, x1 = max(0, ys.min() - pad), min(mask.shape[0], ys.max() + pad + 1), max(0, xs.min() - pad), min(mask.shape[1], xs.max() + pad + 1)
    m = mask[y0:y1, x0:x1]
    hb = h[y0:y1, x0:x1]
    top = float(np.percentile(hb[m], 95))
    seeds = m & (hb >= marker_frac * top)
    k = max(1, int(round(0.5 * cell_mm / mm_per_px)))
    seeds = cv2.morphologyEx(seeds.astype(np.uint8), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))) > 0
    nseed, lab = cv2.connectedComponents(seeds.astype(np.uint8), connectivity=8)
    min_px = min_area_mm2 / mm_per_px ** 2
    sizes = np.bincount(lab.ravel(), minlength=nseed)
    # seeds must be real plateaus (≥ 300 mm² each); thin low tools (hex keys, blades) fragment at 75 % of their
    # top and must never be split
    seed_min = max(min_px * 0.25, 300.0 / mm_per_px ** 2)
    ids = [i for i in range(1, nseed) if sizes[i] >= seed_min]
    if len(ids) < 2 or top < 2.5 * cell_mm:
        return [mask]
    markers = np.zeros(m.shape, np.int32)
    for n, i in enumerate(ids, start=1):
        markers[lab == i] = n
    # flood from the plateaus down the height map; outside the mask is set to the last level so every pixel of the
    # mask is claimed by a plateau before any flood can leak around the outside (a background marker would win
    # the low ramps of a low tool and leave most of it unassigned)
    inv = 255 - np.clip(hb / max(top, 1e-6), 0, 1) * 254
    inv[~m] = 255
    img = cv2.cvtColor(inv.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    ws = cv2.watershed(img, markers.copy())
    ws[~m] = 0
    ws[ws < 0] = 0
    import os
    if os.environ.get("TC_SADDLE_DEBUG"):
        print(f"saddle-dbg: top {top:.1f} seeds {len(ids)} seed areas {[int(sizes[i] * mm_per_px ** 2) for i in ids]} region areas {[int((ws == n).sum() * mm_per_px ** 2) for n in range(1, len(ids) + 1)]} mask {int(m.sum() * mm_per_px ** 2)} unassigned {int((m & (ws == 0)).sum() * mm_per_px ** 2)} mm2")
    # saddle heights between adjacent regions: the max height on their shared boundary
    tops = {n: float(np.percentile(hb[ws == n], 95)) if (ws == n).any() else 0.0 for n in range(1, len(ids) + 1)}
    parent = {n: n for n in tops}
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    dil = {n: cv2.dilate((ws == n).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0 for n in tops}
    for a in tops:
        for b in tops:
            if b <= a:
                continue
            shared = dil[a] & dil[b] & m
            if not shared.any():
                continue
            # a real gap dips toward the floor along most of the shared boundary; a joint (pliers pivot, a neck)
            # only dents it. Merge unless the median of the boundary is below saddle_frac of the lower top AND
            # its high end is still well below that top.
            lo = min(tops[a], tops[b])
            p50, p90 = float(np.percentile(hb[shared], 50)), float(np.percentile(hb[shared], 90))
            # Measured on the synthetic complex drawer: a real 4 mm gap between the wrench and the knife reads
            # p50 0.46–0.48 / p90 0.57–0.63 of the lower top; the pliers' pivot reads 0.71 / 0.82 on a good
            # capture and 0.43 / 0.65 on a sparse 6-frame glide. The rule below separates the gap from the good
            # pivot and only just keeps the sparse pivot together — with fewer frames a joint can still split
            # (the user fixes that with a single click in Detect, which outlines the whole blob).
            # Chosen to always separate the gap (merged neighbours = one pocket for two tools, the worse failure);
            # a sparse capture may then split a pliers pivot, which one click in Detect undoes.
            split = p50 < saddle_frac * lo and p90 < 0.75 * lo   # FINAL_SADDLE_RULE
            import os
            if os.environ.get("TC_SADDLE_DEBUG"):
                print(f"saddle: tops {tops[a]:.1f}/{tops[b]:.1f} boundary {float(shared.sum()) * mm_per_px / 2:.0f} mm p50 {p50:.1f} p90 {p90:.1f} areas {int((ws == a).sum() * mm_per_px ** 2)}/{int((ws == b).sum() * mm_per_px ** 2)} mm2 -> {'SPLIT' if split else 'merge'}")
            if not split:
                parent[find(a)] = find(b)
    groups: Dict[int, np.ndarray] = {}
    for n in tops:
        r = find(n)
        groups[r] = groups.get(r, np.zeros(m.shape, bool)) | (ws == n)
    # watershed lines (ws == -1 → 0) belong to nobody: hand them to the touching group
    unassigned = m & ~np.any(np.stack(list(groups.values())), axis=0) if groups else np.zeros(m.shape, bool)
    if unassigned.any() and groups:
        _, (iy, ix) = ndi.distance_transform_edt(unassigned, return_indices=True)
        for r in groups:
            g = groups[r]
            groups[r] = g | (unassigned & g[iy, ix])
    out = []
    for g in sorted(groups.values(), key=lambda g: -int(g.sum())):
        full = np.zeros(mask.shape, bool)
        full[y0:y1, x0:x1] = g
        if full.sum() >= min_px:
            out.append(full)
    return out or [mask]


def snap_to_base(poly_px: np.ndarray, height_mm: np.ndarray, mm_per_px: float,
                 floor_mm: float = 1.5, max_mm: float = 12.0, smooth_mm: float = 1.5) -> np.ndarray:
    """Slide every vertex along the outline's normal until it sits where the tool meets the mat.

    `topo_footprint` puts the edge at half the height of each wall, which is the right answer for a vertical
    wall but lands part-way UP anything that slopes — so the outline hugs the top of the tool and the pocket
    comes out small. This walks each vertex outward while there is still something above the floor under it,
    and stops at the last point that is: the base perimeter. A vertex that already sits over bare mat walks
    inward instead, so the same pass tightens an outline that was too generous.

    Each vertex is capped at `max_mm` of travel so a vertex over a gap cannot run away to the next tool, and
    the result is smoothed, because a per-vertex march on a noisy height map is jittery by nature.
    """
    poly = np.asarray(poly_px, dtype=np.float64)
    n = len(poly)
    if n < 3 or height_mm is None:
        return poly
    h = np.nan_to_num(height_mm, nan=0.0).astype(np.float32)
    H, W = h.shape

    def sample(pts: np.ndarray) -> np.ndarray:
        x = np.clip(pts[:, 0] - 0.5, 0, W - 1.001)
        y = np.clip(pts[:, 1] - 0.5, 0, H - 1.001)
        x0, y0 = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
        tx, ty = x - x0, y - y0
        return (h[y0, x0] * (1 - tx) + h[y0, x0 + 1] * tx) * (1 - ty) + (h[y0 + 1, x0] * (1 - tx) + h[y0 + 1, x0 + 1] * tx) * ty

    # outward normals: perpendicular to the tangent, flipped to point away from the interior
    tangent = np.roll(poly, -1, axis=0) - np.roll(poly, 1, axis=0)
    ln = np.hypot(tangent[:, 0], tangent[:, 1])
    ln[ln < 1e-9] = 1.0
    nrm = np.column_stack([tangent[:, 1], -tangent[:, 0]]) / ln[:, None]
    # (t_y, -t_x) already points outward for a positively-signed ring; flip it for the other winding
    area = 0.5 * float(np.sum(poly[:, 0] * np.roll(poly, -1, axis=0)[:, 1] - np.roll(poly, -1, axis=0)[:, 0] * poly[:, 1]))
    if area < 0:
        nrm = -nrm

    # Measure the mat AROUND THIS TOOL rather than trusting a fixed floor. A drawer liner, a shadow or a
    # printed marker can sit well above 1.5 mm (p99 of bare mat on a real scan: 3.4 mm), and a march that only
    # has to stay "above 1.5 mm" then walks straight off the tool and onto the neighbour — measured at +85 %
    # area on a 6 mm tool, which had crawled onto an ArUco marker.
    ring_u8 = np.zeros(h.shape, np.uint8)
    cv2.fillPoly(ring_u8, [poly.astype(np.int32)], 1)
    k_in = max(1, int(round(4.0 / mm_per_px)))
    k_out = max(2, int(round(12.0 / mm_per_px)))
    outside = (cv2.dilate(ring_u8, np.ones((2 * k_out + 1,) * 2, np.uint8)) > 0) & (cv2.dilate(ring_u8, np.ones((2 * k_in + 1,) * 2, np.uint8)) == 0)
    if outside.any():
        near = h[outside]
        floor_mm = max(floor_mm, float(np.percentile(near, 97)) + 0.5)
    step = max(0.25, 0.35 / mm_per_px)          # ~0.35 mm per step
    max_px = max_mm / mm_per_px
    steps = int(max_px / step) + 1
    inside0 = sample(poly) > floor_mm
    out = poly.copy()
    # outward for vertices that still have tool under them, inward for those already over bare mat
    for sgn, mask in ((1.0, inside0), (-1.0, ~inside0)):
        if not mask.any():
            continue
        idx = np.nonzero(mask)[0]
        cur = poly[idx].copy()
        best = poly[idx].copy()
        alive = np.ones(len(idx), bool)
        for _ in range(steps):
            cur = cur + sgn * step * nrm[idx]
            over = sample(cur) > floor_mm
            if sgn > 0:
                alive &= over                    # stop the first time we step off the tool
                best[alive] = cur[alive]
            else:
                hit = alive & over               # walking back in: the first point ON the tool is the base
                best[hit] = cur[hit]
                alive &= ~over
            if not alive.any():
                break
        out[idx] = best
    if smooth_mm > 0:
        out = smooth_ring(out, sigma=smooth_mm / mm_per_px, tol=0.0)
    return out
