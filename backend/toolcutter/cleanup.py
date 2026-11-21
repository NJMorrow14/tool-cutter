"""Knowledge-driven outline cleanup.

The split that keeps the millimetres honest: a MODEL may supply knowledge ("this is a steel rule: a rectangle",
"symmetric about its long axis", "the shaft is round") but it never emits a coordinate. THIS module turns such
knowledge into constraints and applies them to the MEASURED trace deterministically — snapping edges that are
already nearly parallel, squaring corners that are already nearly square, averaging halves that already nearly
mirror — and REFUSES a constraint the trace does not support, saying by how much. Every proposal reports the
largest vertex move and the area change, so it can be scored against calipers and shown before it is accepted.

Nothing here is applied silently; the frontend shows the proposal as a ghost and the user accepts or rejects.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .geometry import _resample_closed

# tolerances (mm / degrees): "already nearly" means within these; beyond them a constraint is refused
PAR_TOL_DEG = 4.0        # two edges closer than this to parallel are made parallel
RIGHT_TOL_DEG = 4.0      # two edge directions closer than this to 90 deg are squared
SYM_TOL_MM = 2.0         # a vertex and its mirror partner closer than this are averaged
SYM_MIN_COVER = 0.70     # ... and at least this fraction of vertices must have such a partner
RECT_MIN_IOU = 0.93      # the trace must fill its min-area rectangle this well to BE a rectangle
CIRCLE_MAX_RESID = 0.6   # p95 radial residual (mm) for the trace to BE a circle
LINE_TOL_MM = 0.5        # bow tolerance for a run to count as a straight edge
LOCAL_LINE_MAX_MM = 2.5  # a run the model calls straight may deviate this much from its fitted line (sanity, not a fit test)
LOCAL_ARC_MAX_MM = 2.0   # same for a run it calls an arc
LOCAL_SHIFT_MAX_MM = 5.0 # how far a 'too tight' / 'too loose' run may be moved to the photo edge
LOCAL_RUN_MAX_FRAC = 0.6 # a single local edit may not span more than this much of the ring (a mark range read backwards)
MAX_AREA_CHANGE_LOCAL = 0.20  # the area cap when the model asked for local edits (cutting a shadow spur off a small tool)
SIMPLIFY_MM = 0.2        # Douglas-Peucker tolerance on the returned ring (the constraints run on a 0.5 mm resample)
MAX_AREA_CHANGE = 0.08   # a proposal that changes area more than this is refused unless a primitive was requested


def _resample(poly: np.ndarray, step: float = 0.5) -> np.ndarray:
    return _resample_closed(np.asarray(poly, dtype=np.float64), step)


def _area(poly: np.ndarray) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _corners(r: np.ndarray, step: float, corner_deg: float = 38.0) -> List[int]:
    """Indices of corners on a 2 mm-smoothed copy: signed turn summed over +-2 mm, peaks >= corner_deg, >= 3 mm apart."""
    from scipy.ndimage import gaussian_filter1d
    n = len(r)
    rc = np.column_stack([gaussian_filter1d(r[:, j], max(1.0, 2.0 / step), mode="wrap") for j in range(2)])
    d1 = rc - np.roll(rc, 1, axis=0)
    d2 = np.roll(rc, -1, axis=0) - rc
    ang = np.arctan2(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0], (d1 * d2).sum(axis=1))
    w = max(1, int(round(2.0 / step)))
    turn = np.abs(np.convolve(np.concatenate([ang[-w:], ang, ang[:w]]), np.ones(2 * w + 1), mode="valid"))
    thresh = math.radians(corner_deg)
    gap = max(1, int(round(3.0 / step)))
    taken = np.zeros(n, bool)
    out = []
    for i in np.argsort(-turn):
        if turn[i] < thresh:
            break
        if taken[i]:
            continue
        out.append(int(i))
        taken[np.arange(i - gap, i + gap + 1) % n] = True
    return sorted(out)


def _fit_line(seg: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Least-squares line through a run (trimmed ends): centre, unit direction, bow (low-passed max deviation)."""
    from scipy.ndimage import gaussian_filter1d
    trim = max(1, len(seg) // 6) if len(seg) > 6 else 0
    core = seg[trim:len(seg) - trim] if trim else seg
    c = core.mean(axis=0)
    _, _, vt = np.linalg.svd(core - c, full_matrices=False)
    d = vt[0]
    if d[0] < 0 or (abs(d[0]) < 1e-9 and d[1] < 0):
        d = -d
    nrm = np.array([-d[1], d[0]])
    dev = (seg - c) @ nrm
    lp = gaussian_filter1d(dev, 6.0, mode="nearest") if len(dev) > 6 else dev
    return c, d, float(np.abs(lp).max())


def _angle_deg(d: np.ndarray) -> float:
    return math.degrees(math.atan2(d[1], d[0])) % 180.0


def _rot(d: np.ndarray, deg: float) -> np.ndarray:
    a = math.radians(deg)
    return np.array([d[0] * math.cos(a) - d[1] * math.sin(a), d[0] * math.sin(a) + d[1] * math.cos(a)])


def _intersect(c1, d1, c2, d2) -> Optional[np.ndarray]:
    den = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(den) < 1e-9:
        return None
    t = ((c2[0] - c1[0]) * d2[1] - (c2[1] - c1[1]) * d2[0]) / den
    return c1 + t * d1


def _area_axes(r: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Area centroid and principal axes of the POLYGON (density-independent). A point-centroid/PCA of a resampled
    ring is biased toward whichever side is noisier: a zigzagging flank has a longer path, so it gets more samples —
    on a test capsule that dragged the axis 20 mm off and made a symmetric shape read 17 %% symmetric."""
    x, y = r[:, 0], r[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cross = x * y1 - x1 * y
    A = 0.5 * cross.sum()
    if abs(A) < 1e-9:
        c = r.mean(axis=0); _, _, vt = np.linalg.svd(r - c, full_matrices=False); return c, vt[0], vt[1]
    cx = ((x + x1) * cross).sum() / (6 * A); cy = ((y + y1) * cross).sum() / (6 * A)
    # second moments about the centroid (standard polygon inertia formulas)
    xs, ys, xs1, ys1 = x - cx, y - cy, x1 - cx, y1 - cy
    cr = xs * ys1 - xs1 * ys
    Ixx = (cr * (ys * ys + ys * ys1 + ys1 * ys1)).sum() / 12
    Iyy = (cr * (xs * xs + xs * xs1 + xs1 * xs1)).sum() / 12
    Ixy = (cr * (xs * ys1 + 2 * xs * ys + 2 * xs1 * ys1 + xs1 * ys)).sum() / 24
    cov = np.array([[Iyy, Ixy], [Ixy, Ixx]]) / A        # covariance of the area
    w, v = np.linalg.eigh(cov)
    long_ax, short_ax = v[:, 1], v[:, 0]
    return np.array([cx, cy]), long_ax / np.linalg.norm(long_ax), short_ax / np.linalg.norm(short_ax)


def _symmetrize(r: np.ndarray, axis: str) -> Tuple[Optional[np.ndarray], Dict]:
    """Mirror across the area's principal axis ('long') or minor axis ('short') and average matched partners."""
    c, long_ax, short_ax = _area_axes(r)
    ax = long_ax if axis == "long" else short_ax
    nrm = np.array([-ax[1], ax[0]])
    rel = r - c
    mirrored = rel - 2.0 * np.outer(rel @ nrm, nrm) + c
    # nearest mirrored partner for every vertex
    d = np.linalg.norm(r[:, None, :] - mirrored[None, :, :], axis=2)
    j = np.argmin(d, axis=1)
    dist = d[np.arange(len(r)), j]
    cover = float((dist <= SYM_TOL_MM).mean())
    info = {"type": "mirror_symmetry", "axis": axis, "cover": round(cover, 3), "median_mismatch_mm": round(float(np.median(dist)), 2),
            "p90_mismatch_mm": round(float(np.percentile(dist, 90)), 2)}
    if cover < SYM_MIN_COVER:
        info["refused"] = f"only {cover:.0%} of the outline has a mirror partner within {SYM_TOL_MM} mm"
        return None, info
    out = r.copy()
    ok = dist <= SYM_TOL_MM
    out[ok] = 0.5 * (r[ok] + mirrored[j[ok]])
    info["residual_mm"] = round(float(np.percentile(dist[ok], 90)) / 2, 2)
    return out, info


def _rect_iou(r: np.ndarray) -> Tuple[float, np.ndarray]:
    box = cv2.boxPoints(cv2.minAreaRect(r.astype(np.float32)))
    # raster IoU on a 0.5 mm grid
    allp = np.vstack([r, box])
    x0, y0 = allp.min(axis=0) - 2
    sc = 2.0
    size = (np.ceil((allp.max(axis=0) - allp.min(axis=0) + 4) * sc)).astype(int)[::-1]
    def mask(p):
        m = np.zeros(size, np.uint8)
        cv2.fillPoly(m, [np.round((p - [x0, y0]) * sc).astype(np.int32)], 1)
        return m.astype(bool)
    a, b = mask(r), mask(box)
    return float((a & b).sum() / max(1, (a | b).sum())), box.astype(np.float64)


def _circle_fit(r: np.ndarray) -> Tuple[np.ndarray, float, float]:
    x, y = r[:, 0], r[:, 1]
    A = np.column_stack([x, y, np.ones_like(x)])
    sol, *_ = np.linalg.lstsq(A, x * x + y * y, rcond=None)
    cx, cy = sol[0] / 2, sol[1] / 2
    rad = math.sqrt(max(1e-9, sol[2] + cx * cx + cy * cy))
    res = np.abs(np.hypot(x - cx, y - cy) - rad)
    return np.array([cx, cy]), rad, float(np.percentile(res, 95))


def prepare_ring(poly_mm, step: float = 0.5) -> np.ndarray:
    """The working ring every stage operates on: a constant-step resample, smoothed by 1 mm first when the trace
    zigzags (perimeter > 1.15 x convex hull perimeter) so that sample density follows the shape, not the noise.
    Marks shown to the model are indices into THIS ring, so it must be computed once and passed around."""
    P = np.asarray(poly_mm, dtype=np.float64)
    r = _resample(P, step)
    hull = cv2.convexHull(r.astype(np.float32)).reshape(-1, 2)
    hull_per = float(np.linalg.norm(np.roll(hull, -1, axis=0) - hull, axis=1).sum())
    if len(r) * step > 1.15 * hull_per:
        from scipy.ndimage import gaussian_filter1d
        sm = np.column_stack([gaussian_filter1d(r[:, j], 1.0 / step, mode="wrap") for j in range(2)])
        r = _resample(sm, step)
    return r


MARK_SPACING_MM = 8.0   # marks about this far apart along the outline: a 70 x 30 mm tool gets ~25, a hammer 40


def mark_indices(r: np.ndarray, n_marks: Optional[int] = None) -> List[int]:
    """Evenly spaced reference marks around a constant-step ring (what the model sees numbered on the pictures).
    Mark k sits at r[idx[k]]; a run 'from mark a to mark b' follows the ring forward from idx[a] to idx[b].
    By default the count follows the perimeter (MARK_SPACING_MM apart, 10..40): 32 marks on a 70 x 30 mm tool
    overlapped each other and their own outline on the real drawer."""
    n = len(r)
    if n_marks is None:
        step = float(np.median(np.linalg.norm(np.roll(r, -1, axis=0) - r, axis=1))) or 0.5
        n_marks = int(round(n * step / MARK_SPACING_MM))
        n_marks = max(10, min(40, n_marks))
    n_marks = max(8, min(n_marks, n // 4))
    return [int(v) for v in np.linspace(0, n, n_marks, endpoint=False)]


def _run_indices(n: int, a: int, b: int) -> np.ndarray:
    if b <= a:
        b += n
    return np.arange(a, b + 1) % n


def _outward_sign(r: np.ndarray) -> float:
    x, y = r[:, 0], r[:, 1]
    return 1.0 if (x * np.roll(y, -1) - np.roll(x, -1) * y).sum() > 0 else -1.0


def _normals(r: np.ndarray) -> np.ndarray:
    t = np.roll(r, -1, axis=0) - np.roll(r, 1, axis=0)
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
    nrm = np.column_stack([t[:, 1], -t[:, 0]]) * _outward_sign(r)   # (t_y, -t_x) is outward for a positive ring
    return nrm


def _photo_edge_offsets(r: np.ndarray, idx: np.ndarray, photo: np.ndarray, origin_mm: np.ndarray, mpp: float,
                        direction: int, max_mm: float = LOCAL_SHIFT_MAX_MM) -> Tuple[Optional[float], Dict]:
    """Where, along each point's normal, is the photo's strongest edge in `direction` (+1 outward, -1 inward)?
    Returns the ROBUST (median) offset in mm for the whole run, or None when fewer than half the points saw an edge
    worth the name. The model has said this run is off and in which direction; the photo settles by how much."""
    gray = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY).astype(np.float32) if photo.ndim == 3 else photo.astype(np.float32)
    gray = cv2.GaussianBlur(gray, (0, 0), 0.6 / mpp)
    H, W = gray.shape
    nrm = _normals(r)
    offs = np.arange(0.0, max_mm + 1e-6, 0.25) * direction
    found = []
    for i in idx[::max(1, len(idx) // 60)]:
        p = r[i]; nv = nrm[i]
        pts = (p[None, :] + offs[:, None] * nv[None, :] - origin_mm) / mpp     # local px
        xs, ys = pts[:, 0], pts[:, 1]
        ok = (xs >= 1) & (xs < W - 1) & (ys >= 1) & (ys < H - 1)
        if ok.sum() < 6:
            continue
        prof = cv2.remap(gray, xs.astype(np.float32).reshape(1, -1), ys.astype(np.float32).reshape(1, -1), cv2.INTER_LINEAR).ravel()
        g = np.abs(np.gradient(prof)); g[~ok] = 0
        g[:2] = 0   # ignore the trace's own position (the first 0.5 mm): we are looking for the edge BEYOND it
        j = int(np.argmax(g))
        if g[j] >= 6.0:                      # grey levels per 0.25 mm; the liner's texture is well under this
            found.append(abs(float(offs[j])))
    info = {"points_with_edge": len(found), "points_tested": int(len(idx[::max(1, len(idx) // 60)]))}
    if len(found) < max(3, info["points_tested"] // 2):
        return None, info
    return float(np.median(found)), info


def _taper(k: int, ramp_pts: int = 12) -> np.ndarray:
    """Raised-cosine weights, 0 at both ends of a run and 1 in the middle (ramp ~6 mm at 0.5 mm step), so a local
    edit fades into the untouched ring instead of stepping. On the hammer the first version projected the shaft
    run onto its line and left a visible jog where the run's end met the head's taper."""
    ramp = min(k // 2, ramp_pts)
    w = np.ones(k)
    if ramp > 0:
        t = 0.5 - 0.5 * np.cos(np.linspace(0, math.pi, ramp, endpoint=False))
        w[:ramp] = t; w[-ramp:] = t[::-1]
    return w


def apply_local_edits(r: np.ndarray, marks: List[int], edits: List[Dict], photo: Optional[np.ndarray] = None,
                      origin_mm: Optional[np.ndarray] = None, mpp: Optional[float] = None) -> Tuple[np.ndarray, List[Dict], List[Dict]]:
    """Apply the model's place-by-place verdicts to the working ring. Each edit names a run by two marks and a kind:
      straight   - the run is a straight edge: project its points onto the fitted line
      arc        - a circular arc: project onto the fitted circle
      spur       - the trace sticks OUT beyond the tool (shadow, noise, a neighbour): replace the run by its chord
      notch      - the trace dips INTO the tool: replace the run by its chord
      too_tight  - the outline sits inside the real edge: move the run OUT to the photo's edge
      too_loose  - the outline sits outside the real edge: move the run IN to the photo's edge
    Anything else (merged_neighbour, missing_part, ...) is advice and is passed through untouched.
    The model never gives a coordinate: every number here is measured from the trace or the photo, and each edit is
    refused when the data does not bear it out (a 'straight' run 4 mm from any line, a 'too tight' run with no edge).
    """
    r = r.copy(); n = len(r)
    applied: List[Dict] = []; refused: List[Dict] = []
    for e in edits or []:
        kind = str(e.get("kind") or "").lower()
        try:
            a = marks[int(e.get("from_mark"))]; b = marks[int(e.get("to_mark"))]
        except (TypeError, ValueError, IndexError):
            refused.append({"type": f"local_{kind}", "refused": "bad mark range", "edit": e}); continue
        idx = _run_indices(n, a, b)
        rec = {"type": f"local_{kind}", "from_mark": e.get("from_mark"), "to_mark": e.get("to_mark"), "note": e.get("note", "")}
        if kind in ("merged_neighbour", "missing_part", "other"):
            rec["advice"] = True; applied.append(rec); continue
        if len(idx) > LOCAL_RUN_MAX_FRAC * n or len(idx) < 3:
            rec["refused"] = f"run spans {len(idx)} of {n} ring points"; refused.append(rec); continue
        seg = r[idx]
        if kind == "straight":
            c, d, bow = _fit_line(seg)
            resid = np.abs((seg - c) @ np.array([-d[1], d[0]]))
            if float(np.percentile(resid, 95)) > LOCAL_LINE_MAX_MM:
                rec["refused"] = f"points are up to {resid.max():.1f} mm from any straight line"; refused.append(rec); continue
            r[idx] = seg + _taper(len(idx))[:, None] * ((c + np.outer((seg - c) @ d, d)) - seg)
            rec["max_move_mm"] = round(float(resid.max()), 2); applied.append(rec)
        elif kind == "arc":
            c, rad, p95 = _circle_fit(seg)
            if p95 > LOCAL_ARC_MAX_MM or not np.isfinite(rad) or rad < 1.0:
                rec["refused"] = f"points are {p95:.1f} mm (p95) from any circle"; refused.append(rec); continue
            v = seg - c; v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)
            r[idx] = seg + _taper(len(idx))[:, None] * ((c + v * rad) - seg)
            rec["radius_mm"] = round(float(rad), 2); rec["max_move_mm"] = round(float(p95), 2); applied.append(rec)
        elif kind in ("spur", "notch"):
            chord = np.linspace(seg[0], seg[-1], len(idx))
            before = _area(r); test = r.copy(); test[idx] = chord; after = _area(test)
            d_area = after - before
            if kind == "spur" and d_area > 1e-6:
                rec["refused"] = "that run is a dent, not a spur (bridging it would ADD area)"; refused.append(rec); continue
            if kind == "notch" and d_area < -1e-6:
                rec["refused"] = "that run bulges out, not in (bridging it would REMOVE area)"; refused.append(rec); continue
            r = test
            rec["area_change_mm2"] = round(float(d_area), 1)
            rec["max_move_mm"] = round(float(np.linalg.norm(seg - chord, axis=1).max()), 2); applied.append(rec)
        elif kind in ("too_tight", "too_loose"):
            if photo is None or mpp is None or origin_mm is None:
                rec["refused"] = "no photo to find the edge in"; refused.append(rec); continue
            direction = 1 if kind == "too_tight" else -1
            off, info = _photo_edge_offsets(r, idx, photo, origin_mm, mpp, direction)
            rec.update(info)
            if off is None or off < 0.3:
                rec["refused"] = "the photo shows no clear edge beyond the trace on that run" if off is None else "the photo edge is already where the trace is"
                refused.append(rec); continue
            r[idx] = seg + (off * direction) * _taper(len(idx))[:, None] * _normals(r)[idx]
            rec["moved_mm"] = round(float(off * direction), 2); applied.append(rec)
        else:
            rec["refused"] = f"unknown edit kind '{kind}'"; refused.append(rec)
    return r, applied, refused


def propose(poly_mm, hints: Optional[Dict] = None, step: float = 0.5, ring: Optional[np.ndarray] = None,
            local: Optional[List[Dict]] = None, marks: Optional[List[int]] = None, photo: Optional[np.ndarray] = None,
            origin_mm: Optional[np.ndarray] = None, mpp: Optional[float] = None) -> Dict:
    """Propose a cleaned outline for a traced polygon (mm).

    hints (from the recognition step, all optional): shape_class in {rectangle, rounded_rectangle, capsule, circle,
    L_shape, T_shape, irregular}; symmetric_axis in {long, short, none}; straight_edges, right_angles (bool).
    With no hints every constraint is auto-detected but must be strongly supported by the trace.
    Returns {polygon_mm, applied: [...], refused: [...], max_move_mm, area_before_mm2, area_after_mm2, area_change_pct}.
    """
    hints = hints or {}
    P = np.asarray(poly_mm, dtype=np.float64)
    if len(P) < 4:
        return {"polygon_mm": P.tolist(), "applied": [], "refused": [{"type": "all", "refused": "too few points"}],
                "max_move_mm": 0.0, "area_before_mm2": _area(P), "area_after_mm2": _area(P), "area_change_pct": 0.0}
    area0 = _area(P)
    r = prepare_ring(P, step) if ring is None else np.asarray(ring, dtype=np.float64).copy()
    applied: List[Dict] = []
    refused: List[Dict] = []
    allow_local = False
    if local:
        r, la, lr = apply_local_edits(r, marks or mark_indices(r), local, photo=photo, origin_mm=origin_mm, mpp=mpp)
        applied += la; refused += lr
        allow_local = any(not a.get("advice") for a in la)
    shape_class = str(hints.get("shape_class") or "").lower()

    # ---- circle first: a round trace has no edges to fit. (A rectangle is NOT taken from the min-area box — that
    # hugs the outermost noise and came out 0.6 mm fat on a 100 x 40 test; the fitted-edge path below produces the
    # exact rectangle, and the result is classified as one afterwards.)
    rect_iou, rect_box = _rect_iou(r)
    want_rect = shape_class == "rectangle" or (shape_class == "" and rect_iou >= 0.95)
    if shape_class in ("", "circle"):
        c, rad, p95 = _circle_fit(r)
        lo, hi = sorted(cv2.minAreaRect(r.astype(np.float32))[1])
        if (shape_class == "circle" or (hi / max(lo, 1e-6) < 1.15 and p95 <= CIRCLE_MAX_RESID)) and p95 <= CIRCLE_MAX_RESID and rad >= 2:
            k = max(24, int(2 * math.pi * rad / 1.0))
            th = np.linspace(0, 2 * math.pi, k, endpoint=False)
            out = np.column_stack([c[0] + rad * np.cos(th), c[1] + rad * np.sin(th)])
            applied.append({"type": "circle", "radius_mm": round(rad, 2), "p95_residual_mm": round(p95, 2)})
            return _result(P, out, applied, refused, area0, _max_move(r, out), allow_big=True)
        if shape_class == "circle":
            refused.append({"type": "circle", "refused": f"radial residual p95 {p95:.2f} mm (needs <= {CIRCLE_MAX_RESID})"})

    # ---- symmetry (hinted, or auto when the trace supports it strongly)
    sym_axis = str(hints.get("symmetric_axis") or "").lower()
    if sym_axis in ("long", "short"):
        out, info = _symmetrize(r, sym_axis)
        (applied if out is not None else refused).append(info)
        if out is not None:
            r = out
    elif sym_axis == "" and shape_class not in ("irregular",):
        for ax in ("long", "short"):
            out, info = _symmetrize(r, ax)
            if out is not None and info["cover"] >= 0.95 and info["p90_mismatch_mm"] <= 1.2:   # auto only when unmistakable
                applied.append(info); r = out; break

    # ---- straight edges, parallel groups and right angles
    corners = _corners(r, step)
    if len(corners) >= 2 and hints.get("straight_edges", True):
        n = len(r)
        runs = [np.arange(a, b + 1) % n for a, b in zip(corners, corners[1:] + [corners[0] + n])]
        lines: List[Optional[Tuple[np.ndarray, np.ndarray]]] = []
        for idx in runs:
            seg = r[idx]
            if len(seg) >= 6:
                c, d, bow = _fit_line(seg)
                L = float(np.linalg.norm(seg[-1] - seg[0]))
                lines.append((c, d) if (bow <= LINE_TOL_MM and L >= 6.0) else None)
            else:
                lines.append(None)
        line_ids = [i for i, l in enumerate(lines) if l is not None]
        if len(line_ids) >= 2:
            # parallel groups by direction (mod 180), length-weighted mean
            lengths = {i: float(np.linalg.norm(r[runs[i]][-1] - r[runs[i]][0])) for i in line_ids}
            groups: List[List[int]] = []
            for i in sorted(line_ids, key=lambda k: -lengths[k]):
                ang = _angle_deg(lines[i][1])
                for g in groups:
                    ga = _angle_deg(lines[g[0]][1])
                    dd = abs((ang - ga + 90) % 180 - 90)
                    if dd <= PAR_TOL_DEG:
                        g.append(i); break
                else:
                    groups.append([i])
            # mean direction per group (vector mean of doubled angles)
            gdir: Dict[int, np.ndarray] = {}
            worst_par = 0.0
            for g in groups:
                if len(g) < 2:
                    gdir[id(g)] = lines[g[0]][1]; continue
                a2 = np.array([math.radians(2 * _angle_deg(lines[i][1])) for i in g]); w = np.array([lengths[i] for i in g])
                m = math.atan2((w * np.sin(a2)).sum(), (w * np.cos(a2)).sum()) / 2
                d = np.array([math.cos(m), math.sin(m)])
                for i in g:
                    worst_par = max(worst_par, abs((_angle_deg(lines[i][1]) - _angle_deg(d) + 90) % 180 - 90))
                gdir[id(g)] = d
            if any(len(g) >= 2 for g in groups):
                applied.append({"type": "parallel_edges", "groups": [len(g) for g in groups if len(g) >= 2], "max_correction_deg": round(worst_par, 2)})
            # right angles between the two biggest groups
            if len(groups) >= 2 and hints.get("right_angles", True):
                big = sorted(groups, key=lambda g: -sum(lengths[i] for i in g))[:2]
                d0, d1 = gdir[id(big[0])], gdir[id(big[1])]
                off = abs(((_angle_deg(d1) - _angle_deg(d0)) % 180) - 90)
                if off <= RIGHT_TOL_DEG:
                    gdir[id(big[1])] = _rot(d0, 90)
                    applied.append({"type": "right_angles", "correction_deg": round(off, 2)})
                elif off < 25:
                    refused.append({"type": "right_angles", "refused": f"edges meet at {90 + off:.1f} deg (needs within {RIGHT_TOL_DEG} deg of 90)"})
            # rebuild each straight run on its group direction through its own centre, then re-join
            newlines = {}
            for g in groups:
                for i in g:
                    newlines[i] = (lines[i][0], gdir[id(g)])
            pieces: List[np.ndarray] = []
            m = len(runs)
            for k in range(m):
                idx = runs[k]; seg = r[idx]
                if k in newlines:
                    c, d = newlines[k]
                    # endpoints: intersect with neighbouring straight runs, else project own ends
                    def endpoint(other, fallback):
                        if other in newlines:
                            x = _intersect(c, d, *newlines[other])
                            if x is not None and np.linalg.norm(x - fallback) <= 6.0:
                                return x
                        return c + d * float((fallback - c) @ d)
                    a = endpoint((k - 1) % m, seg[0]); b = endpoint((k + 1) % m, seg[-1])
                    pieces.append(np.vstack([a, b]))
                else:
                    pieces.append(seg[:-1] if len(seg) > 1 else seg)
            out = np.vstack(pieces)
            keep = np.linalg.norm(out - np.roll(out, 1, axis=0), axis=1) > 1e-6
            r = out[keep]

    if want_rect:
        if len(r) == 4 and any(a["type"] == "right_angles" for a in applied):
            w, h = sorted([float(np.linalg.norm(r[1] - r[0])), float(np.linalg.norm(r[2] - r[1]))])
            applied.append({"type": "rectangle", "iou_with_trace": round(rect_iou, 3), "size_mm": [round(w, 1), round(h, 1)]})
            return _result(P, r, applied, refused, area0, _max_move(_resample(P, step), r), allow_big=True)
        if shape_class == "rectangle":
            if rect_iou >= RECT_MIN_IOU:      # the edge fit did not close to four lines; the min-area box is the honest fallback
                applied.append({"type": "rectangle", "iou_with_trace": round(rect_iou, 3), "fallback": "min-area box",
                                "size_mm": [round(float(v), 1) for v in sorted(cv2.minAreaRect(_resample(P, step).astype(np.float32))[1])]})
                return _result(P, rect_box, applied, refused, area0, _max_move(_resample(P, step), rect_box), allow_big=True)
            refused.append({"type": "rectangle", "refused": f"trace fills its min-area rectangle only {rect_iou:.0%} (needs {RECT_MIN_IOU:.0%})"})
    move = _max_move(_resample(P, step), r)
    return _result(P, r, applied, refused, area0, move, allow_big=False, cap=MAX_AREA_CHANGE_LOCAL if allow_local else MAX_AREA_CHANGE)


def _max_move(a: np.ndarray, b: np.ndarray) -> float:
    """Largest distance from any point of ring b to ring a (how far the proposal strays from the trace)."""
    if len(a) == 0 or len(b) == 0:
        return 0.0
    d = np.linalg.norm(b[:, None, :] - a[None, :, :], axis=2)
    return float(d.min(axis=1).max())


def _result(P, out, applied, refused, area0, move, allow_big, cap: float = MAX_AREA_CHANGE) -> Dict:
    area1 = _area(np.asarray(out))
    change = (area1 / max(area0, 1e-9)) - 1.0
    if not allow_big and abs(change) > cap:
        refused.append({"type": "all", "refused": f"proposal would change area by {change:+.1%} (limit {cap:.0%}); trace kept"})
        return {"polygon_mm": P.tolist(), "applied": [], "refused": refused, "max_move_mm": 0.0,
                "area_before_mm2": round(area0, 1), "area_after_mm2": round(area0, 1), "area_change_pct": 0.0}
    # The constraints were applied on a 0.5 mm resample (hundreds to thousands of points). Hand back the SHAPE, not the
    # sampling: a 0.2 mm Douglas-Peucker pass keeps every straight run as 2 points and every arc as a handful, within
    # 0.2 mm of the constrained ring, so an accepted proposal has fewer handles than the trace, never twenty times more
    # (first real-drawer run: 57 -> 1446 points before this).
    out = np.asarray(out, dtype=np.float64)
    if len(out) > 8:
        simp = cv2.approxPolyDP(out.astype(np.float32).reshape(-1, 1, 2), SIMPLIFY_MM, True).reshape(-1, 2).astype(np.float64)
        if len(simp) >= 4:
            out = simp
    return {"polygon_mm": out.tolist(), "applied": applied, "refused": refused, "max_move_mm": round(move, 2),
            "area_before_mm2": round(area0, 1), "area_after_mm2": round(area1, 1), "area_change_pct": round(100 * change, 2)}


def smooth_enclosing(poly_mm, sigma_mm: float = 2.0, tol_mm: float = 0.6, step: float = 0.5) -> Dict:
    """A smooth, low-vertex version of a traced outline that still ENCLOSES the trace (Nolan: "nice smooth outlines
    that will create a professional looking toolboard" — a pocket may be generous by a few tenths, never tight).
    Gaussian smoothing along the ring (sigma_mm) rounds off the sensor's wobble and small spurs; wherever the smoothed
    ring then falls inside the original by more than 0.2 mm, that point is pushed back out along its normal (the
    smoothing of a convex corner cuts it; of a concave one it fills — filling is fine for a pocket, cutting is not);
    then `geometry.clean_ring(tol_mm)` re-expresses it as lines / arcs / curves. Returns {polygon_mm, vertices_before,
    vertices_after, max_inset_mm (how far the result still dips inside the trace, should be ~0), area_change_pct}."""
    from shapely.geometry import Polygon, Point
    from scipy.ndimage import gaussian_filter1d
    from . import geometry
    P = np.asarray(poly_mm, dtype=np.float64)
    if len(P) < 4:
        return {"polygon_mm": P.tolist(), "vertices_before": len(P), "vertices_after": len(P), "max_inset_mm": 0.0, "area_change_pct": 0.0}
    area0 = _area(P)
    r = _resample(P, step)
    # The thing to enclose is the trace's BODY, not its noise: a 0.75 mm blur removes the +-0.4 mm sensor jitter
    # and one-sample teeth while keeping anything a few millimetres wide. Enclosing every noise spike would push
    # the smooth line back out to each of them and give the jitter back (seen: 63 bumpy vertices on a clean capsule).
    body = np.column_stack([gaussian_filter1d(r[:, j], 1.0 / step, mode="wrap") for j in range(2)])
    orig = Polygon(body)
    if not orig.is_valid:
        orig = orig.buffer(0)
    if orig.geom_type == "MultiPolygon":          # a self-touching trace splits on buffer(0); keep the body
        orig = max(orig.geoms, key=lambda g: g.area)
    # Smooth PIECEWISE, between the corners: clean_ring finds the corners on the body, keeps straight runs straight
    # (lines meeting at their intersection) and smooths only the free curves, by `sigma_mm`. A plain Gaussian over the
    # whole ring rounded every corner by ~0.6 sigma and the enclosure step then grew the tool by a millimetre a side.
    out = np.asarray(geometry.clean_ring(body, tol=min(tol_mm, 0.4), curve_sigma_mm=max(0.5, sigma_mm)), dtype=np.float64)
    if len(out) < 4:
        out = body

    def _inset(ring: np.ndarray) -> float:
        worst = 0.0
        for q in _resample(ring, step):
            pt = Point(q)
            if orig.contains(pt):
                worst = max(worst, float(orig.exterior.distance(pt)))
        return worst

    # Smoothing cuts convex corners and tips (and clean_ring's chords sit inside curves by their tolerance). Rather
    # than pushing single points back out — which hands the jitter straight back — offset the WHOLE smooth line
    # outward by the deepest dip, so it stays smooth and encloses the body. A pocket a few tenths generous is right;
    # one that is tight is wrong.
    inset = _inset(out)
    if inset > 0.3:                                  # dips under 0.3 mm are the body's own residual noise bumps
        grown = Polygon(out).buffer(inset - 0.2, join_style=1, resolution=8)
        if grown.geom_type == "MultiPolygon":
            grown = max(grown.geoms, key=lambda g: g.area)
        ring = np.asarray(grown.exterior.coords)[:-1]
        simp = cv2.approxPolyDP(ring.astype(np.float32).reshape(-1, 1, 2), 0.2, True).reshape(-1, 2).astype(np.float64)
        if len(simp) >= 4:
            out = simp
        inset = _inset(out)
    area1 = _area(out)
    return {"polygon_mm": out.tolist(), "vertices_before": int(len(P)), "vertices_after": int(len(out)),
            "max_inset_mm": round(inset, 2), "area_change_pct": round(100 * (area1 / max(area0, 1e-9) - 1), 2)}
