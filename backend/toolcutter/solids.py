"""Cleaning a scanned tool's 3D shape before it is carved — the deterministic half of "use AI to clean the 3D shapes".

Nolan (2026-10-03): "i think i may need to use some sort of AI model to 'clean' the 3D shapes of the scanner
objects/tools." The plan agreed: first the measurable, deterministic steps, with the recognition model only choosing
WHICH solid a tool is; a learned depth refiner later for black objects. Everything here works on one tool's placed
height raster `h` (mm above the floor, on the foam grid) inside its footprint `inside`:

  1. photo-guided filtering  — a guided filter (He et al.) with the photo as the guide: flat where the photo is flat,
                               crisp where the photo has an edge. Plain Gaussian smoothing cannot do both.
  2. mirror symmetry         — most hand tools are symmetric about their long axis; when the two halves agree to
                               within SYM_TOL_MM over most of the footprint, they are averaged.
  3. primitive fitting       — box (constant top), lying cylinder (circular profile across the short axis), extrusion
                               (one profile across, constant along the length). A primitive replaces the surface only
                               when it matches the scan within FIT_TOL_MM (p90), and the semantic hint from the
                               recognition model can prefer one (with a looser SEMANTIC_TOL_MM) or forbid them all.
Every decision is returned in a report so the UI can say "fitted as box, residual 0.4 mm".
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

SYM_TOL_MM = 0.8          # median |h - mirror(h)| for the two halves to be averaged
SYM_MIN_OVERLAP = 0.85    # ... over at least this fraction of the footprint
FIT_TOL_MM = 0.8          # p90 |h - primitive| for an automatic primitive
SEMANTIC_TOL_MM = 1.5     # ... when the recognition model asked for that primitive
GUIDE_RADIUS_MM = 3.0
GUIDE_EPS = 0.08 ** 2     # guide in [0, 1]; smaller = edges preserved more sharply. 0.02 transferred the photo's
                          # TEXTURE into the depth (the black neoprene case's sheen became a 0.41 mm rms static on
                          # its pocket floor, 2026-10-03); 0.08 keeps edges of >= 0.3 contrast and flattens texture
                          # below ~0.1 — measured 0.41 -> see test_solids / CLAUDE.md


# ----------------------------------------------------------------------------- guided filter

def _box(x: np.ndarray, r: int) -> np.ndarray:
    return cv2.blur(x, (2 * r + 1, 2 * r + 1), borderType=cv2.BORDER_REFLECT)


def guided_filter(p: np.ndarray, guide: np.ndarray, radius_px: int, eps: float, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """He, Sun & Tang's guided filter: output q = a*I + b with a, b fitted per window, so q follows the GUIDE's edges
    while smoothing p. With `mask`, only in-mask samples take part (normalised windows) so the floor around a tool
    does not pull its edge down."""
    I = guide.astype(np.float32); P = p.astype(np.float32)
    r = max(1, int(radius_px))
    if mask is None:
        w = np.ones_like(P)
    else:
        w = mask.astype(np.float32)
    N = np.maximum(_box(w, r), 1e-6)
    mean_I = _box(I * w, r) / N
    mean_p = _box(P * w, r) / N
    corr_I = _box(I * I * w, r) / N
    corr_Ip = _box(I * P * w, r) / N
    var_I = corr_I - mean_I * mean_I
    cov_Ip = corr_Ip - mean_I * mean_p
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    mean_a = _box(a * w, r) / N
    mean_b = _box(b * w, r) / N
    return (mean_a * I + mean_b).astype(np.float32)


# ----------------------------------------------------------------------------- helpers

def _axis_frame(inside: np.ndarray):
    """Rotate so the tool's long axis is vertical (y). Returns (M 2x3 forward, Minv, (W, H) of the rotated canvas)."""
    ys, xs = np.nonzero(inside)
    pts = np.column_stack([xs, ys]).astype(np.float32)
    (cx, cy), (w, h), ang = cv2.minAreaRect(pts)
    if w > h:                       # make the long side vertical
        ang += 90.0
    M = cv2.getRotationMatrix2D((float(cx), float(cy)), ang, 1.0)
    # canvas big enough for the rotated footprint
    diag = int(math.ceil(math.hypot(inside.shape[1], inside.shape[0]))) + 4
    M[0, 2] += diag / 2 - cx; M[1, 2] += diag / 2 - cy
    Minv = cv2.invertAffineTransform(M)
    return M, Minv, (diag, diag)


def _warp(img: np.ndarray, M, size, nearest: bool = False) -> np.ndarray:
    return cv2.warpAffine(img, M, size, flags=cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


# ----------------------------------------------------------------------------- the cleaner

def semantic_tolerance(confidence: Optional[float]) -> float:
    """How far a model-named primitive may sit from the scan and still replace it. The semantic layer exists to
    override NOISY data on known solids — the Anker charger (box, 0.72), the pin box (box, 0.90) and the pill bottle
    (cylinder, 0.8) on the desk drawer all failed a flat 1.5 mm rule because their depth is hole-filled sensor noise.
    A confident call earns a wide berth; a guess earns little more than the automatic rule."""
    c = float(confidence or 0.0)
    if c >= 0.85:
        return 4.0
    if c >= 0.7:
        return 2.5
    return SEMANTIC_TOL_MM


def clean_relief(h: np.ndarray, inside: np.ndarray, res_mm: float, guide: Optional[np.ndarray] = None,
                 solid_hint: Optional[str] = None, symmetric_hint: Optional[bool] = None,
                 hint_confidence: Optional[float] = None) -> Tuple[np.ndarray, Dict]:
    """Return (cleaned heights on the same grid, report). `guide` is the photo (any channels) on the same grid."""
    report: Dict = {"solid": "freeform", "steps": []}
    inside = inside.astype(bool)
    if not inside.any():
        return h, report
    out = np.nan_to_num(h.astype(np.float32), nan=0.0)
    # 1. photo-guided filtering inside the footprint
    if guide is not None:
        g = guide
        if g.ndim == 3:
            g = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)
        g = g.astype(np.float32) / 255.0
        r = max(1, int(round(GUIDE_RADIUS_MM / res_mm)))
        q = guided_filter(out, g, r, GUIDE_EPS, mask=inside)
        out = np.where(inside, q, out)
        report["steps"].append({"step": "photo_guided_filter", "radius_mm": GUIDE_RADIUS_MM})
    # work in the tool's own axis frame
    M, Minv, size = _axis_frame(inside)
    hr = _warp(out, M, size)
    mr = _warp(inside.astype(np.uint8), M, size, nearest=True).astype(bool)
    if not mr.any():
        return out, report
    ys, xs = np.nonzero(mr)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    H = hr[y0:y1, x0:x1]; Mm = mr[y0:y1, x0:x1]
    # 2. mirror symmetry about the long (vertical) axis
    if symmetric_hint is not False:
        Hf = H[:, ::-1]; Mf = Mm[:, ::-1]
        both = Mm & Mf
        overlap = float(both.sum()) / float(Mm.sum())
        if overlap >= SYM_MIN_OVERLAP and both.sum() > 20:
            diff = float(np.median(np.abs(H[both] - Hf[both])))
            if diff <= SYM_TOL_MM or (symmetric_hint and diff <= 2 * SYM_TOL_MM):
                H = np.where(both, 0.5 * (H + Hf), H)
                report["steps"].append({"step": "mirror_symmetry", "median_mismatch_mm": round(diff, 2), "overlap": round(overlap, 2)})
    # 3. primitives, fitted on the footprint eroded by ~2 mm (the edge ramp is the sensor's, not the tool's)
    k = max(3, int(round(2.0 / res_mm)) * 2 + 1)
    core = cv2.erode(Mm.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))).astype(bool)
    if core.sum() < 30:
        core = Mm
    fits = {}
    # box: one flat top
    top = float(np.percentile(H[core], 85))
    fits["box"] = (np.full_like(H, top), float(np.percentile(np.abs(H[core] - top), 90)), {"top_mm": round(top, 2)})
    # extrusion: a profile across the short axis, constant along the long axis (column-wise median)
    col_med = np.array([np.median(H[core[:, j], j]) if core[:, j].sum() >= 3 else np.nan for j in range(H.shape[1])], np.float32)
    good = np.isfinite(col_med)
    if good.sum() >= 3:
        jj = np.arange(H.shape[1]); col_med = np.interp(jj, jj[good], col_med[good]).astype(np.float32)
        prof = np.tile(col_med, (H.shape[0], 1))
        fits["extrusion"] = (prof, float(np.percentile(np.abs(H[core] - prof[core]), 90)), {})
        # lying cylinder: circle through the (x, height) profile
        xs_mm = jj[good] * res_mm; zs = col_med[good]
        if good.sum() >= 5 and np.ptp(zs) > 1.0:
            A = np.column_stack([2 * xs_mm, 2 * zs, np.ones_like(xs_mm)])
            bvec = xs_mm ** 2 + zs ** 2
            sol, *_ = np.linalg.lstsq(A, bvec, rcond=None)
            cx, cz = sol[0], sol[1]; R = math.sqrt(max(1e-6, sol[2] + cx * cx + cz * cz))
            xx = jj * res_mm
            under = R * R - (xx - cx) ** 2
            cyl_prof = np.where(under > 0, cz + np.sqrt(np.clip(under, 0, None)), 0.0).astype(np.float32)
            cyl_prof = np.clip(cyl_prof, 0.0, None)
            cyl = np.tile(cyl_prof, (H.shape[0], 1))
            if 3.0 <= R <= 150.0:
                fits["cylinder_lying"] = (cyl, float(np.percentile(np.abs(H[core] - cyl[core]), 90)), {"radius_mm": round(R, 1)})
    # upright cylinder (a round can): the box test already covers it (flat top); the footprint tells the rest
    order = ["box", "cylinder_lying", "extrusion"]
    chosen = None
    hint = (solid_hint or "").lower()
    conf = float(hint_confidence or 0.0)
    sem_tol = semantic_tolerance(hint_confidence)
    # footprint shape: how box-like / bar-like the outline itself is (fill of its min-area rectangle, aspect)
    fill = float(Mm.sum()) / float(max(1, Mm.shape[0] * Mm.shape[1]))
    aspect = Mm.shape[0] / max(1, Mm.shape[1])
    constructed = None
    if hint in ("box", "cylinder_upright") and conf >= 0.8 and fill >= 0.75 and fits["box"][1] > sem_tol:
        # CONSTRUCTED from what the model knows + robust statistics: the pin box (clear lid, pins inside) and the pill
        # bottle read as sensor garbage, so no fit passes — but a confident "box" with a box-shaped footprint IS a flat
        # slab at its measured top (Nolan: "if it sees a hammer, it should make sure the outline looks like a hammer")
        constructed = ("box", np.full_like(H, top), {"top_mm": round(top, 2), "constructed": True})
    elif hint == "cylinder_lying" and conf >= 0.8 and fill >= 0.7 and aspect >= 1.4 and fits.get("cylinder_lying", (None, 99.0))[1] > sem_tol:
        R = 0.5 * Mm.shape[1] * res_mm
        top_h = float(np.percentile(H[core], 90))
        cz = min(max(top_h - R, -0.6 * R), 0.0)          # body lying on the floor: centre at or below floor level
        xx = (np.arange(H.shape[1]) - 0.5 * (H.shape[1] - 1)) * res_mm
        under = R * R - xx ** 2
        prof = np.clip(np.where(under > 0, cz + np.sqrt(np.clip(under, 0, None)), 0.0), 0.0, None).astype(np.float32)
        constructed = ("cylinder_lying", np.tile(prof, (H.shape[0], 1)), {"radius_mm": round(R, 1), "constructed": True})
    if constructed is not None:
        name, model, extra = constructed
        fits[name] = (model, float(np.percentile(np.abs(H[core] - model[core]), 90)), extra)
        chosen = name
    elif hint in fits and fits[hint][1] <= sem_tol:
        chosen = hint
    elif hint == "freeform":
        chosen = None
    elif hint == "cylinder_upright" and fits["box"][1] <= sem_tol:
        chosen = "box"
    else:
        for name in order:
            if name in fits and fits[name][1] <= FIT_TOL_MM:
                chosen = name; break
    report["fits"] = {name: round(v[1], 2) for name, v in fits.items()}
    if chosen is not None:
        model, resid, extra = fits[chosen]
        # keep the sensor's natural edge ramp (so the wall profile still finds the rim) by blending the primitive in
        # over the eroded band: core = primitive, band = original
        Hn = np.where(core, model, H)
        report["solid"] = chosen; report["residual_mm"] = round(resid, 2); report.update(extra)
        report["steps"].append({"step": "primitive", "solid": chosen, "residual_p90_mm": round(resid, 2),
                                "by": "model (constructed)" if extra.get("constructed") else ("model" if hint == chosen else "fit"),
                                "tolerance_mm": sem_tol if hint == chosen else FIT_TOL_MM})
        H = Hn
    hr2 = hr.copy(); hr2[y0:y1, x0:x1] = np.where(Mm, H, hr[y0:y1, x0:x1])
    back = _warp(hr2, Minv, (out.shape[1], out.shape[0]))
    out = np.where(inside, back, out).astype(np.float32)
    return out, report
