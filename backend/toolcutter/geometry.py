"""Mask -> polygon extraction and mm-space outline processing (offset, smoothing, notches)."""
from __future__ import annotations

import math
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
                        min_area_mm2: float = 200.0, above_frac: Optional[np.ndarray] = None) -> Tuple[np.ndarray, List[Dict]]:
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
