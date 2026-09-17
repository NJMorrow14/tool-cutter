"""3D scan (mesh / point cloud) -> top-down height map + color raster.

Pipeline: load with trimesh -> sample surface points -> RANSAC-fit the dominant plane (the mat /
drawer floor) -> express points as (u, v, height) in that plane -> rasterize to a metric grid.
"""
from __future__ import annotations

import io
import logging
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

log = logging.getLogger(__name__)

UNIT_SCALE_TO_MM = {"mm": 1.0, "cm": 10.0, "m": 1000.0, "in": 25.4}
ABOVE_MAT_MM = 1.5  # points higher than this count as "tool", below as "mat" (scan noise floor)


@dataclass
class ScanRaster:
    color_bgr: np.ndarray          # uint8 HxWx3
    height_mm: np.ndarray          # float32 HxW (NaN where no data before fill)
    above_frac: np.ndarray         # float32 HxW, fraction of a cell's points that rise above the mat
    mm_per_px: float
    coverage: np.ndarray           # bool HxW, True where the scan had data
    plane_inlier_fraction: float
    unit_scale: float
    suggested_corners: Optional[np.ndarray]  # 4x2 px, TL TR BR BL, or None


def _load_points(data: bytes, ext: str, target_points: int = 3_000_000):
    """Return (points Nx3 float64, colors Nx3 uint8 or None)."""
    import trimesh  # type: ignore

    ext = ext.lower().lstrip(".")
    loaded = trimesh.load(io.BytesIO(data), file_type=ext, force=None, process=False)
    meshes = []
    clouds = []
    if isinstance(loaded, trimesh.Scene):
        for name, geom in loaded.geometry.items():
            tf = loaded.graph.get(name)[0] if name in loaded.graph.nodes_geometry else np.eye(4)
            try:
                tf = loaded.graph[name][0]
            except Exception:  # noqa: BLE001
                tf = np.eye(4)
            g = geom.copy()
            g.apply_transform(tf)
            (meshes if isinstance(g, trimesh.Trimesh) else clouds).append(g)
    elif isinstance(loaded, trimesh.Trimesh):
        meshes.append(loaded)
    elif isinstance(loaded, trimesh.PointCloud):
        clouds.append(loaded)
    else:
        raise RuntimeError(f"Unsupported 3D content: {type(loaded).__name__}")

    pts_list, col_list, has_color = [], [], True
    for m in meshes:
        if len(m.faces) == 0:
            clouds.append(trimesh.PointCloud(m.vertices, colors=getattr(m.visual, "vertex_colors", None)))
            continue
        # sample proportionally to area so thin/high-detail regions are covered
        n = max(50_000, min(target_points, int(target_points * 1.0)))
        samples, face_idx = trimesh.sample.sample_surface(m, n)
        pts_list.append(np.asarray(samples, dtype=np.float64))
        pts_list.append(np.asarray(m.vertices, dtype=np.float64))
        colors = None
        try:
            if m.visual.kind == "texture" and m.visual.uv is not None and m.visual.material is not None:
                vc = m.visual.to_color().vertex_colors
            else:
                vc = m.visual.vertex_colors
            vc = np.asarray(vc)
            if vc.ndim == 2 and len(vc) == len(m.vertices):
                # face-sampled colors: average of the 3 corner vertices
                tri = m.faces[face_idx]
                colors_s = vc[tri][:, :, :3].mean(axis=1)
                colors = np.vstack([colors_s, vc[:, :3]])
        except Exception:  # noqa: BLE001
            colors = None
        if colors is None:
            has_color = False
            col_list.append(None)
        else:
            col_list.append(colors.astype(np.uint8))
    for c in clouds:
        pts_list.append(np.asarray(c.vertices, dtype=np.float64))
        vc = getattr(c, "colors", None)
        if vc is not None and len(vc) == len(c.vertices):
            col_list.append(np.asarray(vc)[:, :3].astype(np.uint8))
        else:
            has_color = False
            col_list.append(None)
    if not pts_list:
        raise RuntimeError("Scan contained no geometry")
    pts = np.vstack(pts_list)
    colors = np.vstack(col_list) if has_color and all(c is not None for c in col_list) else None
    return pts, colors


def _guess_unit_scale(pts: np.ndarray) -> float:
    extent = float(np.max(pts.max(axis=0) - pts.min(axis=0)))
    if extent < 10.0:      # a drawer/mat scan in meters is ~0.3-1.5
        return 1000.0
    if extent < 150.0:     # centimeters
        return 10.0
    return 1.0             # millimeters


def fit_plane_ransac(pts: np.ndarray, iters: int = 300, thresh: float = 2.0, seed: int = 0):
    """RANSAC plane fit. Returns (normal (3,), d) with n.x + d = 0, and an inlier mask."""
    rng = np.random.default_rng(seed)
    n_pts = len(pts)
    sub = pts[rng.choice(n_pts, min(n_pts, 60_000), replace=False)]
    best_inl, best = -1, None
    for _ in range(iters):
        idx = rng.choice(len(sub), 3, replace=False)
        p0, p1, p2 = sub[idx]
        n = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n /= norm
        d = -float(n @ p0)
        dist = np.abs(sub @ n + d)
        cnt = int((dist < thresh).sum())
        if cnt > best_inl:
            best_inl, best = cnt, (n, d)
    if best is None:
        raise RuntimeError("Plane fit failed")
    n, d = best
    # refine with SVD on inliers
    inl = np.abs(sub @ n + d) < thresh
    P = sub[inl]
    c = P.mean(axis=0)
    _, _, vt = np.linalg.svd(P - c, full_matrices=False)
    n = vt[-1]
    n /= np.linalg.norm(n)
    d = -float(n @ c)
    # orientation: tools sit above the plane -> more off-plane points should be positive
    signed = pts @ n + d
    off = signed[np.abs(signed) > thresh]
    if off.size and np.median(off) < 0:
        n, d = -n, -d
    inlier_mask = np.abs(pts @ n + d) < thresh
    return n, d, inlier_mask


def _plane_basis(n: np.ndarray, inlier_pts: np.ndarray):
    """2D basis (u, v) in the plane; u along the principal axis of the inliers."""
    c = inlier_pts.mean(axis=0)
    P = inlier_pts - c
    P = P - np.outer(P @ n, n)
    _, _, vt = np.linalg.svd(P[:: max(1, len(P) // 50_000)], full_matrices=False)
    u = vt[0]
    u = u - (u @ n) * n
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    v /= np.linalg.norm(v)
    return u, v, c


def _fill_nan(height: np.ndarray, max_iter: int = 50) -> np.ndarray:
    """Fill NaN cells from neighbours (iterative dilation), remaining NaN -> 0."""
    h = height.copy()
    valid = np.isfinite(h)
    if valid.all():
        return h
    h[~valid] = 0.0
    hv = h.copy()
    vm = valid.astype(np.float32)
    k = np.ones((3, 3), np.float32)
    for _ in range(max_iter):
        if vm.all():
            break
        s = cv2.filter2D(hv * vm, -1, k, borderType=cv2.BORDER_REFLECT)
        c = cv2.filter2D(vm, -1, k, borderType=cv2.BORDER_REFLECT)
        newv = (c > 0) & (vm == 0)
        hv[newv] = s[newv] / c[newv]
        vm[newv] = 1.0
    return hv


def _suggest_corners(coverage: np.ndarray, inlier_raster: np.ndarray) -> Optional[np.ndarray]:
    """Largest quadrilateral around the plane inlier region (the mat) as TL,TR,BR,BL px."""
    m = (inlier_raster & coverage).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 0.05 * m.size:
        return None
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    from .calibration import order_corners

    return order_corners(box)


def rasterize_scan(data: bytes, ext: str, units: str = "auto", max_side: int = 2000,
                   min_mm_per_px: float = 0.4, pts_colors=None) -> ScanRaster:
    pts, colors = _load_points(data, ext) if pts_colors is None else pts_colors
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) < 100:
        raise RuntimeError("Scan has too few points")
    scale = UNIT_SCALE_TO_MM.get(units, None) if units != "auto" else _guess_unit_scale(pts)
    if scale is None:
        scale = _guess_unit_scale(pts)
    pts = pts * scale

    n, d, inl = fit_plane_ransac(pts, thresh=2.0)
    inlier_frac = float(inl.mean())
    u, v, c = _plane_basis(n, pts[inl])
    rel = pts - c
    U = rel @ u
    V = rel @ v
    Hh = rel @ n  # height above plane (mm)

    # discard points far below the plane (scan garbage) and very tall stuff (walls, ceiling)
    keep = (Hh > -15.0) & (Hh < 400.0)
    U, V, Hh = U[keep], V[keep], Hh[keep]
    inl = inl[keep]
    if colors is not None:
        colors = colors[keep]

    # grid extent from plane inliers, padded so overhanging tools are kept
    pad = 20.0
    u0, u1 = np.percentile(U[inl], [0.5, 99.5]) if inl.any() else (U.min(), U.max())
    v0, v1 = np.percentile(V[inl], [0.5, 99.5]) if inl.any() else (V.min(), V.max())
    u0 -= pad; v0 -= pad; u1 += pad; v1 += pad
    span = max(u1 - u0, v1 - v0)
    mm_per_px = max(min_mm_per_px, span / max_side)
    W = int(math.ceil((u1 - u0) / mm_per_px))
    Hgt = int(math.ceil((v1 - v0) / mm_per_px))
    W, Hgt = max(W, 8), max(Hgt, 8)

    # image x = u, image y = -v so the raster is what a camera above the mat would see (not mirrored)
    ix = np.floor((U - u0) / mm_per_px).astype(np.int64)
    iy = np.floor((v1 - V) / mm_per_px).astype(np.int64)
    inside = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < Hgt)
    ix, iy, Hh, inl = ix[inside], iy[inside], Hh[inside], inl[inside]
    if colors is not None:
        colors = colors[inside]
    flat = iy * W + ix

    height = np.full(W * Hgt, -np.inf, dtype=np.float64)
    np.maximum.at(height, flat, Hh)
    coverage = np.isfinite(height)
    height[~coverage] = np.nan
    height = height.reshape(Hgt, W)
    coverage = coverage.reshape(Hgt, W)

    # robust to spikes: median filter, then clamp the mat itself to 0
    h32 = _fill_nan(height.astype(np.float32))
    h32 = cv2.medianBlur(h32, 3)
    h32[h32 < 0] = 0.0

    inlier_raster = np.zeros(W * Hgt, dtype=np.int64)
    np.add.at(inlier_raster, flat[inl], 1)
    inlier_raster = inlier_raster.reshape(Hgt, W) > 0

    # Fraction of each cell's samples that are clearly above the mat. Its 0.5 contour sits on the
    # true footprint edge (a max-height raster would grow every tool by ~half a cell per side).
    cnt_total = np.zeros(W * Hgt, dtype=np.float64)
    cnt_above = np.zeros(W * Hgt, dtype=np.float64)
    np.add.at(cnt_total, flat, 1.0)
    np.add.at(cnt_above, flat[Hh > ABOVE_MAT_MM], 1.0)
    frac = np.full(W * Hgt, np.nan, dtype=np.float32)
    nz = cnt_total > 0
    frac[nz] = (cnt_above[nz] / cnt_total[nz]).astype(np.float32)
    frac = _fill_nan(frac.reshape(Hgt, W))
    frac = cv2.medianBlur(frac, 3)

    if colors is not None:
        acc = np.zeros((W * Hgt, 3), dtype=np.float64)
        cnt = np.zeros(W * Hgt, dtype=np.float64)
        np.add.at(acc, flat, colors.astype(np.float64))
        np.add.at(cnt, flat, 1.0)
        rgb = np.zeros_like(acc)
        nz = cnt > 0
        rgb[nz] = acc[nz] / cnt[nz, None]
        color = rgb.reshape(Hgt, W, 3).astype(np.uint8)
        color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
        # fill uncovered cells from neighbours so SAM sees a continuous image
        color = cv2.inpaint(color, (~coverage).astype(np.uint8) * 255, 3, cv2.INPAINT_TELEA)
    else:
        from .imaging import height_to_colormap

        color = height_to_colormap(h32)

    corners = _suggest_corners(coverage, inlier_raster)
    return ScanRaster(color_bgr=color, height_mm=h32, above_frac=frac, mm_per_px=float(mm_per_px), coverage=coverage,
                      plane_inlier_fraction=inlier_frac, unit_scale=float(scale), suggested_corners=corners)


def measure_thickness(height_mm: np.ndarray, mask: np.ndarray) -> Optional[dict]:
    """Height statistics of a tool region (mm above the mat)."""
    vals = height_mm[mask]
    vals = vals[np.isfinite(vals)]
    if vals.size < 5:
        return None
    return {
        "p95_mm": float(np.percentile(vals, 95)),
        "median_mm": float(np.median(vals)),
        "max_mm": float(vals.max()),
    }


# ----------------------------------------------------------------------------- single-object scans

PLANE_INLIER_LAYOUT_MIN = 0.3  # below this (and not stacked) the upload is treated as a single tool


@dataclass
class ObjectRaster:
    color_bgr: np.ndarray
    height_mm: np.ndarray
    mm_per_px: float
    footprint: np.ndarray            # bool HxW
    thickness_mm: float
    unit_scale: float
    extent_mm: Tuple[float, float, float]   # footprint w, footprint h, height


def classify_scan(pts_mm: np.ndarray) -> Tuple[str, float]:
    """'layout' (tools on a mat/drawer floor) or 'object' (one standalone tool).

    Two cues: how much of the cloud lies on the dominant plane, and whether the off-plane points
    sit *stacked above* plane points (an object's top face over its bottom face) or *beside* them
    (tools occlude the mat underneath, so their cells hold no plane points).
    """
    n, d, inl = fit_plane_ransac(pts_mm, iters=200, thresh=2.0)
    frac = float(inl.mean())
    if frac < 0.2:
        return "object", frac
    u, v, c = _plane_basis(n, pts_mm[inl])
    rel = pts_mm - c
    U, V, Hh = rel @ u, rel @ v, rel @ n
    cell = 4.0
    iu = np.floor(U / cell).astype(np.int64)
    iv = np.floor(V / cell).astype(np.int64)
    key = (iu - iu.min()) * (iv.max() - iv.min() + 1) + (iv - iv.min())
    plane_cells = np.unique(key[inl])
    off = np.abs(Hh) > 3.0
    if off.sum() < 50:
        return "layout", frac   # a bare mat / nothing above it
    off_cells = np.unique(key[off])
    stacked = float(np.isin(off_cells, plane_cells).mean())
    if stacked > 0.6:
        return "object", frac
    return ("layout" if frac >= PLANE_INLIER_LAYOUT_MIN else "object"), frac


def _resting_up_vector(pts: np.ndarray) -> np.ndarray:
    """Direction the object stands along when lying as flat as possible.

    Candidate 'down' directions are the convex-hull facet normals; the one with the smallest
    extent of the point cloud along it is the thinnest orientation, i.e. lying flat on that face.
    """
    import trimesh  # type: ignore

    rng = np.random.default_rng(0)
    sub = pts[rng.choice(len(pts), min(len(pts), 150_000), replace=False)]
    hull = trimesh.convex.convex_hull(sub[rng.choice(len(sub), min(len(sub), 15_000), replace=False)])
    normals = hull.face_normals
    areas = hull.area_faces
    # dedupe near-parallel normals, keep the larger facet
    key = np.round(normals, 2)
    _, idx = np.unique(key, axis=0, return_index=True)
    normals, areas = normals[idx], areas[idx]
    proj = sub @ normals.T                        # N x F
    extent = proj.max(axis=0) - proj.min(axis=0)  # F
    best = float(extent.min())
    ok = extent <= best * 1.03                    # near-ties: prefer the bigger (more stable) facet
    cand = np.where(ok)[0]
    face = cand[np.argmax(areas[cand])]
    down = normals[face]
    return -down / np.linalg.norm(down)


def rasterize_object(data: bytes, ext: str, units: str = "auto", max_side: int = 1600,
                     min_mm_per_px: float = 0.25, pts_colors=None) -> ObjectRaster:
    if pts_colors is None:
        pts, colors = _load_points(data, ext)
    else:
        pts, colors = pts_colors
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) < 100:
        raise RuntimeError("Model has too few points")
    scale = UNIT_SCALE_TO_MM.get(units, None) if units != "auto" else _guess_unit_scale(pts)
    if scale is None:
        scale = _guess_unit_scale(pts)
    pts = pts * scale

    up = _resting_up_vector(pts)
    h = pts @ up
    h = h - float(np.percentile(h, 0.2))          # robust floor
    # in-plane axes: u along the longest footprint direction, v = up x u; image y = -v
    P = pts - np.outer(pts @ up, up)
    c = P.mean(axis=0)
    _, _, vt = np.linalg.svd((P - c)[:: max(1, len(P) // 100_000)], full_matrices=False)
    u = vt[0] - (vt[0] @ up) * up
    u /= np.linalg.norm(u)
    v = np.cross(up, u)
    U = (pts - c) @ u
    V = (pts - c) @ v
    pad = 3.0
    u0, u1 = U.min() - pad, U.max() + pad
    v0, v1 = V.min() - pad, V.max() + pad
    span = max(u1 - u0, v1 - v0)
    mm_per_px = max(min_mm_per_px, span / max_side)
    W = max(8, int(math.ceil((u1 - u0) / mm_per_px)))
    Hgt = max(8, int(math.ceil((v1 - v0) / mm_per_px)))
    ix = np.clip(np.floor((U - u0) / mm_per_px).astype(np.int64), 0, W - 1)
    iy = np.clip(np.floor((v1 - V) / mm_per_px).astype(np.int64), 0, Hgt - 1)
    flat = iy * W + ix

    height = np.full(W * Hgt, -np.inf)
    np.maximum.at(height, flat, h)
    coverage = np.isfinite(height).reshape(Hgt, W)
    height[~np.isfinite(height)] = np.nan
    height = height.reshape(Hgt, W).astype(np.float32)

    # footprint = silhouette from above: close small sampling gaps, fill holes, keep the main body
    k = max(3, int(round(1.5 / mm_per_px)) | 1)
    fp = cv2.morphologyEx(coverage.astype(np.uint8) * 255, cv2.MORPH_CLOSE,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    contours, _ = cv2.findContours(fp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    footprint = np.zeros_like(fp)
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        cv2.drawContours(footprint, [cnt], -1, 255, thickness=cv2.FILLED)
    footprint = footprint > 0

    h32 = _fill_nan(height)
    h32 = cv2.medianBlur(h32, 3)
    h32[h32 < 0] = 0.0
    h32[~footprint] = 0.0
    vals = h32[footprint]
    thickness = float(np.percentile(vals, 99.0)) if vals.size else 0.0

    if colors is not None:
        acc = np.zeros((W * Hgt, 3))
        cnt_ = np.zeros(W * Hgt)
        np.add.at(acc, flat, colors.astype(np.float64))
        np.add.at(cnt_, flat, 1.0)
        rgb = np.zeros_like(acc)
        nz = cnt_ > 0
        rgb[nz] = acc[nz] / cnt_[nz, None]
        color = cv2.cvtColor(rgb.reshape(Hgt, W, 3).astype(np.uint8), cv2.COLOR_RGB2BGR)
        color[~coverage] = (235, 235, 235)
    else:
        from .imaging import height_to_colormap

        color = height_to_colormap(h32)
        color[~footprint] = (235, 235, 235)
    ys, xs = np.nonzero(footprint)
    ext_w = float((xs.max() - xs.min() + 1) * mm_per_px) if xs.size else 0.0
    ext_h = float((ys.max() - ys.min() + 1) * mm_per_px) if ys.size else 0.0
    return ObjectRaster(color_bgr=color, height_mm=h32, mm_per_px=float(mm_per_px), footprint=footprint,
                        thickness_mm=thickness, unit_scale=float(scale), extent_mm=(ext_w, ext_h, thickness))
