"""3D (form-fit) pockets — the drawer's height map carved into the foam.

Nolan (2026-10-03): "What if I made this more of a 3D workflow? Taking the height map, smoothing it, and cutting
those shapes directly into the foam via CNC." Instead of a flat pocket at one depth per tool, each pocket's floor
follows the tool's scanned top surface (smoothed, with a little clearance), so the tool nests in a cavity shaped
like itself. Everything here works on ONE depth map of the whole foam block:

    D(x, y) = millimetres to remove below the foam's top surface, on a regular grid (res_mm per cell)

`build_depth_map` rasterises the computed layout into it (relief pockets from the per-tool height rasters, flat
pockets at their depth), then the exporters turn that one array into what a CNC workflow consumes:
  * `depth_to_stl`    — the carved foam block as a closed mesh (for 3D CAM / a model check)
  * `depth_to_png16`  — a 16-bit depth map (0 = top, 65535 = deepest) with a JSON sidecar, which relief-capable CAM
                        (Vectric Aspire / VCarve, Carbide Create Pro, Fusion "mesh") imports directly
  * `depth_to_gcode`  — a GRBL-style 3-axis raster finishing program for a FLAT end mill, gouge-free: the cutter
                        centre height at (x, y) is the HIGHEST surface point under the whole cutter disc, so the
                        cutter never cuts below the modelled surface; multi-pass by step-down, feed/plunge/safe-Z
                        parameters; constant-Z runs are emitted as one move.
"""
from __future__ import annotations

import io
import json
import math
import os
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np


# ----------------------------------------------------------------------------- the depth map

def _place_height(h_src: np.ndarray, mask_src: np.ndarray, mpp: float, *, source_centroid_mm, rotation_deg: float,
                  offset_mm, mirror_w: Optional[float], res_mm: float, W: int, H: int, smooth_mm: float,
                  guide_src: Optional[np.ndarray] = None):
    """A tool's scanned height raster, smoothed inside its mask, resampled onto the foam grid through the layout's
    placement (rotate about the outline's centroid, translate, optional mirror). Returns (height_mm, inside) on the
    grid; height is 0 and inside False where the tool is not."""
    h_raw = np.asarray(h_src, dtype=np.float32)
    m = mask_src.astype(bool)
    # TrueDepth returns NO depth on black / glossy surfaces (a WD drive, a power bank, a zipper case: 22 %% of a
    # real drawer's cells, 2026-10-03). Inside the tool's footprint the unknown cells are the TOOL, not the floor:
    # fill them from the depth the sensor did get on that tool (its rim and lit parts, p75) so the pocket becomes a
    # slab at the tool's height instead of a 1 mm clearance scratch. Below 10 %% valid the tool is treated as flat.
    known = np.isfinite(h_raw) & (h_raw > 0.5)
    fill_from = m & known
    # FLOOR READINGS INSIDE A TALL TOOL ARE SENSOR FAILURES, NOT GEOMETRY (the black neoprene case on a237eba87dba,
    # 2026-10-03: 45 %% of its cells read the 22-25 mm top, 42 %% read 0-4 mm — and three frames AGREED on those,
    # so frame support cannot tell them apart; the IR pattern simply vanishes on that material). A tool found by
    # the photo is object everywhere inside its silhouette, so cells under max(1.5 mm, FAIL_FRAC x p90) are holes
    # when they make up >= 10 %% of it — otherwise they are an edge ramp or a genuinely thin part (a 3 mm blade on a
    # 25 mm stock stays: 12 %% > 10 %%). Holes are then filled from their measured NEIGHBOURHOOD, not from one global
    # p75 — a constant fill left 2-3 mm steps against the measured cells (0.41 mm rms "static" on the pocket floor).
    if fill_from.sum() >= 50:
        top = float(np.percentile(h_raw[fill_from], 90))
        if top >= 8.0:
            bad = fill_from & (h_raw <= max(1.5, FAIL_FRAC * top))
            floorish = m & (~known | bad)                 # holes and near-floor readings together say "this material"
            if float(floorish.sum()) / max(1, int(m.sum())) >= 0.10:
                known = known & ~bad
                fill_from = m & known
    cover = float(fill_from.sum()) / max(1, int(m.sum()))
    # even a few hundred measured cells on the rim say how tall the thing is (the Anker power bank had 9.8 %%
    # coverage and a 10 %% rule left it a 1 mm scratch); 3 %% or 200 cells is enough to fill from
    if fill_from.sum() >= 200 or cover >= 0.03:
        from scipy.ndimage import gaussian_filter
        p75 = float(np.percentile(h_raw[fill_from], 75))
        sig = max(2.0, 12.0 / mpp)                        # 12 mm neighbourhood (holes on dark material are ~5 cm across)
        num = gaussian_filter(np.where(fill_from, h_raw, 0.0).astype(np.float32), sig)
        den = gaussian_filter(fill_from.astype(np.float32), sig)
        local = np.where(den > 0.05, num / np.maximum(den, 1e-6), p75)
        h_raw = np.where(m & ~known, local, h_raw)
    h = np.nan_to_num(h_raw, nan=0.0)
    if smooth_mm > 0:
        # normalised convolution so the tool's edge does not bleed down into the floor (and the floor up into it)
        from scipy.ndimage import gaussian_filter
        sig = smooth_mm / mpp
        num = gaussian_filter(np.where(m, h, 0.0), sig)
        den = gaussian_filter(m.astype(np.float32), sig)
        h = np.where(den > 1e-3, num / np.maximum(den, 1e-3), 0.0).astype(np.float32)
    # grid cell centres (mm, y-down like the SVG/layout) -> source px through the inverse placement
    gx = (np.arange(W) + 0.5) * res_mm
    gy = (np.arange(H) + 0.5) * res_mm
    X, Y = np.meshgrid(gx, gy)
    if mirror_w is not None:
        X = mirror_w - X
    cx, cy = source_centroid_mm
    ox, oy = offset_mm
    th = math.radians(rotation_deg or 0.0)
    c, s = math.cos(th), math.sin(th)
    x = X - ox - cx
    y = Y - oy - cy
    sx = (cx + x * c + y * s) / mpp      # inverse rotation
    sy = (cy - x * s + y * c) / mpp
    mapx = sx.astype(np.float32); mapy = sy.astype(np.float32)
    hg = cv2.remap(h, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    mg = cv2.remap(m.astype(np.float32), mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0) >= 0.5
    if guide_src is not None:
        gg = cv2.remap(guide_src, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return hg, mg, gg
    return hg, mg


def build_depth_map(layout: Dict, geometry_for_tool: Callable[[Dict], Optional[Tuple[np.ndarray, np.ndarray, float]]], *,
                    res_mm: float = 1.0, smooth_mm: float = 1.5, z_clearance_mm: float = 1.0, floor_min_mm: float = 2.0,
                    foam_thickness_mm: float = 30.0, default_style: str = "relief", wall_band_mm: float = 5.0,
                    clean_solids: bool = True, image_for_tool: Optional[Callable[[Dict], Optional[np.ndarray]]] = None,
                    solid_hint_for_tool: Optional[Callable[[Dict, np.ndarray, np.ndarray], Optional[Dict]]] = None) -> Dict:
    """One depth map for the whole block. Per tool: `raw.pocket_style` ('relief' | 'flat', default `default_style`);
    relief needs a scanned height raster (geometry_for_tool), else the tool falls back to a flat pocket.
    Relief depth = smoothed tool height + z_clearance, capped at the tool's depth_mm when one is set and always at
    thickness - floor_min; the clearance band around the tool (the layout ring minus the scanned mask) takes the
    depth of the nearest scanned cell so the wall is vertical at the ring. Returns {depth (float32 HxW), res_mm,
    width_mm, height_mm, thickness_mm, tools: [{id, style, max_depth_mm, min_depth_mm}]}."""
    W_mm = float(layout["mat"]["width_mm"]); H_mm = float(layout["mat"]["height_mm"])
    T = float(foam_thickness_mm)
    W = max(2, int(round(W_mm / res_mm))); H = max(2, int(round(H_mm / res_mm)))
    D = np.zeros((H, W), np.float32)
    cap_all = max(0.5, T - floor_min_mm)
    mirror_w = W_mm if layout.get("mirror") else None
    report = []
    for t in layout["tools"]:
        geom = t.get("shapely")
        if geom is None or geom.is_empty:
            continue
        raw = t.get("raw") or {}
        style = str(raw.get("pocket_style") or default_style).lower()
        pocket, cov = _rasterise(geom, res_mm, W, H, coverage=True)
        edge = (cov > 0) & ~pocket            # outside the half-coverage mask but partly inside the outline
        if not pocket.any():
            continue
        d_set = t.get("depth_mm")
        through = d_set is not None and float(d_set) >= T - 0.25
        flat_depth = T if through else (min(max(float(d_set), 0.5), cap_all) if d_set is not None else cap_all)
        # a relief pocket is capped by the person's explicit per-tool override only — the layout's depth RULE
        # ("measured minus 3 mm") is a flat-pocket idea and made every tool taller than the foam a through-cut
        d_override = raw.get("depth_override_mm")
        placed = None
        guide_placed = None
        depth_cover = None
        if style == "relief":
            g = geometry_for_tool(raw)
            if g is not None:
                mask, height, mpp = g
                with np.errstate(invalid="ignore"):
                    known = np.isfinite(height) & (height > 0.5) & mask.astype(bool)
                depth_cover = float(known.sum()) / max(1, int(mask.sum()))
                guide_src = image_for_tool(raw) if (clean_solids and image_for_tool is not None) else None
                res_place = _place_height(height, mask, mpp, source_centroid_mm=t.get("source_centroid_mm") or t.get("centroid_mm") or (0, 0),
                                          rotation_deg=float(raw.get("rotation_deg") or 0.0),
                                          offset_mm=((raw.get("offset_mm") or {}).get("x", 0.0) or 0.0, (raw.get("offset_mm") or {}).get("y", 0.0) or 0.0),
                                          mirror_w=mirror_w, res_mm=res_mm, W=W, H=H, smooth_mm=smooth_mm, guide_src=guide_src)
                placed = res_place[:2]
                guide_placed = res_place[2] if len(res_place) == 3 else None
        low_cover = depth_cover is not None and depth_cover < LOW_COVERAGE
        if placed is not None and placed[1].any() and (low_cover or float((placed[0][placed[1]] > 1.5).mean()) < 0.10):
            # the sensor saw the footprint but read it at FLOOR height almost everywhere (a matte black drive on
            # a237eba87dba: 42 %% coverage, all of it 0.4 mm), or it saw too LITTLE of it to shape a pocket (the
            # glossy power bank: 10 %% of its cells, and once the ghost cells of a misplaced frame were excluded the
            # rest read 3 mm on a 57 mm object) — no relief to carve. Use the typed thickness if the person gave
            # one, else the flat depth rule, and say so.
            typed = raw.get("thickness_mm")
            if typed is not None and float(typed) > 2.0:
                flat_depth = min(cap_all, float(typed) + z_clearance_mm)
            pocket_d = np.where(pocket, flat_depth, np.where(edge, flat_depth * cov, 0.0)).astype(np.float32)
            D = np.maximum(D, pocket_d)
            why = f"flat (depth coverage {round(100 * depth_cover)} %)" if low_cover else "flat (no usable depth)"
            report.append({"id": t["id"], "style": why, "max_depth_mm": round(flat_depth, 2), "min_depth_mm": round(flat_depth, 2),
                           "depth_coverage": None if depth_cover is None else round(depth_cover, 3)})
            continue
        if placed is not None and placed[1].any():
            hg, inside = placed
            solid_report = None
            if clean_solids and float((hg[inside] > 1.5).mean()) >= 0.10:
                from . import solids
                hint = None
                if solid_hint_for_tool is not None:
                    try:
                        hint = solid_hint_for_tool(raw, hg, inside)
                    except Exception:  # noqa: BLE001
                        hint = None
                hg, solid_report = solids.clean_relief(hg, inside, res_mm, guide=guide_placed,
                                                       solid_hint=(hint or {}).get("solid_class"), symmetric_hint=(hint or {}).get("symmetric"),
                                                       hint_confidence=(hint or {}).get("confidence"))
                if hint and solid_report is not None:
                    solid_report["model_says"] = f"{hint.get('tool_name')} · {hint.get('solid_class')} ({float(hint.get('confidence') or 0):.2f})"
            cap = min(cap_all, float(d_override)) if d_override is not None else cap_all
            d_tool = _relief_pocket_depth(hg, inside, pocket, res_mm, z_clearance_mm, cap, wall_band_mm)
            if edge.any():
                _, (iy, ix) = _nearest_index(pocket)
                d_tool = np.where(edge, d_tool[iy, ix] * cov, d_tool)
            D = np.maximum(D, d_tool)
            vals = d_tool[pocket]
            entry = {"id": t["id"], "style": "relief", "max_depth_mm": round(float(vals.max()), 2), "min_depth_mm": round(float(vals.min()), 2)}
            if solid_report is not None:
                entry["solid"] = solid_report.get("solid"); entry["solid_residual_mm"] = solid_report.get("residual_mm")
                entry["clean_steps"] = [st["step"] for st in solid_report.get("steps", [])]
                entry["fits_mm"] = solid_report.get("fits"); entry["model_says"] = solid_report.get("model_says")
            report.append(entry)
        else:
            D = np.where(pocket, np.maximum(D, flat_depth), D)
            D = np.where(edge, np.maximum(D, flat_depth * cov), D)
            report.append({"id": t["id"], "style": "flat" + ("" if style == "flat" else " (no scan)"), "max_depth_mm": round(flat_depth, 2), "min_depth_mm": round(flat_depth, 2)})
    return {"depth": D, "res_mm": float(res_mm), "width_mm": W_mm, "height_mm": H_mm, "thickness_mm": T, "tools": report}


FAIL_FRAC = float(os.environ.get("TC_RELIEF_FAIL_FRAC", "0.10"))   # inside a tall tool, cells under this share of p90 are sensor failures
LOW_COVERAGE = float(os.environ.get("TC_RELIEF_LOW_COVERAGE", "0.25"))   # below this share of measured cells a tool is carved flat
SUPERSAMPLE = 4   # pocket outlines are rasterised at 4x and averaged: edge cells carry FRACTIONAL coverage


def _relief_pocket_depth(hg: np.ndarray, inside: np.ndarray, pocket: np.ndarray, res_mm: float, z_clear: float, cap: float,
                         band_mm: float = 5.0) -> np.ndarray:
    """Depth of one relief pocket with a CLEAN WALL. The pocket's rim used to take the scanned mask's edge depth cell
    by cell, so the sensor's noisy edge ramp became terraces along every wall (Nolan, 2026-10-03: "so many shapes
    that are rugged and not clean"). Now the wall carries a depth PROFILE along the outline: at each boundary point
    the 75th percentile of the tool's relief within `band_mm` + 3 mm inside, gap-filled and smoothed along the
    contour (sigma 8 mm), assigned to every cell within `band_mm` of the rim and blended into the interior relief
    over 3 mm. The interior keeps the (smoothed) scanned shape; the wall is one smooth, vertical surface that
    follows the tool's local height."""
    from scipy import ndimage as ndi
    from scipy.spatial import cKDTree
    H, W = pocket.shape
    valid = inside & np.isfinite(hg) & (hg > 0.5)
    base = np.where(valid, hg, np.nan).astype(np.float32)
    # interior: scanned relief, holes and the clearance ring filled from the nearest valid cell
    if valid.any():
        _, (iy, ix) = ndi.distance_transform_edt(~valid, return_indices=True)
        interior = base[iy, ix]
    else:
        interior = np.zeros_like(hg)
    interior = np.where(pocket, np.clip(interior + z_clear, 0.5, cap), 0.0).astype(np.float32)
    dist_in = ndi.distance_transform_edt(pocket) * res_mm
    contours, _ = cv2.findContours(pocket.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours or not valid.any():
        return interior
    out = interior.copy()
    vy, vx = np.nonzero(valid)
    tree_valid = cKDTree(np.column_stack([vx, vy]))
    r_px = (band_mm + 3.0) / res_mm
    for cnt in contours:
        pts = cnt.reshape(-1, 2)                       # (x, y) along the rim
        if len(pts) < 8:
            continue
        step = max(1, int(round(1.0 / res_mm)))        # sample the rim every ~1 mm
        samp = pts[::step]
        prof = np.full(len(samp), np.nan, np.float32)
        for k, (x, y) in enumerate(samp):
            idx = tree_valid.query_ball_point([x, y], r_px)
            if len(idx) >= 4:
                prof[k] = np.percentile(hg[vy[idx], vx[idx]], 75)
        if not np.isfinite(prof).any():
            continue
        # gap-fill along the contour (nearest finite), then circular smoothing over ~8 mm
        n = len(prof); ii = np.arange(n); good = np.isfinite(prof)
        prof = np.interp(ii, ii[good], prof[good], period=n) if good.sum() < n else prof
        sig = 8.0 / max(res_mm, 1e-3) / step
        prof_s = ndi.gaussian_filter1d(prof, sig, mode="wrap")
        wall = np.clip(prof_s + z_clear, 0.5, cap).astype(np.float32)
        # assign to the band: each band cell takes the profile of its nearest rim sample
        cmask = np.zeros_like(pocket, np.uint8)
        cv2.drawContours(cmask, [cnt], -1, 1, -1)
        region = (cmask > 0) & pocket
        band = region & (dist_in <= band_mm)
        if not band.any():
            continue
        by, bx = np.nonzero(band)
        tree_rim = cKDTree(samp)
        _, nearest = tree_rim.query(np.column_stack([bx, by]))
        wall_d = wall[nearest]
        w = np.clip((band_mm - dist_in[by, bx]) / 3.0, 0.0, 1.0).astype(np.float32)
        out[by, bx] = w * wall_d + (1.0 - w) * out[by, bx]
    return out


def _rasterise(geom, res_mm: float, W: int, H: int, coverage: bool = False):
    """Pocket mask on the grid. With `coverage=True` also returns the fraction of each cell inside the outline
    (anti-aliased, 4x4 sub-samples) — a binary mask made every wall a staircase along the grid ("the cutouts are
    quantized and don't look continuous", Nolan 2026-10-03); scaling the edge cells' depth by their coverage gives a
    one-cell ramp that follows the outline exactly."""
    from shapely.geometry import MultiPolygon, Polygon
    k = SUPERSAMPLE
    img = np.zeros((H * k, W * k), np.uint8)
    polys = list(geom.geoms) if isinstance(geom, MultiPolygon) else [geom]
    for poly in polys:
        if not isinstance(poly, Polygon) or poly.is_empty:
            continue
        ext = np.round(np.asarray(poly.exterior.coords)[:, :2] / res_mm * k).astype(np.int32)
        cv2.fillPoly(img, [ext.reshape(-1, 1, 2)], 1)
        for ring in poly.interiors:
            hole = np.round(np.asarray(ring.coords)[:, :2] / res_mm * k).astype(np.int32)
            cv2.fillPoly(img, [hole.reshape(-1, 1, 2)], 0)
    cov = img.reshape(H, k, W, k).mean(axis=(1, 3)).astype(np.float32)
    mask = cov >= 0.5
    return (mask, cov) if coverage else mask


def _nearest_index(inside: np.ndarray):
    """For every cell, the index of the nearest cell where `inside` is True (distance transform with indices)."""
    from scipy import ndimage as ndi
    dist, idx = ndi.distance_transform_edt(~inside, return_indices=True)
    return dist, (idx[0], idx[1])


# ----------------------------------------------------------------------------- exports

def depth_to_stl(rel: Dict) -> bytes:
    """The carved block as a closed mesh: top surface z = T - D on the grid (walls come out as steep cells), four
    sides and a bottom. Grid coordinates are mm, y flipped to y-up like `layout_to_stl`."""
    import trimesh  # type: ignore
    D = rel["depth"]; res = rel["res_mm"]; T = rel["thickness_mm"]; W_mm = rel["width_mm"]; H_mm = rel["height_mm"]
    H, W = D.shape
    # corner heights = mean of the four adjacent cells: with coverage-scaled edge cells this gives a continuous
    # wall that follows the outline (the earlier max() of neighbours made every wall a grid staircase)
    Dp = np.pad(D, 1, mode="edge")
    corner = 0.25 * (Dp[:-1, :-1] + Dp[:-1, 1:] + Dp[1:, :-1] + Dp[1:, 1:])      # (H+1, W+1)
    xs = np.linspace(0, W_mm, W + 1); ys = np.linspace(0, H_mm, H + 1)
    X, Y = np.meshgrid(xs, ys)
    Z = T - corner
    top = np.column_stack([X.ravel(), (H_mm - Y).ravel(), Z.ravel()])          # y-up
    n_top = top.shape[0]
    idx = np.arange(n_top).reshape(H + 1, W + 1)
    a = idx[:-1, :-1].ravel(); b = idx[:-1, 1:].ravel(); c = idx[1:, 1:].ravel(); d = idx[1:, :-1].ravel()
    # with y flipped the winding flips too; keep normals up: (a, c, b) and (a, d, c)
    faces = [np.column_stack([a, c, b]), np.column_stack([a, d, c])]
    # bottom (z = 0), same grid corners but only the border is needed; use the four corners
    bot = np.array([[0, 0, 0], [W_mm, 0, 0], [W_mm, H_mm, 0], [0, H_mm, 0]], dtype=np.float64)
    bi = n_top + np.arange(4)
    faces.append(np.array([[bi[0], bi[1], bi[2]], [bi[0], bi[2], bi[3]]]))
    # sides: connect the top boundary to the bottom rectangle edges
    def side(top_ids: np.ndarray, p0: int, p1: int, flip: bool):
        # top boundary vertices run along one edge; bottom is a straight segment p0 -> p1
        tri = []
        n = len(top_ids)
        for i in range(n - 1):
            t0, t1 = top_ids[i], top_ids[i + 1]
            tri.append([t0, t1, p0] if not flip else [t1, t0, p0])
        tri.append([top_ids[-1], p1, p0] if not flip else [p1, top_ids[-1], p0])
        return np.array(tri)
    # boundary rows/cols of the top grid in y-up terms: grid row 0 is y = H_mm (top), row H is y = 0
    faces.append(side(idx[H, :], bi[0], bi[1], False))              # y = 0 edge, x increasing
    faces.append(side(idx[:, W][::-1], bi[1], bi[2], False))        # x = W edge, y increasing (rows reversed)
    faces.append(side(idx[0, :][::-1], bi[2], bi[3], False))        # y = H edge, x decreasing
    faces.append(side(idx[:, 0], bi[3], bi[0], False))              # x = 0 edge, y decreasing
    F = np.vstack(faces)
    V = np.vstack([top, bot])
    mesh = trimesh.Trimesh(vertices=V, faces=F, process=True)
    mesh.fix_normals()
    return mesh.export(file_type="stl")


def depth_to_png16(rel: Dict) -> Tuple[bytes, Dict]:
    """16-bit grayscale PNG: 0 = foam top, 65535 = the deepest point. The sidecar says what a grey level is in mm."""
    from PIL import Image
    D = rel["depth"]
    mx = float(D.max()) if D.size else 0.0
    scale = 65535.0 / mx if mx > 0 else 0.0
    img = Image.fromarray(np.clip(D * scale, 0, 65535).astype(np.uint16))   # uint16 -> mode I;16
    buf = io.BytesIO(); img.save(buf, format="PNG")
    meta = {"mm_per_px": rel["res_mm"], "width_mm": rel["width_mm"], "height_mm": rel["height_mm"], "thickness_mm": rel["thickness_mm"],
            "max_depth_mm": round(mx, 3), "grey_to_mm": round(mx / 65535.0, 6), "zero_is": "foam top surface", "tools": rel["tools"]}
    return buf.getvalue(), meta


def cutter_floor(rel: Dict, cutter_mm: float) -> np.ndarray:
    """Gouge-free cutter-centre Z for a FLAT end mill: at each (x, y) the cutter bottom spans a disc of the cutter's
    radius, so it may go no lower than the highest surface point under that disc = T - min(D over the disc)."""
    from scipy.ndimage import minimum_filter
    D = rel["depth"]; res = rel["res_mm"]; T = rel["thickness_mm"]
    r = max(1, int(round(cutter_mm / 2.0 / res)))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    disc = (xx * xx + yy * yy) <= r * r
    return (T - minimum_filter(D, footprint=disc, mode="nearest")).astype(np.float32)


def depth_to_gcode(rel: Dict, *, cutter_mm: float = 6.0, stepover_mm: float = 2.0, stepdown_mm: float = 6.0, feed_mm_min: float = 1500.0,
                   plunge_mm_min: float = 500.0, safe_z_mm: float = 5.0, spindle_rpm: int = 12000, z_tol_mm: float = 0.05) -> Tuple[str, Dict]:
    """GRBL-style raster finishing (X runs, alternating direction) in step-down layers. Z is the foam TOP = 0,
    cutting negative. Each layer k cuts every cell whose floor is below the previous layer, at max(floor, layer z).
    Points along a run are emitted only where Z changes by more than z_tol (a flat pocket floor is one G1)."""
    D = rel["depth"]; res = rel["res_mm"]; T = rel["thickness_mm"]
    zc = cutter_floor(rel, cutter_mm) - T          # cutter-centre z relative to the top (<= 0)
    H, W = zc.shape
    rows = max(1, int(round(stepover_mm / res)))
    deepest = float(-zc.min())
    n_layers = max(1, int(math.ceil((deepest - 1e-6) / max(0.1, stepdown_mm))))
    out: List[str] = [
        "(ToolFoam Pro form-fit relief - flat end mill)",
        f"(block {rel['width_mm']:.1f} x {rel['height_mm']:.1f} x {T:.1f} mm, grid {res} mm, cutter {cutter_mm} mm, stepover {stepover_mm}, stepdown {stepdown_mm})",
        "(X right, Y away from you, Z = 0 at the FOAM TOP, origin at the block's near-left corner)",
        "G21 G90 G17 G94", f"G0 Z{safe_z_mm:.3f}", f"M3 S{int(spindle_rpm)}",
    ]
    moves = 0; cut_len = 0.0
    for k in range(1, n_layers + 1):
        z_layer = -k * stepdown_mm
        prev_layer = -(k - 1) * stepdown_mm
        out.append(f"(layer {k} of {n_layers}: down to {z_layer:.2f})")
        direction = 1
        for j in range(0, H, rows):
            row = zc[j]
            need = row < prev_layer - 1e-6          # material still to remove at this layer
            if not need.any():
                continue
            z_row = np.maximum(row, z_layer)
            y = rel["height_mm"] - (j + 0.5) * res   # y-up for the machine
            cols = np.arange(W) if direction > 0 else np.arange(W - 1, -1, -1)
            # runs of consecutive cells that need cutting
            i = 0
            while i < W:
                if not need[cols[i]]:
                    i += 1; continue
                i0 = i
                while i < W and need[cols[i]]:
                    i += 1
                seg = cols[i0:i]
                if len(seg) < 2:
                    continue
                x0 = (seg[0] + 0.5) * res; z0 = float(z_row[seg[0]])
                out.append(f"G0 X{x0:.3f} Y{y:.3f}")
                out.append(f"G1 Z{z0:.3f} F{plunge_mm_min:.0f}")
                last_z = z0; last_x = x0
                first = True
                for c in seg[1:]:
                    x = (c + 0.5) * res; z = float(z_row[c])
                    if abs(z - last_z) > z_tol_mm:
                        # finish the constant-Z run just before this cell, then step to the new Z
                        if last_x != x0 or not first:
                            out.append(f"G1 X{last_x:.3f}" + (f" F{feed_mm_min:.0f}" if first else "")); first = False
                        out.append(f"G1 X{x:.3f} Z{z:.3f}" + (f" F{feed_mm_min:.0f}" if first else "")); first = False
                        last_z = z
                    last_x = x
                out.append(f"G1 X{last_x:.3f}" + (f" F{feed_mm_min:.0f}" if first else ""))
                out.append(f"G0 Z{safe_z_mm:.3f}")
                moves += 1; cut_len += (len(seg) - 1) * res
            direction = -direction
    out += ["M5", f"G0 Z{safe_z_mm:.3f}", "G0 X0 Y0", "M30"]
    stats = {"layers": n_layers, "runs": moves, "cut_length_mm": round(cut_len, 1), "deepest_mm": round(deepest, 2),
             "est_minutes": round(cut_len / max(1.0, feed_mm_min) + moves * 2 * safe_z_mm / max(1.0, plunge_mm_min), 1), "lines": len(out)}
    return "\n".join(out) + "\n", stats
