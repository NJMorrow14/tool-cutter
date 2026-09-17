#!/usr/bin/env python3
"""ToolCutter API: 3D scans of laid-out tools (or single tool models) -> foam insert cutting files.

Flow:
  POST /api/sessions                       upload a 3D file (ply/obj/glb/stl); classified as a mat layout scan or a
                                           single tool model (form scan_kind=auto|layout|object)
  POST /api/sessions/<id>/calibrate        4 corners of the mat/drawer + its real size -> top-down mm image
  POST /api/sessions/<id>/auto_detect      find tools automatically (color or height), refine with HQ-SAM
  POST /api/sessions/<id>/segment          (re)segment tools from click prompts
  POST /api/layout                         processed outlines in mm (clearance, smoothing, notches); tools carry polygon_mm
  POST /api/export                         SVG / DXF / STL download (same body + format)
  GET  /api/sessions/<id>/image/<stage>    original | rectified | height preview JPEG

Run:  python3 app.py --host 0.0.0.0 --port 8000
Model: put sam_hq_vit_*.pth in backend/ or set HQSAM_CKPT.
"""
from __future__ import annotations

import argparse
import io
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request, send_file
from flask_cors import CORS

from toolcutter import calibration, geometry, scan as scanmod
from toolcutter.exporters import layout_to_dxf, layout_to_stl, layout_to_svg, layout_tools_to_stl
from toolcutter.imaging import MESH_EXTENSIONS, downscale_to, encode_jpeg, height_to_colormap
from toolcutter.segmenter import Segmenter, clean_mask
from toolcutter.sessions import SessionStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("toolcutter")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 400 * 1024 * 1024
CORS(app, resources={r"/api/*": {"origins": "*"}, r"/health": {"origins": "*"}}, expose_headers=["Content-Disposition"])

STORE = SessionStore()
SEGMENTER = Segmenter()
DISPLAY_MAX_SIDE = 1800


# ----------------------------------------------------------------------------- helpers

class ApiError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


@app.errorhandler(ApiError)
def _handle_api_error(err: ApiError):
    return jsonify({"error": str(err)}), err.status


@app.errorhandler(Exception)
def _handle_unexpected(err: Exception):  # noqa: BLE001
    log.exception("Unhandled error")
    return jsonify({"error": f"{type(err).__name__}: {err}"}), 500


def _session(sid: str):
    s = STORE.get(sid)
    if s is None:
        raise ApiError("Unknown session (server restarted?). Upload again.", 404)
    return s


def _require_rectified(s):
    if s.rectified is None:
        raise ApiError("Calibrate the mat corners first.")
    return s


def _f(d: Dict, key: str, default: Optional[float] = None) -> Optional[float]:
    v = d.get(key, default)
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        raise ApiError(f"Field '{key}' must be a number")


def _points(raw: Any) -> List[Dict]:
    out = []
    for p in raw or []:
        try:
            out.append({"x": float(p["x"]), "y": float(p["y"]), "label": 1 if p.get("label", "pos") in ("pos", 1, True) else 0})
        except (KeyError, TypeError, ValueError):
            raise ApiError("Each point needs numeric x, y and a label")
    return out


def _tool_result(s, tool_id: str, mask: np.ndarray, points: List[Dict], box: Optional[List[float]] = None) -> Dict:
    poly = geometry.mask_to_polygon(mask)
    area_px = float(mask.sum())
    res: Dict[str, Any] = {
        "id": tool_id,
        "session_id": s.id,
        "points": points,
        "box": box,
        "polygon_px": poly.tolist() if poly is not None else [],
        "polygon_mm": (poly * s.mm_per_px).tolist() if poly is not None else [],
        "area_mm2": area_px * (s.mm_per_px ** 2),
        "measured_thickness_mm": None,
        "height_stats": None,
    }
    if poly is not None:
        xs, ys = poly[:, 0], poly[:, 1]
        res["bbox_px"] = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
    if s.rect_height is not None and mask.any():
        stats = scanmod.measure_thickness(s.rect_height, mask)
        res["height_stats"] = stats
        if stats:
            res["measured_thickness_mm"] = round(stats["p95_mm"], 1)
    s.masks[tool_id] = mask
    return res


def _sam_mask(s, points: List[Dict], box: Optional[List[float]], hq_token_only: bool) -> np.ndarray:
    SEGMENTER.set_image(f"{s.id}:{s.version}", s.rectified)
    coords = [(p["x"], p["y"]) for p in points] or None
    labels = [p["label"] for p in points] or None
    mask = SEGMENTER.predict(coords, labels, box=box, hq_token_only=hq_token_only)
    positives = [(p["x"], p["y"]) for p in points if p["label"] == 1]
    min_area_px = 20.0 / (s.mm_per_px ** 2)
    return clean_mask(mask, positives, min_area_px=min_area_px, fill_holes=True)


# ----------------------------------------------------------------------------- routes

@app.get("/health")
def health():
    return jsonify({"ok": True, "model": SEGMENTER.info(), "time": time.time()})


@app.post("/api/sessions")
def create_session():
    f = request.files.get("file") or request.files.get("image")
    if f is None:
        raise ApiError('Send the upload as multipart field "file"')
    data = f.read()
    if not data:
        raise ApiError("Empty upload")
    filename = f.filename or "upload"
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    units = (request.form.get("units") or "auto").lower()
    scan_kind = (request.form.get("scan_kind") or "auto").lower()   # auto | layout | object
    extra_tools: List[Dict] = []
    if ext in MESH_EXTENSIONS:
        try:
            pts, colors = scanmod._load_points(data, ext)
            pts = pts[np.isfinite(pts).all(axis=1)]
            scale = scanmod.UNIT_SCALE_TO_MM.get(units) if units != "auto" else scanmod._guess_unit_scale(pts)
            inlier_frac = None
            if scan_kind == "auto":
                scan_kind, inlier_frac = scanmod.classify_scan(pts * (scale or 1.0))
                log.info("Scan classified as %s (plane inliers %.2f)", scan_kind, inlier_frac)
            if scan_kind == "object":
                obj = scanmod.rasterize_object(data, ext, units=units, pts_colors=(pts, colors))
            else:
                raster = scanmod.rasterize_scan(data, ext, units=units, pts_colors=(pts, colors))
        except Exception as exc:  # noqa: BLE001
            log.exception("scan rasterization failed")
            raise ApiError(f"Could not process scan: {exc}")
        if scan_kind == "object":
            s = STORE.create(
                source_kind="object", filename=filename, original=obj.color_bgr,
                original_height=obj.height_mm, original_mm_per_px=obj.mm_per_px,
                scan_meta={"plane_inlier_fraction": inlier_frac, "unit_scale": obj.unit_scale},
                object_meta={"footprint_w_mm": obj.extent_mm[0], "footprint_h_mm": obj.extent_mm[1],
                             "thickness_mm": obj.thickness_mm},
            )
            # a single tool: the raster is already a top-down metric view, no calibration needed
            s.rectified = obj.color_bgr
            s.rect_height = obj.height_mm
            s.mm_per_px = obj.mm_per_px
            s.version = 1
            base = re.sub(r"\.[^.]+$", "", filename).strip() or "Tool"
            tool = _tool_result(s, f"o{s.id}", obj.footprint, [], None)
            tool["measured_thickness_mm"] = round(obj.thickness_mm, 1)
            tool["name"] = base
            extra_tools.append(tool)
        else:
            s = STORE.create(
                source_kind="scan", filename=filename, original=raster.color_bgr,
                original_height=raster.height_mm, original_frac=raster.above_frac, original_mm_per_px=raster.mm_per_px,
                suggested_corners=raster.suggested_corners.tolist() if raster.suggested_corners is not None else None,
                scan_meta={"plane_inlier_fraction": raster.plane_inlier_fraction, "unit_scale": raster.unit_scale,
                           "coverage": float(raster.coverage.mean())},
            )
    else:
        raise ApiError("Only 3D files are accepted: PLY, OBJ, STL, GLB/GLTF, OFF or XYZ. "
                       "Scan the whole layout (mat + tools) or a single tool and export a mesh or point cloud.")
    info = s.info()
    info["model"] = SEGMENTER.info()
    info["tools"] = extra_tools
    return jsonify(info), 201


@app.get("/api/sessions/<sid>")
def get_session(sid: str):
    s = _session(sid)
    info = s.info()
    info["model"] = SEGMENTER.info()
    return jsonify(info)


@app.get("/api/sessions/<sid>/image/<stage>")
def get_image(sid: str, stage: str):
    s = _session(sid)
    if stage == "original":
        img = s.original
    elif stage == "rectified":
        img = _require_rectified(s).rectified
    elif stage == "height":
        src = s.rect_height if s.rectified is not None else s.original_height
        if src is None:
            raise ApiError("This session has no height data.", 404)
        img = height_to_colormap(src)
    elif stage == "height_original":
        if s.original_height is None:
            raise ApiError("No height data", 404)
        img = height_to_colormap(s.original_height)
    else:
        raise ApiError("stage must be original | rectified | height", 404)
    small, _ = downscale_to(img, DISPLAY_MAX_SIDE)
    resp = Response(encode_jpeg(small, 86), mimetype="image/jpeg")
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.post("/api/sessions/<sid>/calibrate")
def calibrate(sid: str):
    s = _session(sid)
    if s.source_kind == "object":
        raise ApiError("This upload is a single tool model; it is already to scale and needs no calibration.")
    body = request.get_json(force=True, silent=True) or {}
    corners = body.get("corners")
    if not corners or len(corners) != 4:
        raise ApiError("corners must be 4 [x, y] points in original-image pixels")
    try:
        pts = [[float(c[0]), float(c[1])] for c in corners]
    except (TypeError, ValueError, IndexError):
        raise ApiError("corners must be 4 [x, y] points")
    ordered = calibration.order_corners(pts)
    # quarter turns clockwise: lets the user pick which physical corner becomes top-left
    turns = int(body.get("rotate_quarter_turns", 0) or 0) % 4
    if turns:
        ordered = np.roll(ordered, turns, axis=0)
    width_mm = _f(body, "width_mm")
    height_mm = _f(body, "height_mm")
    if s.source_kind == "scan" and (not width_mm or not height_mm):
        # metric raster: measure the rectangle
        hpx, vpx = calibration.edge_lengths_px(ordered)
        width_mm = width_mm or hpx * s.original_mm_per_px
        height_mm = height_mm or vpx * s.original_mm_per_px
    if not width_mm or not height_mm or width_mm <= 0 or height_mm <= 0:
        raise ApiError("width_mm and height_mm (real size of the mat / drawer) are required")
    extra = [s.original_height, s.original_frac] if s.original_height is not None else None
    warped, mm_per_px, H, extras = calibration.rectify(s.original, ordered, width_mm, height_mm, extra=extra,
                                                       already_ordered=True)
    s.rectified = warped
    s.rect_height = extras[0] if extras else None
    s.rect_frac = extras[1] if extras and len(extras) > 1 else None
    s.mm_per_px = mm_per_px
    s.mat_mm = (float(width_mm), float(height_mm))
    s.corners = ordered.tolist()
    s.homography = H
    s.version += 1
    s.masks.clear()
    info = s.info()
    info["model"] = SEGMENTER.info()
    return jsonify(info)


@app.post("/api/sessions/<sid>/auto_detect")
def auto_detect(sid: str):
    s = _require_rectified(_session(sid))
    body = request.get_json(force=True, silent=True) or {}
    mode = body.get("mode", "auto")
    min_area_mm2 = _f(body, "min_area_mm2", 200.0) or 200.0
    thr_mm = _f(body, "height_threshold_mm", 2.0) or 2.0
    refine = bool(body.get("refine_with_sam", s.rect_height is None))
    if mode == "auto":
        mode = "height" if s.rect_height is not None else "color"
    if mode == "height":
        if s.rect_height is None:
            raise ApiError("No height data; use mode 'color'")
        fg, blobs = geometry.detect_blobs_height(s.rect_height, s.mm_per_px, threshold_mm=thr_mm,
                                                 min_area_mm2=min_area_mm2, above_frac=s.rect_frac)
    elif mode == "color":
        fg, blobs = geometry.detect_blobs_color(s.rectified, s.mm_per_px, min_area_mm2=min_area_mm2)
    else:
        raise ApiError("mode must be auto | color | height")
    blobs = blobs[:40]
    tools = []
    use_sam = refine and SEGMENTER.available
    sam_error = None
    if use_sam:
        try:
            SEGMENTER.set_image(f"{s.id}:{s.version}", s.rectified)
        except Exception as exc:  # noqa: BLE001
            log.warning("SAM unavailable for auto-detect: %s", exc)
            use_sam = False
            sam_error = str(exc)
    prefix = body.get("id_prefix") or f"t{int(time.time()) % 100000}_"
    for i, b in enumerate(blobs):
        tid = f"{prefix}{i + 1}"
        points = [{"x": b["seed"][0], "y": b["seed"][1], "label": 1}]
        box = [float(v) for v in b["box"]]
        mask = b["mask"]
        if use_sam:
            try:
                sam = _sam_mask(s, points, box, bool(body.get("hq_token_only", False)))
                # guard against SAM grabbing the whole mat or losing the blob entirely
                ratio = float(sam.sum()) / max(1.0, float(mask.sum()))
                if 0.5 <= ratio <= 2.5:
                    mask = sam
            except Exception as exc:  # noqa: BLE001
                log.warning("SAM refine failed for blob %d: %s", i, exc)
        tools.append(_tool_result(s, tid, mask, points, box))
    return jsonify({"tools": tools, "mode": mode, "sam_used": use_sam, "sam_error": sam_error,
                    "foreground_fraction": float(fg.mean())})


@app.post("/api/sessions/<sid>/segment")
def segment(sid: str):
    s = _require_rectified(_session(sid))
    body = request.get_json(force=True, silent=True) or {}
    tools_in = body.get("tools") or []
    if not tools_in:
        raise ApiError("tools[] with prompt points is required")
    hq_token_only = bool(body.get("hq_token_only", False))
    if not SEGMENTER.available:
        raise ApiError("No HQ-SAM checkpoint on the server; click-to-segment is unavailable. Use auto-detect.", 503)
    results = []
    for t in tools_in:
        tid = str(t.get("id") or f"t{len(results) + 1}")
        points = _points(t.get("points"))
        box = t.get("box")
        if box is not None:
            try:
                box = [float(v) for v in box]
            except (TypeError, ValueError):
                raise ApiError("box must be [x0, y0, x1, y1]")
        if not points and box is None:
            results.append({"id": tid, "points": [], "box": None, "polygon_px": [], "area_mm2": 0.0,
                            "measured_thickness_mm": None, "height_stats": None})
            continue
        if not any(p["label"] == 1 for p in points) and box is None:
            raise ApiError(f"Tool {tid} needs at least one positive point")
        mask = _sam_mask(s, points, box, hq_token_only)
        results.append(_tool_result(s, tid, mask, points, box))
    return jsonify({"tools": results})


# ----------------------------------------------------------------------------- layout + export

def _tool_polygon_mm(t: Dict) -> List[List[float]]:
    """Tools carry mm polygons; older clients may send polygon_px + session_id instead."""
    poly = t.get("polygon_mm")
    if poly:
        return poly
    px = t.get("polygon_px") or []
    sid = t.get("session_id")
    sess = STORE.get(sid) if sid else None
    if px and sess is not None and sess.mm_per_px:
        return (np.asarray(px, dtype=np.float64) * sess.mm_per_px).tolist()
    return []


def _compute_layout(body: Dict) -> Dict:
    mat = body.get("mat") or {}
    width_mm = _f(mat, "width_mm")
    height_mm = _f(mat, "height_mm")
    if not width_mm or not height_mm:
        raise ApiError("mat.width_mm and mat.height_mm are required")
    smoothing = _f(body, "smoothing_mm", 0.6) or 0.0
    simplify = _f(body, "simplify_mm", 0.15) or 0.15
    default_clearance = _f(body, "default_clearance_mm", 1.0) or 0.0
    mirror = bool(body.get("mirror", False))
    tools_out = []
    for t in body.get("tools") or []:
        if not t.get("include", True):
            continue
        poly = _tool_polygon_mm(t)
        if len(poly) < 3:
            continue
        off = t.get("offset_mm") or {}
        notch = t.get("notch") or None
        res = geometry.process_outline(
            poly, 1.0,
            rotation_deg=_f(t, "rotation_deg", 0.0) or 0.0,
            offset_mm=(_f(off, "x", 0.0) or 0.0, _f(off, "y", 0.0) or 0.0),
            clearance_mm=_f(t, "clearance_mm", default_clearance) if t.get("clearance_mm") is not None else default_clearance,
            smoothing_mm=smoothing,
            notch=notch,
            simplify_mm=simplify,
        )
        depth = _f(t, "depth_mm", None)
        entry = {
            "id": str(t.get("id")),
            "name": t.get("name") or str(t.get("id")),
            "depth_mm": depth,
            "raw": t,
            "source_centroid_mm": res.get("source_centroid_mm"),
            "rings": res["rings"],
            "area_mm2": res["area_mm2"],
            "bbox_mm": res["bbox_mm"],
            "centroid_mm": res["centroid_mm"],
            "notch": res.get("notch"),
            "shapely": res.get("shapely"),
        }
        if mirror:
            from shapely import affinity

            entry["rings"] = geometry.mirror_rings(entry["rings"], width_mm)
            if entry["centroid_mm"]:
                entry["centroid_mm"] = [width_mm - entry["centroid_mm"][0], entry["centroid_mm"][1]]
            if entry["bbox_mm"]:
                b = entry["bbox_mm"]
                entry["bbox_mm"] = [width_mm - b[2], b[1], width_mm - b[0], b[3]]
            if entry["shapely"] is not None:
                entry["shapely"] = affinity.scale(entry["shapely"], xfact=-1.0, origin=(width_mm / 2.0, 0))
        tools_out.append(entry)
    # flag overlaps and out-of-mat tools for the UI
    from shapely.geometry import box as sbox

    mat_poly = sbox(0, 0, width_mm, height_mm)
    for i, a in enumerate(tools_out):
        ga = a.get("shapely")
        a["outside_mat"] = bool(ga is not None and not ga.is_empty and not mat_poly.contains(ga))
        overlaps = []
        for j, b in enumerate(tools_out):
            if i == j:
                continue
            gb = b.get("shapely")
            if ga is not None and gb is not None and not ga.is_empty and not gb.is_empty and ga.intersects(gb):
                if ga.intersection(gb).area > 0.5:
                    overlaps.append(b["id"])
        a["overlaps"] = overlaps
    return {"mat": {"width_mm": width_mm, "height_mm": height_mm}, "tools": tools_out, "mirror": mirror}


def _strip_private(layout: Dict) -> Dict:
    return {"mat": layout["mat"], "mirror": layout["mirror"],
            "tools": [{k: v for k, v in t.items() if k not in ("shapely", "raw", "source_centroid_mm")} for t in layout["tools"]]}


def _geometry_for_tool(raw: Dict):
    """(mask, height raster, mm_per_px) for a tool whose upload is still in memory, else None."""
    sid = raw.get("session_id")
    sess = STORE.get(sid) if sid else None
    if sess is None or sess.rect_height is None:
        return None
    mask = sess.masks.get(str(raw.get("id")))
    if mask is None or mask.shape != sess.rect_height.shape:
        return None
    return mask, sess.rect_height, sess.mm_per_px


@app.post("/api/layout")
@app.post("/api/sessions/<sid>/layout")
def layout(sid: Optional[str] = None):
    body = request.get_json(force=True, silent=True) or {}
    return jsonify(_strip_private(_compute_layout(body)))


@app.post("/api/export")
@app.post("/api/sessions/<sid>/export")
def export(sid: Optional[str] = None):
    body = request.get_json(force=True, silent=True) or {}
    fmt = (body.get("format") or "svg").lower()
    opts = body.get("export") or {}
    lay = _compute_layout(body)
    base = re.sub(r"[^A-Za-z0-9_-]+", "_", (opts.get("filename") or "tool_foam_layout")).strip("_") or "tool_foam_layout"
    if fmt == "svg":
        svg = layout_to_svg(lay, include_mat=bool(opts.get("include_mat", True)),
                            include_labels=bool(opts.get("include_labels", True)),
                            fill_mode=opts.get("fill_mode", "none"),
                            stroke_mm=float(opts.get("stroke_mm", 0.2)))
        return send_file(io.BytesIO(svg.encode("utf-8")), mimetype="image/svg+xml", as_attachment=True,
                         download_name=f"{base}.svg")
    if fmt == "dxf":
        data = layout_to_dxf(lay, include_mat=bool(opts.get("include_mat", True)),
                             include_labels=bool(opts.get("include_labels", True)))
        return send_file(io.BytesIO(data), mimetype="application/dxf", as_attachment=True, download_name=f"{base}.dxf")
    if fmt == "stl":
        data = layout_to_stl(lay, mat_thickness_mm=float(opts.get("mat_thickness_mm", 30.0)),
                             floor_min_mm=float(opts.get("floor_min_mm", 2.0)))
        inline = bool(body.get("inline", False))
        return send_file(io.BytesIO(data), mimetype="model/stl", as_attachment=not inline, download_name=f"{base}.stl")
    if fmt == "stl_tools":
        # the tools themselves as 3D bodies sitting in their pockets (for the assembled preview)
        data = layout_tools_to_stl(lay, mat_thickness_mm=float(opts.get("mat_thickness_mm", 30.0)),
                                   geometry_for_tool=_geometry_for_tool)
        inline = bool(body.get("inline", False))
        return send_file(io.BytesIO(data), mimetype="model/stl", as_attachment=not inline, download_name=f"{base}_tools.stl")
    raise ApiError("format must be svg | dxf | stl | stl_tools")


def main():
    ap = argparse.ArgumentParser(description="ToolCutter API server")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--preload", action="store_true", help="Load the HQ-SAM model at startup")
    args = ap.parse_args()
    log.info("Model: %s", SEGMENTER.info())
    if args.preload and SEGMENTER.available:
        SEGMENTER.ensure_loaded()
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()
