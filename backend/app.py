#!/usr/bin/env python3
"""ToolCutter API: 3D scans of laid-out tools (or single tool models) -> foam insert cutting files.

Flow:
  POST /api/sessions                       upload a 3D file (ply/obj/glb/stl); classified as a mat layout scan or a
                                           single tool model (form scan_kind=auto|layout|object)
  POST /api/captures                       phone capture: photo (+ LiDAR depth + intrinsics) of a drawer with ArUco
                                           corner markers -> metric top-down session, auto-calibrated
  POST /api/captures/multi                 several stills of one drawer (manifest + frames) fused on one grid
  POST /api/sweeps                         multi-frame sweep -> photogrammetry job (mac/Photogrammetry worker)
  GET  /api/jobs/<id>                      job status; carries session_id when done
  GET  /api/marker_sheet.svg               printable corner markers at true scale
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
import json
import shutil
import tempfile
import threading
import uuid
from pathlib import Path
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional

import math

import cv2
import hashlib
import numpy as np
from flask import Flask, Response, jsonify, request, send_file
from flask_cors import CORS

from toolcutter import calibration, capture as capmod, geometry, photogrammetry as pgm, scan as scanmod
from toolcutter.registration import floor_alignment, warp_translation
from toolcutter.tool_views import choose_tool_frame, coverage_report, discovery_windows
from toolcutter.exporters import layout_to_dxf, layout_to_stl, layout_to_svg, layout_tools_to_stl
from toolcutter.imaging import MESH_EXTENSIONS, decode_image, downscale_to, encode_jpeg, height_to_colormap
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
    log.warning("%s %s -> %d: %s", request.method, request.path, err.status, err)      # why a phone upload was refused
    return jsonify({"error": str(err)}), err.status


@app.errorhandler(Exception)
def _handle_unexpected(err: Exception):  # noqa: BLE001
    from werkzeug.exceptions import HTTPException
    if isinstance(err, HTTPException):            # 404 for /favicon.ico etc.: not a server error
        return jsonify({"error": err.description}), err.code or 500
    log.exception("Unhandled error")
    return jsonify({"error": f"{type(err).__name__}: {err}"}), 500


CAPTURE_DIR = Path(os.environ.get("TC_CAPTURE_DIR") or Path(__file__).parent / "captures")
_REBUILD_LOCK = threading.Lock()
_CAPTURE_LOCKS = {}
_CAPTURE_LOCKS_GUARD = threading.Lock()


def _capture_lock(sid):
    with _CAPTURE_LOCKS_GUARD:
        return _CAPTURE_LOCKS.setdefault(sid, threading.Lock())


def _session(sid: str):
    s = STORE.get(sid)
    if s is None and re.fullmatch(r"[0-9a-f]{12}", sid or "") and (CAPTURE_DIR / sid / "capture.json").exists():
        # a phone capture from before a restart (or pushed out of the in-memory store): rebuild it from disk
        with _capture_lock(sid):
            s = STORE.get(sid)
            if s is None:
                from toolcutter.processed_cache import load_session, load_detection
                s = load_session(CAPTURE_DIR / sid)
                if s is not None:
                    s.photo_result_cache = load_detection(CAPTURE_DIR / sid)
                    STORE.restore(s)
                else:
                    with _REBUILD_LOCK:
                        s = _rebuild_capture(sid)
    if s is None:
        raise ApiError("Unknown session (server restarted?). Upload again.", 404)
    return s


def _save_capture(sid: str, frames_meta: List[Dict], form: Dict, blobs: Dict[str, bytes], info: Dict) -> None:
    """Keep the raw upload of a phone capture so the session can be rebuilt later (and reused as test data)."""
    try:
        d = CAPTURE_DIR / sid
        d.mkdir(parents=True, exist_ok=True)
        for name, data in blobs.items():
            (d / Path(name).name).write_bytes(data)
        meta = {"id": sid, "created": time.time(), "form": {k: str(v) for k, v in form.items()}, "frames": frames_meta,
                "mat_mm": info.get("mat_mm"), "frames_used": (info.get("scan") or {}).get("frames_used"),
                "filename": info.get("filename")}
        (d / "capture.json").write_text(json.dumps(meta))
        from toolcutter.processed_cache import save_session
        session = STORE.get(sid)
        if session is not None:
            save_session(d, session)
    except Exception:  # noqa: BLE001
        log.exception("could not save capture %s", sid)


def _rebuild_capture(sid: str):
    d = CAPTURE_DIR / sid
    from toolcutter.processed_cache import load_session, save_session, load_detection
    cached = load_session(d)
    if cached is not None:
        cached.photo_result_cache = load_detection(d)
        log.info("opened processed capture %s without rebuilding", sid)
        return STORE.restore(cached)
    meta = json.loads((d / "capture.json").read_text())
    blobs = {}
    for f in meta["frames"]:
        for key in ("image", "depth"):
            if f.get(key):
                blobs[f[key]] = (d / Path(f[key]).name).read_bytes()
    log.info("rebuilding capture %s from disk (%d frames)", sid, len(meta["frames"]))
    s = _build_multi_session(meta["frames"], meta.get("form") or {}, blobs, session_id=sid)
    # the manifest caches what the listing shows; a rebuild under newer code can measure the drawer
    # differently (the marker-order fix moved one scan from 870 x 910 to 922 x 265), so refresh it
    mat = {"width": round(s.mat_mm[0], 1), "height": round(s.mat_mm[1], 1)} if s.mat_mm else None
    used = len(s.frames) if s.frames else None
    if mat != meta.get("mat_mm") or used != meta.get("frames_used"):
        meta["mat_mm"], meta["frames_used"] = mat, used
        try:
            (d / "capture.json").write_text(json.dumps(meta))
        except Exception:  # noqa: BLE001
            log.exception("could not refresh capture manifest %s", sid)
    save_session(d, s)
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


def _tool_result(s, tool_id: str, mask: np.ndarray, points: List[Dict], box: Optional[List[float]] = None,
                 color_silhouette: bool = True, frame_idx: Optional[int] = None, blob: Optional[np.ndarray] = None) -> Dict:
    support = (mask | blob) if blob is not None else mask
    stats = scanmod.measure_thickness(s.rect_height, support) if (s.rect_height is not None and support.any()) else None
    geom, H_after = s.capture_geom, s.homography
    if s.frames:
        if frame_idx is None:
            ys, xs = np.nonzero(mask)
            frame_idx = _nearest_frame(s, float(xs.mean()), float(ys.mean())) if xs.size else 0
        geom, H_after = s.frames[frame_idx]["geom"], s.frames[frame_idx]["H"]
    if color_silhouette and geom is not None and stats is not None:
        # the photo shows the tool's top surface; pull it back to the true footprint at its height
        poly = capmod.footprint_from_mask(mask, geom, stats["p95_mm"], H_after,
                                          height_raster=s.rect_height, mm_per_px=s.mm_per_px, blob=blob)
    else:
        poly = geometry.mask_to_polygon(mask)
    area_px = float(mask.sum())
    if poly is not None and color_silhouette and s.capture_geom is not None:
        area_px = float(cv2.contourArea(poly.astype(np.float32)))
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
    if stats is not None:
        res["height_stats"] = stats
        res["measured_thickness_mm"] = round(stats["p95_mm"], 1)
    s.masks[tool_id] = mask
    return res


def _depth_cell_mm(s) -> float:
    """Footprint of one depth pixel on the floor (mm) for this session's scan."""
    if s.frames:
        cells = [f["geom"].depth_cell_mm for f in s.frames if f.get("geom") is not None]
        if cells:
            return float(np.median(cells))
    if s.capture_geom is not None:
        return float(s.capture_geom.depth_cell_mm)
    return 3.0


def _topo_tool_result(s, tool_id: str, mask: np.ndarray, points: List[Dict], box: Optional[List[float]] = None,
                      signed: Optional[np.ndarray] = None) -> Dict:
    """Tool from a topographic footprint mask (no photo involved). `signed` is topo_footprint's
    height-minus-keep-level field, which puts the outline on the sub-pixel edge instead of on pixel corners."""
    # Hairline spikes and notches (1-2 px wide, several mm long) are below anything the sensor can resolve, and
    # contour-domain smoothing cannot remove them: a spike is thin in WIDTH but long along the CONTOUR (out and back),
    # so along-the-curve filters treat it as a feature. Nolan's screwdriver knob (2026-09-23) carried a 4.7 mm hair
    # that survived three rounds of polygon cleaning and showed as a "razor tooth". Mask morphology sees it for what
    # it is: open removes slivers narrower than HAIR_MM, close fills notches narrower than that. Both are well under
    # the 6 mm shaft of the thinnest tool and the 2 mm inner-corner limit noted for reconnection closings.
    mask = geometry.remove_hairs(mask, s.mm_per_px, HAIR_MM)
    stats = scanmod.measure_thickness(s.rect_height, mask) if (s.rect_height is not None and mask.any()) else None
    # the edge wobbles at roughly the depth-sensor cell size, so smooth at that scale, not the raster's;
    # straighten_ring then snaps the straight runs back, so corners come from intersecting runs, not rounding
    poly = geometry.topo_polygon(mask, s.mm_per_px, signed=signed,
                                 sigma_mm=max(1.0, TOPO_SMOOTH_CELLS * _depth_cell_mm(s)))
    # Calibrated on the FRONT TrueDepth camera only (tape 92.1 -> 89.5). The rear LiDAR's edge ramp is different:
    # applying the same trim there cost the arc harness 0.906 -> 0.870 IoU and added a smoke failure, so the trim
    # is gated to TrueDepth captures until it is made self-calibrating from the measured ramp.
    sensors = (s.scan_meta or {}).get("sensors") or []
    if poly is not None and s.rect_height is not None and EDGE_TRIM_MM > 0 \
            and any(str(x).startswith("truedepth") for x in sensors):
        poly = geometry.trim_edge_bias(poly, s.rect_height, s.mm_per_px, trim_mm=EDGE_TRIM_MM)
    area_px = float(cv2.contourArea(poly.astype(np.float32))) if poly is not None else float(mask.sum())
    res: Dict[str, Any] = {
        "id": tool_id, "session_id": s.id, "points": points, "box": box,
        "polygon_px": poly.tolist() if poly is not None else [],
        "polygon_mm": (poly * s.mm_per_px).tolist() if poly is not None else [],
        "area_mm2": area_px * (s.mm_per_px ** 2), "measured_thickness_mm": None, "height_stats": None, "edge_source": "topo",
    }
    if poly is not None:
        xs, ys = poly[:, 0], poly[:, 1]
        res["bbox_px"] = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
    if stats is not None:
        res["height_stats"] = stats
        res["measured_thickness_mm"] = round(stats["p95_mm"], 1)
    s.masks[tool_id] = mask
    return res


def _edge_source(s, body: Dict) -> str:
    """'topo' (scan topography only, default whenever a height map exists) or 'photo' (HQ-SAM + LiDAR band)."""
    src = body.get("edge_source")
    if src in ("topo", "photo"):
        if src == "topo" and s.rect_height is None:
            raise ApiError("This session has no height data; topographic edges need a scan.")
        return src
    if body.get("refine_with_sam") is True:
        return "photo"
    return "topo" if s.rect_height is not None else "photo"


RANK_PENALTY_MM = 0.0     # preferring exactly placed frames for SAM/mosaic was tried (150 mm): fixed one tool, lost two


def _nearest_frame(s, x: float, y: float) -> Optional[int]:
    """Multi-still sessions: the frame that saw (x, y) whose camera foot point is closest to it."""
    if not s.frames:
        return None
    H, W = s.rectified.shape[:2]
    xi, yi = int(np.clip(round(x), 0, W - 1)), int(np.clip(round(y), 0, H - 1))
    d = []
    for f in s.frames:
        dist = float(np.hypot(f["nadir_px"][0] - x, f["nadir_px"][1] - y))
        valid = f.get("valid")
        if valid is not None and not valid[yi, xi]:
            dist += 1e6
        # SAM's outline inherits the frame's registration: an exactly placed (two-marker) frame is preferred over
        # a one-marker / pose-only frame unless the latter is more than RANK_PENALTY_MM closer to the tool
        dist += f.get("rank", 0) * RANK_PENALTY_MM / s.mm_per_px
        dist += f.get("quality_penalty_px", 0)
        d.append(dist)
    return int(np.argmin(d))


SAM_CROP_MARGIN_MM = 18.0
SAM_CROP_TARGET = 1024


def _sam_mask(s, points: List[Dict], box: Optional[List[float]], hq_token_only: bool, frame_idx: Optional[int] = None,
              _coarse: Optional[bool] = None) -> np.ndarray:
    """HQ-SAM mask for one tool. The model works at ~1024 px, so instead of the whole drawer it sees a crop
    around the tool (prompt box or points + margin) upsampled to that size: 4-8x finer edges."""
    if s.frames:
        if frame_idx is None:
            pos = [(p["x"], p["y"]) for p in points if p["label"] == 1]
            if pos:
                cx, cy = np.mean(pos, axis=0)
            elif box is not None:
                cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            else:
                cx, cy = s.rectified.shape[1] / 2, s.rectified.shape[0] / 2
            frame_idx = _nearest_frame(s, cx, cy)
        image = s.frames[frame_idx]["color"]
        key = f"{s.id}:{s.version}:f{frame_idx}"
    else:
        image = s.rectified
        key = f"{s.id}:{s.version}"
    H, W = image.shape[:2]
    # crop window around the prompts
    pts = np.array([(p["x"], p["y"]) for p in points], dtype=np.float64) if points else np.zeros((0, 2))
    if box is not None:
        bx0, by0, bx1, by1 = box
    elif len(pts) and _coarse is None:
        # a click alone says nothing about the tool's extent: coarse pass on the whole image first, then
        # refine on a crop around what it found
        coarse = _sam_mask(s, points, None, hq_token_only, frame_idx=frame_idx, _coarse=True)
        ys, xs = np.nonzero(coarse)
        if xs.size:
            bx0, by0, bx1, by1 = min(xs.min(), pts[:, 0].min()), min(ys.min(), pts[:, 1].min()), max(xs.max() + 1, pts[:, 0].max()), max(ys.max() + 1, pts[:, 1].max())
        else:
            return coarse
    else:
        bx0, by0, bx1, by1 = 0, 0, W, H
    m = SAM_CROP_MARGIN_MM / s.mm_per_px
    x0, y0 = int(max(0, np.floor(bx0 - m))), int(max(0, np.floor(by0 - m)))
    x1, y1 = int(min(W, np.ceil(bx1 + m))), int(min(H, np.ceil(by1 + m)))
    cw, ch = x1 - x0, y1 - y0
    use_crop = cw >= 16 and ch >= 16 and (cw < W * 0.8 or ch < H * 0.8) and not os.environ.get("TC_SAM_NOCROP")
    if use_crop:
        scale = min(SAM_CROP_TARGET / max(cw, ch), 4.0)      # upsample small tools, never beyond 4x
        crop = image[y0:y1, x0:x1]
        if scale != 1.0:
            crop = cv2.resize(crop, (int(round(cw * scale)), int(round(ch * scale))), interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)
        SEGMENTER.set_image(f"{key}:crop:{x0},{y0},{x1},{y1},{scale:.3f}", crop)
        coords = [((p["x"] - x0) * scale, (p["y"] - y0) * scale) for p in points] or None
        cbox = [(bx0 - x0) * scale, (by0 - y0) * scale, (bx1 - x0) * scale, (by1 - y0) * scale] if box is not None else None
    else:
        SEGMENTER.set_image(key, image)
        scale = 1.0
        coords = [(p["x"], p["y"]) for p in points] or None
        cbox = box
    labels = [p["label"] for p in points] or None
    mask_c = SEGMENTER.predict(coords, labels, box=cbox, hq_token_only=hq_token_only)
    if use_crop:
        mask = np.zeros((H, W), bool)
        mc = mask_c.astype(np.uint8)
        if scale != 1.0:
            mc = cv2.resize(mc.astype(np.float32), (cw, ch), interpolation=cv2.INTER_LINEAR) > 0.5
        mask[y0:y1, x0:x1] = mc.astype(bool)
    else:
        mask = mask_c
    positives = [(p["x"], p["y"]) for p in points if p["label"] == 1]
    min_area_px = 20.0 / (s.mm_per_px ** 2)
    return clean_mask(mask, positives, min_area_px=min_area_px, fill_holes=True)


# ----------------------------------------------------------------------------- routes

@app.get("/health")
def health():
    return jsonify({"ok": True, "model": SEGMENTER.info(), "photogrammetry": pgm.available(), "time": time.time(),
                    "capture_capabilities": ["rear_lidar", "truedepth_lens_v1"], "stitching_version": 4, "outline_version": 3, "reconstruction_version": 5, "processed_scan_cache": True, "detection_cache_version": 2})


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
            pts, colors, normals = scanmod._load_points_full(data, ext)
            finite = np.isfinite(pts).all(axis=1)
            pts = pts[finite]
            normals = normals[finite] if normals is not None else None
            scale = scanmod.UNIT_SCALE_TO_MM.get(units) if units != "auto" else scanmod._guess_unit_scale(pts)
            inlier_frac = None
            if scan_kind == "auto":
                scan_kind, inlier_frac = scanmod.classify_scan(pts * (scale or 1.0))
                log.info("Scan classified as %s (plane inliers %.2f)", scan_kind, inlier_frac)
            if scan_kind == "object":
                obj = scanmod.rasterize_object(data, ext, units=units, pts_colors=(pts, colors))
            else:
                raster = scanmod.rasterize_scan(data, ext, units=units, pts_colors=(pts, colors), normals=normals)
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


@app.post("/api/captures")
def create_capture():
    """Phone capture: multipart `image` (+ optional `depth` float32 LE meters with depth_width/depth_height,
    `intrinsics` JSON {fx,fy,cx,cy,width,height} for the image), marker_size_mm, marker_dict, inset_mm,
    optional drawer_width_mm/drawer_height_mm override. Auto-calibrates when the 4 corner markers are seen."""
    f = request.files.get("image")
    if f is None:
        raise ApiError('Send the photo as multipart field "image"')
    data = f.read()
    if not data:
        raise ApiError("Empty image")
    try:
        img = decode_image(data)
    except Exception as exc:  # noqa: BLE001
        raise ApiError(f"Unsupported image: {exc}")
    form = request.form
    marker_size = float(form.get("marker_size_mm") or 50.0)
    marker_dict = form.get("marker_dict") or "4X4_50"
    inset = float(form.get("inset_mm") or 0.0)
    markers = capmod.corner_markers(capmod.detect_markers(img, marker_dict))
    K = None
    intr_raw = form.get("intrinsics")
    if intr_raw:
        try:
            intr = json.loads(intr_raw)
            K = np.array([[float(intr["fx"]), 0, float(intr["cx"])], [0, float(intr["fy"]), float(intr["cy"])], [0, 0, 1]])
            iw, ih = float(intr.get("width") or img.shape[1]), float(intr.get("height") or img.shape[0])
            if abs(iw - img.shape[1]) > 1 or abs(ih - img.shape[0]) > 1:
                # the photo was decoded at a different size (EXIF rotation / downscale): rescale intrinsics
                sx, sy = img.shape[1] / iw, img.shape[0] / ih
                K[0, :] *= sx
                K[1, :] *= sy
        except (KeyError, TypeError, ValueError) as exc:
            raise ApiError(f"Bad intrinsics: {exc}")
    depth = None
    df = request.files.get("depth")
    if df is not None:
        raw = df.read()
        try:
            if df.filename and df.filename.endswith(".npy"):
                depth = np.load(io.BytesIO(raw)).astype(np.float32)
            else:
                dw, dh = int(form.get("depth_width")), int(form.get("depth_height"))
                depth = np.frombuffer(raw, dtype="<f4").reshape(dh, dw).copy()
        except Exception as exc:  # noqa: BLE001
            raise ApiError(f"Bad depth map: {exc}")
        if K is None:
            raise ApiError("A depth map needs camera intrinsics")
    try:
        if depth is not None:
            res = capmod.rectify_rgbd(img, depth, K, markers, marker_size)
        else:
            if len(markers) == 0:
                raise ApiError("No ArUco markers found and no depth map: place the marker sheet in the drawer, or capture with LiDAR")
            res = capmod.rectify_markers_only(img, markers, marker_size)
    except ApiError:
        raise
    except Exception as exc:  # noqa: BLE001
        log.exception("capture processing failed")
        raise ApiError(f"Could not process capture: {exc}")
    corners = None
    if res.drawer_corners is not None:
        c = calibration.order_corners(res.drawer_corners)
        if inset:
            # markers sit `inset` mm inside the true drawer corners: push each corner outward along the edges
            c = capmod.expand_rectangle(c, inset / res.mm_per_px)
        corners = c
    s = STORE.create(
        source_kind="capture", filename=f.filename or "capture.jpg", original=res.color_bgr,
        original_height=res.height_mm, original_mm_per_px=res.mm_per_px,
        suggested_corners=corners.tolist() if corners is not None else None,
        scan_meta={**res.meta, "markers_found": res.markers_found, "has_depth": depth is not None},
    )
    s.capture_geom = res.geometry
    if corners is not None:
        hpx, vpx = calibration.edge_lengths_px(corners)
        width_mm = round(float(form.get("drawer_width_mm") or hpx * res.mm_per_px), 1)
        height_mm = round(float(form.get("drawer_height_mm") or vpx * res.mm_per_px), 1)
        _apply_calibration(s, corners, width_mm, height_mm)
        s.auto_calibrated = True
    info = s.info()
    info["model"] = SEGMENTER.info()
    info["tools"] = []
    return jsonify(info), 201


# ----------------------------------------------------------------------------- sweeps (photogrammetry jobs)

JOBS: Dict[str, Dict[str, Any]] = {}
JOBS_LOCK = threading.Lock()


def _job_update(jid: str, **kw):
    with JOBS_LOCK:
        JOBS[jid].update(kw)


def _session_from_mesh_file(path: Path, filename: str, marker_size_mm: float, inset_mm: float, units: str = "auto",
                            drawer_size_mm: Optional[Tuple[float, float]] = None):
    """Layout-scan session from a mesh; markers in the colour raster orient it and give the drawer rectangle."""
    ext = path.suffix.lstrip(".").lower()
    pts, colors, normals = scanmod._load_points_full(path, ext)        # by path: OBJ textures resolve
    raster = scanmod.rasterize_scan(b"", ext, units=units, pts_colors=(pts, colors), normals=normals)
    markers = capmod.detect_markers(raster.color_bgr)
    mirrored = False
    if not markers and len(capmod.detect_markers(cv2.flip(raster.color_bgr, 1))) >= 3:
        # markers only decode when seen from above: the plane normal was pointing down, redo it
        log.warning("mesh raster looked at the floor from below (markers decode only mirrored); flipping the plane normal")
        raster = scanmod.rasterize_scan(b"", ext, units=units, pts_colors=(pts, colors), normals=normals, flip_normal=True)
        markers = capmod.detect_markers(raster.color_bgr)
        mirrored = True
    s = STORE.create(
        source_kind="scan", filename=filename, original=raster.color_bgr,
        original_height=raster.height_mm, original_frac=raster.above_frac, original_mm_per_px=raster.mm_per_px,
        suggested_corners=raster.suggested_corners.tolist() if raster.suggested_corners is not None else None,
        scan_meta={"plane_inlier_fraction": raster.plane_inlier_fraction, "unit_scale": raster.unit_scale,
                   "coverage": float(raster.coverage.mean()), "photogrammetry": True},
    )
    s.scan_meta["normal_flipped_by_markers"] = mirrored
    markers = capmod.corner_markers(markers)
    s.scan_meta["markers_found"] = sorted(markers)
    if markers:
        # the raster is metric: the markers' plane coordinates are just raster px * mm_per_px. Work in the
        # marker-oriented frame (id 0 top-left) so corner roles are unambiguous; rotate back at the end.
        raw_corners = {i: c * raster.mm_per_px for i, c in markers.items()}
        # which corner each id sits on comes from the geometry, not from the printed numbering
        raw_corners = capmod.relabel_markers(raw_corners, capmod.corner_order({i: c.mean(axis=0) for i, c in raw_corners.items()}))
        s.scan_meta["markers_found"] = sorted(raw_corners)
        A = capmod._orient_for_markers(raw_corners)
        k = int(round(math.atan2(A[1, 0], A[0, 0]) / (math.pi / 2))) % 4
        R = A[:, :2]
        plane_corners = {i: c @ R.T for i, c in raw_corners.items()}
        sides = [np.linalg.norm(np.roll(pc, -1, axis=0) - pc, axis=1).mean() for pc in plane_corners.values()]
        s.scan_meta["marker_scale_check"] = float(marker_size_mm / np.mean(sides)) if sides else None
        rect = capmod.drawer_rectangle_from_markers(plane_corners)
        if drawer_size_mm and len(plane_corners) >= 2:
            # photogrammetry scale drifts by a percent or two: the known drawer size plus the marker spacing
            # (or the full rectangle) fixes it. Measure the visible marker-to-marker edges.
            outer = {i: capmod.outer_corner(c, i) for i, c in plane_corners.items()}
            meas, known = [], []
            for a, b, size in ((0, 1, drawer_size_mm[0]), (3, 2, drawer_size_mm[0]), (0, 3, drawer_size_mm[1]), (1, 2, drawer_size_mm[1])):
                if a in outer and b in outer:
                    meas.append(float(np.linalg.norm(outer[b] - outer[a]))); known.append(size - 2 * inset_mm)
            if meas:
                fix = float(np.sum(known) / np.sum(meas))
                if 0.9 < fix < 1.1:
                    raster.mm_per_px *= fix
                    plane_corners = {i: c * fix for i, c in plane_corners.items()}
                    rect = capmod.drawer_rectangle_from_markers(plane_corners)
                    s.original_mm_per_px = raster.mm_per_px
                    s.scan_meta["scale_fix_from_drawer_size"] = round(fix, 4)
        if rect is None and drawer_size_mm and len(plane_corners) >= 2:
            # two adjacent markers + the known drawer size (from an earlier capture of this drawer)
            rect = capmod.rectangle_from_two_markers(plane_corners, drawer_size_mm[0] - 2 * inset_mm,
                                                     drawer_size_mm[1] - 2 * inset_mm)
            if rect is not None:
                s.scan_meta["rectangle_from_two_markers"] = True
        if rect is not None:
            # rect is TL,TR,BR,BL in the oriented frame; rotate back to raster coordinates (px)
            rect_raw = (rect @ R) / raster.mm_per_px
            corners = calibration.order_corners(rect_raw)
            if k:
                corners = np.roll(corners, k, axis=0)
            if inset_mm:
                corners = capmod.expand_rectangle(corners, inset_mm / raster.mm_per_px)
            hpx, vpx = calibration.edge_lengths_px(corners)
            # depth-based photogrammetry is metric; the marker-size check is only reported (soft texture edges
            # make it imprecise), the user can still override the drawer size in the UI
            _apply_calibration(s, corners, round(hpx * raster.mm_per_px, 1), round(vpx * raster.mm_per_px, 1))
            s.suggested_corners = corners.tolist()
            s.auto_calibrated = True
    return s


def _run_sweep_job(jid: str, job_dir: Path, filename: str, marker_size_mm: float, inset_mm: float, detail: str,
                   drawer_size_mm: Optional[Tuple[float, float]] = None):
    try:
        _job_update(jid, status="running", message="reconstructing")
        out = job_dir / "model.obj"
        pgm.run_photogrammetry(job_dir / "frames", out, detail=detail,
                               progress=lambda f, msg: _job_update(jid, progress=f if f >= 0 else JOBS[jid].get("progress", 0.0), message=msg))
        _job_update(jid, message="building drawer map")
        s = _session_from_mesh_file(out, filename, marker_size_mm, inset_mm, drawer_size_mm=drawer_size_mm)
        _job_update(jid, status="done", progress=1.0, session_id=s.id, message="done")
    except Exception as exc:  # noqa: BLE001
        log.exception("sweep job failed")
        _job_update(jid, status="error", error=str(exc), message=str(exc))
    finally:
        shutil.rmtree(job_dir / "frames", ignore_errors=True)


@app.post("/api/sweeps")
def create_sweep():
    """Multi-frame capture for photogrammetry: multipart `manifest` (JSON, see mac/Photogrammetry) plus the
    frame/depth files it names. Returns a job id to poll at /api/jobs/<id>."""
    mf = request.files.get("manifest")
    if mf is None:
        raise ApiError('Send a "manifest" JSON file plus the frames it references')
    if not pgm.available():
        raise ApiError("Photogrammetry worker is not built on this Mac (mac/Photogrammetry). Build it with swift build -c release.", 503)
    try:
        manifest = json.loads(mf.read().decode("utf-8"))
        frames = manifest["frames"]
        assert isinstance(frames, list) and len(frames) >= 3
    except Exception as exc:  # noqa: BLE001
        raise ApiError(f"Bad manifest: {exc}")
    job_dir = Path(tempfile.mkdtemp(prefix="tc_sweep_"))
    frames_dir = job_dir / "frames"
    frames_dir.mkdir()
    saved = 0
    for f in frames:
        for key in ("image", "depth"):
            name = f.get(key)
            if not name:
                continue
            up = request.files.get(name)
            if up is None:
                shutil.rmtree(job_dir, ignore_errors=True)
                raise ApiError(f"Manifest names {name} but it was not uploaded")
            up.save(frames_dir / Path(name).name)
            saved += 1
    (frames_dir / "manifest.json").write_text(json.dumps(manifest))
    jid = uuid.uuid4().hex[:12]
    with JOBS_LOCK:
        JOBS[jid] = {"id": jid, "status": "queued", "progress": 0.0, "message": "queued", "frames": len(frames), "files": saved}
    t = threading.Thread(target=_run_sweep_job, args=(
        jid, job_dir, request.form.get("filename") or "sweep.obj",
        float(request.form.get("marker_size_mm") or 50.0), float(request.form.get("inset_mm") or 0.0),
        request.form.get("detail") or "medium",
        (float(request.form["drawer_width_mm"]), float(request.form["drawer_height_mm"]))
        if request.form.get("drawer_width_mm") and request.form.get("drawer_height_mm") else None), daemon=True)
    t.start()
    return jsonify(JOBS[jid]), 202


@app.get("/api/jobs/<jid>")
def get_job(jid: str):
    with JOBS_LOCK:
        job = JOBS.get(jid)
    if job is None:
        raise ApiError("Unknown job", 404)
    return jsonify(job)


def _rectify_frames(results, corners_list, width_mm: float, height_mm: float, ppm: float) -> List[Dict]:
    frames = []
    for res, c in zip(results, corners_list):
        ones = np.ones(res.color_bgr.shape[:2], dtype=np.float32)
        extra = [ones] + ([res.height_mm] if res.height_mm is not None else [])
        warped, mm_per_px, H, extras = calibration.rectify(res.color_bgr, c, width_mm, height_mm, extra=extra,
                                                           already_ordered=True, ppm_override=ppm)
        Hh_, W_ = warped.shape[:2]
        # which part of the common grid this frame actually saw (warp fills the rest by replication)
        valid = cv2.warpPerspective(ones, H, (W_, Hh_), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0.5
        valid = cv2.erode(valid.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        hgt = extras[1] if len(extras) > 1 else None
        if hgt is not None:
            hgt = hgt.astype(np.float32)
            hgt[~valid] = np.nan
            lev, _ = geometry.level_height_raster(hgt)   # unknown cells must not vote as zero-height floor
            lev[~valid] = np.nan
            hgt = lev
        if res.geometry is not None:
            n0 = res.geometry.plane_to_raster(np.array([[0.0, 0.0]]))
            nadir = cv2.perspectiveTransform(n0.reshape(-1, 1, 2), H).reshape(-1)
        else:
            nadir = np.array([warped.shape[1] / 2.0, warped.shape[0] / 2.0])
        frames.append({"color": warped, "height": hgt, "geom": res.geometry, "H": H, "nadir_px": (float(nadir[0]), float(nadir[1])),
                       "valid": valid, "markers": res.markers_found, "markers_px": res.markers_px,
                       "use": res.meta.get("use", "both"),
                       "camera_height_mm": res.meta.get("camera_height_mm")})
    return frames


def _shift_source_corners(frame: Dict, grid_w: int, grid_h: int, dx: float, dy: float) -> np.ndarray:
    """Source-raster corners that would place this frame's rectified image shifted by (dx, dy) grid px:
    the grid rectangle corners moved by (-dx, -dy), mapped back through the frame's homography."""
    grid = np.array([[0, 0], [grid_w, 0], [grid_w, grid_h], [0, grid_h]], dtype=np.float64) - np.array([dx, dy])
    return cv2.perspectiveTransform(grid.reshape(-1, 1, 2), np.linalg.inv(frame["H"])).reshape(-1, 2)


def _floor_texture(frame: Dict) -> np.ndarray:
    """High-passed grey floor texture of a rectified frame (NaN off-frame and on tool tops, whose position
    moves with the viewpoint). Exact at floor level, so it registers frames like the height map does — and
    it is dense where the height map is flat."""
    g = cv2.cvtColor(frame["color"], cv2.COLOR_BGR2GRAY).astype(np.float32)
    hp = g - cv2.GaussianBlur(g, (0, 0), 6)
    hp[~frame["valid"]] = np.nan
    if frame["height"] is not None:
        hp[np.nan_to_num(frame["height"], nan=0.0) > 2.0] = np.nan
    return hp


SEAM_FEATHER_MM = 4.0     # width of the cross-fade between neighbouring frames in the display mosaic
_BLEND_DEBUG: Dict = {}
TALL_OBJECT_MM = 4.0      # above this a pixel is "on a tool": one frame owns the whole blob, no join across it


def _blend_mosaic(frames: List[Dict], best: np.ndarray, mm_per_px: float,
                  height: Optional[np.ndarray] = None) -> np.ndarray:
    """The display mosaic: each pixel from the frame whose camera was most nearly overhead, but with each
    frame's exposure matched, the joins cross-faded, and every raised object taken whole from ONE frame.

    Three things made the old hard cut look like patchwork:
      * phones re-expose between frames (13% brightness spread over one real glide) and a glossy tool or a
        varnished drawer reflects the room differently from each viewpoint, so even perfectly registered
        frames disagree in colour -> per-frame gain, then a cross-fade at the joins;
      * the mosaic is a projection onto the FLOOR, so anything with height appears displaced by h*r/D, in a
        different direction in every frame (11 mm for a 25 mm object 200 mm off-nadir at 450 mm). A join
        crossing a tool therefore steps — or, once cross-faded, ghosts. So each raised blob is assigned in
        one piece to the single frame that saw it most nearly overhead, and no join crosses it.
    Geometry is untouched: outlines come from the fused height map, and the per-frame images the tool
    outliner uses are left alone."""
    n = len(frames)
    gray = [np.where(f["valid"], cv2.cvtColor(f["color"], cv2.COLOR_BGR2GRAY).astype(np.float32), np.nan) for f in frames]
    with np.errstate(all="ignore"):
        consensus = np.nanmedian(np.stack(gray), axis=0)
        gains = []
        for g in gray:
            m = np.isfinite(g) & np.isfinite(consensus) & (consensus > 1)
            r = float(np.nanmedian(g[m] / consensus[m])) if m.sum() > 500 else 1.0
            gains.append(np.clip(r, 0.7, 1.4) if np.isfinite(r) else 1.0)
    yy, xx = np.mgrid[0:best.shape[0], 0:best.shape[1]]
    tall_raw = None if height is None else (np.nan_to_num(height, nan=0.0) > TALL_OBJECT_MM).astype(np.uint8)
    if tall_raw is not None and not tall_raw.any():
        tall_raw = None

    # A frame may only supply floor colour where IT sees floor. Each frame's own height map marks its raised
    # footprints; in its photo those tops appear pushed off the footprint to nadir + (1 + h/D)(p - nadir),
    # so both the footprint and that displaced image are barred. Using each frame's OWN heights rather than
    # the fused ones matters twice over: the displacement needs the per-pixel height (one scan had a 25 mm
    # keyboard beside a 110 mm upright can, and using the tallest for everything threw most of the mosaic
    # away), and a frame that is simply misplaced carries its tool along, so its own map is what says where
    # that frame would paint a tool onto what everything else agrees is bare floor.
    halos: List[Optional[np.ndarray]] = []
    Hh_g, W_g = best.shape
    for f in frames:
        fh = f.get("height")
        if fh is None:
            halos.append(None)
            continue
        own = np.nan_to_num(fh, nan=0.0)
        ty, tx = np.nonzero(own > TALL_OBJECT_MM)
        if not len(ty):
            halos.append(None)
            continue
        D = float(f.get("camera_height_mm") or 0) or 450.0
        nx, ny = float(f["nadir_px"][0]), float(f["nadir_px"][1])
        sc = D / np.maximum(D - own[ty, tx], 0.1 * D)
        qx = np.rint(nx + (tx - nx) * sc).astype(np.int64)
        qy = np.rint(ny + (ty - ny) * sc).astype(np.int64)
        ok_q = (qx >= 0) & (qx < W_g) & (qy >= 0) & (qy < Hh_g)
        g = np.zeros(best.shape, np.uint8)
        g[ty, tx] = 1                                    # the footprint itself
        g[qy[ok_q], qx[ok_q]] = 1                        # ...and where its top appears
        halo = cv2.dilate(g, np.ones((5, 5), np.uint8)) > 0          # close the scatter's gaps
        if tall_raw is not None:
            # only bar it from ground the others call FLOOR. Barring a frame from a real tool's own
            # footprint leaves nobody able to paint that tool, and it comes out shredded into wedges.
            halo &= ~(cv2.dilate(tall_raw, np.ones((5, 5), np.uint8)) > 0)
        halos.append(halo)

    # pick per pixel again, this time refusing a frame that would paint a ghost here
    pick = np.full(best.shape, -1, np.int32)
    score = np.full(best.shape, np.inf, np.float32)
    for i, f in enumerate(frames):
        s = np.where(f["valid"], np.hypot(xx - f["nadir_px"][0], yy - f["nadir_px"][1]).astype(np.float32)
                     + f.get("quality_penalty_px", 0), np.inf)
        if halos[i] is not None:
            s = np.where(halos[i], s + 1e6, s)     # only if nothing clean covers this pixel
        better = s < score
        score[better] = s[better]
        pick[better] = i
    pick = np.where(pick < 0, best, pick)

    # a raised blob comes whole from the single frame that saw it most nearly overhead: a join across it
    # would step by the parallax difference between the two frames
    solid = np.full(best.shape, -1, np.int32)
    if tall_raw is not None:
        grow = max(1, int(round(4.0 / mm_per_px)))
        blobs = cv2.dilate(tall_raw, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1,) * 2))
        n_lab, lab = cv2.connectedComponents(blobs)
        for k in range(1, n_lab):
            m = lab == k
            if m.sum() < 200:
                continue
            cy, cx = float(yy[m].mean()), float(xx[m].mean())
            owner = choose_tool_frame(frames, m, mm_per_px)
            if owner is not None:
                # Ownership must cover the visible TOP too, not just its floor
                # footprint. Otherwise feathering cuts off or doubles the far
                # edge of a tall tool even though its footprint has one owner.
                f = frames[owner]
                D = float(f.get("camera_height_mm") or 450)
                nx, ny = f["nadir_px"]
                py, px = np.nonzero(m)
                heights = np.nan_to_num(height[py, px], nan=0.0)
                factor = D / np.maximum(D - heights, 0.1 * D)
                qx = np.rint(nx + (px - nx) * factor).astype(np.int64)
                qy = np.rint(ny + (py - ny) * factor).astype(np.int64)
                inside = (qx >= 0) & (qx < W_g) & (qy >= 0) & (qy < Hh_g)
                visible = m.astype(np.uint8)
                visible[qy[inside], qx[inside]] = 1
                visible = cv2.morphologyEx(visible, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) > 0
                solid[visible & f["valid"]] = owner

    sigma = max(1.0, SEAM_FEATHER_MM / mm_per_px)
    acc = np.zeros(frames[0]["color"].shape, np.float32)
    wsum = np.zeros(best.shape, np.float32)
    blend_here = solid < 0
    for i, f in enumerate(frames):
        w = cv2.GaussianBlur(((pick == i) & blend_here).astype(np.float32), (0, 0), sigma)
        w[~f["valid"]] = 0.0                      # never borrow colour a frame does not have
        w[~blend_here] = 0.0                      # ...never bleed a neighbour's parallax onto a tool
        if halos[i] is not None:
            w[halos[i]] = 0.0                     # ...and never blend in a ghost
        if w.max() <= 0:
            continue
        acc += (w[..., None] * f["color"].astype(np.float32)) / gains[i]
        wsum += w
    out = np.zeros_like(acc)
    ok = wsum > 1e-6
    out[ok] = acc[ok] / wsum[ok, None]
    for i, f in enumerate(frames):
        m = (solid == i) | ((~ok) & blend_here & (pick == i) & f["valid"])
        if m.any():
            out[m] = f["color"][m].astype(np.float32) / gains[i]
    # If views disagree strongly, averaging creates a translucent second tool
    # edge. Keep the selected source there; feather only consistent overlaps.
    disagreement = np.zeros(best.shape, bool)
    for i, f in enumerate(frames):
        color = f["color"].astype(np.float32) / gains[i]
        delta = np.max(np.abs(color - out), axis=2)
        disagreement |= blend_here & f["valid"] & (delta > 35)
    for i, f in enumerate(frames):
        keep = disagreement & (pick == i) & f["valid"]
        out[keep] = f["color"][keep].astype(np.float32) / gains[i]
    # The protected tool patch includes a floor margin. Fade its OUTSIDE edge
    # into the surrounding floor, leaving the entire tool untouched. Hard-cut
    # ownership otherwise leaves dark islands from viewpoint-dependent shadows.
    for i, f in enumerate(frames):
        owned = solid == i
        if not owned.any():
            continue
        distance = cv2.distanceTransform((~owned).astype(np.uint8), cv2.DIST_L2, 3)
        alpha = np.clip(1.0 - distance / sigma, 0, 1)
        allowed = blend_here & f["valid"] & ~disagreement
        if halos[i] is not None:
            allowed &= ~halos[i]
        alpha[~allowed] = 0
        out = out * (1 - alpha[..., None]) + (f["color"].astype(np.float32) / gains[i]) * alpha[..., None]
    if tall_raw is not None:
        unresolved = (tall_raw > 0) & (solid < 0)
        for i, f in enumerate(frames):
            owned = unresolved & (pick == i) & f["valid"]
            out[owned] = f["color"][owned].astype(np.float32) / gains[i]
    if os.environ.get("TC_BLEND_DEBUG"):
        _BLEND_DEBUG.update(pick=pick, solid=solid, halos=halos, tall=tall_raw, gains=gains)
    return np.clip(out, 0, 255).astype(np.uint8)


REFIT_LOW_MM = float(os.environ.get("TC_REFIT_LOW_MM", "8.0"))   # re-fit blobs shorter than this once more
# A detected tool must be at least this tall at its 90th percentile. Bare mat reads 0.08 mm (p90) on a clean
# scan and up to 3.4 mm (p99) over an ArUco marker; a real 3 mm steel rule reads 2.5 mm. TC_TOOL_MIN_H overrides.
TOOL_MIN_HEIGHT_MM = float(os.environ.get("TC_TOOL_MIN_H", "1.5"))
# Mask features thinner than this are hairs (sub-resolution slivers/notches), removed before an outline is traced.
HAIR_MM = float(os.environ.get("TC_HAIR_MM", "1.6"))
# Outward edge bias of the depth sensor on a tall wall, trimmed back per vertex (0 below 3 mm, full above 15 mm).
# Fitted to calipers: tape measure traced 92.1 (after hair removal) vs 89.5 true = 1.3 mm per side. TC_EDGE_TRIM_MM=0 disables.
EDGE_TRIM_MM = float(os.environ.get("TC_EDGE_TRIM_MM", "1.3"))
# The marker exclusion zone may erase pixels only up to this height: paper, even folded, is not this tall.
MARKER_MASK_MAX_MM = float(os.environ.get("TC_MARKER_MASK_MAX_MM", "5.0"))
TOPO_SMOOTH_CELLS = float(os.environ.get("TC_TOPO_SMOOTH", "1.0"))   # outline smoothing, in depth cells
MIN_SHIFT_GAIN = float(os.environ.get("TC_MIN_GAIN", "1.0"))    # never apply a shift that fits WORSE than standing still
APPLY_SHIFT_MAX_MM = float(os.environ.get("TC_APPLY_MAX_MM", "8.0"))   # ...nor one too big to be drift (see below)


def _shift_gain(a: np.ndarray, b: np.ndarray, mask: np.ndarray, dx: float, dy: float) -> float:
    """How much better `a` matches `b` after moving it by (dx, dy): residual before / residual after.
    1.0 = no improvement. Guards against phase correlation's confident lock onto a periodic pattern,
    which matches just as well one period over but tears everything around it."""
    if abs(dx) + abs(dy) < 1e-6:
        return 1.0
    sa = warp_translation(a, dx, dy)
    sm = (warp_translation(mask.astype(np.uint8), dx, dy, interpolation=cv2.INTER_NEAREST) > 0) & mask
    if sm.sum() < 500:
        return 0.0
    before = float(np.abs(a[sm] - b[sm]).mean())
    after = float(np.abs(sa[sm] - b[sm]).mean())
    return before / after if after > 1e-6 else 0.0


def _refine_frame_alignment(frames: List[Dict], fixed: Sequence[bool], order: Sequence[int], mm_per_px: float,
                            max_shift_mm: float = 12.0) -> List[Tuple[float, float, float]]:
    """Translation (grid px) that best aligns each frame in `order` with the frames already placed: the
    fixed ones (marker-registered at the anchor's end) plus every frame processed before it. Phase correlation
    over the overlap of a combined signal (height map + floor texture). Returns (dx, dy, response) per
    frame; (0, 0, 0) for fixed / skipped frames.

    Only a frame whose own shift was accepted joins the reference. Letting every processed frame seed it
    (to close the "overlap 0 px" hole in the middle of a glide, where no marker frame reaches) was tried and
    reverted 2026-09-20: frames then align to a reference built from drifted pose placements and the error
    propagates — two synthetic glides lost 0.04 IoU and 1.4 mm. A mid-glide frame with nothing solid to
    align against is better left where its pose put it."""
    n = len(frames)
    out = [(0.0, 0.0, 0.0)] * n
    if not order:
        return out
    fixed = list(fixed)
    if not any(fixed):
        fixed[0] = True                         # nothing fixed by markers: the first frame is the origin
    tex = [_floor_texture(f) for f in frames]
    from toolcutter.depth_fusion import fuse_heights
    ref_h = fuse_heights([f["height"] for f, a in zip(frames, fixed) if a and f["height"] is not None])
    if ref_h is None:
        ref_h = np.full(tex[0].shape, np.nan, np.float32)
    ref_c = fuse_heights([t for t, a in zip(tex, fixed) if a])
    max_px = max_shift_mm / mm_per_px

    def seed(idx: int, dx: float = 0.0, dy: float = 0.0) -> None:
        """Add a placed frame to the reference, filling only what the reference does not have yet."""
        nonlocal ref_h, ref_c
        with np.errstate(all="ignore"):
            hh = frames[idx]["height"]
            if hh is not None:
                ref_h = np.where(np.isfinite(ref_h), ref_h, warp_translation(hh, dx, dy, border=float("nan")))
            ref_c = np.where(np.isfinite(ref_c), ref_c, warp_translation(tex[idx], dx, dy, border=float("nan")))

    for i in order:
        if fixed[i]:
            continue
        h = frames[i]["height"]
        c = tex[i]
        vh = np.isfinite(h) & np.isfinite(ref_h) if h is not None else np.zeros(c.shape, bool)
        vc = np.isfinite(c) & np.isfinite(ref_c)
        v = vh | vc
        if v.sum() < 2000:
            log.info("refine frame %d: skipped (overlap %d px)", i, int(v.sum()))
            continue
        ys, xs = np.nonzero(v)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        a = np.zeros((y1 - y0, x1 - x0), np.float32)
        b = np.zeros_like(a)
        used = np.zeros(a.shape, bool)
        parts = []
        if vh.any():
            hs = float(np.nanstd(ref_h[y0:y1, x0:x1][vh[y0:y1, x0:x1]]))
            if hs > 0.5:
                m = vh[y0:y1, x0:x1]
                a += np.where(m, np.nan_to_num(h[y0:y1, x0:x1], nan=0.0), 0.0) / hs
                b += np.where(m, np.nan_to_num(ref_h[y0:y1, x0:x1], nan=0.0), 0.0) / hs
                used |= m
                parts.append("height")
        if vc.any() and not parts:                 # floor texture only where the height map is flat
            cs = float(np.nanstd(ref_c[y0:y1, x0:x1][vc[y0:y1, x0:x1]]))
            if cs > 1e-3:
                m = vc[y0:y1, x0:x1]
                a += np.where(m, np.nan_to_num(c[y0:y1, x0:x1], nan=0.0), 0.0) / cs
                b += np.where(m, np.nan_to_num(ref_c[y0:y1, x0:x1], nan=0.0), 0.0) / cs
                used |= m
                parts.append("texture")
        if not parts:
            log.info("refine frame %d: skipped (flat overlap)", i)
            continue
        win = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
        (dx, dy), resp = cv2.phaseCorrelate(a - a.mean(), b - b.mean(), win)
        if resp < 0.05 or math.hypot(dx, dy) > max_px:
            log.info("refine frame %d: rejected shift (%.1f, %.1f) px resp %.3f [%s]", i, dx, dy, resp, "+".join(parts))
            continue
        # The peak is not proof. Phase correlation locks happily onto a REPEATING pattern — a keyboard's
        # keys, a row of sockets, a drawer of identical spanners — and reports a confident shift of one
        # period, which tears the mosaic. So check the move actually fits better than standing still.
        gain = _shift_gain(a, b, used, dx, dy)
        if gain < MIN_SHIFT_GAIN:
            log.info("refine frame %d: rejected shift (%.1f, %.1f) px resp %.3f gain %.2f (no better than leaving it) [%s]",
                     i, dx, dy, resp, gain, "+".join(parts))
            continue
        log.info("refine frame %d: shift (%.1f, %.1f) px = (%.1f, %.1f) mm resp %.3f gain %.2f [%s]", i, dx, dy, dx * mm_per_px, dy * mm_per_px, resp, gain, "+".join(parts))
        out[i] = (float(dx), float(dy), float(resp))
        seed(i, dx, dy)                          # now a reference for the frames after it, where it landed
    return out


def _parse_frame_upload(f: Dict, blobs: Dict[str, bytes], form_intr: Optional[Dict] = None):
    """Decode one frame (image + optional depth + intrinsics) named in a multi-still manifest."""
    up = blobs.get(f["image"])
    if up is None:
        raise ApiError(f"Frame image {f['image']} not uploaded")
    img = decode_image(up)
    K = None
    intr = f.get("intrinsics") or form_intr
    if intr:
        K = np.array([[float(intr["fx"]), 0, float(intr["cx"])], [0, float(intr["fy"]), float(intr["cy"])], [0, 0, 1]])
        iw, ih = float(intr.get("width") or img.shape[1]), float(intr.get("height") or img.shape[0])
        if abs(iw - img.shape[1]) > 1 or abs(ih - img.shape[0]) > 1:
            K[0, :] *= img.shape[1] / iw
            K[1, :] *= img.shape[0] / ih
    depth = None
    if f.get("depth"):
        raw = blobs.get(f["depth"])
        if raw is None:
            raise ApiError(f"Frame depth {f['depth']} not uploaded")
        depth = np.frombuffer(raw, dtype="<f4").reshape(int(f["depth_height"]), int(f["depth_width"])).copy()
        if K is None:
            raise ApiError("Depth frames need intrinsics")
        depth[~np.isfinite(depth) | (depth <= 0)] = np.nan
        if f.get("lens_calibration") is not None:
            from toolcutter.depth_camera import rectify_lens
            try:
                img, depth = rectify_lens(img, depth, f["lens_calibration"])
            except (ValueError, TypeError) as exc:
                raise ApiError(f"Invalid depth camera calibration: {exc}") from exc
    # ARKit's FRONT camera delivers a MIRRORED image (the selfie convention), and a mirror image breaks
    # everything that cares about handedness: ArUco cannot decode a reflected marker, and ORB cannot match a
    # reflected raster against a normally-oriented one. On Nolan's capture 136c77400c3f all 102 front frames
    # found ZERO markers with a marker plainly in shot, and every one decoded once flipped. Un-mirror here, after
    # the lens model (which is radial about a centre measured in the mirrored frame, so it must be applied first),
    # and move the principal point with the pixels. Plain "truedepth" (AVFoundation) frames are NOT mirrored —
    # capture b35103e40780 detects markers as delivered — so this is keyed to the ARKit path exactly.
    if str(f.get("sensor") or "") == "truedepth_tracked":
        img = np.ascontiguousarray(img[:, ::-1])
        if depth is not None:
            depth = np.ascontiguousarray(depth[:, ::-1])
        if K is not None:
            K[0, 2] = img.shape[1] - 1 - K[0, 2]
    return img, depth, K


ARKIT_TO_CV = np.diag([1.0, -1.0, -1.0])   # ARKit camera: x right, y up, z back  ->  OpenCV: x right, y down, z forward


# The ARKit FRONT camera's world transform is NOT usable the way the rear one is. Its image arrives mirrored
# (see _parse_frame_upload), and the pose does not survive that reflection: on capture 136c77400c3f, honouring
# these poses placed 4 frames and got 23 REJECTED as "pose inconsistent with the floor plane" (30-132 mm), for
# 85 of 108 frames used; ignoring them and letting markers + neighbour matching do the work places 108 of 108
# with nothing skipped and an equally clean height map. The transforms are still sent and stored, so the
# convention can be worked out later from saved captures — but nothing depends on them today.
NO_POSE_SENSORS = {"truedepth_tracked"}


def _pose_cv(f: Dict) -> Optional[np.ndarray]:
    """4x4 camera-to-world pose in OpenCV camera convention from a manifest frame's ARKit `transform`
    (16 floats, column-major as simd_float4x4 stores it, or a 4x4 nested list)."""
    if str(f.get("sensor") or "") in NO_POSE_SENSORS:
        return None
    t = f.get("transform")
    if t is None:
        return None
    arr = np.asarray(t, dtype=np.float64)
    M = arr.reshape(4, 4).T if arr.ndim == 1 else arr.reshape(4, 4)
    if abs(M[3, 3] - 1.0) > 1e-6 and abs(M[3, 3]) < 1e-9:   # row-major 16 floats
        M = arr.reshape(4, 4)
    R, tr = M[:3, :3], M[:3, 3]
    # X_world = R * X_arkitcam + t ;  X_arkitcam = ARKIT_TO_CV * X_cv  (ARKIT_TO_CV is its own inverse)
    R_cv = R @ ARKIT_TO_CV
    out = np.eye(4)
    out[:3, :3] = R_cv
    out[:3, 3] = tr * (1000.0 if f.get("transform_units", "m") == "m" else 1.0)
    return out


CORNER_NAMES = ["top-left", "top-right", "bottom-right", "bottom-left"]


def _corner_map_for_frames(frames_meta: List[Dict], blobs: Dict[str, bytes], marker_dict: str,
                           marker_size_mm: float) -> Optional[Dict[int, int]]:
    """Global marker id -> corner slot (0 TL, 1 TR, 2 BR, 3 BL) for a multi-frame capture.

    A printed id only means "this corner" if the sheet was laid out in the printed order. Nolan's first real
    scans had ids 0 and 2 together at one end of the drawer and 1 and 3 at the other, which the old code read
    as a bow-tie and turned into a 0.9 m square "drawer". So work the order out from where the markers are:
    lift each one into ARKit world coordinates (solvePnP + the frame's pose) and flatten along gravity, which
    gives one top-down, un-mirrored picture of the whole sheet even when no single frame sees three markers.
    With no usable poses, the frame that sees the most markers decides. None = leave the printed ids alone.

    Cheap on purpose (reduced-size decode, no depth): it runs before the real per-frame rectification, which
    needs the answer to orient each raster.
    """
    obj = np.array([[-0.5, 0.5, 0.0], [0.5, 0.5, 0.0], [0.5, -0.5, 0.0], [-0.5, -0.5, 0.0]]) * marker_size_mm
    world: Dict[int, List[np.ndarray]] = {}
    best: Dict[int, np.ndarray] = {}
    for f in frames_meta:
        raw = blobs.get(f.get("image"))
        if raw is None:
            continue
        small = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_REDUCED_COLOR_2)
        if small is None:
            continue
        mk = capmod.corner_markers(capmod.detect_markers(small, marker_dict))
        if not mk:
            continue
        intr, pose = f.get("intrinsics"), _pose_cv(f)
        if intr is None or pose is None:
            if len(mk) > len(best):
                best = mk
            continue
        # detection pixels -> the intrinsics' own pixel frame
        sx = float(intr.get("width") or small.shape[1] * 2) / small.shape[1]
        sy = float(intr.get("height") or small.shape[0] * 2) / small.shape[0]
        mk = {i: c * np.array([sx, sy]) for i, c in mk.items()}
        if len(mk) > len(best):
            best = mk
        K = np.array([[float(intr["fx"]), 0, float(intr["cx"])], [0, float(intr["fy"]), float(intr["cy"])], [0, 0, 1]])
        for i, c in mk.items():
            ok, _, tvec = cv2.solvePnP(obj, c.astype(np.float64), K, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if ok:
                world.setdefault(i, []).append(pose[:3, :3] @ tvec.reshape(3) + pose[:3, 3])
    if len(world) >= 3:
        # ARKit world is y-up and right-handed, so (x, z) is the top-down frame with z running "down" the page
        flat = {i: np.array([float(np.median([p[0] for p in v])), float(np.median([p[2] for p in v]))])
                for i, v in world.items()}
        cmap = capmod.corner_order(flat)
        if cmap:
            return cmap
        log.warning("multi-still: %d markers do not form a rectangle in world coordinates: %s",
                    len(flat), {i: np.round(p).tolist() for i, p in flat.items()})
    if len(best) >= 3:
        return capmod.corner_order({i: c.mean(axis=0) for i, c in best.items()})
    return None


@app.post("/api/captures/multi")
def create_multi_capture():
    """Several stills of the same drawer from different positions (manifest like /api/sweeps, depth optional).
    Every frame is rectified onto one common drawer grid (markers give the rectangle); heights are fused
    (median) and each tool is outlined in the frame whose camera was closest to it, so perspective
    displacement stays small even on long drawers."""
    mf = request.files.get("manifest")
    if mf is None:
        raise ApiError('Send a "manifest" JSON plus the frames it names')
    try:
        frames_meta = json.loads(mf.read().decode("utf-8"))["frames"]
        assert isinstance(frames_meta, list) and frames_meta
    except Exception as exc:  # noqa: BLE001
        raise ApiError(f"Bad manifest: {exc}")
    blobs = {name: fs.read() for name, fs in request.files.items() if name != "manifest"}
    form = {k: v for k, v in request.form.items()}
    s = _build_multi_session(frames_meta, form, blobs)
    info = s.info()
    info["model"] = SEGMENTER.info()
    info["tools"] = []
    _save_capture(s.id, frames_meta, form, blobs, info)
    return jsonify(info), 201


def _build_multi_session(frames_meta: List[Dict], form: Dict, blobs: Dict[str, bytes], session_id: Optional[str] = None):
    """The multi-frame fusion itself (see create_multi_capture); also used to rebuild a saved capture from disk."""
    marker_size = float(form.get("marker_size_mm") or 50.0)
    inset = float(form.get("inset_mm") or 0.0)
    marker_dict = form.get("marker_dict") or "4X4_50"
    corner_map = _corner_map_for_frames(frames_meta, blobs, marker_dict, marker_size)
    reordered = corner_map is not None and any(i != s for i, s in corner_map.items())
    if reordered:
        log.info("multi-still: marker ids re-ordered from their positions — %s",
                 ", ".join(f"id {i} is the {CORNER_NAMES[s]} corner" for i, s in sorted(corner_map.items())))
    results = []
    poses = []
    depth_matches = []
    skipped = []
    # "truedepth" (AVFoundation) and "truedepth_tracked" (ARKit, may carry a pose) are the SAME SENSOR: the same
    # sparse map with the same holes. Every depth-handling decision below keys on the sensor, not on the pose.
    def is_truedepth(f):
        return str(f.get("sensor") or "").startswith("truedepth")

    for i, f in enumerate(frames_meta):
        img, depth, K = _parse_frame_upload(f, blobs)
        markers = capmod.corner_markers(capmod.detect_markers(img, marker_dict))
        pose = _pose_cv(f)
        try:
            if depth is not None:
                res = capmod.rectify_rgbd(img, depth, K, markers, marker_size, corner_map=corner_map,
                                          preserve_unknown=is_truedepth(f))
            else:
                if not markers:
                    skipped.append((i, "no markers")); continue
                res = capmod.rectify_markers_only(img, markers, marker_size, corner_map=corner_map)
        except Exception as exc:  # noqa: BLE001
            skipped.append((i, str(exc))); continue
        # A frame may be here for its DEPTH, its COLOUR, or both. Two-pass capture (Nolan, 2026-09-21): sweep
        # the drawer with the front TrueDepth camera for topography, then shoot sharp stills with the rear
        # camera, triggered by hand so nothing is taken mid-move. Both passes still register off the markers,
        # so they land on the same grid without needing a pose chain between them.
        res.meta["use"] = str(f.get("use") or "both").lower()
        res.meta["has_depth"] = depth is not None
        results.append(res)
        poses.append(pose)
        if is_truedepth(f) and depth is not None and res.geometry is not None:
            from toolcutter.rgbd_registration import depth_features
            depth_matches.append(depth_features(img, depth, K, res.geometry))
        else:
            depth_matches.append(None)
    # DEPTH SCALE CALIBRATION AGAINST THE MARKERS (2026-09-23, from Nolan's calipers). A depth frame gets its
    # metric scale from the depth itself, and the front TrueDepth depth is not quite metric: on capture
    # b59fc5fdd2d1 the 50 mm markers measured 52.11 mm in the depth-derived rasters (+4.2 %), and the black tape
    # measure came out 93.3 mm against a caliper 87.5 — 3.7 mm of that 5.8 mm was this. The markers are a known
    # length in EVERY frame that sees one, so measure them there and rescale every depth frame (mm/px and heights,
    # which are depth units too) by the one robust factor. Photogrammetry already does the same from marker
    # spacing (_session_from_mesh_file). TC_DEPTH_CAL=0 disables; the factor is reported as scan_meta.depth_scale.
    depth_scale = 1.0
    if os.environ.get("TC_DEPTH_CAL", "0") == "1":   # OFF by default: marker-placed frames are already metric in-plane; see CLAUDE.md
        meas = []
        for res in results:
            if not res.meta.get("has_depth") or not res.markers_px:
                continue
            for c in res.markers_px.values():
                c = np.asarray(c, dtype=np.float64).reshape(-1, 2)
                if len(c) == 4:
                    meas.append(float(np.mean([np.linalg.norm(c[j] - c[(j + 1) % 4]) for j in range(4)])) * res.mm_per_px)
        if len(meas) >= 3:
            factor = marker_size / float(np.median(meas))
            if 0.90 <= factor <= 1.10 and abs(factor - 1.0) >= 0.004:
                depth_scale = factor
                for res in results:
                    if res.meta.get("has_depth"):
                        res.mm_per_px *= factor
                        if res.height_mm is not None:
                            res.height_mm = res.height_mm * np.float32(factor)
                        if res.geometry is not None:
                            res.geometry.mm_per_px *= factor
                log.info("multi-still: depth scale calibrated against %d marker sightings: %.2f mm read for a %.1f mm "
                         "marker -> x%.4f", len(meas), float(np.median(meas)), marker_size, factor)
            elif not (0.90 <= factor <= 1.10):
                log.warning("multi-still: markers read %.1f mm for a declared %.1f mm size — not calibrating (check the print and the marker size setting)",
                            float(np.median(meas)), marker_size)
    visual_corners = {}
    registration_info = None
    # TrueDepth supplies synchronized metric depth, but no ARKit world pose.
    # Place overlapping floor rasters jointly before completing drawer corners.
    depth_indices = {i for i, features in enumerate(depth_matches) if features is not None}
    if len(depth_indices) > 1:
        from toolcutter.rgbd_registration import register_depth_subset
        visual_corners, inferred_map, measured_size, registration_info = register_depth_subset(
            results, depth_matches, infer_order=corner_map is None)
        if visual_corners:
            if inferred_map:
                corner_map = inferred_map
                reordered = any(i != slot for i, slot in corner_map.items())
                for res in results:
                    res.markers_px = capmod.relabel_markers(res.markers_px, corner_map)
                    res.markers_found = sorted(res.markers_px)
            form = dict(form)
            if not form.get("drawer_width_mm"):
                form["drawer_width_mm"] = str(measured_size[0] + 2 * inset)
            if not form.get("drawer_height_mm"):
                form["drawer_height_mm"] = str(measured_size[1] + 2 * inset)
            log.info("TrueDepth: registered %d/%d frames through %d depth-corrected overlaps", len(visual_corners), len(results), registration_info["matched_pairs"])
        registration_info["applied"] = bool(visual_corners)
        registration_info["frames_registered"] = len(visual_corners)
        if not visual_corners and not any(r.drawer_corners is not None for r in results):
            raise ApiError("The TrueDepth views could not be aligned reliably. Capture a steady overview with three corner markers, "
                           "then pause at each closer view with at least half of the previous view still visible. Use bright, even lighting.")
    full = [r for r in results if r.drawer_corners is not None]
    user_size = bool(form.get("drawer_width_mm") and form.get("drawer_height_mm"))   # typed by the user: trusted
    drawer_from = "rgbd_overlap" if visual_corners else ("markers" if full else ("form" if user_size else "pairs"))

    if not full and any(p is not None for p in poses):
        # No frame sees the whole drawer (a close glide over a long drawer). Triangulate: put every marker
        # corner observed in any frame into ONE frame's plane coordinates via the camera poses, take the
        # median per marker id, and complete the rectangle there.
        anchor_i = next((i for i, (r, p) in enumerate(zip(results, poses)) if p is not None and r.geometry is not None and r.markers_px), None)
        if anchor_i is not None:
            a_res, a_pose = results[anchor_i], poses[anchor_i]
            a_inv = np.linalg.inv(a_pose)
            obs: Dict[int, List[np.ndarray]] = {}
            for res, pose in zip(results, poses):
                if pose is None or res.geometry is None:
                    continue
                for mid, mpx in res.markers_px.items():
                    corner_px = capmod.outer_corner(mpx, mid)
                    uv = res.geometry.raster_to_plane(corner_px.reshape(1, 2))
                    cam = res.geometry.plane_to_cam(uv, 0.0)
                    world = (pose[:3, :3] @ cam.T).T + pose[:3, 3]
                    a_cam = (a_inv[:3, :3] @ world.T).T + a_inv[:3, 3]
                    a_uv, _ = a_res.geometry.cam_to_plane(a_cam)
                    obs.setdefault(mid, []).append(a_uv[0])
            pts = {mid: np.median(np.stack(v), axis=0) for mid, v in obs.items()}
            if len(pts) >= 3:
                if len(pts) == 3:
                    missing = [i for i in capmod.CORNER_IDS if i not in pts][0]
                    opp, a, b = (missing + 2) % 4, (missing + 1) % 4, (missing + 3) % 4
                    pts[missing] = pts[a] + pts[b] - pts[opp]
                rect_uv = np.asarray([pts[i] for i in capmod.CORNER_IDS])
                # the poses are only good for a few mm, so use the triangulation for the drawer SIZE alone;
                # every frame is then placed by its own markers (exact) plus that size
                tri_w = float(np.mean([np.linalg.norm(rect_uv[1] - rect_uv[0]), np.linalg.norm(rect_uv[2] - rect_uv[3])])) + 2 * inset
                tri_h = float(np.mean([np.linalg.norm(rect_uv[3] - rect_uv[0]), np.linalg.norm(rect_uv[2] - rect_uv[1])])) + 2 * inset
                form = dict(form)
                form.setdefault("drawer_width_mm", str(round(tri_w, 1)))
                form.setdefault("drawer_height_mm", str(round(tri_h, 1)))
                drawer_from = "triangulated"
                log.info("multi-still: drawer size %.1f x %.1f triangulated from %d marker observations across frames", tri_w, tri_h, sum(len(v) for v in obs.values()))

    def _chain_unanchored(results, width_mm: float, height_mm: float, inset_mm: float) -> int:
        """Place frames that see NO markers and carry no pose, by matching them to a neighbour that IS placed.

        A TrueDepth sweep has neither: AVFoundation gives no world pose, and at 20-50 cm the camera's view is far
        too narrow to keep a corner marker in shot — on Nolan's first two-pass capture 35 of 44 depth frames saw
        no marker at all and were dropped, which is why the bottom half of the drawer was missing. Each frame IS
        individually rectified to a metric top-down raster, though, so consecutive frames differ only by a rigid
        move in the plane. Match features between a placed frame and an unplaced one, recover that move, and carry
        the drawer rectangle across. Anchors spread outward from the frames the markers did place.
        """
        def corners_of(res):
            if res.drawer_corners is not None:
                return calibration.order_corners(res.drawer_corners)
            if not res.markers_px:
                return None
            wpx, hpx = (width_mm - 2 * inset_mm) / res.mm_per_px, (height_mm - 2 * inset_mm) / res.mm_per_px
            c = capmod.rectangle_from_two_markers(res.markers_px, wpx, hpx) if len(res.markers_px) >= 2 else None
            if c is None:
                c = capmod.rectangle_from_one_marker(res.markers_px, wpx, hpx)
            return c

        placed = [corners_of(r) for r in results]
        if not any(c is not None for c in placed):
            return 0
        MM = 1.0                                     # match at a common 1 mm/px so scale is not a free parameter
        orb = cv2.ORB_create(4000)
        cache: Dict[int, tuple] = {}

        def feats(i):
            if i not in cache:
                r = results[i]
                g = cv2.cvtColor(r.color_bgr, cv2.COLOR_BGR2GRAY)
                sc = r.mm_per_px / MM
                g = cv2.resize(g, (max(8, int(g.shape[1] * sc)), max(8, int(g.shape[0] * sc))))
                cache[i] = (orb.detectAndCompute(g, None), sc)
            return cache[i]

        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        debug = os.environ.get("TC_CHAIN_DEBUG")          # why a frame refused to match: features/matches/inliers/scale
        added = 0
        for _ in range(len(results)):
            progress = False
            for i, c in enumerate(placed):
                if c is not None:
                    continue
                # nearest already-placed frame in time
                order = sorted((j for j, d in enumerate(placed) if d is not None), key=lambda j: abs(j - i))
                for j in order[:3]:
                    ((k1, d1), s1), ((k2, d2), s2) = feats(i), feats(j)
                    if d1 is None or d2 is None or len(k1) < 12 or len(k2) < 12:
                        if debug:
                            log.info("chain %d<-%d: too few features (%d, %d)", i, j, len(k1), len(k2))
                        continue
                    good = [m for m, n in bf.knnMatch(d1, d2, k=2) if m.distance < 0.75 * n.distance]
                    if len(good) < 18:
                        if debug:
                            log.info("chain %d<-%d: %d good matches of %d/%d features", i, j, len(good), len(k1), len(k2))
                        continue
                    src = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                    dst = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                    M, inl = cv2.estimateAffinePartial2D(dst, src, method=cv2.RANSAC, ransacReprojThreshold=3.0)
                    if M is None or inl is None or int(inl.sum()) < 15:
                        if debug:
                            log.info("chain %d<-%d: %d good matches but only %s inliers", i, j, len(good),
                                     "no model" if inl is None else int(inl.sum()))
                        continue
                    if abs(math.hypot(M[0, 0], M[1, 0]) - 1.0) > 0.06:      # both rasters are metric: scale must be ~1
                        if debug:
                            log.info("chain %d<-%d: scale %.3f off 1", i, j, math.hypot(M[0, 0], M[1, 0]))
                        continue
                    pts = (np.asarray(placed[j], dtype=np.float32) * s2).reshape(-1, 1, 2)
                    moved = cv2.transform(pts, M).reshape(-1, 2) / s1
                    placed[i] = moved
                    results[i].drawer_corners = moved
                    added += 1
                    progress = True
                    break
            if not progress:
                break
        if added:
            log.info("multi-still: %d frame(s) placed by matching a neighbour (no markers, no pose)", added)
        return added

    def _edge_lengths_from_pairs(res):
        """(width_mm or None, height_mm or None) from adjacent marker pairs visible in one frame."""
        mp = res.markers_px
        if len(mp) < 2:
            return None, None
        outer = {i: capmod.outer_corner(c, i) for i, c in mp.items()}
        w = h = None
        for a, b in ((0, 1), (3, 2)):
            if a in outer and b in outer:
                w = float(np.linalg.norm(outer[b] - outer[a])) * res.mm_per_px + 2 * inset
        for a, b in ((0, 3), (1, 2)):
            if a in outer and b in outer:
                h = float(np.linalg.norm(outer[b] - outer[a])) * res.mm_per_px + 2 * inset
        return w, h

    if not full and not (form.get("drawer_width_mm") and form.get("drawer_height_mm")):
        # long drawers shot up close: one frame sees the left pair, another the right pair, a third the top
        # pair. Each adjacent pair measures one edge exactly.
        ws, hs = [], []
        for res in results:
            w, h = _edge_lengths_from_pairs(res)
            if w: ws.append(w)
            if h: hs.append(h)
        if form.get("drawer_width_mm"): ws = [float(form["drawer_width_mm"])]
        if form.get("drawer_height_mm"): hs = [float(form["drawer_height_mm"])]
        if not ws or not hs:
            raise ApiError("Could not measure the drawer: no frame shows 3 markers, and the visible pairs do not cover both "
                           "a width edge and a depth edge. Add a frame that sees the top or bottom pair, or type the drawer size. "
                           + "; ".join(f"frame {i}: {why}" for i, why in skipped))
        form = dict(form)
        form["drawer_width_mm"] = str(np.median(ws))
        form["drawer_height_mm"] = str(np.median(hs))

    def _inset(c, mm_per_px):
        return capmod.expand_rectangle(c, inset / mm_per_px) if inset else c

    # pass 1: drawer size from frames that see >= 3 markers (or the form)
    sizes = []
    for res in full:
        c = _inset(calibration.order_corners(res.drawer_corners), res.mm_per_px)
        hpx, vpx = calibration.edge_lengths_px(c)
        sizes.append((hpx * res.mm_per_px, vpx * res.mm_per_px))
    width_mm = round(float(form.get("drawer_width_mm") or np.median([w for w, _ in sizes])), 1)
    height_mm = round(float(form.get("drawer_height_mm") or np.median([h for _, h in sizes])), 1)
    # Frames with no markers ride in on a neighbour. A POSE ONLY HELPS IF SOMETHING ANCHORS IT: placing a
    # pose-only frame needs a frame that has markers AND depth geometry AND a pose, to tie that pose's world
    # origin to the drawer. The two-pass capture has no such frame by construction — the front depth sweep is
    # too close (20-50 cm) to see a marker, and the rear photos that do see markers carry no depth, hence no
    # geometry. Worse, the two passes are separate ARSessions with unrelated origins, so their poses could not
    # be chained even if they were all present (Nolan's capture 136c77400c3f: 102 posed depth frames, all
    # dropped as "no camera pose on any marker-registered frame to anchor to", leaving an empty scan).
    # Appearance matching is origin-free, so fall back to it whenever the poses cannot be anchored.
    pose_anchor = any(p is not None and r.geometry is not None and (r.drawer_corners is not None or r.markers_px)
                      for r, p in zip(results, poses))
    chained = 0
    if not visual_corners and any(r.drawer_corners is None and not r.markers_px and (p is None or not pose_anchor)
           for r, p in zip(results, poses)):
        if not pose_anchor and any(p is not None for p in poses):
            log.info("multi-still: poses present but nothing anchors them (no frame has markers + depth); "
                     "placing marker-less frames by matching instead")
        chained = _chain_unanchored(results, width_mm, height_mm, inset)

    def _register(width_mm: float, height_mm: float):
        """Pass 2: place every frame on the common drawer grid. Frames with two adjacent markers (one end of a
        long drawer) get the rectangle completed from the drawer size; frames with fewer markers but an ARKit
        pose are registered through the pose relative to an anchor frame. Then every frame that is not fixed by
        3+ markers is refined against the frames already placed (height-map phase correlation, temporal order),
        and the drawer size is re-measured from where the markers landed."""
        corners_list = []
        kept = []
        anchor = None      # (CaptureResult, corners px, pose) of the first fully-marked frame with a pose
        pending = []
        for i, (res, pose) in enumerate(zip(results, poses)):
            c = None
            if visual_corners and i in depth_indices:
                if i not in visual_corners:
                    skipped.append((i, "no reliable depth-corrected overlap with the drawer")); continue
                c = _inset(visual_corners[i], res.mm_per_px)
            elif res.drawer_corners is not None:
                c = _inset(calibration.order_corners(res.drawer_corners), res.mm_per_px)
            elif len(res.markers_px) >= 1:
                wpx, hpx = (width_mm - 2 * inset) / res.mm_per_px, (height_mm - 2 * inset) / res.mm_per_px
                c2 = capmod.rectangle_from_two_markers(res.markers_px, wpx, hpx) if len(res.markers_px) >= 2 else None
                if c2 is None:
                    # one marker: exact position, axes from the marker's edges (~0.5 deg; the pose chain was
                    # tried for the axes and was worse). Such frames rank below two-marker frames in the fusion.
                    c2 = capmod.rectangle_from_one_marker(res.markers_px, wpx, hpx)
                if c2 is not None:
                    c = _inset(c2, res.mm_per_px)
            if c is None:
                if pose is not None and res.geometry is not None:
                    pending.append((i, res, pose)); continue
                skipped.append((i, f"only {len(res.markers_found)} markers and no camera pose")); continue
            corners_list.append(c)
            kept.append(res)
            if anchor is None and pose is not None and res.geometry is not None and len(res.markers_px) >= 2:
                anchor = (res, c, pose)
        if anchor is None:
            for res, c in zip(kept, corners_list):
                pose = poses[[id(r) for r in results].index(id(res))]
                if pose is not None and res.geometry is not None:
                    anchor = (res, c, pose); break
        if pending:
            if anchor is None:
                for i, *_ in pending:
                    skipped.append((i, "no camera pose on any marker-registered frame to anchor to"))
            else:
                a_res, a_c, a_pose = anchor
                # drawer corners as 3D points: anchor raster px -> plane mm -> anchor camera -> world
                a_uv = a_res.geometry.raster_to_plane(np.asarray(a_c, dtype=np.float64))
                a_cam = a_res.geometry.plane_to_cam(a_uv, 0.0)
                world = (a_pose[:3, :3] @ a_cam.T).T + a_pose[:3, 3]
                for i, res, pose in pending:
                    inv = np.linalg.inv(pose)
                    cam = (inv[:3, :3] @ world.T).T + inv[:3, 3]
                    # project onto this frame's floor plane (drop residual height from pose drift) and into its raster
                    uv, h_res = res.geometry.cam_to_plane(cam)
                    if np.abs(h_res).max() > 25.0:
                        skipped.append((i, f"pose inconsistent with the floor plane ({np.abs(h_res).max():.0f} mm)")); continue
                    corners_list.append(res.geometry.plane_to_raster(uv))
                    kept.append(res)
        if not kept:
            raise ApiError("No usable frames: " + "; ".join(f"frame {i}: {why}" for i, why in skipped))
        # keep temporal order (pending frames were appended after the marker frames)
        pos = {id(r): i for i, r in enumerate(results)}
        order = sorted(range(len(kept)), key=lambda k: pos[id(kept[k])])
        kept = [kept[k] for k in order]
        corners_list = [corners_list[k] for k in order]
        ppm = min(2200.0 / max(width_mm, height_mm), 1.0 / min(r.mm_per_px for r in kept) * 1.15)
        # fixed: frames placed by 3+ markers, and two-marker frames sharing a marker with them (same end of
        # the drawer: their placement is as good as the markers). Pose-only frames and two-marker frames at
        # the OTHER end (placed through the drawer size) are refined against the fixed ones.
        anchor_ids = set()
        for res in kept:
            if res.drawer_corners is not None:
                anchor_ids |= set(res.markers_px)
        if not anchor_ids:
            first = next((res for res in kept if len(res.markers_px) >= 2), None)
            if first is not None:
                anchor_ids = set(first.markers_px)
        fixed = [res.drawer_corners is not None or (len(res.markers_px) >= 2 and bool(set(res.markers_px) & anchor_ids)) for res in kept]
        if visual_corners:
            registered_ids = {id(results[i]) for i in visual_corners}
            fixed = [f or id(r) in registered_ids for f, r in zip(fixed, kept)]
        # marker-placed but not fixed: one-marker frames (axes from a 50 mm edge) and frames at the other end
        # (placed through the drawer size). Only the latter measure the size error.
        far_end = [not f and len(res.markers_px) >= 1 for f, res in zip(fixed, kept)]
        other_end = [fe and not (set(res.markers_px) & anchor_ids) for fe, res in zip(far_end, kept)]
        frames = _rectify_frames(kept, corners_list, width_mm, height_mm, ppm)
        # Correct small rotation/translation drift from floor texture, using only
        # directly marker-anchored references. Never re-scale the drawer or chain
        # uncertain frames into new reference frames.
        visually_aligned = set()
        for i, frame in enumerate(frames):
            if fixed[i] or len(kept[i].markers_px) >= 2:
                continue
            candidates = sorted((j for j in range(len(frames)) if fixed[j]),
                                key=lambda j: -int((frame["valid"] & frames[j]["valid"]).sum()))[:2]
            for j in candidates:
                correction = floor_alignment(frame, frames[j], 1.0 / ppm)
                if correction is None:
                    continue
                Hnew = correction @ frame["H"]
                hh, ww = frame["valid"].shape
                grid = np.array([[0, 0], [ww, 0], [ww, hh], [0, hh]], np.float64)
                corners_list[i] = cv2.perspectiveTransform(grid[None], np.linalg.inv(Hnew))[0]
                visually_aligned.add(i)
                break
        if visually_aligned:
            frames = _rectify_frames(kept, corners_list, width_mm, height_mm, ppm)
            log.info("multi-still: visually aligned %d frames to marker-anchored floor texture", len(visually_aligned))
        n_moved = 0
        far_shifts = []
        if not all(fixed):
            order = [i for i in range(len(kept)) if not fixed[i] and not far_end[i]] + [i for i in range(len(kept)) if far_end[i]]
            # Color-only views were handled by floor texture alignment above.
            # They cannot refine height placement or measure a height-based size correction.
            order = [i for i in order if i not in visually_aligned and frames[i]["height"] is not None]
            shifts = _refine_frame_alignment(frames, fixed, order, 1.0 / ppm, max_shift_mm=20.0)
            Wg, Hg = frames[0]["color"].shape[1], frames[0]["color"].shape[0]
            # Height-map correlation across viewpoints carries a ~1-2 mm bias (each frame sees a tool's far side
            # smeared away from the camera), so frames placed by their markers are never moved by it. Frames
            # without a marker (pose only) are, but only by a credible amount: pose drift over one glide is a
            # few mm (every genuine correction measured on the synthetic glides is under 5.5 mm), whereas phase
            # correlation on a REPEATING subject locks a whole period away and reports it confidently — a real
            # scan of a keyboard produced four "corrections" of 14-19 mm, one key pitch, which tore the mosaic
            # into offset strips. Their residual gain was 1.7-2.1, better than the genuine 1.02-1.08 ones, so
            # only the size gives them away. Frames at the other end still MEASURE the drawer-size error over
            # the full 20 mm window; that is accepted only when they agree with each other.
            for i, (dx, dy, resp) in enumerate(shifts):
                if i in visually_aligned:
                    continue
                if resp <= 0 or abs(dx) + abs(dy) <= 0.25:
                    continue
                if other_end[i]:
                    far_shifts.append((dx / ppm, dy / ppm))
                if far_end[i]:
                    continue
                if math.hypot(dx, dy) / ppm > APPLY_SHIFT_MAX_MM:
                    log.info("multi-still: frame %d not moved, %.1f mm is too far to be drift", i, math.hypot(dx, dy) / ppm)
                    continue
                corners_list[i] = _shift_source_corners(frames[i], Wg, Hg, dx, dy)
                n_moved += 1
            if n_moved:
                frames = _rectify_frames(kept, corners_list, width_mm, height_mm, ppm)
                log.info("multi-still: refined %d frames by height/texture alignment", n_moved)
        # drawer size as the markers now land on the grid (outer corners, per id median)
        landed: Dict[int, List[np.ndarray]] = {}
        for res, f in zip(kept, frames):
            for mid, mpx in res.markers_px.items():
                oc = capmod.outer_corner(mpx, mid).reshape(1, 1, 2)
                landed.setdefault(mid, []).append(cv2.perspectiveTransform(oc, f["H"]).reshape(2))
        pts = {mid: np.median(np.stack(v), axis=0) for mid, v in landed.items()}
        est_w = est_h = None
        ws = [pts[b][0] - pts[a][0] for a, b in ((0, 1), (3, 2)) if a in pts and b in pts]
        hs = [pts[b][1] - pts[a][1] for a, b in ((0, 3), (1, 2)) if a in pts and b in pts]
        # only trust the re-measured size when the far-end frames agree on the shift
        consistent = bool(far_shifts) and (len(far_shifts) == 1 or np.ptp(np.asarray(far_shifts), axis=0).max() < 1.5)
        if consistent:
            mean_shift = np.mean(np.asarray(far_shifts), axis=0)
            # the far-end markers land where they were placed (unshifted): the size error is the shift itself
            if ws and abs(mean_shift[0]) > 3.0:
                est_w = float(np.mean(ws)) / ppm + 2 * inset + float(mean_shift[0]) * (1 if any(1 in res.markers_px or 2 in res.markers_px for res, oe in zip(kept, other_end) if oe) else -1)
            if hs and abs(mean_shift[1]) > 3.0:
                est_h = float(np.mean(hs)) / ppm + 2 * inset + float(mean_shift[1]) * (1 if any(2 in res.markers_px or 3 in res.markers_px for res, oe in zip(kept, other_end) if oe) else -1)
        for res, f in zip(kept, frames):
            f["rank"] = 0 if (visual_corners or res.drawer_corners is not None or len(res.markers_px) >= 2) else (1 if res.markers_px else 2)
        if visual_corners:
            # A blurred close view should not displace a sharper, slightly more
            # oblique view of the same tool merely because its camera is nearer.
            sharpness = []
            for res in kept:
                gray = cv2.cvtColor(res.color_bgr, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, None, fx=640 / max(gray.shape), fy=640 / max(gray.shape), interpolation=cv2.INTER_AREA)
                sharpness.append(float(cv2.Laplacian(gray, cv2.CV_32F).var()))
            reference = max(float(np.percentile(sharpness, 80)), 1.)
            for f, value in zip(frames, sharpness):
                f["quality_penalty_px"] = 60 * ppm * max(0., 1. - value / reference)
        return kept, corners_list, frames, ppm, fixed, est_w, est_h

    results_kept, corners_list, frames, ppm, anchored, est_w, est_h = _register(width_mm, height_mm)
    if not user_size and not all(anchored):
        new_w = round(est_w, 1) if est_w and abs(est_w - width_mm) < 40 else width_mm
        new_h = round(est_h, 1) if est_h and abs(est_h - height_mm) < 40 else height_mm
        if abs(new_w - width_mm) > 0.3 or abs(new_h - height_mm) > 0.3:
            log.info("multi-still: drawer size re-measured from aligned frames %.1f x %.1f -> %.1f x %.1f", width_mm, height_mm, new_w, new_h)
            width_mm, height_mm = new_w, new_h
            drawer_from += "+aligned"
            results_kept, corners_list, frames, ppm, anchored, _, _ = _register(width_mm, height_mm)
    results = results_kept
    heights = [f["height"] for f in frames if f["height"] is not None]
    from toolcutter.depth_fusion import fuse_heights
    fused = fuse_heights(heights)
    # Fusion can leave a residual floor offset/tilt even when each camera's plane
    # fit passed. Use the same floor reference as single captures; otherwise a
    # few millimetres of raised floor become bridges between unrelated tools.
    floor_shift = 0.0
    if fused is not None:
        levelled, shift = geometry.level_height_raster(fused)
        if shift > 1.0:
            fused = levelled
            floor_shift = shift
    # display colour: each pixel from the nearest-foot-point frame among those that saw it (least perspective)
    Hh, W = frames[0]["color"].shape[:2]
    yy, xx = np.mgrid[0:Hh, 0:W]
    # exactly placed frames win the mosaic wherever they cover (rank penalty far beyond any real distance)
    paint = [f for f in frames if f.get("use", "both") != "depth"] or frames
    dists = np.stack([np.where(f["valid"], np.hypot(xx - f["nadir_px"][0], yy - f["nadir_px"][1]) + f["rank"] * RANK_PENALTY_MM * ppm, np.inf) for f in paint])
    best = np.argmin(dists, axis=0)
    mosaic = _blend_mosaic(paint, best, 1.0 / ppm, fused)
    s = STORE.create(session_id=session_id, source_kind="capture", filename=form.get("filename") or "multi_capture.jpg", original=mosaic,
                     original_height=fused, original_mm_per_px=1.0 / ppm,
                     scan_meta={"mode": "multi_still", "frames_used": len(frames), "frames_skipped": skipped,
                                "floor_relevel_mm": round(floor_shift, 1),
                                "rgbd_registration": registration_info,
                                "sensors": sorted({str(f.get("sensor", "unknown")) for f in frames_meta}),
                                "depth_scale": round(depth_scale, 4),
                                "drawer_from": drawer_from,
                                # how each frame earned its place, so a new capture path can be diagnosed from the result
                                "photo_coverage": coverage_report(paint, fused.shape if fused is not None else mosaic.shape[:2]),
                                "placed_by": {"markers_2plus": sum(1 for f in frames if f["rank"] == 0) - chained,
                                              "one_marker": sum(1 for f in frames if f["rank"] == 1),
                                              "pose_only": sum(1 for f in frames if f["rank"] == 2),
                                              "matched_to_neighbour": chained},
                                "sensors": sorted({str(f.get("sensor") or "?") for f in frames_meta}),
                                "marker_corners": None if not corner_map else {str(i): CORNER_NAMES[s_] for i, s_ in sorted(corner_map.items())},
                                "markers_reordered": reordered,
                                "has_depth": fused is not None, "has_height": fused is not None,
                                "camera_heights_mm": [f["camera_height_mm"] for f in frames],
                                "drawer_sizes_mm": [[round(w, 1), round(h, 1)] for w, h in sizes]})
    s.frames = frames
    s.rectified = mosaic
    s.rect_height = fused
    s.rect_frac = None
    s.mm_per_px = 1.0 / ppm
    s.mat_mm = (width_mm, height_mm)
    # the session's "original" IS the fused, already-rectified mosaic: its calibration corners are the mosaic's own
    # rectangle (frame-0 source-raster corners here would draw the handles far off the image and, if re-applied,
    # re-crop the drawer to nonsense)
    Hm_px, Wm_px = mosaic.shape[:2]
    s.corners = [[0.0, 0.0], [float(Wm_px), 0.0], [float(Wm_px), float(Hm_px)], [0.0, float(Hm_px)]]
    s.suggested_corners = s.corners
    s.original_mm_per_px = 1.0 / ppm
    s.homography = None
    s.version = 1
    s.auto_calibrated = True
    return s


@app.get("/api/marker_sheet.svg")
def marker_sheet():
    """Printable corner markers (ids 0-3) at true scale, with a 100 mm check bar. Print at 100%."""
    size = float(request.args.get("marker_mm") or 50.0)
    page_w, page_h = 215.9, 279.4  # US Letter
    d = cv2.aruco.getPredefinedDictionary(capmod.ARUCO_DICTS.get(request.args.get("dict") or "4X4_50", cv2.aruco.DICT_4X4_50))
    labels = ["TL  (id 0)", "TR  (id 1)", "BR  (id 2)", "BL  (id 3)"]
    quiet = size * 0.25
    cell_pitch = size + 2 * quiet
    cols = 2
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{page_w}mm" height="{page_h}mm" viewBox="0 0 {page_w} {page_h}" font-family="Helvetica, Arial, sans-serif">',
           f'<text x="{page_w/2}" y="14" font-size="6" text-anchor="middle" font-weight="bold">ToolCutter drawer markers · {size:g} mm · print at 100% scale</text>',
           f'<text x="{page_w/2}" y="21" font-size="3.4" text-anchor="middle">Cut each square on the dashed line and put one in each corner of the drawer, flat and upright, with the marker\'s outer corner touching the drawer corner. Any corner will do: which is which is read off their positions, not their numbers.</text>']
    x0 = (page_w - cols * cell_pitch - 10) / 2
    y0 = 30.0
    for i in range(4):
        cx = x0 + (i % cols) * (cell_pitch + 10)
        cy = y0 + (i // cols) * (cell_pitch + 14)
        bits = cv2.aruco.generateImageMarker(d, i, 6)  # 6x6 incl. black border
        n = bits.shape[0]
        cell = size / n
        out.append(f'<rect x="{cx}" y="{cy}" width="{cell_pitch}" height="{cell_pitch}" fill="#fff" stroke="#999" stroke-width="0.25" stroke-dasharray="2 1.5"/>')
        for r in range(n):
            for c in range(n):
                if bits[r, c] < 128:
                    out.append(f'<rect x="{cx + quiet + c*cell:.3f}" y="{cy + quiet + r*cell:.3f}" width="{cell:.3f}" height="{cell:.3f}" fill="#000"/>')
        out.append(f'<text x="{cx + cell_pitch/2}" y="{cy + cell_pitch + 5}" font-size="4" text-anchor="middle">{labels[i]}</text>')
    yb = y0 + 2 * (cell_pitch + 14) + 6
    out.append(f'<line x1="{page_w/2-50}" y1="{yb}" x2="{page_w/2+50}" y2="{yb}" stroke="#000" stroke-width="0.4"/>')
    for xm in (page_w/2-50, page_w/2+50):
        out.append(f'<line x1="{xm}" y1="{yb-3}" x2="{xm}" y2="{yb+3}" stroke="#000" stroke-width="0.4"/>')
    out.append(f'<text x="{page_w/2}" y="{yb+7}" font-size="3.4" text-anchor="middle">this bar must measure exactly 100 mm — if not, fix the print scale</text>')
    out.append("</svg>")
    resp = Response("\n".join(out), mimetype="image/svg+xml")
    resp.headers["Content-Disposition"] = 'inline; filename="toolcutter_markers.svg"'
    return resp


@app.get("/api/sessions")
def list_sessions():
    """Recent uploads: sessions in memory plus phone captures saved on disk (rebuilt on demand)."""
    out: Dict[str, Dict] = {}
    if CAPTURE_DIR.exists():
        for d in CAPTURE_DIR.iterdir():
            try:
                meta = json.loads((d / "capture.json").read_text())
                out[meta["id"]] = {"id": meta["id"], "created": meta.get("created"), "source_kind": "capture", "filename": meta.get("filename"),
                                   "mat_mm": meta.get("mat_mm"), "frames": meta.get("frames_used") or len(meta.get("frames") or []),
                                   "in_memory": False, "saved": True}
            except Exception:  # noqa: BLE001
                continue
    for s in STORE.items():
        e = out.setdefault(s.id, {"id": s.id, "created": getattr(s, "created", None), "source_kind": s.source_kind, "filename": s.filename,
                                  "mat_mm": ({"width": s.mat_mm[0], "height": s.mat_mm[1]} if s.mat_mm else None),
                                  "frames": len(s.frames) if s.frames else None, "saved": False})
        e["in_memory"] = True
    items = sorted(out.values(), key=lambda e: e.get("created") or 0, reverse=True)
    return jsonify({"sessions": items[:30]})


@app.get("/api/sessions/<sid>")
def get_session(sid: str):
    s = _session(sid)
    info = s.info()
    info["model"] = SEGMENTER.info()
    return jsonify(info)


@app.delete("/api/sessions/<sid>")
def delete_session(sid: str):
    """Forget a session and delete its saved raw frames. Irreversible: the frames are the only copy of a
    phone scan once it is off the phone."""
    if not re.fullmatch(r"[0-9a-f]{12}", sid or ""):
        raise ApiError("Not a session id")
    dropped = STORE.drop(sid)
    d = CAPTURE_DIR / sid
    removed_frames = 0
    with _capture_lock(sid), _REBUILD_LOCK:   # never delete under a rebuild that is reading the frames
        if d.is_dir() and d.parent == CAPTURE_DIR:
            removed_frames = len([p for p in d.iterdir() if p.suffix in (".jpg", ".f32")])
            shutil.rmtree(d)
    if not dropped and not removed_frames:
        raise ApiError(f"No session {sid}", 404)
    log.info("deleted session %s (in memory: %s, saved frames: %d)", sid, dropped, removed_frames)
    return jsonify({"id": sid, "deleted": True, "was_in_memory": dropped, "frames_removed": removed_frames})


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


def _heightfield_payload(h: np.ndarray, mm_per_px: float, step_mm: float, x0_mm: float = 0.0, y0_mm: float = 0.0) -> Dict[str, Any]:
    """Height raster (mm, floor = 0) resampled to a step_mm grid, as base64 little-endian float32 row-major."""
    import base64
    h = np.asarray(h, dtype=np.float32)
    cols = max(2, int(round(h.shape[1] * mm_per_px / step_mm)))
    rows = max(2, int(round(h.shape[0] * mm_per_px / step_mm)))
    observed = np.isfinite(h)
    weight = cv2.resize(observed.astype(np.float32), (cols, rows), interpolation=cv2.INTER_AREA)
    total = cv2.resize(np.where(observed, h, 0), (cols, rows), interpolation=cv2.INTER_AREA)
    valid = weight >= .5
    small = np.divide(total, weight, out=np.zeros_like(total), where=valid)
    return {"cols": cols, "rows": rows, "step_mm": float(h.shape[1] * mm_per_px / cols), "step_y_mm": float(h.shape[0] * mm_per_px / rows),
            "x0_mm": x0_mm, "y0_mm": y0_mm, "width_mm": float(h.shape[1] * mm_per_px), "height_mm": float(h.shape[0] * mm_per_px),
            "valid_b64": base64.b64encode(valid.astype(np.uint8).tobytes()).decode("ascii"),
            "max_mm": float(small.max()) if small.size else 0.0, "heights_b64": base64.b64encode(small.astype("<f4").tobytes()).decode("ascii")}


@app.get("/api/sessions/<sid>/heightfield")
def get_heightfield(sid: str):
    """The fused scan of the whole drawer as a coarse height grid (for the 3D scan view)."""
    s = _require_rectified(_session(sid))
    if s.rect_height is None:
        raise ApiError("This session has no height data.", 404)
    step = float(request.args.get("step_mm") or 2.0)
    step = min(max(step, 0.5), 20.0)
    return jsonify(_heightfield_payload(s.rect_height, s.mm_per_px, step))


@app.get("/api/sessions/<sid>/tools/<tid>/image")
def tool_image(sid: str, tid: str):
    s = _require_rectified(_session(sid))
    mask = s.masks.get(tid)
    if mask is None or not mask.any():
        raise ApiError("Detect this tool again to create its overhead image.", 404)
    index = s.tool_image_frames.get(tid)
    image = s.frames[index]["color"] if index is not None else s.rectified
    ys, xs = np.nonzero(mask)
    pad = max(2, int(round(8 / s.mm_per_px)))
    x0, y0 = max(0, xs.min()-pad), max(0, ys.min()-pad)
    x1, y1 = min(image.shape[1], xs.max()+pad+1), min(image.shape[0], ys.max()+pad+1)
    crop = image[y0:y1, x0:x1].copy()
    contours, _ = cv2.findContours(mask[y0:y1, x0:x1].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(crop, contours, -1, (30, 190, 255), max(1, round(0.4/s.mm_per_px)))
    crop, _ = downscale_to(crop, 1200)
    return Response(encode_jpeg(crop, 94), mimetype="image/jpeg", headers={"Cache-Control": "no-store"})


@app.get("/api/sessions/<sid>/tools/<tid>/heightfield")
def get_tool_heightfield(sid: str, tid: str):
    """One tool's scanned top surface (heights inside its mask, 0 outside), cropped, in mat-mm coordinates."""
    s = _require_rectified(_session(sid))
    if s.rect_height is None:
        raise ApiError("This session has no height data.", 404)
    mask = s.masks.get(tid)
    if mask is None or mask.shape != s.rect_height.shape or not mask.any():
        raise ApiError("No scanned mask for this tool.", 404)
    step = min(max(float(request.args.get("step_mm") or 1.0), 0.3), 10.0)
    ys, xs = np.nonzero(mask)
    pad = max(2, int(round(step / s.mm_per_px)))
    y0, y1 = max(0, ys.min() - pad), min(mask.shape[0], ys.max() + pad + 1)
    x0, x1 = max(0, xs.min() - pad), min(mask.shape[1], xs.max() + pad + 1)
    inside = mask[y0:y1, x0:x1]
    h = np.where(inside, s.rect_height[y0:y1, x0:x1], 0.0).astype(np.float32)
    # Preserve sensor gaps: a fabricated half-height plateau is not a scanned surface.
    payload = _heightfield_payload(h, s.mm_per_px, step, x0_mm=float(x0 * s.mm_per_px), y0_mm=float(y0 * s.mm_per_px))
    measured = h[inside & np.isfinite(h)]
    payload["median_mm"] = float(np.median(measured)) if measured.size else 0.0
    return jsonify(payload)


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
        width_mm = width_mm or round(hpx * s.original_mm_per_px, 1)
        height_mm = height_mm or round(vpx * s.original_mm_per_px, 1)
    if not width_mm or not height_mm or width_mm <= 0 or height_mm <= 0:
        raise ApiError("width_mm and height_mm (real size of the mat / drawer) are required")
    _apply_calibration(s, ordered, width_mm, height_mm)
    s.auto_calibrated = False
    info = s.info()
    info["model"] = SEGMENTER.info()
    return jsonify(info)


def _apply_calibration(s, ordered: np.ndarray, width_mm: float, height_mm: float) -> None:
    extra = [s.original_height, s.original_frac] if s.original_height is not None else None
    warped, mm_per_px, H, extras = calibration.rectify(s.original, ordered, width_mm, height_mm, extra=extra,
                                                       already_ordered=True)
    s.rectified = warped
    s.rect_height = extras[0] if extras else None
    s.rect_frac = extras[1] if extras and len(extras) > 1 else None
    if s.rect_height is not None:
        # reference heights to the floor actually visible inside the mat rectangle
        levelled, shift = geometry.level_height_raster(s.rect_height)
        if shift > 1.0:
            log.info("height raster re-levelled to the mat floor (plane offset up to %.1f mm)", shift)
            s.rect_height = levelled
            s.rect_frac = None            # computed against the old plane; the 50 %-of-height rule takes over
            s.scan_meta["floor_relevel_mm"] = round(shift, 1)
    if s.frames:
        for f in s.frames:
            # Retain the original common raster so repeated recalibration does
            # not resample an already warped image or accumulate transforms.
            if "calibration_original" not in f:
                f["calibration_original"] = {k: f.get(k) for k in ("color", "valid", "height", "nadir_px", "H")}
            original = f["calibration_original"]
            size = (warped.shape[1], warped.shape[0])
            f["color"] = cv2.warpPerspective(original["color"], H, size, flags=cv2.INTER_LINEAR)
            f["valid"] = cv2.warpPerspective(original["valid"].astype(np.uint8), H, size, flags=cv2.INTER_NEAREST) > 0
            if original["height"] is not None:
                f["height"] = cv2.warpPerspective(original["height"], H, size, flags=cv2.INTER_NEAREST, borderValue=float('nan'))
            f["nadir_px"] = cv2.perspectiveTransform(np.asarray(original["nadir_px"], np.float64).reshape(1,1,2), H)[0,0]
            if original["H"] is not None:
                f["H"] = H @ original["H"]
        paint = [f for f in s.frames if f.get("use", "both") != "depth"] or s.frames
        s.scan_meta["photo_coverage"] = coverage_report(paint, warped.shape[:2])
    s.tool_view_cache.clear()
    s.photo_result_cache = None
    s.photo_discovery_cache = None
    s.tool_image_frames.clear()
    s.mm_per_px = mm_per_px
    s.mat_mm = (float(width_mm), float(height_mm))
    s.corners = ordered.tolist()
    s.homography = H
    s.version += 1
    s.masks.clear()


def _marker_mask(s) -> np.ndarray:
    """Exclude observed calibration squares from automatic tool discovery.

    These are capture references, including the printed paper quiet border;
    folded paper can otherwise appear as several thin tools in the height map.
    """
    mask = np.zeros(s.rectified.shape[:2], np.uint8)
    for frame in s.frames or []:
        for corners in frame.get("markers_px", {}).values():
            quad = cv2.perspectiveTransform(np.asarray(corners, np.float64)[None], frame["H"])[0]
            if np.isfinite(quad).all():
                # marker_sheet.svg prints a quiet border of 25% of the marker
                # side on each edge: total paper side is 1.5 times the marker.
                center = quad.mean(axis=0)
                quad = center + 1.5 * (quad - center)
                cv2.fillConvexPoly(mask, np.rint(quad).astype(np.int32), 1)
    radius = max(1, int(round(1. / s.mm_per_px)))
    out = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2)) > 0
    # The zone is PAPER, and paper reads ~0 mm (a folded edge, 2-4 mm). Anything taller under it is a tool that
    # happens to sit next to a marker, and erasing it carves a rectangular notch out of the tool: Nolan's hammer
    # (2026-09-23, capture b59fc5fdd2d1) lost the corner of its striking face this way — 4.8 cm2 of a 28 mm-tall
    # head zeroed by the top-left marker's quiet border. Keep the mask only where the height says paper.
    if s.rect_height is not None:
        out &= ~(np.nan_to_num(s.rect_height, nan=0.0) > MARKER_MASK_MAX_MM)
    return out


def _detect_photo_tools(s, body):
    # Coalesce overlapping clicks and reuse the complete result, including the
    # expensive per-tool image encodings, polygons and source-photo choices.
    import copy
    from toolcutter.processed_cache import save_detection
    key = (s.version, _f(body, "min_area_mm2", 200.0) or 200.0)
    with s.detection_lock:
        cached = s.photo_result_cache
        if cached is None or cached["key"] != key:
            result = _detect_photo_tools_uncached(s, {**body, "id_prefix": "cached_"})
            cached = {"key": key, "result": result,
                      "masks": {t["id"]: s.masks[t["id"]] for t in result["tools"]},
                      "frames": {t["id"]: s.tool_image_frames.get(t["id"]) for t in result["tools"]}}
            s.photo_result_cache = cached
            if (CAPTURE_DIR / s.id / "capture.json").exists():
                save_detection(CAPTURE_DIR / s.id, s)
        result = copy.deepcopy(cached["result"])
        prefix = body.get("id_prefix") or f"p{uuid.uuid4().hex[:8]}_"
        for i, tool in enumerate(result["tools"]):
            old, new = tool["id"], f"{prefix}{i + 1}"
            tool["id"] = new
            tool["image_url"] = f"/api/sessions/{s.id}/tools/{new}/image"
            s.masks[new] = cached["masks"][old].copy()
            s.tool_image_frames[new] = cached["frames"][old]
        return result


def _detect_photo_tools_uncached(s, body):
    """Outline the displayed mosaic, without projecting it through another lens's depth."""
    from toolcutter.photo_outlines import select_silhouettes
    # Bound decoding cost while retaining sub-millimetre sampling on a normal drawer.
    target = min(4400, max(1100, round(max(s.rectified.shape[:2]) * min(1., 2 * s.mm_per_px))))
    image, _ = downscale_to(s.rectified, target)
    scale = s.rectified.shape[1] / image.shape[1]
    markers = cv2.resize(_marker_mask(s).astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    # Rectification can crop away the white quiet border needed by ArUco.
    pad = 40
    padded = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    for corners in capmod.detect_markers(padded).values():
        corners = corners - pad
        center = corners.mean(axis=0)
        quad = center + 1.5 * (corners - center)
        cv2.fillConvexPoly(markers, np.rint(quad).astype(np.int32), 1)
    min_area = _f(body, "min_area_mm2", 200.0) or 200.0
    cache_key = (s.version, min_area, target)
    cached = s.photo_discovery_cache
    if cached is not None and cached[0] == cache_key:
        masks = cached[1]
    else:
        overview, _ = downscale_to(image, 1100)
        proposals = []
        for p in SEGMENTER.automatic_masks(overview, key=f"{s.id}:{s.version}:overview-v2"):
            m = cv2.resize(p['segmentation'].astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
            proposals.append({'segmentation': m, 'area': int(m.sum())})
        if max(image.shape[:2]) > 1100:
            for x0, y0, x1, y1 in discovery_windows(image.shape[1], image.shape[0]):
                for p in SEGMENTER.automatic_masks(image[y0:y1, x0:x1], key=f"{s.id}:{s.version}:tile:{x0}:{y0}"):
                    m = p['segmentation']
                    # A mask cut by an internal tile boundary is only part of a
                    # tool. The overlapping tile or overview supplies it whole.
                    if ((x0 > 0 and m[:, :2].any()) or (y0 > 0 and m[:2].any()) or
                        (x1 < image.shape[1] and m[:, -2:].any()) or (y1 < image.shape[0] and m[-2:].any())):
                        continue
                    full = np.zeros(image.shape[:2], bool)
                    full[y0:y1, x0:x1] = m
                    proposals.append({'segmentation': full, 'area': int(full.sum())})
                # Bound retained memory between tiles, removing nested parts.
                reduced = select_silhouettes(proposals, s.mm_per_px * scale, markers > 0, min_area)
                proposals = [{'segmentation': m, 'area': int(m.sum())} for m in reduced]
        masks = select_silhouettes(proposals, s.mm_per_px * scale, markers > 0, min_area)
        s.photo_discovery_cache = (cache_key, masks)
    prefix = body.get("id_prefix") or f"p{uuid.uuid4().hex[:8]}_"
    tools = []
    sensors = set((s.scan_meta or {}).get("sensors", []))
    # Unregistered front depth and rear photos cannot safely supply pocket depths.
    unregistered = "rear_photo" in sensors and any(x.startswith("truedepth") for x in sensors) and not (s.scan_meta or {}).get("rgbd_registration")
    for i, small in enumerate(masks):
        mask = cv2.resize(small.astype(np.uint8), (s.rectified.shape[1], s.rectified.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        frame_index = choose_tool_frame(s.frames, mask, s.mm_per_px) if s.frames else None
        source = "stitched"
        view_key = (s.version, frame_index, hashlib.sha256(np.packbits(mask).tobytes()).digest())
        cached_view = s.tool_view_cache.get(view_key)
        if cached_view is not None:
            mask, frame_index, source = cached_view
        elif frame_index is not None:
            ys, xs = np.nonzero(mask)
            box = [float(xs.min()), float(ys.min()), float(xs.max()+1), float(ys.max()+1)]
            dt = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
            sy, sx = np.unravel_index(int(dt.argmax()), dt.shape)
            try:
                refined = _sam_mask(s, [{"x": float(sx), "y": float(sy), "label": 1}], box, False, frame_idx=frame_index)
                ratio = refined.sum() / max(1, mask.sum())
                intersection = np.count_nonzero(refined & mask)
                iou = intersection / max(1, np.count_nonzero(refined | mask))
                if .7 <= ratio <= 1.4 and iou >= .65:
                    mask = refined
                    source = "single_photo"
                else:
                    frame_index = None
            except Exception:
                log.warning("Keeping stitched outline after individual photo refinement failed", exc_info=True)
                frame_index = None
        # Limit this cache to the current detected tool set, not every edit history.
        if len(s.tool_view_cache) >= 80:
            s.tool_view_cache.clear()
        s.tool_view_cache[view_key] = (mask, frame_index, source)
        poly = geometry.topo_polygon(mask, s.mm_per_px, sigma_mm=0.5)
        if poly is None:
            continue
        tid = f"{prefix}{i + 1}"
        dt = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
        y, x = np.unravel_index(int(dt.argmax()), dt.shape)
        stats = None if unregistered or s.rect_height is None else scanmod.measure_thickness(s.rect_height, mask)
        tools.append({"id": tid, "session_id": s.id, "points": [{"x": float(x), "y": float(y), "label": 1}],
                      "box": [float(poly[:, 0].min()), float(poly[:, 1].min()), float(poly[:, 0].max()), float(poly[:, 1].max())],
                      "polygon_px": poly.tolist(), "polygon_mm": (poly * s.mm_per_px).tolist(),
                      "area_mm2": float(cv2.contourArea(poly.astype(np.float32))) * s.mm_per_px ** 2,
                      "measured_thickness_mm": round(stats["p95_mm"], 1) if stats else None,
                      "height_stats": stats, "edge_source": "photo", "image_source": source,
                      "image_url": f"/api/sessions/{s.id}/tools/{tid}/image"})
        s.masks[tid] = mask
        s.tool_image_frames[tid] = frame_index
    return {"tools": tools, "mode": "color", "edge_source": "photo", "sam_used": True, "sam_error": None,
            "foreground_fraction": float(np.logical_or.reduce(masks).mean()) if masks else 0.0}


@app.post("/api/sessions/<sid>/auto_detect")
def auto_detect(sid: str):
    s = _require_rectified(_session(sid))
    body = request.get_json(force=True, silent=True) or {}
    mode = body.get("mode", "auto")
    if mode not in ("auto", "height", "color"):
        raise ApiError("mode must be auto | color | height")
    if mode != "height" and SEGMENTER.available and _edge_source(s, body) == "photo":
        try:
            return jsonify(_detect_photo_tools(s, body))
        except Exception as exc:
            log.exception("Photo tool discovery failed")
            raise ApiError("Photo detection failed. Try again or select depth-only detection in Settings.", 503) from exc
    min_area_mm2 = _f(body, "min_area_mm2", 200.0) or 200.0
    thr_mm = _f(body, "height_threshold_mm", 2.0) or 2.0
    refine = bool(body.get("refine_with_sam", s.rect_height is None or s.capture_geom is not None or bool(s.frames)))
    if mode == "auto":
        mode = "height" if s.rect_height is not None else "color"
    if mode == "height":
        if s.rect_height is None:
            raise ApiError("No height data; use mode 'color'")
        detection_height = s.rect_height
        marker_mask = _marker_mask(s)
        if marker_mask.any():
            detection_height = detection_height.copy()
            detection_height[marker_mask] = 0
        fg, blobs = geometry.detect_blobs_height(detection_height, s.mm_per_px, threshold_mm=thr_mm,
                                                 min_area_mm2=min_area_mm2, above_frac=s.rect_frac,
                                                 edge_rule="fixed" if (s.capture_geom is not None or s.frames) else "half_height")
    elif mode == "color":
        fg, blobs = geometry.detect_blobs_color(s.rectified, s.mm_per_px, min_area_mm2=min_area_mm2)
    else:
        raise ApiError("mode must be auto | color | height")
    blobs = blobs[:40]
    tools = []
    prefix = body.get("id_prefix") or f"t{int(time.time()) % 100000}_"
    min_px = min_area_mm2 / (s.mm_per_px ** 2)
    if mode == "height" and _edge_source(s, body) == "topo":
        # borders from the scan topography alone: each blob re-thresholded at half its local top; merged
        # neighbours fall apart at the dip between them and become separate tools
        cell = _depth_cell_mm(s)
        i = 0
        for b in blobs:
            fld: Dict = {}
            tm = geometry.topo_footprint(detection_height, b["mask"], s.mm_per_px, cell_mm=cell, threshold_mm=thr_mm, field_out=fld)
            pieces = []      # (mask, signed field it came from) — each re-derived part has its own
            # topographic masks are tight (no ramp): a tool that passed the blob size test may come out well under
            # min_area here, so only slivers are dropped at this stage
            for comp in geometry.split_components(tm, 0.35 * min_px):
                parts = geometry.split_at_saddles(comp, detection_height, s.mm_per_px, cell_mm=cell, min_area_mm2=0.35 * min_area_mm2)
                if len(parts) > 1:
                    # re-derive each part's edges from its own topography (the watershed line ran through the ramp)
                    for part in parts:
                        sub_fld: Dict = {}
                        sub = geometry.topo_footprint(detection_height, part, s.mm_per_px, cell_mm=cell, threshold_mm=thr_mm, restrict=part, field_out=sub_fld)
                        got = geometry.split_components(sub, 0.35 * min_px)
                        pieces.extend([(g, sub_fld.get("signed")) for g in got] or [(part, None)])
                else:
                    pieces.extend((p_, fld.get("signed")) for p_ in parts)
            for comp, signed in pieces:
                i += 1
                if i > 60:
                    break
                # A LOW tool is where the first pass is weakest: its seed comes from a fixed-threshold blob that
                # is too generous, and one more pass seeded by the result converges on the true band (a 3 mm
                # steel rule went 34.5 -> 24.5 mm wide, perimeter/hull 1.15 -> 1.07). For everything taller the
                # second pass is a fixed point, so it is only worth paying for down here.
                if REFIT_LOW_MM > 0 and s.rect_height is not None and comp.any() and float(np.nanmax(np.where(comp, s.rect_height, np.nan))) < REFIT_LOW_MM:
                    fld2: Dict = {}
                    again = geometry.topo_footprint(s.rect_height, comp, s.mm_per_px, cell_mm=cell, threshold_mm=thr_mm, field_out=fld2)
                    if again.any():
                        comp, signed = again, fld2.get("signed")
                # A TOOL HAS TO HAVE HEIGHT. On Nolan's drawer 136c77400c3f, 2 of 13 "tools" sat on bare floor
                # (p90 of the fused height inside them was 0.00 mm, one of them 16.6 cm²) and 4 more were
                # fragments. They come from artefacts in the band along the drawer's back wall, where the height
                # map is noisy. topo_footprint re-thresholds against a blob's OWN local top, so a patch of noise
                # is happily re-derived into a confident outline — the check has to be against the real height,
                # not against the blob's internal contrast. The steel rule (a genuine 3 mm tool) reads 2.5 mm
                # here, so the floor is set below that and well above the 0.08-0.8 mm the bare mat reads.
                if TOOL_MIN_HEIGHT_MM > 0 and s.rect_height is not None:
                    hs = np.where(comp, s.rect_height, np.nan)
                    if not np.isfinite(hs).any() or float(np.nanpercentile(hs, 90)) < TOOL_MIN_HEIGHT_MM:
                        i -= 1
                        continue
                dt = cv2.distanceTransform(comp.astype(np.uint8), cv2.DIST_L2, 3)
                yy, xx = np.unravel_index(int(np.argmax(dt)), dt.shape)
                ys, xs = np.nonzero(comp)
                box = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
                tools.append(_topo_tool_result(s, f"{prefix}{i}", comp, [{"x": float(xx), "y": float(yy), "label": 1}], box, signed=signed))
        return jsonify({"tools": tools, "mode": mode, "edge_source": "topo", "sam_used": False, "sam_error": None,
                        "foreground_fraction": float(fg.mean())})
    use_sam = refine and SEGMENTER.available
    sam_error = None
    if use_sam and not s.frames:
        try:
            SEGMENTER.set_image(f"{s.id}:{s.version}", s.rectified)
        except Exception as exc:  # noqa: BLE001
            log.warning("SAM unavailable for auto-detect: %s", exc)
            use_sam = False
            sam_error = str(exc)
    work = list(blobs)
    i = 0
    while work and i < 60:
        b = work.pop(0)
        i += 1
        tid = f"{prefix}{i}"
        points = [{"x": b["seed"][0], "y": b["seed"][1], "label": 1}]
        box = [float(v) for v in b["box"]]
        mask = b["mask"]
        from_color = mode == "color"
        blob_for_tool = mask
        fidx = _nearest_frame(s, b["seed"][0], b["seed"][1]) if s.frames else None
        if use_sam:
            try:
                sam = _sam_mask(s, points, box, bool(body.get("hq_token_only", False)), frame_idx=fidx)
                ratio = float(sam.sum()) / max(1.0, float(mask.sum()))
                if 0.5 <= ratio <= 2.5:
                    mask = sam
                    from_color = True
                elif 0.08 <= ratio < 0.5 and sam.any():
                    # SAM took one tool out of a blob that merged neighbours (tools almost touching): accept it
                    # and queue the remainder of the blob as another candidate
                    k = max(1, int(round(3.0 / s.mm_per_px)))
                    rest = mask & ~(cv2.dilate(sam.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))) > 0)
                    num, lab, st, _ = cv2.connectedComponentsWithStats(rest.astype(np.uint8), connectivity=8)
                    for j in range(1, num):
                        if st[j, cv2.CC_STAT_AREA] >= min_px:
                            comp = lab == j
                            dt = cv2.distanceTransform(comp.astype(np.uint8), cv2.DIST_L2, 3)
                            yy, xx = np.unravel_index(int(np.argmax(dt)), dt.shape)
                            x, y, bw, bh = [int(st[j, q]) for q in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT)]
                            work.append({"box": [x, y, x + bw, y + bh], "seed": [float(xx), float(yy)], "area_px": float(st[j, cv2.CC_STAT_AREA]), "mask": comp})
                    blob_for_tool = mask & ~rest          # this tool's share of the merged blob
                    mask = sam
                    from_color = True
            except Exception as exc:  # noqa: BLE001
                log.warning("SAM refine failed for blob %d: %s", i, exc)
        tools.append(_tool_result(s, tid, mask, points, box, color_silhouette=from_color, frame_idx=fidx,
                                  blob=blob_for_tool if from_color else None))
    return jsonify({"tools": tools, "mode": mode, "edge_source": "photo", "sam_used": use_sam, "sam_error": sam_error,
                    "foreground_fraction": float(fg.mean())})


def _segment_topo(s, body: Dict) -> List[Dict]:
    """Click-to-outline on the scan: a positive point picks the height blob under it (a box picks everything
    inside), the footprint comes from the topography, and an exclude point carves its part away."""
    h = np.nan_to_num(s.rect_height, nan=0.0)
    thr = float(body.get("height_threshold_mm") or 2.0)
    cell = _depth_cell_mm(s)
    cell_px = max(1.0, cell / s.mm_per_px)
    fg = (h > thr).astype(np.uint8)
    num, lab = cv2.connectedComponents(fg, connectivity=8)
    out = []
    for t in body.get("tools") or []:
        tid = str(t.get("id") or uuid.uuid4().hex[:8])
        points = _points(t.get("points"))
        box = t.get("box")
        if box is not None:
            try:
                box = [float(v) for v in box]
            except (TypeError, ValueError):
                raise ApiError("box must be [x0, y0, x1, y1]")
        pos = [p for p in points if p["label"] == 1]
        neg = [p for p in points if p["label"] == 0]
        seed = np.zeros(h.shape, bool)
        restrict = None
        if box is not None:
            x0, y0, x1, y1 = [int(round(v)) for v in box]
            restrict = np.zeros(h.shape, bool)
            restrict[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = True
            seed |= restrict & (fg > 0)
        for p in pos:
            x, y = int(round(p["x"])), int(round(p["y"]))
            if 0 <= y < h.shape[0] and 0 <= x < h.shape[1]:
                l = lab[y, x]
                if l > 0:
                    seed |= lab == l
                else:
                    # clicked just beside a blob (the ramp): take the nearest blob within a cell
                    k = int(round(cell_px)) | 1
                    near = cv2.dilate((lab > 0).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1)))
                    if near[y, x]:
                        ys_, xs_ = np.nonzero(lab > 0)
                        j = int(np.argmin((xs_ - x) ** 2 + (ys_ - y) ** 2))
                        seed |= lab == lab[ys_[j], xs_[j]]
        if not seed.any():
            out.append({"id": tid, "points": points, "box": box, "polygon_px": [], "polygon_mm": [], "area_mm2": 0.0,
                        "measured_thickness_mm": None, "height_stats": None, "edge_source": "topo"})
            continue
        fld: Dict = {}
        mask = geometry.topo_footprint(h, seed, s.mm_per_px, cell_mm=cell, threshold_mm=thr, restrict=restrict, field_out=fld)
        for p in neg:
            x, y = int(round(p["x"])), int(round(p["y"]))
            if not (0 <= y < h.shape[0] and 0 <= x < h.shape[1]):
                continue
            n2, lab2 = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
            l = lab2[y, x]
            keep_pos = any(0 <= int(round(q["y"])) < h.shape[0] and 0 <= int(round(q["x"])) < h.shape[1]
                           and lab2[int(round(q["y"])), int(round(q["x"]))] == l for q in pos)
            if l > 0 and not keep_pos:
                mask &= lab2 != l                    # a separate piece: drop it whole
            else:
                # same piece as the tool: carve a cell-sized hole around the click so the piece splits, then keep
                # the part(s) that still hold a positive point
                r = int(round(1.2 * cell_px))
                cv2.circle(mask.view(np.uint8), (x, y), r, 0, -1)
                n3, lab3 = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
                keep = set(lab3[int(round(q["y"])), int(round(q["x"]))] for q in pos
                           if 0 <= int(round(q["y"])) < h.shape[0] and 0 <= int(round(q["x"])) < h.shape[1])
                keep.discard(0)
                if keep:
                    mask = np.isin(lab3, list(keep))
        if pos and mask.any():
            # keep only the piece(s) holding a positive point
            n4, lab4 = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
            keep = set(lab4[int(round(q["y"])), int(round(q["x"]))] for q in pos
                       if 0 <= int(round(q["y"])) < h.shape[0] and 0 <= int(round(q["x"])) < h.shape[1])
            keep.discard(0)
            if keep:
                mask = np.isin(lab4, list(keep))
        out.append(_topo_tool_result(s, tid, mask, points, box, signed=fld.get("signed")))
    return out


@app.post("/api/sessions/<sid>/snap_base")
def snap_base(sid: str):
    """Push an outline out (or in) to where the tool meets the mat.

    The topographic edge sits at half the height of each wall, which hugs the TOP of anything that slopes;
    this walks each vertex along the outline normal to the last point with something above the floor under
    it. Nolan, 2026-09-20: "the vertexes seem to start on top of the object and I have to spread them out
    so they are around it".
    """
    s_ = _session(sid)
    if s_.rect_height is None:
        raise ApiError("This capture has no height data, so there is no base to snap to.")
    body = request.get_json(force=True) or {}
    poly = np.asarray(body.get("polygon_px") or [], dtype=np.float64)
    if poly.ndim != 2 or len(poly) < 3:
        raise ApiError("Send polygon_px with at least 3 points")
    out = geometry.snap_to_base(poly, s_.rect_height, s_.mm_per_px,
                                floor_mm=float(body.get("floor_mm") or 1.5),
                                max_mm=float(body.get("max_mm") or 12.0),
                                smooth_mm=float(body.get("smooth_mm") or 1.5))
    return jsonify({"polygon_px": out.tolist(), "polygon_mm": (out * s_.mm_per_px).tolist()})


@app.post("/api/sessions/<sid>/split")
def split_tool(sid: str):
    """Cut one detected tool into two along a line the user dragged across it.

    body: {tool_id, line: [[x0, y0], [x1, y1]] (rectified px), edge_source?}. The tool's mask is divided by the
    (infinite) line; each side becomes a new tool whose border is re-derived from the topography (or the plain
    mask outline when there is no height data). Returns {tools: [two tools]}.
    """
    s = _require_rectified(_session(sid))
    body = request.get_json(force=True, silent=True) or {}
    tid = str(body.get("tool_id") or "")
    mask = s.masks.get(tid)
    if mask is None:
        raise ApiError("Unknown tool (nothing to split); detect or click it first.", 404)
    try:
        (x0, y0), (x1, y1) = [(float(p[0]), float(p[1])) for p in body["line"]]
    except (KeyError, TypeError, ValueError, IndexError):
        raise ApiError("line must be [[x0, y0], [x1, y1]] in image pixels")
    if math.hypot(x1 - x0, y1 - y0) < 2:
        raise ApiError("Drag a longer line across the tool")
    H, W = mask.shape
    yy, xx = np.mgrid[0:H, 0:W]
    side = (x1 - x0) * (yy - y0) - (y1 - y0) * (xx - x0)          # sign = which side of the line
    # a 1.5 mm-wide seam along the cut so the two parts do not touch
    seam = np.abs(side) / math.hypot(x1 - x0, y1 - y0) < max(1.0, 0.75 / s.mm_per_px)
    parts = [mask & (side > 0) & ~seam, mask & (side < 0) & ~seam]
    min_px = 40.0 / s.mm_per_px ** 2
    out = []
    prefix = body.get("id_prefix") or f"s{int(time.time()) % 100000}_"
    cell = _depth_cell_mm(s)
    n = 0
    for part in parts:
        comps = geometry.split_components(part, min_px)
        if not comps:
            continue
        # keep every real piece on that side (a cut can leave a tool in two components if it was U-shaped)
        piece = np.zeros_like(mask)
        for c in comps:
            if c.sum() >= 0.15 * part.sum():
                piece |= c
        if not piece.any():
            continue
        n += 1
        nid = f"{prefix}{n}"
        dt = cv2.distanceTransform(piece.astype(np.uint8), cv2.DIST_L2, 3)
        py, px = np.unravel_index(int(np.argmax(dt)), dt.shape)
        pts = [{"x": float(px), "y": float(py), "label": 1}]
        ys, xs = np.nonzero(piece)
        box = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
        if s.rect_height is not None and _edge_source(s, body) == "topo":
            fld = {}
            tm = geometry.topo_footprint(s.rect_height, piece, s.mm_per_px, cell_mm=cell, restrict=piece | seam, field_out=fld)
            out.append(_topo_tool_result(s, nid, tm if tm.any() else piece, pts, box, signed=fld.get("signed")))
        else:
            out.append(_tool_result(s, nid, piece, pts, box, color_silhouette=False))
    if len(out) < 2:
        raise ApiError("That line does not divide the tool into two pieces — drag it across the join.")
    s.masks.pop(tid, None)
    return jsonify({"tools": out, "removed": tid})


@app.post("/api/sessions/<sid>/merge")
def merge_tools(sid: str):
    """Combine several tools into one — the inverse of /split.

    body: {tool_ids: [...], bridge_mm?}. The tools' stored masks are unioned and the seam between them is
    healed by a closing of `bridge_mm` (default 2.0, capped at 6.0 — a wide closing fills T/L inner corners,
    which is why the reconnection elsewhere in this file is held to 2 mm). Parts that are still apart after
    that are joined by a straight bridge of the same width between their nearest points, so the call always
    yields ONE outline; `bridged_mm` reports the widest gap that had to be crossed, and the UI says so.

    The parts' own edges are kept as they were derived — the union is NOT re-thresholded. Each part was already
    traced against its own local topography, and re-deriving the whole would judge a small part against the
    tall one's top and eat it. Returns {tools: [one], removed: [ids]}.
    """
    s = _require_rectified(_session(sid))
    body = request.get_json(force=True, silent=True) or {}
    ids = [str(t) for t in (body.get("tool_ids") or []) if str(t)]
    if len(ids) < 2:
        raise ApiError("Select at least two tools to combine.")
    missing = [t for t in ids if s.masks.get(t) is None]
    if missing:
        raise ApiError(f"Unknown tool(s) in this scan: {', '.join(missing)}. "
                       "Hand-drawn shapes and tools from another capture cannot be combined here.", 404)
    masks = [s.masks[t] for t in ids]
    union = np.zeros_like(masks[0], dtype=bool)
    for m in masks:
        union |= m.astype(bool)
    bridge_mm = min(6.0, max(0.0, _f(body, "bridge_mm", 2.0) or 0.0))
    if bridge_mm > 0:
        k = max(3, int(round(bridge_mm / s.mm_per_px)) | 1)
        union = cv2.morphologyEx(union.astype(np.uint8), cv2.MORPH_CLOSE,
                                 cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))).astype(bool)
    # Anything still separate is joined explicitly: the user asked for one tool, so produce one rather than
    # returning a polygon with holes in the middle of it.
    bridged_mm = 0.0
    comps = geometry.split_components(union, 1.0)
    while len(comps) > 1:
        a = comps[0]
        ya, xa = np.nonzero(a)
        best = None
        for other in comps[1:]:
            yb, xb = np.nonzero(other)
            step = max(1, len(xa) // 2000), max(1, len(xb) // 2000)
            pa = np.stack([xa[::step[0]], ya[::step[0]]], 1).astype(np.float32)
            pb = np.stack([xb[::step[1]], yb[::step[1]]], 1).astype(np.float32)
            d = np.linalg.norm(pa[:, None, :] - pb[None, :, :], axis=2)
            i, j = np.unravel_index(int(np.argmin(d)), d.shape)
            if best is None or d[i, j] < best[0]:
                best = (float(d[i, j]), tuple(pa[i].astype(int)), tuple(pb[j].astype(int)), other)
        gap, p1, p2, other = best
        bridged_mm = max(bridged_mm, gap * s.mm_per_px)
        link = np.zeros(union.shape, np.uint8)
        cv2.line(link, p1, p2, 1, max(1, int(round(max(1.0, bridge_mm) / s.mm_per_px))))
        union |= link.astype(bool)
        comps = geometry.split_components(union, 1.0)
    if not union.any():
        raise ApiError("Those tools have no scanned area to combine.")
    nid = str(body.get("id") or f"m{int(time.time()) % 100000}_1")
    dt = cv2.distanceTransform(union.astype(np.uint8), cv2.DIST_L2, 3)
    py, px = np.unravel_index(int(np.argmax(dt)), dt.shape)
    ys, xs = np.nonzero(union)
    box = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
    tool = _topo_tool_result(s, nid, union, [{"x": float(px), "y": float(py), "label": 1}], box)
    for t in ids:
        s.masks.pop(t, None)
    return jsonify({"tools": [tool], "removed": ids, "bridged_mm": round(bridged_mm, 1)})


@app.post("/api/autolayout")
def autolayout():
    """Pack the tools onto the mat: {mat: {width_mm, height_mm}, tools: [{id, polygon_mm, include?}], gap_mm?,
    margin_mm?, allow_rotate?}. Returns per tool {id, rotation_deg, offset_mm: {x, y}} in the same convention as
    /api/layout (rotate about the outline's own centroid, then translate); tools that do not fit are listed in
    `unplaced` and keep their current placement."""
    body = request.get_json(force=True, silent=True) or {}
    mat = body.get("mat") or {}
    W = _f(mat, "width_mm", None)
    Hm = _f(mat, "height_mm", None)
    if not W or not Hm:
        raise ApiError("mat.width_mm and mat.height_mm are required")
    gap = _f(body, "gap_mm", 8.0) or 8.0
    margin = _f(body, "margin_mm", 10.0) or 10.0
    allow_rotate = bool(body.get("allow_rotate", True))
    # "columns" (default): tools stand upright (long side vertical) side by side across the drawer, left to right,
    # wrapping to a new band below; "rows": tools lie along the drawer (long side horizontal), stacked top to bottom,
    # wrapping to a new column to the right. The other orientation is used only when the preferred one fits nowhere.
    vertical = (body.get("direction") or "columns") != "rows"
    step = 2.0                                          # occupancy grid (mm)
    from shapely.geometry import Polygon as _P
    from shapely import affinity as _aff
    items = []
    for t in body.get("tools") or []:
        if t.get("include") is False:
            continue
        poly = t.get("polygon_mm") or []
        if len(poly) < 3:
            continue
        p = _P(poly).buffer(0)
        if p.is_empty:
            continue
        if p.geom_type == "MultiPolygon":
            p = max(p.geoms, key=lambda g: g.area)
        items.append((str(t.get("id")), p))
    if not items:
        return jsonify({"placements": [], "unplaced": []})
    gw, gh = int(math.ceil(W / step)), int(math.ceil(Hm / step))
    occ = np.zeros((gh, gw), np.uint8)
    # the margin: block the border
    mpx = int(round(margin / step))
    if mpx > 0:
        occ[:mpx, :] = 1; occ[-mpx:, :] = 1; occ[:, :mpx] = 1; occ[:, -mpx:] = 1
    half_gap_px = max(1, int(round(gap / 2 / step)))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * half_gap_px + 1, 2 * half_gap_px + 1))

    def principal_angle(p) -> float:
        rect = cv2.minAreaRect(np.asarray(p.exterior.coords, dtype=np.float32))
        (cx, cy), (rw, rh), ang = rect
        # angle that makes the long side horizontal (add 90 for vertical)
        return -ang if rw >= rh else -(ang + 90.0)

    def raster(p):
        minx, miny, maxx, maxy = p.bounds
        w = int(math.ceil((maxx - minx) / step)) + 1
        h = int(math.ceil((maxy - miny) / step)) + 1
        img = np.zeros((h, w), np.uint8)
        pts = ((np.asarray(p.exterior.coords) - [minx, miny]) / step).astype(np.int32)
        cv2.fillPoly(img, [pts], 1)
        for ring in p.interiors:
            cv2.fillPoly(img, [((np.asarray(ring.coords) - [minx, miny]) / step).astype(np.int32)], 0)
        return img, (minx, miny)

    # biggest first (by long side), like a human would
    order = sorted(items, key=lambda it: -max(it[1].bounds[2] - it[1].bounds[0], it[1].bounds[3] - it[1].bounds[1]))
    placements, unplaced = [], []
    for tid, p in order:
        c = (p.centroid.x, p.centroid.y)
        base = principal_angle(p) + (90.0 if vertical else 0.0)      # preferred: long side along the fill direction
        cands = [base, base + 90.0] if allow_rotate else [0.0]
        best = None
        for ang in cands:                       # preferred orientation first; the other only if this one fits nowhere
            r = _aff.rotate(p, ang, origin=c)
            img, (minx, miny) = raster(r)
            fat = cv2.dilate(img, ker)                                  # tool + half the gap all round
            fh, fw = fat.shape
            if fw > gw or fh > gh:
                continue
            found = None
            if vertical:
                # upright tools side by side: fill each band left to right, then the next band down
                for gy in range(0, gh - fh + 1, 2):
                    for gx in range(0, gw - fw + 1, 2):
                        if not np.any(occ[gy:gy + fh, gx:gx + fw] & fat):
                            found = (gx, gy)
                            break
                    if found:
                        break
            else:
                # lying tools stacked: fill each column top to bottom, then the next column to the right
                for gx in range(0, gw - fw + 1, 2):
                    for gy in range(0, gh - fh + 1, 2):
                        if not np.any(occ[gy:gy + fh, gx:gx + fw] & fat):
                            found = (gx, gy)
                            break
                    if found:
                        break
            if found is None:
                continue
            gx, gy = found
            best = (None, ang, gx, gy, minx, miny, fat, fw, fh)
            break
        if best is None:
            unplaced.append(tid)
            continue
        _, ang, gx, gy, minx, miny, fat, fw, fh = best
        occ[gy:gy + fh, gx:gx + fw] |= fat
        # the rotated outline's bbox min must land at grid (gx, gy) plus the dilation offset
        tx = (gx + half_gap_px) * step - minx
        ty = (gy + half_gap_px) * step - miny
        placements.append({"id": tid, "rotation_deg": round(float(ang) % 360.0, 1), "offset_mm": {"x": round(tx, 1), "y": round(ty, 1)}})
    return jsonify({"placements": placements, "unplaced": unplaced, "gap_mm": gap, "margin_mm": margin})


@app.post("/api/sessions/<sid>/segment")
def segment(sid: str):
    s = _require_rectified(_session(sid))
    body = request.get_json(force=True, silent=True) or {}
    tools_in = body.get("tools") or []
    if not tools_in:
        raise ApiError("tools[] with prompt points is required")
    hq_token_only = bool(body.get("hq_token_only", False))
    if _edge_source(s, body) == "topo":
        return jsonify({"tools": _segment_topo(s, body)})
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
    poly = raw.get("polygon_mm")
    if poly and len(poly) >= 3 and sess.homography is None:
        # a hand-edited outline: the body follows the edited polygon; where it reaches outside the scanned mask the
        # top is filled at the tool's median height
        pts = (np.asarray(poly, dtype=np.float64) / sess.mm_per_px)
        edited = np.zeros(mask.shape, np.uint8)
        cv2.fillPoly(edited, [np.round(pts).astype(np.int32)], 1)
        edited = edited > 0
        if edited.any() and not np.array_equal(edited, mask):
            h = sess.rect_height
            fill = float(np.nanmedian(h[mask])) if mask.any() else float(np.nanmedian(h[edited]))
            h2 = np.where(mask, h, fill).astype(np.float32)
            return edited, h2, sess.mm_per_px
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


def _warm_saved_captures():
    """Restore ready snapshots only; do not compete with detection by rebuilding old scans."""
    from toolcutter.processed_cache import load_session, load_detection
    if not CAPTURE_DIR.exists():
        return
    for directory in CAPTURE_DIR.iterdir():
        if STORE.get(directory.name) is not None:
            continue
        cached = load_session(directory)
        if cached is not None:
            cached.photo_result_cache = load_detection(directory)
            STORE.restore(cached)
            log.info("restored processed capture %s", cached.id)


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
    if not os.environ.get("TC_NO_WARM") and not (args.debug and os.environ.get("WERKZEUG_RUN_MAIN") != "true"):
        threading.Thread(target=_warm_saved_captures, name="warm-captures", daemon=True).start()
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()
