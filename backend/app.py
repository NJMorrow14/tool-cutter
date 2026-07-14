#!/usr/bin/env python3
"""
Flask API that wraps Meta's HQ-SAM tooling to segment tool images and export SVG cutouts.

Primary endpoints:
 - POST /api/upload_image -> accepts an image upload (JPEG/PNG/HEIC/etc.)
 - POST /api/preview      -> returns overlay + mask previews based on provided prompts
 - POST /api/export_svg   -> generates an SVG for download using the latest mask
 - GET  /health           -> readiness probe

Run:
  python app.py --image ./images/overhead_aruco.jpg --host 127.0.0.1 --port 8000

Dependencies:
  pip install flask flask-cors numpy opencv-python svgwrite pillow pillow-heif (optional)
"""
from __future__ import annotations

import argparse
import base64
import io
import os
import subprocess
import sys
import tempfile
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Tuple, List, Optional

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import cv2
import numpy as np

BACKEND_ROOT = Path(__file__).resolve().parent
# Allow bundled SAM/HQ-SAM sources to be imported without separate installs.
SAM_HQ_ROOTS = (
    BACKEND_ROOT / 'sam-hq',
    BACKEND_ROOT / 'sam-hq' / 'sam-hq2',
)
for _path in SAM_HQ_ROOTS:
    if _path.exists():
        p = str(_path)
        if p not in sys.path:
            sys.path.insert(0, p)

import tool_image_to_svg as core

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}, r"/health": {"origins": "*"}})


def bgr_to_png_bytes(img_bgr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode('.png', img_bgr)
    if not ok:
        raise RuntimeError('PNG encode failed')
    return buf.tobytes()


def png_bytes_to_b64uri(png_bytes: bytes) -> str:
    b64 = base64.b64encode(png_bytes).decode('ascii')
    return f"data:image/png;base64,{b64}"


_HEIF_SETUP_DONE = False


def _ensure_heif_opener() -> None:
    """Register HEIF decoders with Pillow if available."""
    global _HEIF_SETUP_DONE
    if _HEIF_SETUP_DONE:
        return
    _HEIF_SETUP_DONE = True
    try:
        from pillow_heif import register_heif_opener  # type: ignore

        register_heif_opener()
    except Exception:
        # Pillow may not have HEIF support installed; ignore silently.
        pass


def _decode_heif_with_pillow(data: bytes) -> Optional[np.ndarray]:
    try:
        from PIL import Image  # type: ignore
        from PIL import UnidentifiedImageError  # type: ignore
    except Exception:
        return None

    _ensure_heif_opener()
    try:
        with Image.open(io.BytesIO(data)) as pil_img:
            pil_rgb = pil_img.convert('RGB')
            return cv2.cvtColor(np.array(pil_rgb), cv2.COLOR_RGB2BGR)
    except UnidentifiedImageError:
        return None
    except Exception:
        return None


def _decode_heif_with_sips(data: bytes) -> Optional[np.ndarray]:
    """macOS fallback: use `sips` to convert HEIC to JPEG."""
    with tempfile.NamedTemporaryFile(suffix='.heic', delete=False) as src:
        src_path = src.name
        src.write(data)
    dest_path = f"{src_path}.jpg"
    try:
        result = subprocess.run(
            ['sips', '-s', 'format', 'jpeg', src_path, '--out', dest_path],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if result.returncode != 0:
            return None
        img = cv2.imread(dest_path, cv2.IMREAD_COLOR)
        return img
    except FileNotFoundError:
        return None
    except Exception:
        return None
    finally:
        try:
            os.remove(src_path)
        except OSError:
            pass
        try:
            os.remove(dest_path)
        except OSError:
            pass


def decode_image_bytes_to_bgr(data: bytes) -> Tuple[Optional[np.ndarray], bool]:
    """Decode raw image bytes into a BGR np.ndarray with HEIC fallback.

    Returns (image, converted_to_jpeg) where the boolean indicates that we had
    to route through Pillow (typically for HEIC/HEIF) and normalize to RGB
    before converting back to OpenCV's BGR layout.
    """
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is not None:
        return img, False
    pil_bgr = _decode_heif_with_pillow(data)
    if pil_bgr is not None:
        return pil_bgr, True
    sips_bgr = _decode_heif_with_sips(data)
    if sips_bgr is not None:
        return sips_bgr, True
    return None, False


def load_image_from_path(path: str) -> Tuple[np.ndarray, bool]:
    """Load an image from disk with HEIC->JPEG fallback."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is not None:
        return img, False
    with open(path, 'rb') as f:
        data = f.read()
    decoded, converted = decode_image_bytes_to_bgr(data)
    if decoded is None:
        raise RuntimeError(f'Unsupported image format: {path}')
    return decoded, converted


def rotate_bgr_image(img: np.ndarray, rotation: int) -> np.ndarray:
    """Rotate an image (any channel count) in 90° increments clockwise."""
    rot = int(rotation) % 360
    if rot == 0:
        return img.copy()
    if rot == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if rot == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    if rot == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError('Rotation must be a multiple of 90 degrees')


def _ensure_image_orientation(target_rotation: int) -> None:
    """Rotate the stored image/mask in app.config to match target rotation."""
    target = int(target_rotation) % 360
    if target % 90 != 0:
        target = (round(target / 90) * 90) % 360
    current = int(app.config.get('IMAGE_ROTATION', 0)) % 360
    if target == current:
        return
    img = app.config.get('IMAGE')
    if img is None:
        app.config['IMAGE_ROTATION'] = target
        return
    delta = (target - current) % 360
    if delta:
        rotated_img = rotate_bgr_image(img, delta)
        app.config['IMAGE'] = rotated_img
        raw_mask = app.config.get('RAW_MASK')
        if isinstance(raw_mask, np.ndarray):
            app.config['RAW_MASK'] = rotate_bgr_image(raw_mask, delta)
        app.config['IMAGE_BASE_SHAPE'] = (int(rotated_img.shape[0]), int(rotated_img.shape[1]))
    app.config['IMAGE_ROTATION'] = target




def get_hqsam_predictor(checkpoint: str, model_type: str, force_device: Optional[str] = None):
    """Load and cache a predictor for SAM/HQ-SAM. Tries HQ-SAM, then base SAM."""
    if not checkpoint:
        raise RuntimeError('Missing HQ-SAM checkpoint path')
    device_pref = force_device or app.config.get('HQSAM_FORCE_DEVICE')
    cache = app.config.setdefault('HQSAM_CACHE', {})
    # Select device
    try:
        import torch  # type: ignore
        if device_pref:
            device = device_pref
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        elif torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
    except Exception:
        device = 'cpu'

    key = (checkpoint, model_type, device)
    if key in cache:
        return cache[key]

    predictor = None
    # Try HQ-SAM
    try:
        try:
            from segment_anything_hq import sam_model_registry as hq_registry  # type: ignore
            from segment_anything_hq import SamHQImagePredictor as HQPred  # type: ignore
            build_fn = hq_registry[model_type]
            try:
                sam = build_fn(checkpoint=None)
            except TypeError:
                sam = build_fn()
            import torch  # type: ignore
            state = torch.load(checkpoint, map_location='cpu')
            if isinstance(state, dict) and 'state_dict' in state:
                state = state['state_dict']
            try:
                sam.load_state_dict(state, strict=True)
            except Exception:
                sam.load_state_dict(state, strict=False)
        except Exception:
            from segment_anything_hq.build_sam_hq import build_sam_hq  # type: ignore
            from segment_anything_hq.sam_hq_image_predictor import SamHQImagePredictor as HQPred  # type: ignore
            sam = build_sam_hq(model_type, checkpoint, device='cpu')
        try:
            sam.to(device)
        except Exception:
            pass
        predictor = HQPred(sam)
    except Exception:
        predictor = None

    # Fallback: base SAM
    if predictor is None:
        try:
            from segment_anything import sam_model_registry, SamPredictor  # type: ignore
            sam = sam_model_registry[model_type](checkpoint=checkpoint)
            try:
                sam.to(device)
            except Exception:
                pass
            predictor = SamPredictor(sam)
        except Exception:
            raise RuntimeError('HQ-SAM/SAM not available. Check your environment and checkpoint path.')

    setattr(predictor, '_tc_device', device)
    cache[key] = predictor
    return predictor




def _call_to_svg_compat(contours, hierarchy, out_path, mm_per_px, margin_mm, include_holes, sort_desc, stroke_width_mm: float):
    """Call core.to_svg with backward compatibility.
    Newer versions accept stroke_only and stroke_width; older ones don't.
    """
    try:
        return core.to_svg(
            contours,
            hierarchy,
            out_path,
            mm_per_px,
            margin_mm,
            include_holes,
            sort_desc,
            stroke_only=True,
            stroke_width=stroke_width_mm,
        )
    except TypeError:
        # Fall back to older signature without styling args
        return core.to_svg(
            contours,
            hierarchy,
            out_path,
            mm_per_px,
            margin_mm,
            include_holes,
            sort_desc,
        )
# Fallbacks if tool_image_to_svg is missing newer helpers
def _local_detect_quarter_ellipse(img_bgr: np.ndarray):
    import math as _math
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    try:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
    except Exception:
        pass
    g = cv2.GaussianBlur(gray, (5, 5), 0)
    h, w = gray.shape[:2]
    minR = max(8, int(min(h, w) * 0.015))
    maxR = int(min(h, w) * 0.25)
    def refine_in_roi(x: int, y: int, r: int):
        x0 = max(0, x - int(1.5 * r)); x1 = min(w, x + int(1.5 * r))
        y0 = max(0, y - int(1.5 * r)); y1 = min(h, y + int(1.5 * r))
        roi = gray[y0:y1, x0:x1]
        med = float(np.median(roi))
        lo = int(max(0, 0.66 * med)); hi = int(min(255, 1.33 * med + 30))
        edges = cv2.Canny(roi, lo, hi)
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cnts:
            return None
        cxr = x - x0; cyr = y - y0
        best = None; best_score = -1.0
        for c in cnts:
            if len(c) < 5:
                continue
            area = cv2.contourArea(c)
            if area < 50:
                continue
            peri = cv2.arcLength(c, True)
            if peri <= 0:
                continue
            circ = 4 * _math.pi * (area / (peri * peri))
            (ecx, ecy), (MA, ma), angle = cv2.fitEllipse(c)
            if MA <= 0 or ma <= 0:
                continue
            axis_ratio = MA / ma if MA >= ma else ma / MA
            ratio_penalty = abs(1.0 - (1.0 / axis_ratio))
            pts = c.reshape(-1, 2)
            d2 = np.min((pts[:, 0] - cxr) ** 2 + (pts[:, 1] - cyr) ** 2)
            center_bonus = 1.0 / (1.0 + d2)
            score = float(circ - 0.5 * ratio_penalty + 0.2 * center_bonus)
            if score > best_score:
                best_score = score
                best = (ecx + x0, ecy + y0, MA, ma, angle)
        if best is None:
            return None
        cxo, cyo, MA, ma, angle = best
        axis_ratio = MA / ma if MA >= ma else ma / MA
        score = max(0.0, 1.0 - abs(1.0 - (1.0 / axis_ratio)))
        return (score, float(cxo), float(cyo), float(MA), float(ma), float(angle))
    best_candidate = None
    for dp in (1.2, 1.4):
        for p2 in (20, 30, 40):
            try:
                circles = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT, dp=dp, minDist=min(h, w) // 6,
                                           param1=120, param2=p2, minRadius=minR, maxRadius=maxR)
            except Exception:
                circles = None
            if circles is None:
                continue
            for (x, y, r) in np.round(circles[0, :]).astype(int):
                cand = refine_in_roi(x, y, r)
                if cand is None:
                    continue
                if (best_candidate is None) or (cand[0] > best_candidate[0]):
                    best_candidate = cand
    if best_candidate is not None:
        _, cx, cy, MA, ma, angle = best_candidate
        return float(cx), float(cy), float(MA), float(ma), float(angle)
    med = float(np.median(g))
    lo = int(max(0, 0.66 * med)); hi = int(min(255, 1.33 * med + 30))
    edges = cv2.Canny(g, lo, hi)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    best = None; best_score = -1.0
    import math as _m
    for c in cnts:
        if len(c) < 5:
            continue
        area = cv2.contourArea(c)
        if area < 100:
            continue
        peri = cv2.arcLength(c, True)
        if peri <= 0:
            continue
        circ = 4 * _m.pi * (area / (peri * peri))
        (cx, cy), (MA, ma), angle = cv2.fitEllipse(c)
        if MA <= 0 or ma <= 0:
            continue
        axis_ratio = MA / ma if MA >= ma else ma / MA
        ratio_penalty = abs(1.0 - (1.0 / axis_ratio))
        score = float(circ - 0.5 * ratio_penalty)
        if score > best_score:
            best_score = score
            best = (cx, cy, MA, ma, angle)
    if best is not None:
        cx, cy, MA, ma, angle = best
        return float(cx), float(cy), float(MA), float(ma), float(angle)
    return None


def _local_affine_rectify_ellipse_to_circle(cx: float, cy: float, MA: float, ma: float, angle_deg: float) -> np.ndarray:
    import math as _m
    a = MA / 2.0; b = ma / 2.0
    if a <= 0 or b <= 0:
        return np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    R = (a + b) / 2.0
    sx = R / a; sy = R / b
    th = _m.radians(angle_deg)
    c = _m.cos(th); s = _m.sin(th)
    T = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], dtype=np.float64)
    Rm = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)
    S = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], dtype=np.float64)
    Rp = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]], dtype=np.float64)
    Tb = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]], dtype=np.float64)
    M = Tb @ Rp @ S @ Rm @ T
    return M[:2, :].astype(np.float32)


def _refine_quarter_circle(img_bgr: np.ndarray, cx: float, cy: float, r_init: float):
    try:
        if not np.isfinite([cx, cy, r_init]).all() or r_init <= 2:
            return cx, cy, r_init
        h, w = img_bgr.shape[:2]
        band = max(6.0, r_init * 0.15)
        x0 = max(0, int(np.floor(cx - r_init - band)))
        y0 = max(0, int(np.floor(cy - r_init - band)))
        x1 = min(w, int(np.ceil(cx + r_init + band)))
        y1 = min(h, int(np.ceil(cy + r_init + band)))
        if x1 <= x0 or y1 <= y0:
            return cx, cy, r_init
        roi = img_bgr[y0:y1, x0:x1]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        med = float(np.median(gray))
        lo = int(max(0, 0.66 * med))
        hi = int(min(255, 1.33 * med + 30))
        edges = cv2.Canny(gray, lo, hi)
        ys, xs = np.nonzero(edges)
        if len(xs) < 50:
            return cx, cy, r_init
        xs = xs.astype(np.float64) + x0
        ys = ys.astype(np.float64) + y0
        d = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
        mask = np.abs(d - r_init) <= band
        xs = xs[mask]; ys = ys[mask]
        if xs.size < 30:
            return cx, cy, r_init
        if xs.size > 4000:
            import numpy as _np
            idx = _np.random.choice(xs.size, 4000, replace=False)
            xs = xs[idx]; ys = ys[idx]
        A = np.column_stack([xs, ys, np.ones_like(xs)])
        b = -(xs**2 + ys**2)
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        Aco, Bco, Cco = sol
        cx2 = -Aco / 2.0
        cy2 = -Bco / 2.0
        r2 = float(np.sqrt(max(1e-6, (Aco*Aco + Bco*Bco) / 4.0 - Cco)))
        d2 = np.sqrt((xs - cx2) ** 2 + (ys - cy2) ** 2)
        resid = np.abs(d2 - r2)
        medr = float(np.median(resid))
        thr = max(2.0, 2.5 * medr)
        inl = resid <= thr
        if inl.sum() >= 25 and inl.sum() >= xs.size * 0.3:
            xs2 = xs[inl]; ys2 = ys[inl]
            A = np.column_stack([xs2, ys2, np.ones_like(xs2)])
            b = -(xs2**2 + ys2**2)
            sol, *_ = np.linalg.lstsq(A, b, rcond=None)
            Aco, Bco, Cco = sol
            cx2 = -Aco / 2.0
            cy2 = -Bco / 2.0
            r2 = float(np.sqrt(max(1e-6, (Aco*Aco + Bco*Bco) / 4.0 - Cco)))
        if abs(r2 - r_init) > max(8.0, 0.35 * r_init):
            return cx, cy, r_init
        return float(cx2), float(cy2), float(r2)
    except Exception:
        return cx, cy, r_init

def rectify_by_quarter_if_requested(
    img: np.ndarray,
    use_quarter: bool,
    quarter_mm: float,
    quarter_roi: Optional[Dict[str, int]] = None,
    quarter_manual: Optional[Dict[str, float]] = None,
) -> Tuple[np.ndarray, Optional[float], Optional[Dict[str, float]]]:
    if not use_quarter:
        return img, None, None
    # If user provided a manual ellipse, prefer it
    if isinstance(quarter_manual, dict):
        try:
            cx = float(quarter_manual.get('cx'))
            cy = float(quarter_manual.get('cy'))
            MA = float(quarter_manual.get('MA'))
            ma = float(quarter_manual.get('ma'))
            angle = float(quarter_manual.get('angle', 0.0))
            a = MA / 2.0; b = ma / 2.0; R0 = (a + b) / 2.0
            cxr, cyr, Rr = _refine_quarter_circle(img, float(cx), float(cy), float(R0))
            R = Rr if Rr > 0 else R0
            mm_per_px = quarter_mm / (2.0 * R) if R > 0 else None
            vis = {"cx": float(cxr), "cy": float(cyr), "r": float(R)}
            return img, mm_per_px, vis
        except Exception:
            pass
    H, W = img.shape[:2]
    max_dim = max(H, W)
    det_cap = 1200
    sf = 1.0
    det_img = img
    if max_dim > det_cap:
        sf = det_cap / float(max_dim)
        det_img = cv2.resize(img, (int(W * sf), int(H * sf)), interpolation=cv2.INTER_AREA)
    # ROI-guided detection if provided
    det_for_search = det_img
    roi_off_x = 0
    roi_off_y = 0
    if isinstance(quarter_roi, dict):
        try:
            x0 = int(max(0, quarter_roi.get('x0', 0)))
            y0 = int(max(0, quarter_roi.get('y0', 0)))
            x1 = int(quarter_roi.get('x1', W))
            y1 = int(quarter_roi.get('y1', H))
            if x1 < x0:
                x0, x1 = x1, x0
            if y1 < y0:
                y0, y1 = y1, y0
            if sf != 1.0:
                x0 = int(round(x0 * sf)); y0 = int(round(y0 * sf))
                x1 = int(round(x1 * sf)); y1 = int(round(y1 * sf))
            x0 = max(0, min(x0, det_img.shape[1] - 1))
            y0 = max(0, min(y0, det_img.shape[0] - 1))
            x1 = max(x0 + 1, min(x1, det_img.shape[1]))
            y1 = max(y0 + 1, min(y1, det_img.shape[0]))
            det_for_search = det_img[y0:y1, x0:x1]
            roi_off_x, roi_off_y = x0, y0
        except Exception:
            pass
    detect_fn = getattr(core, 'detect_quarter_ellipse', None)
    if detect_fn is None:
        detect_fn = _local_detect_quarter_ellipse
    ellipse = detect_fn(det_for_search)
    if ellipse is None and det_for_search is not det_img:
        roi_off_x = 0; roi_off_y = 0
        ellipse = detect_fn(det_img)
    if ellipse is None:
        return img, None, None
    cx, cy, MA, ma, angle = ellipse
    # Account for ROI offset and scale back to original coords
    cx += roi_off_x; cy += roi_off_y
    if sf != 1.0:
        inv = 1.0 / sf
        cx *= inv; cy *= inv; MA *= inv; ma *= inv
    # Compute scale from average radius and refine with edges
    a = MA / 2.0; b = ma / 2.0; R0 = (a + b) / 2.0
    cxr, cyr, Rr = _refine_quarter_circle(img, float(cx), float(cy), float(R0))
    cx, cy, R = cxr, cyr, Rr
    mm_per_px = quarter_mm / (2.0 * R) if R > 0 else None
    vis = {"cx": float(cx), "cy": float(cy), "r": float(R)}
    return img, mm_per_px, vis


@app.route('/')
def index():
    return INDEX_HTML


@app.post('/api/preview')
def api_preview():
    data = request.json or {}
    desired_rotation = int(data.get('image_rotation', app.config.get('IMAGE_ROTATION', 0)))
    _ensure_image_orientation(desired_rotation)
    data = dict(data)
    data['image_rotation'] = int(app.config.get('IMAGE_ROTATION', 0))
    img = app.config.get('IMAGE')
    if img is None:
        # Placeholder prompting upload
        H, W = 600, 900
        overlay = np.full((H, W, 3), 245, dtype=np.uint8)
        try:
            cv2.putText(overlay, 'Drop an image here or click Upload', (36, H//2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (60,60,60), 2, cv2.LINE_AA)
        except Exception:
            pass
        mask_bgr = np.zeros_like(overlay)
        stats = {'contour_count': 0, 'total_area_px2': 0.0}
        overlay_b64 = png_bytes_to_b64uri(bgr_to_png_bytes(overlay))
        mask_b64 = png_bytes_to_b64uri(bgr_to_png_bytes(mask_bgr))
        return jsonify({'overlay_png': overlay_b64, 'mask_png': mask_b64, 'stats': stats, 'scale_down': 1.0})
    try:
        overlay, mask_bgr, sstats = run_hqsam_preview(img, data)
    except Exception as e:  # noqa: BLE001
        return jsonify({'error': f'Preview failed: {e}'}), 400
    # Derive stats from mask
    mask = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours((mask > 0).astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total_area = float(sum(cv2.contourArea(c) for c in contours))
    stats = {'contour_count': int(len(contours)), 'total_area_px2': total_area}
    # Build kept contours for SVG
    kept = []
    epsilon_frac = 0.002
    min_area = 350.0
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        eps = max(1.0, epsilon_frac * peri)
        ap = cv2.approxPolyDP(cnt, eps, True)
        if len(ap) >= 3 and cv2.contourArea(ap) >= min_area:
            kept.append(ap)
    # Generate inline SVG preview
    svg_data_uri = None
    if kept:
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.svg', delete=False) as tmp:
            tmp_path = tmp.name
        try:
            mm_per_px = (sstats or {}).get('mm_per_px')
            margin_mm = 5.0
            stroke_mm = 1.4
            if mm_per_px is None:
                px_per_mm = 3.7795275591
                stroke_width = stroke_mm * px_per_mm
            else:
                stroke_width = stroke_mm
            include_holes = False
            sort_desc = True
            _call_to_svg_compat(kept, hierarchy, tmp_path, mm_per_px, margin_mm, include_holes, sort_desc, stroke_width)
            with open(tmp_path, 'rb') as f:
                svg_bytes = f.read()
            svg_b64 = base64.b64encode(svg_bytes).decode('ascii')
            svg_data_uri = f"data:image/svg+xml;base64,{svg_b64}"
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
    # Resize preview if large
    max_w = 1000
    scale = 1.0
    if overlay.shape[1] > max_w:
        scale = max_w / overlay.shape[1]
        overlay = cv2.resize(overlay, (int(overlay.shape[1] * scale), int(overlay.shape[0] * scale)))
        mask_bgr = cv2.resize(mask_bgr, (overlay.shape[1], overlay.shape[0]))

    overlay_b64 = png_bytes_to_b64uri(bgr_to_png_bytes(overlay))
    mask_b64 = png_bytes_to_b64uri(bgr_to_png_bytes(mask_bgr))
    resp = {
        'overlay_png': overlay_b64,
        'mask_png': mask_b64,
        'stats': stats,
        'scale_down': scale,
    }
    if svg_data_uri is not None:
        resp['svg_data_uri'] = svg_data_uri
    return jsonify(resp)


@app.post('/api/upload_image')
def api_upload_image():
    try:
        f = request.files.get('image')
        if f is None:
            return jsonify({'error': 'No file part named "image"'}), 400
        data = f.read()
        if not data:
            return jsonify({'error': 'Empty file'}), 400
        img, converted = decode_image_bytes_to_bgr(data)
        if img is None:
            return jsonify({'error': 'Unsupported image format'}), 400
        auto_rotation = 0
        try:
            if img.shape[1] < img.shape[0]:
                img = rotate_bgr_image(img, 90)
                auto_rotation = 90
        except Exception:
            auto_rotation = 0
        app.config['IMAGE'] = img
        app.config['IMAGE_BASE_SHAPE'] = (int(img.shape[0]), int(img.shape[1]))
        app.config['IMAGE_ROTATION'] = auto_rotation
        app.config.pop('RAW_MASK', None)
        resp: Dict[str, Any] = {
            'ok': True,
            'shape': [int(img.shape[0]), int(img.shape[1])],
            'auto_rotation_deg': auto_rotation,
        }
        if converted:
            resp['converted_to'] = 'jpg'
        return jsonify(resp), 200
    except Exception as e:
        return jsonify({'error': f'Upload failed: {e}'}), 400

@app.post('/api/export_svg')
def api_export_svg():
    data = request.json or {}
    desired_rotation = int(data.get('image_rotation', app.config.get('IMAGE_ROTATION', 0)))
    _ensure_image_orientation(desired_rotation)
    data = dict(data)
    data['image_rotation'] = int(app.config.get('IMAGE_ROTATION', 0))
    img = app.config.get('IMAGE')
    if img is None:
        return jsonify({'error': 'No image loaded. Upload or drop an image first.'}), 400
    try:
        export_data = dict(data)
        export_data['compute_mask'] = True
        overlay, mask_bgr, sstats = run_hqsam_preview(img, export_data)
    except Exception as e:  # noqa: BLE001
        return jsonify({'error': f'Export failed: {e}'}), 400
    # Build contours from mask_bgr
    mask = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours((mask > 0).astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    kept = []
    epsilon_frac = 0.002
    min_area = 350.0
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        eps = max(1.0, epsilon_frac * peri)
        ap = cv2.approxPolyDP(cnt, eps, True)
        if len(ap) >= 3 and cv2.contourArea(ap) >= min_area:
            kept.append(ap)
    if not kept:
        return jsonify({'error': 'No contours found with current mask'}), 400
    mm_per_px = (sstats or {}).get('mm_per_px')
    with tempfile.NamedTemporaryFile(suffix='.svg', delete=False) as tmp:
        tmp_path = tmp.name
    try:
        margin_mm = 5.0
        stroke_mm = 1.4
        if mm_per_px is None:
            px_per_mm = 3.7795275591
            stroke_width = stroke_mm * px_per_mm
        else:
            stroke_width = stroke_mm
        include_holes = False
        sort_desc = True
        _call_to_svg_compat(kept, hierarchy, tmp_path, mm_per_px, margin_mm, include_holes, sort_desc, stroke_width)
        return send_file(tmp_path, mimetype='image/svg+xml', as_attachment=True, download_name='tool_cutouts.svg')
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def main():
    ap = argparse.ArgumentParser(description='Run the ToolCutter segmentation API server')
    ap.add_argument('--image', '-i', required=False, default=None, help='Path to overhead tool image (optional)')
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=8000)
    ap.add_argument('--debug', action='store_true')
    args = ap.parse_args()

    if args.image:
        try:
            img, converted = load_image_from_path(args.image)
        except RuntimeError as exc:
            raise SystemExit(str(exc))
        app.config['IMAGE'] = img
        if converted:
            app.logger.info('Converted %s to JPEG for processing', args.image)
    app.run(host=args.host, port=args.port, debug=args.debug)


def run_hqsam_preview(base_img: np.ndarray, data: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    rotation = int(data.get('image_rotation', app.config.get('IMAGE_ROTATION', 0))) % 360
    working_img = base_img.copy()

    use_quarter = bool(data.get('use_quarter', False))
    quarter_mm = float(data.get('quarter_diameter_mm', 24.26))
    quarter_roi = data.get('quarter_roi') if isinstance(data.get('quarter_roi'), dict) else None
    quarter_manual = data.get('quarter_manual') if isinstance(data.get('quarter_manual'), dict) else None
    img, mm_per_px, vis = rectify_by_quarter_if_requested(working_img, use_quarter, quarter_mm, quarter_roi, quarter_manual)
    full_frame = img.copy()  # keep reference to the rectified image before any cropping
    crop_bounds: Optional[Tuple[int, int, int, int]] = None

    # Optional crop
    crop_rect = data.get('crop_rect')
    if isinstance(crop_rect, dict):
        try:
            x0 = int(max(0, crop_rect.get('x0', 0)))
            y0 = int(max(0, crop_rect.get('y0', 0)))
            x1 = int(crop_rect.get('x1', img.shape[1]))
            y1 = int(crop_rect.get('y1', img.shape[0]))
            if x1 < x0:
                x0, x1 = x1, x0
            if y1 < y0:
                y0, y1 = y1, y0
            x0 = max(0, min(x0, img.shape[1] - 1))
            y0 = max(0, min(y0, img.shape[0] - 1))
            x1 = max(x0 + 1, min(x1, img.shape[1]))
            y1 = max(y0 + 1, min(y1, img.shape[0]))
            crop_bounds = (x0, y0, x1, y1)
            img = img[y0:y1, x0:x1].copy()
            if vis is not None:
                vis = {"cx": float(vis["cx"]) - float(x0), "cy": float(vis["cy"]) - float(y0), "r": float(vis.get("r", 0.0))}
        except Exception:
            pass

    compute = bool(data.get('compute_mask', True))
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    if not compute:
        raw_prev = app.config.get('RAW_MASK')
        if isinstance(raw_prev, np.ndarray) and raw_prev.shape[:2] == img.shape[:2]:
            mask = raw_prev.copy()
    if compute:
        ckpt = data.get('hqsam_checkpoint', '')
        model_type = data.get('hqsam_model_type', 'vit_h')
        if not ckpt:
            ckpt = os.environ.get('HQSAM_CKPT', '') or '/Users/NolanMorrow/Programming/ToolCutter/backend/sam_hq_vit_h.pth'

        H0, W0 = img.shape[:2]
        max_side = float(data.get('sam_max_side', 640.0))
        max_side = max(512.0, min(max_side, 2048.0))
        scale_factor = 1.0
        work_img = img
        if max(H0, W0) > max_side:
            scale_factor = max_side / float(max(H0, W0))
            new_w = max(1, int(round(W0 * scale_factor)))
            new_h = max(1, int(round(H0 * scale_factor)))
            work_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        img_rgb = cv2.cvtColor(work_img, cv2.COLOR_BGR2RGB)

        def infer_with_predictor(pred) -> np.ndarray:
            try:
                import torch  # type: ignore
                ctx = torch.inference_mode
            except Exception:
                ctx = None
            context = ctx() if callable(ctx) else nullcontext()
            with context:
                pred.set_image(img_rgb)
                points = data.get('sam_points', [])
                sam_auto = bool(data.get('sam_auto', False))
                local_mask = np.zeros(work_img.shape[:2], dtype=np.uint8)
                if points:
                    pts = np.array([[p['x'], p['y']] for p in points], dtype=np.float32)
                    if scale_factor != 1.0:
                        pts *= scale_factor
                    labs = np.array([1 if str(p.get('label','pos'))=='pos' else 0 for p in points], dtype=np.int32)
                    multimask = bool(data.get('sam_multimask', True))
                    union_masks = bool(data.get('sam_union', True))
                    masks, scores, _ = pred.predict(point_coords=pts, point_labels=labs, multimask_output=multimask)
                    if masks is not None and len(masks) > 0:
                        if union_masks and multimask:
                            local_mask = (np.any(masks > 0, axis=0)).astype(np.uint8) * 255
                        else:
                            k = int(np.argmax(scores)) if scores is not None else 0
                            local_mask = (masks[k] > 0).astype(np.uint8) * 255
                elif sam_auto:
                    H, W = work_img.shape[:2]
                    box = np.array([0, 0, W-1, H-1], dtype=np.float32)
                    multimask = bool(data.get('sam_multimask', True))
                    union_masks = bool(data.get('sam_union', True))
                    try:
                        masks, scores, _ = pred.predict(point_coords=None, point_labels=None, box=box, multimask_output=multimask)
                    except TypeError:
                        masks, scores, _ = pred.predict(point_coords=None, point_labels=None, box=box[None, :], multimask_output=multimask)
                    if masks is not None and len(masks) > 0:
                        if union_masks and multimask:
                            local_mask = (np.any(masks > 0, axis=0)).astype(np.uint8) * 255
                        else:
                            k = int(np.argmax(scores)) if scores is not None else 0
                            local_mask = (masks[k] > 0).astype(np.uint8) * 255
                if scale_factor != 1.0 and local_mask.size:
                    local_mask = cv2.resize(local_mask, (W0, H0), interpolation=cv2.INTER_NEAREST)
                return local_mask

        predictor = get_hqsam_predictor(ckpt, model_type)
        try:
            mask = infer_with_predictor(predictor)
        except RuntimeError as exc:
            msg = str(exc).lower()
            retriable = any(token in msg for token in ('mps', 'metal', 'command buffer', 'outofmemory'))
            if not retriable:
                raise
            # MPS occasionally runs out of memory on large crops; fall back to CPU for reliability.
            app.logger.warning('HQ-SAM predictor on %s failed (%s). Retrying on CPU.', getattr(predictor, '_tc_device', 'unknown'), exc)
            cache = app.config.get('HQSAM_CACHE', {})
            pred_device = getattr(predictor, '_tc_device', None)
            if pred_device is not None:
                cache.pop((ckpt, model_type, pred_device), None)
            predictor = get_hqsam_predictor(ckpt, model_type, force_device='cpu')
            mask = infer_with_predictor(predictor)
        app.config['RAW_MASK'] = mask.copy()
        # Post-process (fixed defaults)
        thr = 45
        blur = 1
        dil = 4
        ero = 2
        if blur > 0:
            k = max(1, int(blur) * 2 + 1)
            mask = cv2.GaussianBlur(mask, (k, k), 0)
            _, mask = cv2.threshold(mask, thr, 255, cv2.THRESH_BINARY)
        if dil > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dil, dil))
            mask = cv2.dilate(mask, k, iterations=1)
        if ero > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ero, ero))
            mask = cv2.erode(mask, k, iterations=1)

    overlay = img.copy()
    if np.any(mask > 0):
        color = (0, 160, 255)
        alpha = 0.5
        color_img = np.zeros_like(overlay); color_img[:] = color
        overlay = np.where((mask > 0)[..., None], (overlay * (1 - alpha) + color_img * alpha).astype(np.uint8), overlay)
    if vis is not None:
        cx_i, cy_i, rr = int(vis['cx']), int(vis['cy']), int(max(1, vis.get('r', 0)))
        cv2.circle(overlay, (cx_i, cy_i), rr, (0, 255, 0), 3)
        cv2.circle(overlay, (cx_i, cy_i), 3, (0, 255, 0), -1)
        cv2.line(overlay, (cx_i - 12, cy_i), (cx_i + 12, cy_i), (0, 255, 0), 2)
        cv2.line(overlay, (cx_i, cy_i - 12), (cx_i, cy_i + 12), (0, 255, 0), 2)
        try:
            cv2.putText(overlay, 'Quarter', (cx_i + rr + 8, max(15, cy_i - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        except Exception:
            pass

    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    if crop_bounds is not None:
        x0, y0, x1, y1 = crop_bounds
        base_overlay = full_frame.copy()
        base_overlay[y0:y1, x0:x1] = overlay
        overlay = base_overlay
        full_mask = np.zeros_like(base_overlay)
        full_mask[y0:y1, x0:x1] = mask_bgr
        mask_bgr = full_mask
        if vis is not None:
            vis = {
                "cx": float(vis["cx"]) + float(x0),
                "cy": float(vis["cy"]) + float(y0),
                "r": float(vis.get("r", 0.0)),
            }

    stats = {
        'mm_per_px': mm_per_px,
        'quarter_vis': vis,
        'quarter_found': bool(vis is not None),
        'image_rotation': rotation,
    }
    return overlay, mask_bgr, stats

@app.get('/health')
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    main()
