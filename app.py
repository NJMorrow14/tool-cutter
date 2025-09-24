#!/usr/bin/env python3
"""
Interactive web UI to segment tools using Meta's HQ-SAM (high-quality SAM) with point prompts,
preview in real-time, optionally scale/rectify using a US quarter, crop the image, and export SVG cutouts.

Endpoints:
 - GET /              -> HTML page with controls and live preview
 - POST /api/preview  -> JSON with base64 PNGs for overlay and mask + stats
 - POST /api/export_svg -> generates SVG and serves it for download

Run:
  python app.py --image ./images/overhead_aruco.jpg --host 127.0.0.1 --port 8000

Dependencies:
  pip install flask numpy opencv-python svgwrite
"""
from __future__ import annotations

import argparse
import base64
import io
import os
import subprocess
import tempfile
from typing import Any, Dict, Tuple, List, Optional

from flask import Flask, jsonify, request, send_file
import cv2
import numpy as np

import tool_image_to_svg as core

app = Flask(__name__)


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





def get_hqsam_predictor(checkpoint: str, model_type: str):
    """Load and cache a predictor for SAM/HQ-SAM. Tries HQ-SAM, then base SAM."""
    key = (checkpoint, model_type)
    cache = app.config.setdefault('HQSAM_CACHE', {})
    if key in cache:
        return cache[key]
    # Select device
    try:
        import torch  # type: ignore
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        elif torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
    except Exception:
        device = 'cpu'

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
        app.config['IMAGE'] = img
        app.config.pop('RAW_MASK', None)
        resp: Dict[str, Any] = {'ok': True, 'shape': [int(img.shape[0]), int(img.shape[1])]}
        if converted:
            resp['converted_to'] = 'jpg'
        return jsonify(resp), 200
    except Exception as e:
        return jsonify({'error': f'Upload failed: {e}'}), 400

@app.post('/api/export_svg')
def api_export_svg():
    data = request.json or {}
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
    import tempfile
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
    ap = argparse.ArgumentParser(description='Run the interactive parameter tuner web app')
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
    use_quarter = bool(data.get('use_quarter', False))
    quarter_mm = float(data.get('quarter_diameter_mm', 24.26))
    quarter_roi = data.get('quarter_roi') if isinstance(data.get('quarter_roi'), dict) else None
    quarter_manual = data.get('quarter_manual') if isinstance(data.get('quarter_manual'), dict) else None
    img, mm_per_px, vis = rectify_by_quarter_if_requested(base_img, use_quarter, quarter_mm, quarter_roi, quarter_manual)

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
            ckpt = os.environ.get('HQSAM_CKPT', '') or '/Users/NolanMorrow/Programming/ToolCutter/sam_hq_vit_h.pth'
        predictor = get_hqsam_predictor(ckpt, model_type)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        predictor.set_image(img_rgb)
        points = data.get('sam_points', [])
        sam_auto = bool(data.get('sam_auto', False))
        if points:
            pts = np.array([[p['x'], p['y']] for p in points], dtype=np.float32)
            labs = np.array([1 if str(p.get('label','pos'))=='pos' else 0 for p in points], dtype=np.int32)
            multimask = bool(data.get('sam_multimask', True))
            union_masks = bool(data.get('sam_union', True))
            masks, scores, _ = predictor.predict(point_coords=pts, point_labels=labs, multimask_output=multimask)
            if masks is not None and len(masks) > 0:
                if union_masks and multimask:
                    mask = (np.any(masks > 0, axis=0)).astype(np.uint8) * 255
                else:
                    k = int(np.argmax(scores)) if scores is not None else 0
                    mask = (masks[k] > 0).astype(np.uint8) * 255
        elif sam_auto:
            H, W = img.shape[:2]
            box = np.array([0, 0, W-1, H-1], dtype=np.float32)
            multimask = bool(data.get('sam_multimask', True))
            union_masks = bool(data.get('sam_union', True))
            try:
                masks, scores, _ = predictor.predict(point_coords=None, point_labels=None, box=box, multimask_output=multimask)
            except TypeError:
                masks, scores, _ = predictor.predict(point_coords=None, point_labels=None, box=box[None, :], multimask_output=multimask)
            if masks is not None and len(masks) > 0:
                if union_masks and multimask:
                    mask = (np.any(masks > 0, axis=0)).astype(np.uint8) * 255
                else:
                    k = int(np.argmax(scores)) if scores is not None else 0
                    mask = (masks[k] > 0).astype(np.uint8) * 255
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

    stats = {
        'mm_per_px': mm_per_px,
        'quarter_vis': vis,
        'quarter_found': bool(vis is not None),
    }
    return overlay, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), stats




INDEX_HTML = """
<!DOCTYPE html>
<meta charset="utf-8">
<title>Tool Cutout Tuner</title>
<style>
  body{font-family:system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin:16px;}
  .grid{display:grid; grid-template-columns: 360px 1fr; gap:16px;}
  fieldset{border:1px solid #ddd; border-radius:8px; padding:12px; margin-bottom:12px;}
  legend{font-weight:600;}
  .row{display:flex; align-items:center; gap:8px; margin:6px 0;}
  label{width:140px; font-size:13px; color:#333;}
  input[type=range]{width:100%;}
  .preview{display:flex; gap:12px; align-items:flex-start;}
  .panel{border:1px solid #ddd; border-radius:8px; padding:8px;}
  img{max-width:100%; height:auto;}
  .btns{display:flex; gap:8px; margin-top:8px;}
  .stat{font-size:12px; color:#444;}
  .pill{padding:2px 6px; background:#eee; border-radius:999px; font-size:12px;}
</style>
<div class="grid">
  <div>
    <fieldset>
      <legend>View</legend>
      <div class="row">
        <label>Crop</label>
        <button id="crop_reset" style="margin-left:8px;">Reset crop</button>
        <span class="stat">Drag on the image to crop.</span>
      </div>
      <div class="row"><label>Zoom</label>
        <input type="range" id="zoom" min="0.5" max="4" step="0.1" value="1">
      </div>
    </fieldset>
    <fieldset>
      <legend>Image</legend>
      <div class="row"><label>Upload</label>
        <input type="file" id="file_input" accept="image/*">
      </div>
      <div class="row"><span class="stat">Tip: drag & drop onto the overlay</span></div>
    </fieldset>
    <fieldset>
      <legend>Segmentation</legend>
      <div class="row"><label>Quarter scale</label>
        <input type="checkbox" id="use_quarter">
      </div>
      <div class="row"><label>Mark quarter</label>
        <input type="checkbox" id="qmark_mode"> <button id="qmark_clear" style="margin-left:8px;">Clear</button>
      </div>
      <div class="row"><label>Manual quarter</label>
        <input type="checkbox" id="qmanual_mode"> <button id="qmanual_clear" style="margin-left:8px;">Clear</button>
        <span class="stat">Click center, drag to size (hold Shift for ellipse)</span>
      </div>
    </fieldset>
    <fieldset id="sam_group">
      <legend>HQ-SAM</legend>
      <div class="row"><label>Checkpoint</label>
        <input id="hqsam_checkpoint" value="/Users/NolanMorrow/Programming/ToolCutter/sam_hq_vit_h.pth" placeholder="/path/to/sam_hq_vit_h.pth" style="width:100%"/>
      </div>
      <div class="row"><label>Model</label>
        <select id="hqsam_model_type">
          <option value="vit_h" selected>vit_h</option>
          <option value="vit_l">vit_l</option>
          <option value="vit_b">vit_b</option>
        </select>
      </div>
      <div class="row"><label>Multimask</label>
        <input type="checkbox" id="sam_multimask" checked>
      </div>
      <div class="row"><label>Union masks</label>
        <input type="checkbox" id="sam_union" checked>
      </div>
      <div class="row"><label>Auto (no clicks)</label>
        <input type="checkbox" id="sam_auto" checked>
      </div>
      <div class="row"><label>Click label</label>
        <select id="sam_label">
          <option value="pos" selected>positive</option>
          <option value="neg">negative</option>
        </select>
      </div>
      <div class="row"><button id="sam_undo">Undo last point</button><button id="sam_clear">Clear points</button></div>
    </fieldset>
    <div class="btns">
      <button id="compute">Compute mask</button>
      <button id="download_svg">Download SVG</button>
    </div>
    <div class="stat" id="stats"></div>
  </div>
  <div>
    <div class="preview" style="flex-direction:column; align-items:stretch;">
      <div class="panel" id="overlay_panel" style="position:relative; overflow:auto"><div>Overlay</div>
        <div id="loading" style="display:none;position:absolute;right:8px;top:8px;background:rgba(255,255,255,0.85);padding:6px 10px;border-radius:6px;border:1px solid #ccc;">Loading…</div>
        <div id="drop_hint" style="display:none;position:absolute;left:0;top:0;right:0;bottom:0;background:rgba(0,0,0,0.35);color:#fff;align-items:center;justify-content:center;font-weight:600;font-size:18px;">Drop image to load</div>
        <img id="overlay" alt="overlay" style="cursor: crosshair; display:block; max-width:none;">
      </div>
            <div class="panel" id="svg_panel" style="display:none"><div>SVG</div><img id="svg_img" alt="svg preview"></div>
    </div>
  </div>
</div>

<script>
const els = {
  eps: document.getElementById('epsilon_frac'),
  area: document.getElementById('min_area'),
  sam_group: document.getElementById('sam_group'),
  hqsam_checkpoint: document.getElementById('hqsam_checkpoint'),
  hqsam_model_type: document.getElementById('hqsam_model_type'),
  sam_multimask: document.getElementById('sam_multimask'),
  sam_union: document.getElementById('sam_union'),
  sam_auto: document.getElementById('sam_auto'),
  sam_label: document.getElementById('sam_label'),
  sam_undo: document.getElementById('sam_undo'),
  sam_clear: document.getElementById('sam_clear'),
  overlay: document.getElementById('overlay'),
  file_input: document.getElementById('file_input'),
  overlay_panel: document.getElementById('overlay_panel'),
  drop_hint: document.getElementById('drop_hint'),
  qmark_mode: document.getElementById('qmark_mode'),
  qmark_clear: document.getElementById('qmark_clear'),
  qmanual_mode: document.getElementById('qmanual_mode'),
  qmanual_clear: document.getElementById('qmanual_clear'),
  overlay_points: null,
  svg_img: document.getElementById('svg_img'),
  svg_panel: document.getElementById('svg_panel'),
  stats: document.getElementById('stats'),
  use_quarter: document.getElementById('use_quarter'),
  loading: document.getElementById('loading'),
  zoom: document.getElementById('zoom'),
  stroke_mm: document.getElementById('stroke_mm'),
  mask_blur: document.getElementById('mask_blur'),
  mask_dilate: document.getElementById('mask_dilate'),
  mask_erode: document.getElementById('mask_erode'),
  mask_threshold: document.getElementById('mask_threshold'),
  crop_mode: document.getElementById('crop_mode'),
  crop_reset: document.getElementById('crop_reset'),
};
const labels = {};

// Fixed params are used server-side; no sliders here.

(() => {
  const img = els.overlay;
  const panel = img && img.parentElement;
  if(!img || !panel) return;
  const c = document.createElement('canvas');
  c.id = 'overlay_points';
  c.style.position = 'absolute';
  c.style.left = '0'; c.style.top='0';
  c.style.pointerEvents = 'none';
  panel.appendChild(c);
  els.overlay_points = c;
})();

function uiToParams(){
  const p = {};
  p.backend = 'hqsam';
  p.use_quarter = els.use_quarter.checked;
  p.quarter_diameter_mm = 24.26;
  p.compute_mask = computeFlag;
  if(currentCrop){
    p.crop_rect = { x0: currentCrop.x0|0, y0: currentCrop.y0|0, x1: currentCrop.x1|0, y1: currentCrop.y1|0 };
  }
  if(quarterROI){
    p.quarter_roi = { x0: quarterROI.x0|0, y0: quarterROI.y0|0, x1: quarterROI.x1|0, y1: quarterROI.y1|0 };
  }
  if(quarterManual){
    p.quarter_manual = { cx: quarterManual.cx|0, cy: quarterManual.cy|0, MA: quarterManual.MA|0, ma: quarterManual.ma|0, angle: +quarterManual.angle || 0 };
  }
  const rect = els.overlay.getBoundingClientRect();
  const scaleDisplayToImg = (els.overlay.naturalWidth > 0 && rect.width > 0)
    ? (els.overlay.naturalWidth / rect.width) / currentScaleDown
    : (1.0 / currentScaleDown);
  p.sam_points = imgPoints.map(pt=>({x: pt.x, y: pt.y, label: pt.label}));
  p.hqsam_checkpoint = (els.hqsam_checkpoint && els.hqsam_checkpoint.value) || '';
  p.hqsam_model_type = (els.hqsam_model_type && els.hqsam_model_type.value) || 'vit_h';
  p.sam_multimask = !!(els.sam_multimask && els.sam_multimask.checked);
  p.sam_union = !!(els.sam_union && els.sam_union.checked);
  p.sam_auto = !!(els.sam_auto && els.sam_auto.checked);
  return p;
}

function setModeVisibility(){
  if(els.sam_group){ els.sam_group.style.display = ''; }
  if(els.overlay_points){ els.overlay_points.style.display = ''; }
  if(els.svg_panel){ els.svg_panel.style.display = ''; }
}

let pending = null;
let currentScaleDown = 1.0;
let computeFlag = false;
let imgPoints = [];
let currentCrop = null;
let dragCrop = null;
let quarterROI = null;
let quarterDrag = null;
let quarterManual = null; // {cx, cy, MA, ma, angle} in image coords
let quarterManualDrag = null; // {cx, cy, x, y, shift} during drag in display coords
let quarterManualMove = null; // {startX, startY, origCx, origCy, prev}
let actions = []; // undo stack

function isPointInsideManualEllipse(xd, yd){
  if(!quarterManual) return false;
  const imgEl2 = els.overlay;
  const rect2 = imgEl2.getBoundingClientRect();
  const scaleDisplayToImg2 = (imgEl2.naturalWidth > 0 && rect2.width > 0)
    ? (imgEl2.naturalWidth / rect2.width) / currentScaleDown
    : (1.0 / currentScaleDown);
  const sc = 1.0 / scaleDisplayToImg2;
  let cx = quarterManual.cx * sc; let cy = quarterManual.cy * sc;
  let rx = (quarterManual.MA * 0.5) * sc; let ry = (quarterManual.ma * 0.5) * sc;
  let ang = (quarterManual.angle||0) * Math.PI/180.0;
  if(currentCrop){ cx -= currentCrop.x0 * sc; cy -= currentCrop.y0 * sc; }
  if(rx < 1 || ry < 1) return false;
  const dx = xd - cx, dy = yd - cy;
  const ca = Math.cos(-ang), sa = Math.sin(-ang);
  const xr = dx * ca - dy * sa;
  const yr = dx * sa + dy * ca;
  const val = (xr*xr)/(rx*rx) + (yr*yr)/(ry*ry);
  return val <= 1.0;
}

function undoLast(){
  if(actions.length === 0) return;
  const a = actions.pop();
  switch(a.type){
    case 'point_add':
      imgPoints.pop();
      drawPoints();
      break;
    case 'crop':
      currentCrop = a.prev ? { ...a.prev } : null;
      imgPoints = [];
      refresh();
      break;
    case 'quarter_roi':
      quarterROI = a.prev ? { ...a.prev } : null;
      refresh();
      break;
    case 'quarter_manual':
      quarterManual = a.prev ? { ...a.prev } : null;
      refresh();
      break;
    default:
      break;
  }
}

function syncCanvas(){
  const img = els.overlay;
  const c = els.overlay_points;
  if(!img || !c) return;
  c.style.left = (img.offsetLeft || 0) + 'px';
  c.style.top = (img.offsetTop || 0) + 'px';
  const rct = img.getBoundingClientRect();
  c.width = Math.max(1, Math.floor(rct.width));
  c.height = Math.max(1, Math.floor(rct.height));
}

function drawPoints(){
  const c = els.overlay_points; if(!c) return;
  const ctx = c.getContext('2d'); if(!ctx) return;
  ctx.clearRect(0,0,c.width,c.height);
  const imgEl = els.overlay;
  const rect = imgEl.getBoundingClientRect();
  const scaleDisplayToImg = (imgEl.naturalWidth > 0 && rect.width > 0)
    ? (imgEl.naturalWidth / rect.width) / currentScaleDown
    : (1.0 / currentScaleDown);
  const scaleImgToDisplay = 1.0 / scaleDisplayToImg;
  for(const pt of imgPoints){
    const col = pt.label === 'neg' ? '#ff3333' : '#00cc66';
    ctx.strokeStyle = col;
    ctx.fillStyle = col;
    const dx = pt.x * scaleImgToDisplay;
    const dy = pt.y * scaleImgToDisplay;
    ctx.beginPath(); ctx.arc(dx, dy, 5, 0, Math.PI*2); ctx.fill();
    ctx.beginPath(); ctx.arc(dx, dy, 10, 0, Math.PI*2); ctx.lineWidth = 2; ctx.stroke();
  }
  if(dragCrop){
    const {x0,y0,x1,y1} = dragCrop;
    const ctx2 = ctx;
    ctx2.save();
    ctx2.strokeStyle = '#3366ff';
    ctx2.setLineDash([6,4]);
    ctx2.lineWidth = 2;
    ctx2.strokeRect(Math.min(x0,x1), Math.min(y0,y1), Math.abs(x1-x0), Math.abs(y1-y0));
    ctx2.restore();
  }
  // Draw manual quarter ellipse (persistent)
  if(quarterManual){
    const imgEl2 = els.overlay;
    const rect2 = imgEl2.getBoundingClientRect();
    const scaleDisplayToImg2 = (imgEl2.naturalWidth > 0 && rect2.width > 0)
      ? (imgEl2.naturalWidth / rect2.width) / currentScaleDown
      : (1.0 / currentScaleDown);
    const sc = 1.0 / scaleDisplayToImg2;
    let cx = quarterManual.cx * sc; let cy = quarterManual.cy * sc;
    let rx = (quarterManual.MA * 0.5) * sc; let ry = (quarterManual.ma * 0.5) * sc;
    let ang = (quarterManual.angle||0) * Math.PI/180.0;
    if(currentCrop){ cx -= currentCrop.x0 * sc; cy -= currentCrop.y0 * sc; }
    const c3 = ctx; c3.save(); c3.strokeStyle = '#00aa88'; c3.lineWidth = 2;
    if(c3.ellipse){ c3.beginPath(); c3.ellipse(cx, cy, Math.max(1,rx), Math.max(1,ry), ang, 0, Math.PI*2); c3.stroke(); }
    c3.restore();
  }
  // Draw manual quarter drag preview
  if(quarterManualDrag){
    const {cx, cy, x, y, shift} = quarterManualDrag;
    const dx = x - cx, dy = y - cy;
    let rx, ry, ang;
    if(shift){ rx = Math.abs(dx); ry = Math.abs(dy); ang = Math.atan2(dy, dx); }
    else { const r = Math.hypot(dx,dy); rx = r; ry = r; ang = 0; }
    const c3 = ctx; c3.save(); c3.strokeStyle = '#00aa88'; c3.setLineDash([5,3]); c3.lineWidth = 2;
    if(c3.ellipse){ c3.beginPath(); c3.ellipse(cx, cy, Math.max(1,rx), Math.max(1,ry), ang, 0, Math.PI*2); c3.stroke(); }
    c3.restore();
  }
  // Draw persistent quarter ROI (orange)
  if(quarterROI){
    const imgEl2 = els.overlay;
    const rect2 = imgEl2.getBoundingClientRect();
    const scaleDisplayToImg2 = (imgEl2.naturalWidth > 0 && rect2.width > 0)
      ? (imgEl2.naturalWidth / rect2.width) / currentScaleDown
      : (1.0 / currentScaleDown);
    const sc = 1.0 / scaleDisplayToImg2;
    let x0 = quarterROI.x0 * sc, y0 = quarterROI.y0 * sc;
    let x1 = quarterROI.x1 * sc, y1 = quarterROI.y1 * sc;
    if(currentCrop){ x0 -= currentCrop.x0 * sc; x1 -= currentCrop.x0 * sc; y0 -= currentCrop.y0 * sc; y1 -= currentCrop.y0 * sc; }
    ctx.save(); ctx.strokeStyle = '#ff8800'; ctx.setLineDash([4,3]); ctx.lineWidth = 2;
    ctx.strokeRect(Math.min(x0,x1), Math.min(y0,y1), Math.abs(x1-x0), Math.abs(y1-y0));
    ctx.restore();
  }
  // Draw quarter drag rectangle (orange)
  if(quarterDrag){
    const {x0,y0,x1,y1} = quarterDrag;
    ctx.save(); ctx.strokeStyle = '#ff8800'; ctx.setLineDash([4,3]); ctx.lineWidth = 2;
    ctx.strokeRect(Math.min(x0,x1), Math.min(y0,y1), Math.abs(x1-x0), Math.abs(y1-y0));
    ctx.restore();
  }
}


function applyZoom(){
  const z = (els.zoom && parseFloat(els.zoom.value)) || 1.0;
  const img = els.overlay;
  if(!img) return;
  const natW = img.naturalWidth || 0;
  if(natW > 0){ img.style.width = (natW * z) + 'px'; }
  syncCanvas(); drawPoints();
}

async function refresh(){
  if(pending) clearTimeout(pending);
  pending = setTimeout(async () => {
    try {
      if(els.loading) els.loading.style.display = computeFlag ? 'block' : 'none';
      const resp = await fetch('/api/preview', {
        method: 'POST', headers: { 'Content-Type':'application/json' }, body: JSON.stringify(uiToParams())
      });
      const data = await resp.json();
      if(data.error){ throw new Error(data.error); }
      els.overlay.onload = ()=>{ applyZoom(); };
      els.overlay.src = data.overlay_png;
      if(els.svg_img){
        if(data.svg_data_uri){
          els.svg_img.src = data.svg_data_uri;
          if(els.svg_panel) els.svg_panel.style.display = '';
        } else {
          if(els.svg_panel) els.svg_panel.style.display = 'none';
        }
      }
      const s = data.stats;
      let qst = (s && s.quarter_found === true) ? 'found' : (els.use_quarter && els.use_quarter.checked ? 'not found' : 'off');
      els.stats.textContent = `Contours: ${s.contour_count} | Total area (px^2): ${s.total_area_px2.toFixed(0)} | Quarter: ${qst}`;
      currentScaleDown = data.scale_down || 1.0;
    } catch (err) {
      console.error(err);
      alert('Preview failed: ' + (err && err.message ? err.message : err));
    } finally {
      if(els.loading) els.loading.style.display = 'none';
      computeFlag = false;
    }
  }, 80);
}

function attach(){
  // No polygon/mask slider bindings
  els.use_quarter.addEventListener('change', ()=>{ /* toggle only, compute on button */ });
  if(els.hqsam_checkpoint){ els.hqsam_checkpoint.addEventListener('change', ()=>{ /* wait for compute */ }); }
  if(els.qmanual_clear){ els.qmanual_clear.addEventListener('click', ()=>{ const prev = quarterManual ? { ...quarterManual } : null; quarterManual = null; quarterManualDrag = null; actions.push({type:'quarter_manual', prev}); refresh(); }); }
  if(els.qmark_clear){ els.qmark_clear.addEventListener('click', ()=>{ const prev = quarterROI ? { ...quarterROI } : null; quarterROI = null; actions.push({type:'quarter_roi', prev}); refresh(); }); }
  if(els.hqsam_model_type){ els.hqsam_model_type.addEventListener('change', ()=>{ /* wait for compute */ }); }
  if(els.sam_multimask){ els.sam_multimask.addEventListener('change', ()=>{ /* wait for compute */ }); }
  if(els.sam_union){ els.sam_union.addEventListener('change', ()=>{ /* wait for compute */ }); }
  const computeBtn = document.getElementById('compute');
  computeBtn.addEventListener('click', ()=>{ computeFlag = true; refresh(); });
  if(els.zoom){ els.zoom.addEventListener('input', ()=>{ applyZoom(); }); }
  if(els.file_input){ els.file_input.addEventListener('change', (e)=>{ const f = e.target.files && e.target.files[0]; if(f){ const fd = new FormData(); fd.append('image', f); fetch('/api/upload_image', {method:'POST', body: fd}).then(async (resp)=>{ if(!resp.ok){ const err = await resp.json().catch(()=>null); alert((err && err.error) || 'Upload failed'); return; } imgPoints=[]; currentCrop=null; refresh(); }); } }); }
    // Global prevent default to avoid browser opening the image on drop
  window.addEventListener('dragover', (e)=>{ e.preventDefault(); });
  window.addEventListener('drop', (e)=>{ e.preventDefault(); });
  // Robust drag counters for hint
  if(els.overlay_panel){
    const oh = els.drop_hint; let dragCtr = 0;
    const show=()=>{ if(oh) oh.style.display='flex'; }; const hide=()=>{ if(oh) oh.style.display='none'; }
    const onDragEnter=(e)=>{ e.preventDefault(); dragCtr++; show(); };
    const onDragLeave=(e)=>{ e.preventDefault(); dragCtr=Math.max(0,dragCtr-1); if(dragCtr===0) hide(); };
    const onDragOver=(e)=>{ e.preventDefault(); };
    const onDrop=(e)=>{ e.preventDefault(); dragCtr=0; hide(); const dt=e.dataTransfer; const files=(dt&& dt.files && dt.files.length)? dt.files : null; if(files && files[0]){ const fd=new FormData(); fd.append('image', files[0]); fetch('/api/upload_image',{method:'POST', body: fd}).then(async (resp)=>{ if(!resp.ok){ const err = await resp.json().catch(()=>null); alert((err && err.error) || 'Upload failed'); return; } imgPoints=[]; currentCrop=null; quarterROI=null; quarterManual=null; actions=[]; refresh(); }); } };
    els.overlay_panel.addEventListener('dragenter', onDragEnter);
    els.overlay_panel.addEventListener('dragleave', onDragLeave);
    els.overlay_panel.addEventListener('dragover', onDragOver);
    els.overlay_panel.addEventListener('drop', onDrop);
    // Also bind on the image itself
    if(els.overlay){ els.overlay.addEventListener('dragenter', onDragEnter); els.overlay.addEventListener('dragleave', onDragLeave); els.overlay.addEventListener('dragover', onDragOver); els.overlay.addEventListener('drop', onDrop); }
  }
  if(els.overlay_panel){
    els.overlay_panel.addEventListener('wheel', (e)=>{
      // Zoom only when Cmd (mac) or Ctrl (win) is held
      if(!(e.metaKey || e.ctrlKey)) return;
      e.preventDefault();
      const panel = els.overlay_panel; const img = els.overlay; if(!panel || !img) return;
      const panelRect = panel.getBoundingClientRect();
      const mouseXPanel = e.clientX - panelRect.left;
      const mouseYPanel = e.clientY - panelRect.top;
      // Account for panel padding when converting between offset and scroll coords
      const cs = window.getComputedStyle(panel);
      const padL = parseFloat(cs.paddingLeft)||0; const padT = parseFloat(cs.paddingTop)||0;
      // Compute normalized point relative to image BEFORE zoom
      const curW = img.clientWidth || img.naturalWidth || 1;
      const curH = img.clientHeight || img.naturalHeight || 1;
      const imgLeft = img.offsetLeft - padL;
      const imgTop  = img.offsetTop  - padT;
      const contentX = panel.scrollLeft + mouseXPanel - imgLeft;
      const contentY = panel.scrollTop  + mouseYPanel - imgTop;
      let u = contentX / curW; let v = contentY / curH;
      if(!Number.isFinite(u)) u = 0.5; if(!Number.isFinite(v)) v = 0.5;
      u = Math.max(0, Math.min(1, u)); v = Math.max(0, Math.min(1, v));
      // Update zoom value
      const cur = parseFloat((els.zoom && els.zoom.value) || '1');
      const factor = (e.deltaY < 0) ? 1.1 : 0.9;
      let next = cur * factor; next = Math.max(0.5, Math.min(4.0, next));
      if(els.zoom){ els.zoom.value = next.toFixed(2); }
      applyZoom();
      // Keep same image point under cursor AFTER zoom
      const newW = img.clientWidth || img.naturalWidth || 1;
      const newH = img.clientHeight || img.naturalHeight || 1;
      const newImgLeft = img.offsetLeft - padL;
      const newImgTop  = img.offsetTop  - padT;
      const targetScrollLeft = (newImgLeft + u * newW) - mouseXPanel;
      const targetScrollTop  = (newImgTop  + v * newH) - mouseYPanel;
      const maxSL = Math.max(0, panel.scrollWidth - panel.clientWidth);
      const maxST = Math.max(0, panel.scrollHeight - panel.clientHeight);
      panel.scrollLeft = Math.max(0, Math.min(maxSL, targetScrollLeft));
      panel.scrollTop  = Math.max(0, Math.min(maxST, targetScrollTop));
    }, { passive: false });
  }
  document.getElementById('download_svg').addEventListener('click', async ()=>{
    const resp = await fetch('/api/export_svg', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(uiToParams())});
    if(resp.ok){
      const blob = await resp.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url; a.download = 'tool_cutouts.svg';
      document.body.appendChild(a); a.click(); a.remove(); URL.revokeObjectURL(url);
    } else {
      alert('Export failed');
    }
  });
  // Click to add SAM points
  els.overlay.addEventListener('click', (e)=>{
        const rect = els.overlay.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const imgEl = els.overlay;
    const scaleDisplayToImg = (imgEl.naturalWidth > 0 && rect.width > 0)
      ? (imgEl.naturalWidth / rect.width) / currentScaleDown
      : (1.0 / currentScaleDown);
    const label = els.sam_label ? els.sam_label.value : 'pos';
    imgPoints.push({x: x * scaleDisplayToImg, y: y * scaleDisplayToImg, label});
    actions.push({type:'point_add'});
    syncCanvas(); drawPoints();
  });
  // Right-click = negative point
  els.overlay.addEventListener('contextmenu', (e)=>{
    e.preventDefault();
        const rect = els.overlay.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const imgEl = els.overlay;
    const scaleDisplayToImg = (imgEl.naturalWidth > 0 && rect.width > 0)
      ? (imgEl.naturalWidth / rect.width) / currentScaleDown
      : (1.0 / currentScaleDown);
    imgPoints.push({x: x * scaleDisplayToImg, y: y * scaleDisplayToImg, label: 'neg'});
    syncCanvas(); drawPoints();
    return false;
  });

  // Crop interactions
  if(els.crop_reset){ els.crop_reset.addEventListener('click', ()=>{ currentCrop = null; imgPoints = []; refresh(); }); }
  let isDragging = false;
  els.overlay.addEventListener('mousedown', (e)=>{
    // Manual quarter ellipse takes precedence, then quarter ROI, then crop
    if(els.qmanual_mode && els.qmanual_mode.checked){
      const rect = els.overlay.getBoundingClientRect();
      const mx = e.clientX - rect.left, my = e.clientY - rect.top;
      if(quarterManual && isPointInsideManualEllipse(mx, my)){
        // begin move existing ellipse
        quarterManualMove = { startX: mx, startY: my, origCx: quarterManual.cx, origCy: quarterManual.cy, prev: { ...(quarterManual) } };
        drawPoints(); e.preventDefault(); return;
      } else {
        // begin creating new ellipse
        quarterManualDrag = { cx: mx, cy: my, x: mx, y: my, shift: e.shiftKey };
        drawPoints(); e.preventDefault(); return;
      }
    }
    if(els.qmark_mode && els.qmark_mode.checked){
      const rect = els.overlay.getBoundingClientRect();
      quarterDrag = { x0: e.clientX - rect.left, y0: e.clientY - rect.top, x1: e.clientX - rect.left, y1: e.clientY - rect.top };
      drawPoints(); e.preventDefault(); return;
    }
    const rect = els.overlay.getBoundingClientRect();
    isDragging = true;
    dragCrop = { x0: e.clientX - rect.left, y0: e.clientY - rect.top, x1: e.clientX - rect.left, y1: e.clientY - rect.top };
    drawPoints();
    e.preventDefault();
  });
  window.addEventListener('mousemove', (e)=>{
    if(quarterManualMove){
      const rect = els.overlay.getBoundingClientRect();
      const mx = e.clientX - rect.left, my = e.clientY - rect.top;
      const dx = mx - quarterManualMove.startX;
      const dy = my - quarterManualMove.startY;
      const imgEl = els.overlay;
      const scaleDisplayToImg = (imgEl.naturalWidth > 0 && rect.width > 0)
        ? (imgEl.naturalWidth / rect.width) / currentScaleDown
        : (1.0 / currentScaleDown);
      const sc = scaleDisplayToImg;
      quarterManual.cx = Math.round(quarterManualMove.origCx + dx * sc);
      quarterManual.cy = Math.round(quarterManualMove.origCy + dy * sc);
      drawPoints(); return;
    }
    if(quarterManualDrag){
      const rect = els.overlay.getBoundingClientRect();
      quarterManualDrag.x = e.clientX - rect.left;
      quarterManualDrag.y = e.clientY - rect.top;
      quarterManualDrag.shift = e.shiftKey;
      drawPoints(); return;
    }
    if(quarterDrag){
      const rect = els.overlay.getBoundingClientRect();
      quarterDrag.x1 = e.clientX - rect.left;
      quarterDrag.y1 = e.clientY - rect.top;
      drawPoints(); return;
    }
    if(!isDragging || !dragCrop) return;
    const rect = els.overlay.getBoundingClientRect();
    dragCrop.x1 = e.clientX - rect.left;
    dragCrop.y1 = e.clientY - rect.top;
    drawPoints();
  });
  window.addEventListener('mouseup', (e)=>{
    if(quarterManualMove){
      actions.push({type:'quarter_manual', prev: quarterManualMove.prev || null});
      quarterManualMove = null; drawPoints(); refresh(); return;
    }
    if(quarterManualDrag){
      const rect = els.overlay.getBoundingClientRect();
      const imgEl = els.overlay;
      const scaleDisplayToImg = (imgEl.naturalWidth > 0 && rect.width > 0)
        ? (imgEl.naturalWidth / rect.width) / currentScaleDown
        : (1.0 / currentScaleDown);
      const sc = scaleDisplayToImg;
      const {cx, cy, x, y, shift} = quarterManualDrag;
      const dx = x - cx, dy = y - cy;
      let rx, ry, ang;
      if(shift){ rx = Math.abs(dx); ry = Math.abs(dy); ang = Math.atan2(dy, dx); }
      else { const r = Math.hypot(dx,dy); rx = r; ry = r; ang = 0; }
      let icx = Math.round(cx * sc), icy = Math.round(cy * sc);
      if(currentCrop){ icx += currentCrop.x0; icy += currentCrop.y0; }
      const MA = Math.max(2, Math.round(rx * 2 * sc));
      const ma = Math.max(2, Math.round(ry * 2 * sc));
      const angleDeg = ang * 180.0 / Math.PI;
      const prev = quarterManual ? { ...quarterManual } : null;
      quarterManual = { cx: icx, cy: icy, MA: MA, ma: ma, angle: angleDeg };
      actions.push({type:'quarter_manual', prev});
      quarterManualDrag = null; drawPoints(); refresh(); return;
    }
    if(quarterDrag){
      const rect = els.overlay.getBoundingClientRect();
      const x0d = Math.max(0, Math.min(quarterDrag.x0, quarterDrag.x1));
      const y0d = Math.max(0, Math.min(quarterDrag.y0, quarterDrag.y1));
      const x1d = Math.min(rect.width, Math.max(quarterDrag.x0, quarterDrag.x1));
      const y1d = Math.min(rect.height, Math.max(quarterDrag.y0, quarterDrag.y1));
      quarterDrag = null; drawPoints();
      const imgEl = els.overlay;
      const scaleDisplayToImg = (imgEl.naturalWidth > 0 && rect.width > 0)
        ? (imgEl.naturalWidth / rect.width) / currentScaleDown
        : (1.0 / currentScaleDown);
      let nx0 = Math.round(x0d * scaleDisplayToImg);
      let ny0 = Math.round(y0d * scaleDisplayToImg);
      let nx1 = Math.round(x1d * scaleDisplayToImg);
      let ny1 = Math.round(y1d * scaleDisplayToImg);
      if(nx1 > nx0 && ny1 > ny0){
        if(currentCrop){ nx0 += currentCrop.x0; nx1 += currentCrop.x0; ny0 += currentCrop.y0; ny1 += currentCrop.y0; }
        const prev = quarterROI ? { ...quarterROI } : null;
        quarterROI = { x0: nx0, y0: ny0, x1: nx1, y1: ny1 };
        actions.push({type:'quarter_roi', prev});
        refresh();
      }
      return;
    }
    if(!isDragging || !dragCrop) return;
    isDragging = false;
    const rect = els.overlay.getBoundingClientRect();
    const x0d = Math.max(0, Math.min(dragCrop.x0, dragCrop.x1));
    const y0d = Math.max(0, Math.min(dragCrop.y0, dragCrop.y1));
    const x1d = Math.min(rect.width, Math.max(dragCrop.x0, dragCrop.x1));
    const y1d = Math.min(rect.height, Math.max(dragCrop.y0, dragCrop.y1));
    dragCrop = null; drawPoints();
    const imgEl = els.overlay;
    const scaleDisplayToImg = (imgEl.naturalWidth > 0 && rect.width > 0)
      ? (imgEl.naturalWidth / rect.width) / currentScaleDown
      : (1.0 / currentScaleDown);
    let nx0 = Math.round(x0d * scaleDisplayToImg);
    let ny0 = Math.round(y0d * scaleDisplayToImg);
    let nx1 = Math.round(x1d * scaleDisplayToImg);
    let ny1 = Math.round(y1d * scaleDisplayToImg);
    if(nx1 <= nx0 || ny1 <= ny0) return;
    if(currentCrop){
      nx0 += currentCrop.x0; nx1 += currentCrop.x0;
      ny0 += currentCrop.y0; ny1 += currentCrop.y0;
    }
    const prev = currentCrop ? { ...currentCrop } : null;
    currentCrop = { x0: nx0, y0: ny0, x1: nx1, y1: ny1 };
    actions.push({type:'crop', prev});
    imgPoints = [];
    refresh();
  });

  if(els.sam_undo){ els.sam_undo.addEventListener('click', ()=>{ imgPoints.pop(); drawPoints(); }); }
  if(els.sam_clear){ els.sam_clear.addEventListener('click', ()=>{ imgPoints = []; drawPoints(); }); }
  window.addEventListener('keydown', (e)=>{
    // Undo: Ctrl+Z / Cmd+Z
    if((e.key === 'z' || e.key === 'Z') && (e.ctrlKey || e.metaKey)){ e.preventDefault(); undoLast(); return; }
    if(!els.sam_label) return;
    if(e.key === 'p' || e.key === 'P') els.sam_label.value = 'pos';
    if(e.key === 'n' || e.key === 'N') els.sam_label.value = 'neg';
  });
  window.addEventListener('resize', ()=>{ syncCanvas(); drawPoints(); });
  setModeVisibility();
  refresh();
}
attach();
</script>
"""


if __name__ == '__main__':
    main()
