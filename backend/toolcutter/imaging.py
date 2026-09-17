"""Image decode/encode helpers (EXIF-aware, HEIC-capable)."""
from __future__ import annotations

import io
from typing import Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps

try:  # iPhone HEIC support if pillow-heif is installed
    from pillow_heif import register_heif_opener  # type: ignore

    register_heif_opener()
except Exception:  # pragma: no cover - optional dependency
    pass

IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "heic", "heif", "webp", "tif", "tiff", "bmp"}
MESH_EXTENSIONS = {"ply", "obj", "stl", "glb", "gltf", "off", "xyz"}


def decode_image(data: bytes) -> np.ndarray:
    """Decode bytes to a BGR uint8 array, honoring EXIF orientation."""
    with Image.open(io.BytesIO(data)) as im:
        im = ImageOps.exif_transpose(im)
        rgb = im.convert("RGB")
        arr = np.asarray(rgb)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def encode_jpeg(img_bgr: np.ndarray, quality: int = 88) -> bytes:
    ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return buf.tobytes()


def encode_png(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("PNG encode failed")
    return buf.tobytes()


def downscale_to(img: np.ndarray, max_side: int) -> Tuple[np.ndarray, float]:
    """Return (image, scale) with the longest side <= max_side. scale = out/in."""
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return img, 1.0
    scale = max_side / float(longest)
    out = cv2.resize(img, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA)
    return out, scale


def height_to_colormap(height_mm: np.ndarray, vmax: float | None = None) -> np.ndarray:
    """Render a float height map (mm, NaN allowed) to a BGR color image."""
    h = np.nan_to_num(height_mm, nan=0.0)
    if vmax is None:
        valid = h[np.isfinite(height_mm)] if np.isfinite(height_mm).any() else h
        vmax = float(np.percentile(valid, 99.5)) if valid.size else 1.0
    vmax = max(vmax, 1e-3)
    norm = np.clip(h / vmax, 0.0, 1.0)
    u8 = (norm * 255.0).astype(np.uint8)
    return cv2.applyColorMap(u8, cv2.COLORMAP_VIRIDIS)
