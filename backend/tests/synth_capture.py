"""Synthetic phone capture of a drawer: photo + aligned LiDAR-style depth + intrinsics, with ground truth.

Scene: drawer floor 400 x 300 mm with four ArUco markers (ids 0-3, 50 mm) in the corners and three
box tools of known footprint and height. Camera ~450 mm above, slightly tilted and rolled.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

DRAWER_W, DRAWER_H = 400.0, 300.0
MARKER_MM = 50.0
IMG_W, IMG_H = 1920, 1440
DEPTH_W, DEPTH_H = 256, 192
FX = 1450.0


def tools() -> Dict[str, Tuple[np.ndarray, float]]:
    """name -> (footprint polygon in drawer mm (x right, y down from TL), height mm)."""
    return {
        "bar": (np.array([[70, 60], [250, 60], [250, 95], [70, 95]], float), 35.0),
        "block": (np.array([[290, 80], [360, 80], [360, 180], [290, 180]], float), 20.0),
        "plate": (np.array([[60, 160], [220, 160], [220, 250], [60, 250]], float), 8.0),
    }


def camera() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Intrinsics K and world->camera (R, t). World: drawer TL corner at origin, x right, y down, z UP."""
    K = np.array([[FX, 0, IMG_W / 2 + 12], [0, FX, IMG_H / 2 - 8], [0, 0, 1]], float)
    # camera 450 mm above the drawer center, looking down, tilted 6 deg about x and rolled 4 deg
    rx = np.deg2rad(6.0); rz = np.deg2rad(4.0)
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    # base: camera x = world x, camera y = world y (down), camera z = -world z (looking down)
    base = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]], float)
    R = Rx @ Rz @ base
    cam_pos = np.array([DRAWER_W / 2 + 20, DRAWER_H / 2 - 10, 450.0])
    t = -R @ cam_pos
    return K, R, t


def project(K, R, t, Xw: np.ndarray) -> np.ndarray:
    Xc = Xw @ R.T + t
    return np.column_stack([K[0, 0] * Xc[:, 0] / Xc[:, 2] + K[0, 2], K[1, 1] * Xc[:, 1] / Xc[:, 2] + K[1, 2]])


def floor_texture(ppm: float = 4.0, marker_ids: Optional[Dict[int, int]] = None) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Drawer floor image (mm * ppm) with markers in the corners; returns texture + marker outer corners (mm).

    `marker_ids` maps corner slot (0 TL, 1 TR, 2 BR, 3 BL) to the id printed there, for testing a sheet laid
    out in the wrong order (the server works the order out from the positions)."""
    W, H = int(DRAWER_W * ppm), int(DRAWER_H * ppm)
    rng = np.random.default_rng(5)
    tex = np.full((H, W, 3), 190, np.uint8)
    tex = np.clip(tex.astype(np.int16) + rng.integers(-8, 9, (H, W, 1)), 0, 255).astype(np.uint8)
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    m = int(MARKER_MM * ppm)
    pad = int(6 * ppm)
    spots = {0: (pad, pad), 1: (W - m - pad, pad), 2: (W - m - pad, H - m - pad), 3: (pad, H - m - pad)}
    for i, (x, y) in spots.items():
        img = cv2.aruco.generateImageMarker(d, (marker_ids or {}).get(i, i), m)
        border = int(4 * ppm)
        cv2.rectangle(tex, (x - border, y - border), (x + m + border, y + m + border), (255, 255, 255), -1)
        tex[y:y + m, x:x + m] = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return tex, []


def render(marker_ids: Optional[Dict[int, int]] = None) -> Tuple[bytes, np.ndarray, Dict]:
    """Returns (jpeg bytes, depth map meters (DEPTH_H x DEPTH_W), intrinsics dict)."""
    K, R, t = camera()
    tex, _ = floor_texture(marker_ids=marker_ids)
    ppm = tex.shape[1] / DRAWER_W
    # background (desk around the drawer) then floor via homography of the z=0 plane
    img = np.full((IMG_H, IMG_W, 3), (60, 70, 90), np.uint8)
    floor_mm = np.array([[0, 0], [DRAWER_W, 0], [DRAWER_W, DRAWER_H], [0, DRAWER_H]], float)
    floor_px = project(K, R, t, np.column_stack([floor_mm, np.zeros(4)]))
    Hf = cv2.getPerspectiveTransform((floor_mm * ppm).astype(np.float32), floor_px.astype(np.float32))
    warped = cv2.warpPerspective(tex, Hf, (IMG_W, IMG_H))
    mask = cv2.warpPerspective(np.full(tex.shape[:2], 255, np.uint8), Hf, (IMG_W, IMG_H))
    img = np.where(mask[..., None] > 0, warped, img)

    # depth: per pixel ray -> floor plane, overridden by tool tops (sides ignored)
    vs, us = np.mgrid[0:IMG_H, 0:IMG_W]
    rays_c = np.stack([(us - K[0, 2]) / K[0, 0], (vs - K[1, 2]) / K[1, 1], np.ones_like(us, float)], axis=-1)
    cam_pos = -R.T @ t
    rays_w = rays_c @ R      # R^T applied to each ray (rows): ray_w = R^T ray_c
    def plane_depth(z0):
        lam = (z0 - cam_pos[2]) / rays_w[..., 2]
        return lam  # depth along camera z equals lam since ray_c z = 1
    depth = plane_depth(0.0)
    for name, (poly, h) in tools().items():
        top_px = project(K, R, t, np.column_stack([poly, np.full(len(poly), h)]))
        cv2.fillPoly(img, [np.round(top_px).astype(np.int32)], (52, 54, 58))
        m = np.zeros((IMG_H, IMG_W), np.uint8)
        cv2.fillPoly(m, [np.round(top_px).astype(np.int32)], 255)
        dh = plane_depth(h)
        depth = np.where(m > 0, dh, depth)
        # draw the visible side walls dark too (approximate): footprint at z=0 -> hull with top
        bot_px = project(K, R, t, np.column_stack([poly, np.zeros(len(poly))]))
        hull = cv2.convexHull(np.vstack([top_px, bot_px]).astype(np.float32))
        side = np.zeros((IMG_H, IMG_W), np.uint8)
        cv2.fillPoly(side, [np.round(hull).astype(np.int32)], 255)
        side[m > 0] = 0
        img[side > 0] = (40, 42, 46)
        # side-wall depth: where the ray enters the axis-aligned box [x0,x1]x[y0,y1]x[0,h] (slab test), so
        # the LiDAR points lie ON the wall like a real scan
        x0, x1 = poly[:, 0].min(), poly[:, 0].max()
        y0, y1 = poly[:, 1].min(), poly[:, 1].max()
        with np.errstate(divide="ignore", invalid="ignore"):
            tx = np.sort(np.stack([(x0 - cam_pos[0]) / rays_w[..., 0], (x1 - cam_pos[0]) / rays_w[..., 0]]), axis=0)
            ty = np.sort(np.stack([(y0 - cam_pos[1]) / rays_w[..., 1], (y1 - cam_pos[1]) / rays_w[..., 1]]), axis=0)
            tz = np.sort(np.stack([(0.0 - cam_pos[2]) / rays_w[..., 2], (h - cam_pos[2]) / rays_w[..., 2]]), axis=0)
            t_in = np.maximum.reduce([tx[0], ty[0], tz[0]])
            t_out = np.minimum.reduce([tx[1], ty[1], tz[1]])
        hit = (side > 0) & (t_in < t_out) & np.isfinite(t_in)
        depth = np.where(hit, t_in, depth)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    rng = np.random.default_rng(9)
    depth_small = cv2.resize(depth.astype(np.float32), (DEPTH_W, DEPTH_H), interpolation=cv2.INTER_AREA)
    depth_small += rng.normal(0, 1.5, depth_small.shape).astype(np.float32)   # ~1.5 mm LiDAR noise
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 93])
    intr = {"fx": K[0, 0], "fy": K[1, 1], "cx": K[0, 2], "cy": K[1, 2], "width": IMG_W, "height": IMG_H}
    return buf.tobytes(), depth_small / 1000.0, intr
