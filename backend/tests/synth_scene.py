"""Synthetic drawer scenes rendered from real triangle meshes: photo + LiDAR-style depth + intrinsics.

Tools are trimesh meshes (prisms, lying cylinders, spheres) so depth is exact, including sloped
sides. Ground truth footprints are shapely polygons. Used by tests/capture_eval.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import cv2
import numpy as np
import trimesh
from shapely import affinity
from shapely.geometry import Point, Polygon, box
from shapely.ops import unary_union

MARKER_MM = 50.0
IMG_W, IMG_H = 1920, 1440
DEPTH_W, DEPTH_H = 256, 192
FX = 1450.0


@dataclass
class Tool:
    name: str
    mesh: trimesh.Trimesh          # world mm, z up, resting on z=0
    footprint: Polygon             # truth, drawer mm (x right, y down from TL)
    height: float
    color: Tuple[int, int, int] = (56, 58, 62)


@dataclass
class Scene:
    drawer_w: float
    drawer_h: float
    tools: List[Tool]
    cam_height: float
    tilt_deg: float = 5.0
    roll_deg: float = 3.0
    cam_offset: Tuple[float, float] = (15.0, -10.0)   # camera xy offset from the drawer centre
    marker_inset: float = 6.0
    name: str = "scene"


# ----------------------------------------------------------------------------- tool builders

def prism(name: str, poly: Polygon, h: float, color=(56, 58, 62)) -> Tool:
    m = trimesh.creation.extrude_polygon(poly, h)
    return Tool(name, m, poly, h, color)


def lying_cylinder(name: str, x: float, y: float, length: float, radius: float, angle_deg: float = 0.0, color=(70, 72, 78)) -> Tool:
    m = trimesh.creation.cylinder(radius=radius, height=length, sections=64)
    m.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))   # axis along x
    m.apply_transform(trimesh.transformations.rotation_matrix(np.deg2rad(angle_deg), [0, 0, 1]))
    m.apply_translation([x, y, radius])
    fp = affinity.rotate(box(-length / 2, -radius, length / 2, radius), angle_deg, origin=(0, 0), use_radians=False)
    fp = affinity.translate(fp, x, y)
    return Tool(name, m, fp, 2 * radius, color)


def sphere(name: str, x: float, y: float, r: float, color=(90, 60, 50)) -> Tool:
    m = trimesh.creation.icosphere(subdivisions=3, radius=r)
    m.apply_translation([x, y, r])
    return Tool(name, m, Point(x, y).buffer(r, 64), 2 * r, color)


def wrench_poly(x: float, y: float, length: float, bar_w: float, head_r: float, angle_deg: float) -> Polygon:
    p = unary_union([box(0, -bar_w / 2, length, bar_w / 2), Point(0, 0).buffer(head_r, 48), Point(length, 0).buffer(head_r * 0.8, 48)])
    p = affinity.rotate(p, angle_deg, origin=(0, 0))
    return affinity.translate(p, x, y)


def l_bracket(x: float, y: float, a: float, b: float, t: float, angle_deg: float) -> Polygon:
    p = unary_union([box(0, 0, a, t), box(0, 0, t, b)])
    p = affinity.rotate(p, angle_deg, origin=(0, 0))
    return affinity.translate(p, x, y)


# ----------------------------------------------------------------------------- scenes

def scene_small() -> Scene:
    tools = [
        prism("bar", box(70, 60, 250, 95), 35.0),
        prism("block", box(290, 80, 360, 180), 20.0),
        prism("plate", box(60, 160, 220, 250), 8.0),
    ]
    return Scene(400, 300, tools, cam_height=450, name="small_boxes")


def scene_complex_wide() -> Scene:
    tools = [
        prism("wrench", wrench_poly(120, 90, 210, 20, 24, 12), 9.0),
        lying_cylinder("handle", 470, 110, 160, 16, angle_deg=-20),
        prism("bracket", l_bracket(560, 250, 110, 90, 22, 35), 25.0),
        prism("hexnut", affinity.rotate(Point(150, 300).buffer(30, 6), 15, origin=(150, 300)), 14.0),
        prism("ruler", box(230, 360, 560, 390), 3.0),
        sphere("ball", 330, 250, 22),
        prism("tallblock", box(600, 60, 660, 130), 45.0),
        prism("thinplate", affinity.rotate(box(60, 380, 160, 420), -8, origin=(110, 400)), 5.0),
    ]
    return Scene(700, 450, tools, cam_height=780, tilt_deg=6, roll_deg=-4, cam_offset=(30, 20), name="complex_wide")


def scene_low_camera() -> Scene:
    """Same drawer as complex_wide but captured too close: stresses the perspective correction."""
    s = scene_complex_wide()
    s.cam_height = 560
    s.name = "complex_wide_low"
    return s


def scene_long_shallow() -> Scene:
    """Drawer 4x wider than deep (900 x 225 mm) with tall tools at both ends, far off the camera axis."""
    tools = [
        prism("endblock_L", box(75, 70, 145, 160), 55.0),                       # 55 mm tall, ~340 mm off-axis
        prism("endcyl_R", Point(830, 120).buffer(22, 64), 60.0),               # standing cylinder, 60 mm tall, ~380 mm off-axis
        prism("midblock", box(420, 40, 500, 100), 40.0),
        prism("wrench", wrench_poly(230, 185, 170, 18, 22, -6), 9.0),
        lying_cylinder("handle", 620, 160, 170, 17, angle_deg=4),
        sphere("ball", 760, 178, 24),
        prism("bracket", l_bracket(300, 55, 90, 70, 20, 10), 25.0),
        prism("plate", affinity.rotate(box(530, 40, 620, 90), 5, origin=(575, 65)), 5.0),
    ]
    return Scene(900, 225, tools, cam_height=800, tilt_deg=3, roll_deg=2, cam_offset=(10, 5), name="long_shallow")


def compound(name: str, parts: List[Tool], color=None) -> Tool:
    """Union of several primitive tools into one (mesh concatenation; footprint = polygon union)."""
    mesh = trimesh.util.concatenate([p.mesh for p in parts])
    fp = unary_union([p.footprint for p in parts])
    h = max(p.height for p in parts)
    return Tool(name, mesh, fp, h, color or parts[0].color)


def pliers_poly(x: float, y: float, angle_deg: float) -> Polygon:
    """Two tapered handles meeting at a pivot, short jaws; a gap of ~6 mm between the handles."""
    handle_a = Polygon([(0, -4), (95, -14), (95, -24), (0, -8)])
    handle_b = Polygon([(0, 4), (95, 14), (95, 24), (0, 8)])
    pivot = Point(0, 0).buffer(9, 32)
    jaw_a = Polygon([(-2, -3), (-38, -9), (-40, -2), (-4, 0)])
    jaw_b = Polygon([(-2, 3), (-38, 9), (-40, 2), (-4, 0)])
    p = unary_union([handle_a, handle_b, pivot, jaw_a, jaw_b])
    p = affinity.rotate(p, angle_deg, origin=(0, 0))
    return affinity.translate(p, x, y)


def adjustable_wrench_poly(x: float, y: float, angle_deg: float) -> Polygon:
    handle = affinity.rotate(box(0, -9, 150, 9), 0, origin=(0, 0))
    head = Polygon([(150, -22), (188, -22), (200, -8), (200, 10), (186, 26), (150, 26)])
    jaw_notch = Polygon([(178, 4), (200, 4), (200, -6), (178, -6)])          # the adjustable jaw opening
    p = unary_union([handle, head]).difference(jaw_notch)
    p = affinity.rotate(p, angle_deg, origin=(0, 0))
    return affinity.translate(p, x, y)


def utility_knife_poly(x: float, y: float, angle_deg: float) -> Polygon:
    body = Polygon([(0, 0), (120, 6), (150, 0), (158, -8), (150, -20), (120, -24), (60, -26), (0, -18)]).buffer(3, 16)
    p = affinity.rotate(body, angle_deg, origin=(0, 0))
    return affinity.translate(p, x, y)


def scene_complex_tools() -> Scene:
    """Realistic tool shapes with fine features: gaps, notches, thin shafts, small round parts."""
    tools = [
        prism("pliers", pliers_poly(120, 90, 15), 12.0),
        prism("adj_wrench", adjustable_wrench_poly(230, 300, -12), 14.0, color=(80, 82, 88)),
        compound("hammer", [prism("head", box(520, 40, 630, 72), 26.0), lying_cylinder("handle", 575, 175, 210, 13, angle_deg=90)], color=(70, 60, 55)),
        compound("screwdriver", [lying_cylinder("grip", 380, 420, 95, 14, angle_deg=5), lying_cylinder("shaft", 500, 430.5, 160, 3.5, angle_deg=5)]),
        prism("hex_key", l_bracket(60, 380, 80, 30, 6, -20), 6.0),
        prism("tape", box(640, 330, 705, 395).buffer(6, 16), 36.0, color=(95, 70, 45)),
        prism("socket_a", Point(470, 250).buffer(12, 48), 26.0),
        prism("socket_b", Point(510, 250).buffer(9, 48), 22.0),
        prism("knife", utility_knife_poly(80, 220, 30), 15.0),
    ]
    return Scene(760, 480, tools, cam_height=780, tilt_deg=4, roll_deg=-3, cam_offset=(20, 10), name="complex_tools")


SCENES = {"small_boxes": scene_small, "complex_tools": scene_complex_tools, "complex_wide": scene_complex_wide, "complex_wide_low": scene_low_camera,
          "long_shallow": scene_long_shallow}


# ----------------------------------------------------------------------------- rendering

def camera(scene: Scene):
    K = np.array([[FX, 0, IMG_W / 2 + 12], [0, FX, IMG_H / 2 - 8], [0, 0, 1]], float)
    rx = np.deg2rad(scene.tilt_deg); rz = np.deg2rad(scene.roll_deg)
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    base = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]], float)
    R = Rx @ Rz @ base
    cam_pos = np.array([scene.drawer_w / 2 + scene.cam_offset[0], scene.drawer_h / 2 + scene.cam_offset[1], scene.cam_height])
    t = -R @ cam_pos
    return K, R, t, cam_pos


def floor_texture(scene: Scene, ppm: float = 3.0) -> np.ndarray:
    W, H = int(scene.drawer_w * ppm), int(scene.drawer_h * ppm)
    rng = np.random.default_rng(5)
    tex = np.full((H, W, 3), 185, np.uint8)
    tex = np.clip(tex.astype(np.int16) + rng.integers(-10, 11, (H, W, 1)), 0, 255).astype(np.uint8)
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    m = int(MARKER_MM * ppm)
    pad = int(scene.marker_inset * ppm)
    spots = {0: (pad, pad), 1: (W - m - pad, pad), 2: (W - m - pad, H - m - pad), 3: (pad, H - m - pad)}
    for i, (x, y) in spots.items():
        img = cv2.aruco.generateImageMarker(d, i, m)
        b = int(4 * ppm)
        cv2.rectangle(tex, (x - b, y - b), (x + m + b, y + m + b), (255, 255, 255), -1)
        tex[y:y + m, x:x + m] = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return tex


def render(scene: Scene, depth_noise_mm: float = 1.5, seed: int = 9):
    """Returns (jpeg bytes, depth meters (DEPTH_H x DEPTH_W), intrinsics dict, full-res depth mm)."""
    blur_px = 0.0
    K, R, t, cam_pos = camera(scene)
    tex = floor_texture(scene)
    ppm = tex.shape[1] / scene.drawer_w
    img = np.full((IMG_H, IMG_W, 3), (60, 70, 90), np.uint8)
    # floor via homography
    floor_mm = np.array([[0, 0], [scene.drawer_w, 0], [scene.drawer_w, scene.drawer_h], [0, scene.drawer_h]], float)
    floor_px = _project(K, R, t, np.column_stack([floor_mm, np.zeros(4)]))
    Hf = cv2.getPerspectiveTransform((floor_mm * ppm).astype(np.float32), floor_px.astype(np.float32))
    warped = cv2.warpPerspective(tex, Hf, (IMG_W, IMG_H))
    fmask = cv2.warpPerspective(np.full(tex.shape[:2], 255, np.uint8), Hf, (IMG_W, IMG_H))
    img = np.where(fmask[..., None] > 0, warped, img)
    # depth buffer: floor plane
    vs, us = np.mgrid[0:IMG_H, 0:IMG_W]
    rays_c = np.stack([(us - K[0, 2]) / K[0, 0], (vs - K[1, 2]) / K[1, 1], np.ones_like(us, float)], axis=-1)
    rays_w = rays_c @ R
    zbuf = ((0.0 - cam_pos[2]) / rays_w[..., 2]).astype(np.float64)   # depth (camera z) to the floor
    light = np.array([0.3, -0.4, 0.87]); light /= np.linalg.norm(light)
    rng = np.random.default_rng(seed)
    for tool in scene.tools:
        mesh = tool.mesh
        V = mesh.vertices
        Vc = V @ R.T + t                       # camera coords
        P = np.column_stack([K[0, 0] * Vc[:, 0] / Vc[:, 2] + K[0, 2], K[1, 1] * Vc[:, 1] / Vc[:, 2] + K[1, 2]])
        normals = mesh.face_normals
        base = np.array(tool.color, float)
        for fi, face in enumerate(mesh.faces):
            # back-face cull: triangle facing away from camera
            nc = R @ normals[fi]
            if nc[2] > 0:   # normal pointing away (camera looks along +z)
                continue
            tri = P[face]
            x0, y0 = np.floor(tri.min(axis=0)).astype(int); x1, y1 = np.ceil(tri.max(axis=0)).astype(int)
            x0, y0 = max(x0, 0), max(y0, 0); x1, y1 = min(x1 + 1, IMG_W), min(y1 + 1, IMG_H)
            if x1 <= x0 or y1 <= y0:
                continue
            sub = np.zeros((y1 - y0, x1 - x0), np.uint8)
            cv2.fillConvexPoly(sub, np.round(tri - [x0, y0]).astype(np.int32), 255)
            if not sub.any():
                continue
            # depth of the face plane along each ray in the sub-window
            n_c = nc
            p0 = Vc[face[0]]
            rr = rays_c[y0:y1, x0:x1]
            denom = rr @ n_c
            with np.errstate(divide="ignore", invalid="ignore"):
                lam = (p0 @ n_c) / denom
            zsub = zbuf[y0:y1, x0:x1]
            hit = (sub > 0) & np.isfinite(lam) & (lam > 0) & (lam < zsub)
            zsub[hit] = lam[hit]
            shade = 0.55 + 0.45 * max(0.0, float(normals[fi] @ light))
            col = np.clip(base * shade + rng.normal(0, 3, 3), 0, 255)
            img[y0:y1, x0:x1][hit] = col.astype(np.uint8)
            zbuf[y0:y1, x0:x1] = zsub
    img = cv2.GaussianBlur(img, (3, 3), 0)
    if blur_px and blur_px > 0.5:
        # directional motion blur, as a hand moving during exposure produces
        k = int(round(blur_px)) | 1
        ang = rng.uniform(0, np.pi)
        kern = np.zeros((k, k), np.float32)
        c = k // 2
        for i in range(k):
            x = int(round(c + (i - c) * np.cos(ang))); y = int(round(c + (i - c) * np.sin(ang)))
            kern[min(max(y, 0), k - 1), min(max(x, 0), k - 1)] = 1.0
        kern /= kern.sum()
        img = cv2.filter2D(img, -1, kern)
    depth_small = cv2.resize(zbuf.astype(np.float32), (DEPTH_W, DEPTH_H), interpolation=cv2.INTER_AREA)
    depth_small += rng.normal(0, depth_noise_mm, depth_small.shape).astype(np.float32)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 93])
    intr = {"fx": K[0, 0], "fy": K[1, 1], "cx": K[0, 2], "cy": K[1, 2], "width": IMG_W, "height": IMG_H}
    return buf.tobytes(), depth_small / 1000.0, intr, zbuf


def _project(K, R, t, Xw):
    Xc = Xw @ R.T + t
    return np.column_stack([K[0, 0] * Xc[:, 0] / Xc[:, 2] + K[0, 2], K[1, 1] * Xc[:, 1] / Xc[:, 2] + K[1, 2]])


# ----------------------------------------------------------------------------- multi-frame sweeps

def render_from_pose(scene: Scene, cam_pos: np.ndarray, look_at: np.ndarray, roll_deg: float = 0.0,
                     depth_noise_mm: float = 1.5, seed: int = 0, texture: bool = True, blur_px: float = 0.0):
    """Render one frame from an arbitrary camera position looking at `look_at` (world mm, z up).

    Returns (jpeg bytes, depth m (DEPTH_H x DEPTH_W), intrinsics, gravity in camera coords (unit, ARKit
    convention: camera x right, y up, z backward), K, R, t).
    """
    # Right-handed world for multi-view work: drawer x right, y down (as on paper), and UP = -z. The
    # scene meshes are authored with +z up, so mirror them and the camera through z=0.
    scene = _flip_scene_z(scene)
    cam_pos = np.array([cam_pos[0], cam_pos[1], -cam_pos[2]], float)
    look_at = np.array([look_at[0], look_at[1], -look_at[2]], float)
    K = np.array([[FX, 0, IMG_W / 2], [0, FX, IMG_H / 2], [0, 0, 1]], float)
    fwd = look_at - cam_pos; fwd /= np.linalg.norm(fwd)          # camera +z (OpenCV) = forward
    ref = np.array([0.0, 1.0, 0.0])                              # image down ~ world +y
    right = np.cross(ref, fwd); right /= np.linalg.norm(right)   # hmm: want right = down x forward? use cross(fwd, ref)?
    right = np.cross(fwd, ref) * -1
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right); down /= np.linalg.norm(down)
    R = np.vstack([right, down, fwd])                            # rows: camera axes in world
    if roll_deg:
        rz = np.deg2rad(roll_deg)
        Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
        R = Rz @ R
    t = -R @ cam_pos
    sc = Scene(scene.drawer_w, scene.drawer_h, scene.tools, scene.cam_height, marker_inset=scene.marker_inset, name=scene.name)
    jpg, depth_small, intr, zbuf = _render_with_camera(sc, K, R, t, cam_pos, depth_noise_mm, seed, texture, blur_px)
    # gravity (world +z, since up is -z) in ARKit camera coords: ARKit camera has x right, y up, z backward
    g_cv = R @ np.array([0.0, 0.0, 1.0])
    g_arkit = np.array([g_cv[0], -g_cv[1], -g_cv[2]])
    return jpg, depth_small, intr, g_arkit, K, R, t


def _flip_scene_z(scene: Scene) -> Scene:
    tools = []
    for tl in scene.tools:
        m = tl.mesh.copy()
        v = m.vertices.copy(); v[:, 2] *= -1.0; m.vertices = v
        m.invert()
        tools.append(Tool(tl.name, m, tl.footprint, tl.height, tl.color))
    return Scene(scene.drawer_w, scene.drawer_h, tools, scene.cam_height, marker_inset=scene.marker_inset, name=scene.name)


def _render_with_camera(scene: Scene, K, R, t, cam_pos, depth_noise_mm, seed, texture, blur_px: float = 0.0):
    tex = floor_texture(scene)
    ppm = tex.shape[1] / scene.drawer_w
    img = np.full((IMG_H, IMG_W, 3), (60, 70, 90), np.uint8)
    # desk texture around the drawer so photogrammetry has features outside too
    rng = np.random.default_rng(seed + 100)
    img = np.clip(img.astype(np.int16) + rng.integers(-12, 13, (IMG_H, IMG_W, 1)), 0, 255).astype(np.uint8)
    floor_mm = np.array([[0, 0], [scene.drawer_w, 0], [scene.drawer_w, scene.drawer_h], [0, scene.drawer_h]], float)
    floor_px = _project(K, R, t, np.column_stack([floor_mm, np.zeros(4)]))
    Hf = cv2.getPerspectiveTransform((floor_mm * ppm).astype(np.float32), floor_px.astype(np.float32))
    warped = cv2.warpPerspective(tex, Hf, (IMG_W, IMG_H))
    fmask = cv2.warpPerspective(np.full(tex.shape[:2], 255, np.uint8), Hf, (IMG_W, IMG_H))
    img = np.where(fmask[..., None] > 0, warped, img)
    vs, us = np.mgrid[0:IMG_H, 0:IMG_W]
    rays_c = np.stack([(us - K[0, 2]) / K[0, 0], (vs - K[1, 2]) / K[1, 1], np.ones_like(us, float)], axis=-1)
    rays_w = rays_c @ R
    with np.errstate(divide="ignore", invalid="ignore"):
        zbuf = ((0.0 - cam_pos[2]) / rays_w[..., 2]).astype(np.float64)
    zbuf[~np.isfinite(zbuf) | (zbuf < 0)] = 1e6
    light = np.array([0.3, -0.4, 0.87]); light /= np.linalg.norm(light)
    rng = np.random.default_rng(seed)
    toolmask = np.zeros((IMG_H, IMG_W), bool)
    for tool in scene.tools:
        mesh = tool.mesh
        Vc = mesh.vertices @ R.T + t
        P = np.column_stack([K[0, 0] * Vc[:, 0] / Vc[:, 2] + K[0, 2], K[1, 1] * Vc[:, 1] / Vc[:, 2] + K[1, 2]])
        normals = mesh.face_normals
        base = np.array(tool.color, float)
        for fi, face in enumerate(mesh.faces):
            nc = R @ normals[fi]
            if nc[2] > 0:
                continue
            tri = P[face]
            x0, y0 = np.floor(tri.min(axis=0)).astype(int); x1, y1 = np.ceil(tri.max(axis=0)).astype(int)
            x0, y0 = max(x0, 0), max(y0, 0); x1, y1 = min(x1 + 1, IMG_W), min(y1 + 1, IMG_H)
            if x1 <= x0 or y1 <= y0:
                continue
            sub = np.zeros((y1 - y0, x1 - x0), np.uint8)
            cv2.fillConvexPoly(sub, np.round(tri - [x0, y0]).astype(np.int32), 255)
            if not sub.any():
                continue
            p0 = Vc[face[0]]
            rr = rays_c[y0:y1, x0:x1]
            denom = rr @ nc
            with np.errstate(divide="ignore", invalid="ignore"):
                lam = (p0 @ nc) / denom
            zsub = zbuf[y0:y1, x0:x1]
            hit = (sub > 0) & np.isfinite(lam) & (lam > 0) & (lam < zsub)
            zsub[hit] = lam[hit]
            shade = 0.55 + 0.45 * max(0.0, float(normals[fi] @ light))
            col = np.clip(base * shade, 0, 255)
            img[y0:y1, x0:x1][hit] = col.astype(np.uint8)
            toolmask[y0:y1, x0:x1] |= hit
            zbuf[y0:y1, x0:x1] = zsub
    if texture:
        # speckle so photogrammetry can match features on tool surfaces (real tools have scratches/grain)
        noise = rng.integers(-18, 19, (IMG_H, IMG_W, 1))
        img = np.where(toolmask[..., None], np.clip(img.astype(np.int16) + noise, 0, 255), img).astype(np.uint8)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    if blur_px and blur_px > 0.5:
        # directional motion blur, as a hand moving during exposure produces
        k = int(round(blur_px)) | 1
        ang = rng.uniform(0, np.pi)
        kern = np.zeros((k, k), np.float32)
        c = k // 2
        for i in range(k):
            x = int(round(c + (i - c) * np.cos(ang))); y = int(round(c + (i - c) * np.sin(ang)))
            kern[min(max(y, 0), k - 1), min(max(x, 0), k - 1)] = 1.0
        kern /= kern.sum()
        img = cv2.filter2D(img, -1, kern)
    depth_small = cv2.resize(zbuf.astype(np.float32), (DEPTH_W, DEPTH_H), interpolation=cv2.INTER_AREA)
    depth_small += rng.normal(0, depth_noise_mm, depth_small.shape).astype(np.float32)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 93])
    intr = {"fx": K[0, 0], "fy": K[1, 1], "cx": K[0, 2], "cy": K[1, 2], "width": IMG_W, "height": IMG_H}
    return buf.tobytes(), depth_small / 1000.0, intr, zbuf


def render_sweep(scene: Scene, out_dir, n_frames: int = 36, radius: float = 350.0, height: float = 650.0, seed: int = 0):
    """Frames along an arc above the drawer -> out_dir/frame_XXX.jpg, depth_XXX.f32 and manifest.json."""
    import json
    from pathlib import Path

    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    centre = np.array([scene.drawer_w / 2, scene.drawer_h / 2, 0.0])
    frames = []
    for i in range(n_frames):
        a = 2 * np.pi * i / n_frames
        # two loops at different heights/radii for baseline (heights are +up here; render_from_pose flips z)
        r = radius * (1.0 if i % 2 == 0 else 0.6)
        h = height * (1.0 if i % 3 else 0.85)
        cam = centre + np.array([r * np.cos(a), r * np.sin(a), h])
        jpg, depth, intr, grav, *_ = render_from_pose(scene, cam, centre, seed=seed + i)
        (out / f"frame_{i:03d}.jpg").write_bytes(jpg)
        (out / f"depth_{i:03d}.f32").write_bytes(depth.astype("<f4").tobytes())
        frames.append({"image": f"frame_{i:03d}.jpg", "depth": f"depth_{i:03d}.f32", "depth_width": DEPTH_W,
                       "depth_height": DEPTH_H, "gravity": [float(v) for v in grav], "intrinsics": intr})
    (out / "manifest.json").write_text(json.dumps({"frames": frames}))
    return out


def render_still(scene: Scene, cam_xy: Tuple[float, float], height: float, tilt_toward: Tuple[float, float] = (0.0, 0.0), seed: int = 0):
    """One still from above (right-handed renderer): camera over cam_xy at `height`, looking at the point
    under it shifted by tilt_toward (mm). Returns (jpeg, depth m, intrinsics)."""
    cam = np.array([cam_xy[0], cam_xy[1], height])
    look = np.array([cam_xy[0] + tilt_toward[0], cam_xy[1] + tilt_toward[1], 0.0])
    jpg, depth, intr, *_ = render_from_pose(scene, cam, look, seed=seed)
    return jpg, depth, intr


def render_sweep_path(scene: Scene, out_dir, n_frames: int = 36, rx: float = 350.0, ry: float = 150.0, height: float = 650.0, seed: int = 0):
    """Like render_sweep but on an ellipse (rx, ry) so long drawers get covered end to end."""
    import json
    from pathlib import Path

    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    centre = np.array([scene.drawer_w / 2, scene.drawer_h / 2, 0.0])
    frames = []
    for i in range(n_frames):
        a = 2 * np.pi * i / n_frames
        f = 1.0 if i % 2 == 0 else 0.6
        h = height * (1.0 if i % 3 else 0.85)
        cam = centre + np.array([rx * f * np.cos(a), ry * f * np.sin(a), h])
        look = centre + np.array([0.35 * rx * f * np.cos(a), 0.35 * ry * f * np.sin(a), 0.0])   # look ahead a bit, not always at centre
        jpg, depth, intr, grav, *_ = render_from_pose(scene, cam, look, seed=seed + i)
        (out / f"frame_{i:03d}.jpg").write_bytes(jpg)
        (out / f"depth_{i:03d}.f32").write_bytes(depth.astype("<f4").tobytes())
        frames.append({"image": f"frame_{i:03d}.jpg", "depth": f"depth_{i:03d}.f32", "depth_width": DEPTH_W,
                       "depth_height": DEPTH_H, "gravity": [float(v) for v in grav], "intrinsics": intr})
    (out / "manifest.json").write_text(json.dumps({"frames": frames}))
    return out


def arkit_transform_from_cv(R: np.ndarray, t: np.ndarray, world_scale: float = 0.001) -> list:
    """OpenCV world->camera (R, t in mm) -> ARKit camera-to-world 4x4 (metres, column-major 16 floats)."""
    R_c2w = R.T
    cam_pos = -R.T @ t
    R_arkit = R_c2w @ np.diag([1.0, -1.0, -1.0])       # columns: ARKit camera axes in world
    M = np.eye(4)
    M[:3, :3] = R_arkit
    M[:3, 3] = cam_pos * world_scale
    return [float(v) for v in M.T.ravel()]              # column-major like simd_float4x4


def render_arc_frames(scene: Scene, n_frames: int = 12, height: float = 550.0, margin: float = 120.0, seed: int = 0):
    """Frames along a straight-ish pass over a long drawer, each looking down: (jpg, depth, intr, transform)."""
    frames = []
    xs = np.linspace(margin, scene.drawer_w - margin, n_frames)
    for i, x in enumerate(xs):
        y = scene.drawer_h / 2 + (25.0 if i % 2 else -25.0)
        cam = np.array([x, y, height])
        look = np.array([x + (12.0 if i % 2 else -12.0), scene.drawer_h / 2, 0.0])
        jpg, depth, intr, grav, K, R, t = render_from_pose(scene, cam, look, seed=seed + i)
        frames.append((jpg, depth, intr, arkit_transform_from_cv(R, t)))
    return frames


def render_human_arc(scene: Scene, n_frames: int, seed: int = 0, height: float = 550.0,
                     pose_noise_mm: float = 2.0, pose_noise_deg: float = 0.3):
    """A hand-held glide from one end of the drawer to the other, imperfectly:
    uneven speed, height wobble and drift, lateral wander, random tilt and roll, motion blur on
    some frames, and ARKit-like pose error on the reported transforms.
    Returns frames as (jpg, depth, intr, transform) like render_arc_frames."""
    rng = np.random.default_rng(seed)
    # uneven progress along x (speed varies frame to frame), starting/ending a little past the ends
    steps = rng.uniform(0.4, 1.6, n_frames - 1); steps /= steps.sum()
    prog = np.concatenate([[0.0], np.cumsum(steps)])
    x0, x1 = 60.0 + rng.uniform(-40, 40), scene.drawer_w - 60.0 + rng.uniform(-40, 40)
    xs = x0 + prog * (x1 - x0)
    drift = rng.uniform(-40, 40)
    frames = []
    for i, x in enumerate(xs):
        h = height + drift * (i / max(1, n_frames - 1)) + rng.normal(0, 25)
        y = scene.drawer_h / 2 + rng.normal(0, 30)
        cam = np.array([x, y, h])
        look = np.array([x + rng.normal(0, 45), scene.drawer_h / 2 + rng.normal(0, 40), 0.0])   # tilt wobble
        roll = rng.normal(0, 4.0)
        blur = float(rng.choice([0.0, 0.0, 0.0, 3.0, 5.0]))
        jpg, depth, intr, grav, K, R, t = render_from_pose(scene, cam, look, roll_deg=roll, seed=seed * 100 + i, blur_px=blur)
        # reported pose = true pose + tracking error
        M = np.asarray(arkit_transform_from_cv(R, t)).reshape(4, 4).T
        dth = np.deg2rad(rng.normal(0, pose_noise_deg, 3))
        Rx = np.array([[1, 0, 0], [0, np.cos(dth[0]), -np.sin(dth[0])], [0, np.sin(dth[0]), np.cos(dth[0])]])
        Ry = np.array([[np.cos(dth[1]), 0, np.sin(dth[1])], [0, 1, 0], [-np.sin(dth[1]), 0, np.cos(dth[1])]])
        Rz = np.array([[np.cos(dth[2]), -np.sin(dth[2]), 0], [np.sin(dth[2]), np.cos(dth[2]), 0], [0, 0, 1]])
        M[:3, :3] = Rx @ Ry @ Rz @ M[:3, :3]
        M[:3, 3] += rng.normal(0, pose_noise_mm, 3) / 1000.0
        frames.append((jpg, depth, intr, [float(v) for v in M.T.ravel()]))
    return frames
