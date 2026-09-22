"""Synthetic inputs with known ground truth: a perspective photo and a 3D scan of the same layout."""
from __future__ import annotations

import io
from typing import Dict, List, Tuple

import cv2
import numpy as np
from shapely import affinity
from shapely.geometry import Point, Polygon, box
from shapely.ops import unary_union

MAT_W, MAT_H = 300.0, 200.0  # mm


def tool_shapes() -> Dict[str, Tuple[Polygon, float]]:
    """name -> (shape in mat mm coords, thickness mm)."""
    wrench = unary_union([box(40, 40, 160, 65), Point(160, 52.5).buffer(20, 64)])
    disc = Point(220, 130).buffer(25, 64)
    bar = affinity.rotate(box(110, 132.5, 190, 147.5), 30, origin=(150, 140))
    return {"wrench": (wrench, 8.0), "disc": (disc, 12.0), "bar": (bar, 5.0)}


def render_mat(ppm: float = 6.0) -> np.ndarray:
    W, H = int(MAT_W * ppm), int(MAT_H * ppm)
    rng = np.random.default_rng(1)
    img = np.full((H, W, 3), 205, np.uint8)
    img = np.clip(img.astype(np.int16) + rng.integers(-6, 7, (H, W, 1)), 0, 255).astype(np.uint8)
    for name, (shape, _) in tool_shapes().items():
        pts = (np.asarray(shape.exterior.coords) * ppm).astype(np.int32)
        cv2.fillPoly(img, [pts], (58, 60, 64))
    return img


def photo_corners() -> np.ndarray:
    # trapezoid in a 2400x1800 photo: TL, TR, BR, BL
    return np.array([[420, 380], [2010, 300], [2180, 1520], [300, 1440]], np.float32)


def make_photo() -> Tuple[bytes, np.ndarray]:
    ppm = 6.0
    mat = render_mat(ppm)
    W, H = mat.shape[1], mat.shape[0]
    src = np.array([[0, 0], [W, 0], [W, H], [0, H]], np.float32)
    dst = photo_corners()
    Hm = cv2.getPerspectiveTransform(src, dst)
    rng = np.random.default_rng(2)
    bg = np.full((1800, 2400, 3), (70, 95, 130), np.uint8)  # brownish desk
    bg = np.clip(bg.astype(np.int16) + rng.integers(-10, 11, (1800, 2400, 1)), 0, 255).astype(np.uint8)
    warped = cv2.warpPerspective(mat, Hm, (2400, 1800))
    mask = cv2.warpPerspective(np.full((H, W), 255, np.uint8), Hm, (2400, 1800))
    photo = np.where(mask[..., None] > 0, warped, bg)
    photo = cv2.GaussianBlur(photo, (3, 3), 0)
    ok, buf = cv2.imencode(".jpg", photo, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return buf.tobytes(), dst


def make_scan_ply() -> bytes:
    """Textured single-sided mat plane + extruded tools, in meters, arbitrarily rotated."""
    import trimesh

    parts = []
    # the mat surface with the tool footprints removed (a scanner cannot see under a tool)
    mat_poly = box(0, 0, MAT_W, MAT_H).difference(unary_union([s for s, _ in tool_shapes().values()]))
    v2, f = trimesh.creation.triangulate_polygon(mat_poly, engine="earcut")
    mat = trimesh.Trimesh(vertices=np.column_stack([v2, np.zeros(len(v2))]), faces=f, process=False)
    mat = mat.subdivide().subdivide()
    mat.visual.vertex_colors = np.tile([205, 205, 205, 255], (len(mat.vertices), 1))
    parts.append(mat)
    for name, (shape, thick) in tool_shapes().items():
        solid = trimesh.creation.extrude_polygon(shape, thick)
        solid.update_faces(solid.triangles_center[:, 2] > 1e-6)  # remove the underside (z = 0)
        solid.visual.vertex_colors = np.tile([64, 60, 58, 255], (len(solid.vertices), 1))
        parts.append(solid)
    scene = trimesh.util.concatenate(parts)
    # simulate a drawer floor extending beyond the mat region a bit (still same plane)
    scene.apply_scale(0.001)  # -> meters
    rot = trimesh.transformations.euler_matrix(0.4, -0.7, 1.9)
    rot[:3, 3] = [0.35, -0.2, 1.1]
    scene.apply_transform(rot)
    # add mild measurement noise
    rng = np.random.default_rng(3)
    scene.vertices += rng.normal(0, 0.00025, scene.vertices.shape)
    return scene.export(file_type="ply")


def make_object_ply() -> Tuple[bytes, Tuple[float, float, float]]:
    """A single tool model: a 180 x 30 mm bar 12 mm thick with a 40 mm knob 20 mm tall, randomly posed.

    Returns (ply bytes, (footprint w, footprint h, height)) for the flat resting pose.
    """
    import trimesh

    bar = trimesh.creation.box((180, 30, 12))
    knob = trimesh.creation.cylinder(radius=20, height=20)
    knob.apply_translation((70, 0, 4))  # knob spans z -6..14, bar -6..6 -> total height 20
    solid = trimesh.util.concatenate([bar, knob])
    solid.visual.vertex_colors = np.tile([90, 90, 95, 255], (len(solid.vertices), 1))
    pts, _ = trimesh.sample.sample_surface(solid, 400_000)
    cloud = trimesh.PointCloud(pts, colors=np.tile([90, 90, 95, 255], (len(pts), 1)))
    rot = trimesh.transformations.euler_matrix(1.1, 0.6, -2.0)
    rot[:3, 3] = [500, -120, 80]
    cloud.apply_transform(rot)
    return cloud.export(file_type="ply"), (180.0, 40.0, 20.0)
