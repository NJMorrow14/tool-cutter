"""Phone captures: a top-down still (+ optional LiDAR depth + intrinsics) of a drawer with ArUco markers.

Two geometric paths give a metric top-down raster of the drawer floor:
  * RGB-D: fit the floor plane in the depth point cloud, re-project the full-resolution photo onto
    that plane (exact for the floor), and rasterize the depth points as a height map.
  * Markers only: one ArUco marker of known size fixes the image->plane homography; all visible
    markers refine it by least squares. No heights (the user types them).

Because a tool's silhouette in the photo is its *top* surface, the floor-level rectification shows
it displaced away from the camera nadir by h * r / D. `footprint_from_mask` undoes that once the
tool's height is known (RGB-D path).
"""
from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .calibration import order_corners
from .scan import _fill_nan, fit_plane_ransac

log = logging.getLogger(__name__)

ARUCO_DICTS = {
    "4X4_50": cv2.aruco.DICT_4X4_50,
    "5X5_50": cv2.aruco.DICT_5X5_50,
    "6X6_50": cv2.aruco.DICT_6X6_50,
    "APRILTAG_36H11": cv2.aruco.DICT_APRILTAG_36h11,
}
MIXED_NEAR_FRAC = 0.65  # mixed depth pixel goes to the near surface when its value is within this fraction of the jump
RIM_TRIM_FRAC = 0.35    # LiDAR blob rim pixels below this fraction of the local top are trimmed
CORNER_IDS = [0, 1, 2, 3]  # printed sheet convention: id 0 top-left, 1 top-right, 2 bottom-right, 3 bottom-left


@dataclass
class CaptureGeometry:
    """Camera + floor-plane frame needed to re-project masks at other heights (RGB-D path)."""
    K: np.ndarray                 # 3x3 intrinsics of the photo
    n: np.ndarray                 # unit plane normal pointing toward the camera
    c: np.ndarray                 # plane origin (camera coords, mm)
    e_u: np.ndarray               # plane x axis (image right)
    e_v: np.ndarray               # plane y axis (image down)
    D: float                      # camera height above the plane (mm)
    u0: float                     # raster origin in plane coords (mm)
    v0: float
    mm_per_px: float
    depth_cell_mm: float = 3.0    # footprint of one depth pixel on the floor

    def raster_to_plane(self, px: np.ndarray) -> np.ndarray:
        return np.column_stack([self.u0 + px[:, 0] * self.mm_per_px, self.v0 + px[:, 1] * self.mm_per_px])

    def plane_to_raster(self, uv: np.ndarray) -> np.ndarray:
        return np.column_stack([(uv[:, 0] - self.u0) / self.mm_per_px, (uv[:, 1] - self.v0) / self.mm_per_px])

    def plane_to_cam(self, uv: np.ndarray, h: float = 0.0) -> np.ndarray:
        return self.c[None, :] + uv[:, :1] * self.e_u[None, :] + uv[:, 1:2] * self.e_v[None, :] + h * self.n[None, :]

    def cam_to_plane(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        rel = X - self.c[None, :]
        return np.column_stack([rel @ self.e_u, rel @ self.e_v]), rel @ self.n

    def project(self, X: np.ndarray) -> np.ndarray:
        z = np.clip(X[:, 2], 1e-6, None)
        return np.column_stack([self.K[0, 0] * X[:, 0] / z + self.K[0, 2], self.K[1, 1] * X[:, 1] / z + self.K[1, 2]])

    def footprint_correction(self, raster_pts: np.ndarray, h: float) -> np.ndarray:
        """Raster points of a silhouette seen at floor level -> raster points of the true footprint at height h."""
        uv = self.raster_to_plane(np.asarray(raster_pts, dtype=np.float64))
        X0 = self.plane_to_cam(uv, 0.0)
        lam = (self.D - h) / self.D           # the same image ray meets the height-h plane closer to the camera
        Xh = X0 * lam
        uv_h, _ = self.cam_to_plane(Xh)
        return self.plane_to_raster(uv_h)


@dataclass
class CaptureResult:
    color_bgr: np.ndarray                 # floor-level metric raster
    height_mm: Optional[np.ndarray]       # same grid, from depth (None for markers-only)
    mm_per_px: float
    drawer_corners: Optional[np.ndarray]  # 4x2 raster px (TL,TR,BR,BL) from the markers' outer corners
    markers_found: List[int]
    geometry: Optional[CaptureGeometry]
    meta: Dict = field(default_factory=dict)
    markers_px: Dict[int, np.ndarray] = field(default_factory=dict)   # marker corners in raster px


# ----------------------------------------------------------------------------- markers

def detect_markers(img_bgr: np.ndarray, dict_name: str = "4X4_50") -> Dict[int, np.ndarray]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICTS.get(dict_name, cv2.aruco.DICT_4X4_50))
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(dictionary, params)
    corners, ids, _ = detector.detectMarkers(gray)
    out: Dict[int, np.ndarray] = {}
    if ids is None:
        return out
    for c, i in zip(corners, ids.ravel()):
        out[int(i)] = c.reshape(4, 2).astype(np.float64)
    return out


def _square(size: float) -> np.ndarray:
    return np.array([[0, 0], [size, 0], [size, size], [0, size]], dtype=np.float64)


def homography_from_markers(markers: Dict[int, np.ndarray], marker_size_mm: float) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
    """Image -> metric plane homography (mm) from coplanar markers of known size.

    One marker fixes H (4 point pairs). Extra markers add 4 corners each with unknown in-plane pose
    (x, y, theta); everything is refined jointly by least squares. Returns H and each marker's
    corners in plane mm.
    """
    from scipy.optimize import least_squares

    ids = sorted(markers)
    # start from the biggest marker in the image
    areas = {i: cv2.contourArea(markers[i].astype(np.float32)) for i in ids}
    base = max(ids, key=areas.get)
    H0, _ = cv2.findHomography(markers[base].astype(np.float32), _square(marker_size_mm).astype(np.float32))
    if H0 is None:
        raise RuntimeError("Could not fit a homography to the markers")
    others = [i for i in ids if i != base]
    if not others:
        return H0, {base: _square(marker_size_mm)}

    def apply(H, pts):
        p = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), H).reshape(-1, 2)
        return p

    # initial poses of the other markers from H0
    poses0 = []
    for i in others:
        q = apply(H0, markers[i])
        ang = math.atan2(q[1, 1] - q[0, 1], q[1, 0] - q[0, 0])
        poses0 += [q[0, 0], q[0, 1], ang]
    h0 = (H0 / H0[2, 2]).ravel()[:8]
    x0 = np.concatenate([h0, poses0])
    sq = _square(marker_size_mm)

    def residuals(x):
        H = np.append(x[:8], 1.0).reshape(3, 3)
        res = [apply(H, markers[base]) - sq]
        for k, i in enumerate(others):
            tx, ty, th = x[8 + 3 * k: 11 + 3 * k]
            R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
            target = sq @ R.T + np.array([tx, ty])
            res.append(apply(H, markers[i]) - target)
        return np.concatenate(res).ravel()

    sol = least_squares(residuals, x0, method="lm", max_nfev=2000)
    H = np.append(sol.x[:8], 1.0).reshape(3, 3)
    plane_corners = {i: apply(H, markers[i]) for i in ids}
    return H, plane_corners


_SPARE = -1   # stands in for the one corner marker a 3-marker capture is missing


def corner_markers(markers: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
    """Only the printed sheet's four ids. ArUco reads a 4X4 code out of clutter now and then (keyboard keys
    produced ids 16/17/28/37 in a real scan), and one bogus marker is enough to rotate a frame's raster or
    wreck the markers-only homography, which assumes every marker is `marker_size_mm` across."""
    return {i: c for i, c in markers.items() if i in CORNER_IDS}


def _clockwise_cycle(pts: Dict[int, np.ndarray]) -> List[int]:
    """ids ordered clockwise about their centroid. The frame is x right / y DOWN, where a growing
    atan2 angle turns clockwise on screen."""
    ctr = np.mean(np.stack([pts[i] for i in pts]), axis=0)
    return sorted(pts, key=lambda i: math.atan2(pts[i][1] - ctr[1], pts[i][0] - ctr[0]))


def corner_order(points: Dict[int, np.ndarray]) -> Optional[Dict[int, int]]:
    """Marker id -> corner slot (0 TL, 1 TR, 2 BR, 3 BL), worked out from where the markers actually lie.

    `points` holds 3 or 4 marker centres in a metric top-down frame that is NOT mirrored (x right, y down).
    TL,TR,BR,BL runs clockwise around a rectangle, so the ids sorted clockwise about their centroid already
    are the slots up to which one starts the cycle; the start is picked to agree with the printed sheet
    wherever it can, so a correctly laid-out sheet maps to itself and nothing moves.

    Returns None when the markers do not form a convex quad (a bad detection): better to leave the printed
    ids alone than to invent an order.
    """
    pts = {int(i): np.asarray(p, dtype=np.float64).reshape(2) for i, p in points.items() if int(i) in CORNER_IDS}
    if len(pts) == 3:
        # complete the rectangle: the right-angled corner is the one between the other two, and the missing
        # corner is its reflection through their midpoint
        ids = list(pts)
        cos = {}
        for k in ids:
            a, b = [pts[j] - pts[k] for j in ids if j != k]
            cos[k] = abs(float(a @ b)) / (float(np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9)
        k = min(cos, key=cos.get)
        if cos[k] > 0.35:            # > ~20 deg out of square: not three corners of a rectangle
            return None
        a, b = [j for j in ids if j != k]
        pts[_SPARE] = pts[a] + pts[b] - pts[k]
    if len(pts) != 4:
        return None
    cyc = _clockwise_cycle(pts)
    quad = np.stack([pts[i] for i in cyc])
    e = np.roll(quad, -1, axis=0) - quad
    nxt = np.roll(e, -1, axis=0)
    cross = e[:, 0] * nxt[:, 1] - e[:, 1] * nxt[:, 0]     # all one sign <=> convex, and > 0 for this cycle
    lens = np.linalg.norm(e, axis=1)
    if np.any(cross <= 0) or lens.min() < 0.02 * lens.max():
        return None
    start = max(range(4), key=lambda k: (sum(cyc[(s + k) % 4] == s for s in range(4)), -k))
    return {cyc[(s + start) % 4]: s for s in range(4) if cyc[(s + start) % 4] != _SPARE}


def relabel_markers(markers: Dict[int, np.ndarray], mapping: Optional[Dict[int, int]]) -> Dict[int, np.ndarray]:
    """Re-key detected markers by corner slot. Ids the map does not mention are dropped; a None map
    (too few markers to tell) leaves everything as it is."""
    if not mapping:
        return markers
    return {mapping[i]: c for i, c in markers.items() if i in mapping}


def outer_corner(corners: np.ndarray, marker_id: int) -> np.ndarray:
    """The marker corner that touches the drawer corner, in an ORIENTED frame (id 0 top-left, y down):
    id 0 -> min x & min y, 1 -> max x & min y, 2 -> max x & max y, 3 -> min x & max y."""
    sx = -1 if marker_id in (0, 3) else 1
    sy = -1 if marker_id in (0, 1) else 1
    score = sx * corners[:, 0] + sy * corners[:, 1]
    return corners[int(np.argmax(score))]


def drawer_rectangle_from_markers(plane_corners: Dict[int, np.ndarray]) -> Optional[np.ndarray]:
    """Drawer corners (mm) = each corner marker's outermost corner (TL, TR, BR, BL order).

    Works with all four corner markers, or with three (a tall tool often hides one): the plane
    coordinates are metric and orthographic, so the missing corner of the rectangle follows from
    the other three as a parallelogram completion.
    """
    present = [i for i in CORNER_IDS if i in plane_corners]
    if len(present) < 3:
        return None
    pts: Dict[int, np.ndarray] = {i: outer_corner(plane_corners[i], i) for i in present}
    if len(present) == 3:
        missing = [i for i in CORNER_IDS if i not in pts][0]
        opp = (missing + 2) % 4              # opposite corner
        a, b = (missing + 1) % 4, (missing + 3) % 4   # the two neighbours
        pts[missing] = pts[a] + pts[b] - pts[opp]
    return np.asarray([pts[i] for i in CORNER_IDS])


def rectangle_from_two_markers(markers_px: Dict[int, np.ndarray], width_px: float, height_px: float) -> Optional[np.ndarray]:
    """Drawer corners (TL,TR,BR,BL, raster px) from two ADJACENT corner markers plus the known drawer size.

    Used when a closer shot of a long drawer only shows one end. The raster is metric and oriented
    (marker 0 top-left), so the missing edge is perpendicular to the visible one.
    """
    ids = [i for i in CORNER_IDS if i in markers_px]
    if len(ids) < 2:
        return None
    pts: Dict[int, np.ndarray] = {i: outer_corner(markers_px[i], i) for i in ids}

    def perp_cw(v):   # rotate by +90 deg in image coords (x right, y down): (dx,dy) -> (-dy, dx)
        return np.array([-v[1], v[0]])

    if 0 in pts and 1 in pts:            # top edge known
        e = pts[1] - pts[0]; e /= np.linalg.norm(e); down = perp_cw(e) * height_px
        pts[3] = pts[0] + down; pts[2] = pts[1] + down
    elif 3 in pts and 2 in pts:          # bottom edge
        e = pts[2] - pts[3]; e /= np.linalg.norm(e); down = perp_cw(e) * height_px
        pts[0] = pts[3] - down; pts[1] = pts[2] - down
    elif 0 in pts and 3 in pts:          # left edge
        d = pts[3] - pts[0]; d /= np.linalg.norm(d); right = -perp_cw(d) * width_px
        pts[1] = pts[0] + right; pts[2] = pts[3] + right
    elif 1 in pts and 2 in pts:          # right edge
        d = pts[2] - pts[1]; d /= np.linalg.norm(d); right = -perp_cw(d) * width_px
        pts[0] = pts[1] - right; pts[3] = pts[2] - right
    else:
        return None   # diagonal pair: ambiguous without the rectangle's orientation
    return np.asarray([pts[i] for i in CORNER_IDS])


def marker_axis_angle(corners: np.ndarray) -> float:
    """In-plane angle (rad, image coords) of a marker's +x edge, averaged over its four edges."""
    e = [corners[1] - corners[0], corners[2] - corners[1], corners[3] - corners[2], corners[0] - corners[3]]
    angs = [math.atan2(v[1], v[0]) - k * math.pi / 2 for k, v in enumerate(e)]
    ref = angs[0]
    angs = [a - 2 * math.pi * round((a - ref) / (2 * math.pi)) for a in angs]
    return float(np.mean(angs))


def rectangle_from_one_marker(markers_px: Dict[int, np.ndarray], width_px: float, height_px: float) -> Optional[np.ndarray]:
    """Drawer corners (TL,TR,BR,BL, raster px) from ONE corner marker plus the known drawer size, in an
    oriented metric raster. Its outer corner is a drawer corner; the marker's edges give the axes."""
    ids = [i for i in CORNER_IDS if i in markers_px]
    if not ids:
        return None
    mid = ids[0]
    c = markers_px[mid]
    oc = outer_corner(c, mid)
    th = marker_axis_angle(c)
    ex = np.array([math.cos(th), math.sin(th)]) * width_px
    ey = np.array([-math.sin(th), math.cos(th)]) * height_px
    if mid == 0:
        tl = oc
    elif mid == 1:
        tl = oc - ex
    elif mid == 2:
        tl = oc - ex - ey
    else:
        tl = oc - ey
    return np.asarray([tl, tl + ex, tl + ex + ey, tl + ey])


def expand_rectangle(corners: np.ndarray, inset: float) -> np.ndarray:
    """Push the corners of an ordered rectangle (TL,TR,BR,BL) outward by `inset` along the rectangle's own
    axes (not along the diagonal, which over-expands the long side of an elongated rectangle)."""
    c = np.asarray(corners, dtype=np.float64)
    ex = c[1] - c[0]; ex /= np.linalg.norm(ex) + 1e-9
    ey = c[3] - c[0]; ey /= np.linalg.norm(ey) + 1e-9
    signs = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
    return np.asarray([c[i] + inset * (sx * ex + sy * ey) for i, (sx, sy) in enumerate(signs)])


def _orient_for_markers(plane_corners: Dict[int, np.ndarray], mirror_ok: bool = False) -> np.ndarray:
    """2x3 affine (on plane mm) that puts marker 0 top-left, 1 top-right, 2 bottom-right, 3 bottom-left
    (quarter turns only). Any adjacent pair of markers fixes the turn; with none, identity."""
    # expected direction (angle in image coords, x right / y down) of the vector between adjacent ids
    # adjacent pairs first; a diagonal also fixes the quarter turn, because the 0->2 diagonal of a rectangle
    # always points somewhere between +x and +y, i.e. within 45 deg of the 45 deg listed here, whatever the
    # aspect ratio. (Common once the ids are ordered by position: a glide sees one short edge at a time.)
    expected = {(0, 1): 0.0, (1, 2): math.pi / 2, (2, 3): math.pi, (3, 0): -math.pi / 2,
                (0, 2): math.pi / 4, (1, 3): 3 * math.pi / 4}
    ang = None
    for (a, b), exp in expected.items():
        if a in plane_corners and b in plane_corners:
            ca, cb = plane_corners[a].mean(axis=0), plane_corners[b].mean(axis=0)
            ang = math.atan2(cb[1] - ca[1], cb[0] - ca[0]) - exp
            break
    if ang is None and plane_corners:
        # a single marker: its own top edge (corner 0 -> 1, printed upright) points along the drawer's +x
        c = next(iter(plane_corners.values()))
        ang = math.atan2(c[1, 1] - c[0, 1], c[1, 0] - c[0, 0])
    if ang is None:
        return np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float64)
    k = int(round(-ang / (math.pi / 2))) % 4
    th = k * math.pi / 2
    return np.array([[math.cos(th), -math.sin(th), 0], [math.sin(th), math.cos(th), 0]], dtype=np.float64)


# ----------------------------------------------------------------------------- markers-only path

def rectify_markers_only(img_bgr: np.ndarray, markers: Dict[int, np.ndarray], marker_size_mm: float,
                         max_side: int = 2200, margin_mm: float = 15.0,
                         corner_map: Optional[Dict[int, int]] = None) -> CaptureResult:
    markers = corner_markers(markers)
    if not markers:
        raise RuntimeError("No corner markers (ids 0-3) found in the photo")
    H, plane_corners = homography_from_markers(markers, marker_size_mm)
    # which corner each id sits on comes from the geometry, not from the printed numbering
    cmap = corner_map if corner_map is not None else corner_order({i: c.mean(axis=0) for i, c in plane_corners.items()})
    plane_corners = relabel_markers(plane_corners, cmap)
    A = _orient_for_markers(plane_corners)
    A3 = np.vstack([A, [0, 0, 1]])
    H = A3 @ H
    plane_corners = {i: cv2.transform(c.reshape(-1, 1, 2), A).reshape(-1, 2) for i, c in plane_corners.items()}
    rect = drawer_rectangle_from_markers(plane_corners)
    if rect is not None:
        rect = order_corners(rect)
        umin, vmin = rect.min(axis=0) - margin_mm
        umax, vmax = rect.max(axis=0) + margin_mm
    else:
        # whole photo footprint on the plane
        h, w = img_bgr.shape[:2]
        border = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        pb = cv2.perspectiveTransform(border.reshape(-1, 1, 2), H.astype(np.float32)).reshape(-1, 2)
        umin, vmin = pb.min(axis=0)
        umax, vmax = pb.max(axis=0)
        span = max(umax - umin, vmax - vmin)
        if span > 2000:   # wildly extrapolated corners: clamp around the markers
            allc = np.vstack(list(plane_corners.values()))
            umin, vmin = allc.min(axis=0) - 300
            umax, vmax = allc.max(axis=0) + 300
    ppm = min(max_side / max(umax - umin, vmax - vmin), 12.0)
    W = int(math.ceil((umax - umin) * ppm))
    Hh = int(math.ceil((vmax - vmin) * ppm))
    S = np.array([[ppm, 0, -umin * ppm], [0, ppm, -vmin * ppm], [0, 0, 1]])
    Ht = S @ H
    warped = cv2.warpPerspective(img_bgr, Ht, (W, Hh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    corners_px = None if rect is None else (rect - np.array([umin, vmin])) * ppm
    mpx = {i: (pc - np.array([umin, vmin])) * ppm for i, pc in plane_corners.items()}
    return CaptureResult(color_bgr=warped, height_mm=None, mm_per_px=1.0 / ppm, drawer_corners=corners_px,
                         markers_found=sorted(plane_corners), geometry=None,
                         meta={"mode": "markers", "marker_size_mm": marker_size_mm, "corner_map": cmap}, markers_px=mpx)


# ----------------------------------------------------------------------------- RGB-D path

def remove_flying_pixels(depth_m: np.ndarray, tol_mm: float = 5.0, jump_mm: float = 12.0,
                         nadir_px: Optional[Sequence[float]] = None) -> np.ndarray:
    """Clean depth discontinuities the way a ToF/LiDAR sensor blurs them.

    * A *mixed* pixel straddles a top edge: its 3x3 neighbourhood contains both the near surface
      (tool top, ~dmin) and the far one (floor, ~dmax) and its own value lies in between. Unprojected
      as-is it lands in mid-air, i.e. outside the footprint. It is snapped to whichever surface its
      value is closer to (mostly-covered -> top, mostly-uncovered -> floor), which keeps the edge
      unbiased to within half a pixel.
    * A *wall* pixel (steep side facing the camera) also has an intermediate depth but its neighbours
      are wall pixels too (no near AND far surface both present). Snapping it to the top would land
      far outside the footprint, so it is dropped; the footprint edge on that side comes from the
      top samples, which project orthographically onto the true edge anyway.
    * On the edge FACING the camera foot point (depth increases toward the nadir: floor on the nadir
      side, tool beyond) the intermediate pixels are genuine wall samples, which already sit on the true
      footprint edge; snapping them to the top would slide them along their rays toward the nadir, i.e.
      h*r/D outside the footprint. With `nadir_px` given they are kept as measured.
    Steep-but-continuous surfaces (sphere rims) have small 3x3 depth ranges and are left alone.
    """
    d = depth_m.astype(np.float32)
    valid = np.isfinite(d) & (d > 0)
    if not valid.any():
        return np.full_like(d, np.nan)
    k = np.ones((3, 3), np.uint8)
    dmax = cv2.dilate(np.where(valid, d, -np.inf), k)
    dmin = cv2.erode(np.where(valid, d, np.inf), k)
    tol = tol_mm / 1000.0
    intermediate = valid & (d > dmin + tol) & (d < dmax - tol) & (dmax - dmin > jump_mm / 1000.0)
    # does the neighbourhood contain a near-surface pixel and a far-surface pixel?
    near_present = cv2.dilate((np.abs(d - dmin) <= tol).astype(np.uint8), k) > 0
    far_present = cv2.dilate((np.abs(d - dmax) <= tol).astype(np.uint8), k) > 0
    # the pixel itself is near/far for its neighbours; exclude self by requiring another such neighbour:
    # approximate by requiring both flags (self is neither near nor far when intermediate)
    mixed = intermediate & near_present & far_present
    wall = intermediate & ~mixed
    if nadir_px is not None:
        # Invalid/low-confidence samples are holes, not a zero-distance surface.
        from scipy.ndimage import distance_transform_edt
        _, indices = distance_transform_edt(~valid, return_indices=True)
        gradient_depth = d[tuple(indices)] if not valid.all() else d
        gx = cv2.Sobel(gradient_depth, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gradient_depth, cv2.CV_32F, 0, 1, ksize=3)
        vs, us = np.mgrid[0:d.shape[0], 0:d.shape[1]]
        toward = gx * (float(nadir_px[0]) - us) + gy * (float(nadir_px[1]) - vs)
        facing = intermediate & (toward > 0)
        mixed &= ~facing
        wall &= ~facing
    out = d.copy()
    # a mixed pixel's value is (roughly) area-weighted between the two surfaces: mostly-tool pixels go to
    # the top, mostly-floor pixels to the floor. Snapping every mixed pixel to the top would push each far
    # edge outward by up to a full depth cell.
    mix_frac = float(os.environ.get("TC_MIX_FRAC", MIXED_NEAR_FRAC))
    to_near = mixed & (d - dmin <= mix_frac * (dmax - dmin))
    to_far = mixed & ~to_near
    out[to_near] = dmin[to_near]
    out[to_far] = dmax[to_far]
    out[wall] = np.nan
    out[~valid] = np.nan
    return out


def depth_to_points(depth_m: np.ndarray, K_depth: np.ndarray) -> np.ndarray:
    h, w = depth_m.shape
    vs, us = np.mgrid[0:h, 0:w]
    z = depth_m.astype(np.float64) * 1000.0
    x = (us - K_depth[0, 2]) / K_depth[0, 0] * z
    y = (vs - K_depth[1, 2]) / K_depth[1, 1] * z
    pts = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    return pts[np.isfinite(pts).all(axis=1) & (pts[:, 2] > 50)]


def rectify_rgbd(img_bgr: np.ndarray, depth_m: np.ndarray, K: np.ndarray, markers: Dict[int, np.ndarray],
                 marker_size_mm: Optional[float] = None, max_side: int = 2200, margin_mm: float = 15.0,
                 corner_map: Optional[Dict[int, int]] = None, preserve_unknown: bool = False) -> CaptureResult:
    """Metric top-down raster from a photo + aligned depth map (depth may be lower resolution)."""
    markers = corner_markers(markers)
    ih, iw = img_bgr.shape[:2]
    dh, dw = depth_m.shape
    K_depth = K.copy()
    K_depth[0, :] *= dw / iw
    K_depth[1, :] *= dh / ih
    # a first plane fit on the raw points (RANSAC ignores the flying pixels) gives the camera foot point,
    # which the flying-pixel filter needs to tell camera-facing walls from far edges
    pts_raw = depth_to_points(depth_m, K_depth)
    nadir_px = None
    if len(pts_raw) >= 200:
        n0, d0, _ = fit_plane_ransac(pts_raw, iters=150, thresh=4.0)
        if d0 < 0:
            n0, d0 = -n0, -d0
        foot = -n0 * float(d0)
        if foot[2] > 1e-6:
            nadir_px = (K_depth[0, 0] * foot[0] / foot[2] + K_depth[0, 2], K_depth[1, 1] * foot[1] / foot[2] + K_depth[1, 2])
    depth_clean = remove_flying_pixels(depth_m, nadir_px=nadir_px)
    pts = depth_to_points(depth_clean, K_depth)
    if len(pts) < 200:
        raise RuntimeError("Depth map has too few valid points")
    n, d, inl = fit_plane_ransac(pts, iters=300, thresh=4.0)
    # camera sits at the origin above the floor: make the normal point toward it
    if n @ np.zeros(3) + d < 0:
        n, d = -n, -d
    D = float(d)                                  # distance camera -> plane along the normal
    # plane frame: origin under the camera, x = image right, y = image down (projected onto the plane)
    c = -n * D
    e_u = np.array([1.0, 0.0, 0.0]); e_u -= (e_u @ n) * n; e_u /= np.linalg.norm(e_u)
    e_v = np.array([0.0, 1.0, 0.0]); e_v -= (e_v @ n) * n; e_v -= (e_v @ e_u) * e_u; e_v /= np.linalg.norm(e_v)
    geom = CaptureGeometry(K=K, n=n, c=c, e_u=e_u, e_v=e_v, D=D, u0=0.0, v0=0.0, mm_per_px=1.0)

    uv, hgt = geom.cam_to_plane(pts)
    # markers (if any) in plane coordinates via the exact plane geometry, for the drawer rectangle + orientation
    plane_corners: Dict[int, np.ndarray] = {}
    for i, mc in markers.items():
        rays = np.column_stack([(mc[:, 0] - K[0, 2]) / K[0, 0], (mc[:, 1] - K[1, 2]) / K[1, 1], np.ones(4)])
        lam = (-D) / (rays @ n)        # n.X + d = 0 with X = lam * ray  ->  lam = -d / (n.ray)
        X = rays * lam[:, None]
        plane_corners[i], _ = geom.cam_to_plane(X)
    # which corner each id sits on comes from the geometry, not from the printed numbering
    cmap = corner_map if corner_map is not None else corner_order({i: c.mean(axis=0) for i, c in plane_corners.items()})
    plane_corners = relabel_markers(plane_corners, cmap)
    scale_check = None
    if marker_size_mm and plane_corners:
        sides = [np.linalg.norm(np.roll(pc, -1, axis=0) - pc, axis=1).mean() for pc in plane_corners.values()]
        scale_check = float(marker_size_mm / np.mean(sides))   # >1 means LiDAR reads short
    A = _orient_for_markers(plane_corners)
    R = A[:, :2]
    # rotate the plane frame (quarter turns) so marker 0 is top-left
    e_u_new = R[0, 0] * e_u + R[0, 1] * e_v
    e_v_new = R[1, 0] * e_u + R[1, 1] * e_v
    geom.e_u, geom.e_v = e_u_new, e_v_new
    uv = uv @ R.T
    plane_corners = {i: pc @ R.T for i, pc in plane_corners.items()}

    rect = drawer_rectangle_from_markers(plane_corners)
    if rect is not None:
        rect = order_corners(rect)
        umin, vmin = rect.min(axis=0) - margin_mm
        umax, vmax = rect.max(axis=0) + margin_mm
    else:
        lo, hi = np.percentile(uv[inl], [0.5, 99.5], axis=0)
        umin, vmin = lo - margin_mm
        umax, vmax = hi + margin_mm
    ppm = min(max_side / max(umax - umin, vmax - vmin), 12.0)
    W = int(math.ceil((umax - umin) * ppm))
    Hh = int(math.ceil((vmax - vmin) * ppm))
    geom.u0, geom.v0, geom.mm_per_px = float(umin), float(vmin), 1.0 / ppm
    geom.depth_cell_mm = float(D / K_depth[0, 0])

    # color: inverse-map every raster cell (floor level) into the photo
    jj, ii = np.meshgrid(np.arange(W), np.arange(Hh))
    grid_uv = geom.raster_to_plane(np.column_stack([jj.ravel() + 0.5, ii.ravel() + 0.5]))
    Xf = geom.plane_to_cam(grid_uv, 0.0)
    p = geom.project(Xf)
    map_x = p[:, 0].reshape(Hh, W).astype(np.float32)
    map_y = p[:, 1].reshape(Hh, W).astype(np.float32)
    color = cv2.remap(img_bgr, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    # height: depth points rasterized orthographically (true footprint positions)
    rp = geom.plane_to_raster(uv)
    ix = np.floor(rp[:, 0]).astype(np.int64)
    iy = np.floor(rp[:, 1]).astype(np.int64)
    ok = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < Hh)
    height = np.full(W * Hh, -np.inf)
    np.maximum.at(height, iy[ok] * W + ix[ok], hgt[ok])
    height[~np.isfinite(height)] = np.nan
    height = height.reshape(Hh, W).astype(np.float32)
    # depth is far coarser than the color raster: fill the gaps between depth samples from their
    # neighbours (no max-dilation: that would grow every tool by half a depth cell), then smooth lightly
    cell = max(1, int(round((D / K_depth[0, 0]) * ppm)))            # depth pixel size in raster px
    height = _fill_nearest(height, max_dist_px=0.75 * cell, unknown_value=np.nan if preserve_unknown else 0.)
    height = smooth_observed_height(height, max(0.5, cell * 0.25)) if preserve_unknown else cv2.GaussianBlur(height, (0, 0), max(0.5, cell * 0.25))
    height[height < 0] = 0.0

    corners_px = None if rect is None else geom.plane_to_raster(rect)
    mpx = {i: geom.plane_to_raster(pc) for i, pc in plane_corners.items()}
    return CaptureResult(color_bgr=color, height_mm=height, mm_per_px=geom.mm_per_px, drawer_corners=corners_px,
                         markers_found=sorted(plane_corners), geometry=geom,
                         meta={"mode": "rgbd", "camera_height_mm": D, "plane_inlier_fraction": float(inl.mean()),
                               "marker_scale_check": scale_check, "depth_cell_px": cell, "corner_map": cmap}, markers_px=mpx)


def _lidar_footprint(lidar: np.ndarray, height_raster: np.ndarray, nadir_px: np.ndarray, geom: CaptureGeometry,
                     thr_mm: float, homography_after: Optional[np.ndarray], mm_per_px: float) -> np.ndarray:
    """LiDAR footprint with far edges pulled back by the mixed-pixel displacement.

    Depth samples on edges facing the camera lie on the tool's wall, so those edges are right. On
    edges facing away, the boundary pixel mixes the top surface with the floor and lands outside
    the footprint by about (h_local - thr) * r / D, where h_local is the surface height just inside
    the edge (full h for a box, ~0 for a sloped/round edge). Each far boundary point is moved back
    by exactly that amount.
    """
    u8 = lidar.astype(np.uint8) * 255
    contours, _ = cv2.findContours(u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = np.zeros_like(u8)
    H, W = lidar.shape
    probe_px = 1.5 * geom.depth_cell_mm / mm_per_px
    hr = np.nan_to_num(height_raster, nan=0.0)
    for cnt in contours:
        if len(cnt) < 8:
            continue
        pts = cnt.reshape(-1, 2).astype(np.float64) + 0.5
        k = max(3, min(9, len(pts) // 20))
        tan = np.roll(pts, -k, axis=0) - np.roll(pts, k, axis=0)
        nrm = np.column_stack([tan[:, 1], -tan[:, 0]])
        nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-9
        c = pts.mean(axis=0)
        sign = np.sign(np.einsum("ij,ij->i", nrm, pts - c))
        nrm *= np.where(sign == 0, 1, sign)[:, None]          # outward
        to_nadir = nadir_px[None, :] - pts
        far = np.einsum("ij,ij->i", nrm, to_nadir) < 0
        moved = pts.copy()
        win = max(5, int(round(8.0 / mm_per_px)))            # ~8 mm of contour
        if False and far.any():   # mixed-pixel correction retired: flying pixels are removed before unprojection
            inward = pts - nrm * probe_px
            ix = np.clip(np.round(inward[:, 0]).astype(int), 0, W - 1)
            iy = np.clip(np.round(inward[:, 1]).astype(int), 0, H - 1)
            h_all = np.maximum(0.0, hr[iy, ix] - thr_mm)
            h_all = _circular_smooth(h_all, win)               # the surface height just inside the edge, smoothed along it
            h_local = h_all[far]
            sel = pts[far]
            if homography_after is not None:
                sel = cv2.perspectiveTransform(sel.reshape(-1, 1, 2), np.linalg.inv(homography_after)).reshape(-1, 2)
            # footprint_correction with a per-point height: do it explicitly
            uv = geom.raster_to_plane(sel)
            X0 = geom.plane_to_cam(uv, 0.0)
            lam = ((geom.D - h_local) / geom.D)[:, None]
            uv_h, _ = geom.cam_to_plane(X0 * lam)
            corr = geom.plane_to_raster(uv_h)
            if homography_after is not None:
                corr = cv2.perspectiveTransform(corr.reshape(-1, 1, 2), homography_after).reshape(-1, 2)
            moved[far] = corr
        moved = _circular_smooth(moved, win)                   # remove LiDAR boundary noise (rounds corners ~2 mm)
        cv2.fillPoly(out, [np.round(moved - 0.5).astype(np.int32)], 255)
    return out > 0


def _circular_smooth(arr: np.ndarray, win: int) -> np.ndarray:
    """Moving average along a closed contour (works for 1-D values or N x 2 points)."""
    if win <= 1 or len(arr) < win:
        return arr
    k = np.ones(win) / win
    if arr.ndim == 1:
        pad = np.concatenate([arr[-win:], arr, arr[:win]])
        return np.convolve(pad, k, mode="same")[win:-win]
    out = np.empty_like(arr)
    for j in range(arr.shape[1]):
        pad = np.concatenate([arr[-win:, j], arr[:, j], arr[:win, j]])
        out[:, j] = np.convolve(pad, k, mode="same")[win:-win]
    return out


def smooth_observed_height(height: np.ndarray, sigma: float) -> np.ndarray:
    """Smooth measured heights without treating missing depth as floor.

    Keep unknown cells unknown so another view can supply their measurement.
    """
    valid = np.isfinite(height)
    weight = cv2.GaussianBlur(valid.astype(np.float32), (0, 0), sigma)
    total = cv2.GaussianBlur(np.where(valid, height, 0.).astype(np.float32), (0, 0), sigma)
    out = np.full(height.shape, np.nan, np.float32)
    np.divide(total, weight, out=out, where=valid & (weight > 1e-6))
    return out


def _fill_nearest(height: np.ndarray, max_dist_px: Optional[float] = None, unknown_value: float = 0.) -> np.ndarray:
    """Fill nearby NaN cells; distant cells receive unknown_value (legacy default: zero).

    Depth samples are ~one depth pixel apart, so the only cells farther away than that lie in the gaps
    opened by flying-pixel removal along tool edges. Splitting those gaps at the midpoint would extend
    every tool outward on the side facing away from the camera; treating them as floor keeps the edge
    within half a sample of the last tool sample.
    """
    valid = np.isfinite(height)
    if valid.all():
        return height.astype(np.float32)
    if not valid.any():
        return np.full_like(height, unknown_value, dtype=np.float32)
    from scipy import ndimage

    dist, (iy, ix) = ndimage.distance_transform_edt(~valid, return_indices=True)
    out = height[iy, ix].astype(np.float32)
    if max_dist_px is not None:
        out[dist > max_dist_px] = unknown_value
    return out


def _far_rim_excess(base: np.ndarray, color: np.ndarray, nadir_px: np.ndarray, depth_px: float) -> np.ndarray:
    """Pixels of the LiDAR footprint within `depth_px` of an edge facing away from the nadir that the colour
    outline (present nearby) does not claim."""
    from scipy import ndimage as ndi

    if not base.any() or not color.any():
        return np.zeros_like(base)
    dist = ndi.distance_transform_edt(base)
    rim = base & (dist <= depth_px)
    if not rim.any():
        return np.zeros_like(base)
    gy, gx = np.gradient(ndi.gaussian_filter(dist, 1.0))          # points inward
    ys, xs = np.nonzero(rim)
    ox, oy = -gx[ys, xs], -gy[ys, xs]                               # outward normal
    tx, ty = nadir_px[0] - xs, nadir_px[1] - ys
    far = ox * tx + oy * ty < 0
    k = max(1, int(round(depth_px)))
    near_color = cv2.dilate(color.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))) > 0
    out = np.zeros_like(base)
    out[ys[far], xs[far]] = True
    return out & ~color & near_color


def _keep_connected(base: np.ndarray, final: np.ndarray, reach_px: float, min_piece_px: float = 0.0) -> np.ndarray:
    """The colour band may only trim the LiDAR footprint, never cut it into pieces (a shaft SAM skipped must
    stay attached to its grip): pieces stranded from the main body get the removed LiDAR pixels around them
    (within reach_px) put back until they reconnect. Slivers smaller than min_piece_px (rim noise left over
    by the trim) are dropped instead."""
    removed = base & ~final
    if not removed.any():
        return final
    k = max(1, int(round(reach_px)))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
    out = final.copy()
    for _ in range(4):
        num, lab = cv2.connectedComponents(out.astype(np.uint8), connectivity=8)
        if num <= 2:
            break
        sizes = np.bincount(lab.ravel(), minlength=num); sizes[0] = 0
        main = int(np.argmax(sizes))
        small = (sizes < min_piece_px)[lab] & (lab != main) & (lab > 0)
        out &= ~small
        stranded = out & (lab != main) & ~small
        if not stranded.any():
            break
        bridge = removed & (cv2.dilate(stranded.astype(np.uint8), ker) > 0)
        if not bridge.any():
            break
        out |= bridge
        removed &= ~bridge
    return out


_FP_DEBUG: List[Dict] = []   # filled when TC_FP_DEBUG is set (tests only)


def _trim_rim_half_height(lidar: np.ndarray, height_raster: np.ndarray, cell_px: float, thr_mm: float,
                          support: Optional[np.ndarray]) -> np.ndarray:
    """Drop rim pixels of a LiDAR blob whose height is under half the local top (sampled 0.5-2.5 cells inward
    along the inward normal). Components that lose contact with the tool's own evidence are discarded."""
    from scipy import ndimage as ndi

    if not lidar.any():
        return lidar
    h = np.nan_to_num(height_raster, nan=0.0)
    dist = ndi.distance_transform_edt(lidar)
    rim = lidar & (dist <= 2.5 * cell_px + 1.0)
    if not rim.any():
        return lidar
    gy, gx = np.gradient(ndi.gaussian_filter(dist, max(0.7, 0.3 * cell_px)))
    nrm = np.hypot(gx, gy) + 1e-9
    gx, gy = gx / nrm, gy / nrm
    ys, xs = np.nonzero(rim)
    top = h[ys, xs].copy()
    for k in (0.5, 1.0, 1.5, 2.0, 2.5):
        sy = ys + gy[ys, xs] * k * cell_px
        sx = xs + gx[ys, xs] * k * cell_px
        v = ndi.map_coordinates(h, [sy, sx], order=1, mode="nearest")
        top = np.maximum(top, v)
    keep_thr = np.maximum(thr_mm, float(os.environ.get("TC_TRIM_FRAC", RIM_TRIM_FRAC)) * top)
    out = lidar.copy()
    drop = h[ys, xs] < keep_thr
    out[ys[drop], xs[drop]] = False
    # fill pin holes the trim may open, then keep only components still touching the tool's evidence
    out = cv2.morphologyEx(out.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) > 0
    num, labels = cv2.connectedComponents(out.astype(np.uint8), connectivity=8)
    if num > 2 and support is not None:
        overlap = np.bincount(labels[support & out].ravel(), minlength=num); overlap[0] = 0
        if overlap.max() > 0:
            out = (overlap > 0)[labels]
    return out if out.any() else lidar


def footprint_from_mask(mask: np.ndarray, geom: CaptureGeometry, height_mm: float, homography_after: Optional[np.ndarray],
                        height_raster: Optional[np.ndarray] = None, mm_per_px: float = 1.0, eps_px: float = 0.6,
                        band_mm: float = 2.0, blob: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """Silhouette mask (session raster) -> true footprint polygon at the tool's height.

    The photo shows the top surface displaced away from the nadir by h*r/D plus whatever side walls
    face the camera, so it cannot be trusted alone. Footprint =
      1. LiDAR mask (height above a low threshold), far edges corrected per point (see _lidar_footprint);
      2. refined by the colour outline within a +/- band_mm band: the colour candidate is the
         silhouette intersected with its copy scaled by (D-h)/D (exact for vertical walls).
    Without a height raster only the colour candidate is used.
    """
    from .geometry import mask_to_polygon

    poly = mask_to_polygon(mask, eps_px=eps_px)
    if poly is None or height_mm is None or height_mm <= 0.3:
        return poly
    H, W = mask.shape
    nadir_px = geom.plane_to_raster(np.array([[0.0, 0.0]]))
    if homography_after is not None:
        nadir_px = cv2.perspectiveTransform(nadir_px.reshape(-1, 1, 2), homography_after).reshape(-1, 2)
    nadir_px = nadir_px[0]
    # colour candidate: the silhouette with its FAR edges (facing away from the camera foot point, i.e. the
    # top surface's overhang) pulled back by the perspective scaling, near edges (wall bases) kept. Doing it
    # per boundary point instead of intersecting with a scaled copy keeps thin features at full width.
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    color_u8 = np.zeros((H, W), np.uint8)
    for cnt in contours:
        if cv2.contourArea(cnt) < 4:
            continue
        pts = cnt.reshape(-1, 2).astype(np.float64) + 0.5
        if len(pts) >= 8:
            k = max(3, min(9, len(pts) // 20))
            tan = np.roll(pts, -k, axis=0) - np.roll(pts, k, axis=0)
            nrm = np.column_stack([tan[:, 1], -tan[:, 0]])
            nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-9
            c = pts.mean(axis=0)
            sign = np.sign(np.einsum("ij,ij->i", nrm, pts - c)); nrm *= np.where(sign == 0, 1, sign)[:, None]
            far = np.einsum("ij,ij->i", nrm, nadir_px[None, :] - pts) < 0
        else:
            far = np.ones(len(pts), bool)
        moved = pts.copy()
        if far.any():
            sel = pts[far]
            if homography_after is not None:
                sel = cv2.perspectiveTransform(sel.reshape(-1, 1, 2), np.linalg.inv(homography_after)).reshape(-1, 2)
            corr = geom.footprint_correction(sel, float(height_mm))
            if homography_after is not None:
                corr = cv2.perspectiveTransform(corr.reshape(-1, 1, 2), homography_after).reshape(-1, 2)
            moved[far] = corr
        cv2.fillPoly(color_u8, [np.round(moved - 0.5).astype(np.int32)], 255)
    color = color_u8 > 0
    final = color
    if height_raster is not None and height_raster.shape == mask.shape:
        # search window for the LiDAR footprint: the height blob when known (it covers parts the colour mask
        # may have skipped, e.g. a screwdriver shaft), else the colour mask
        region = (mask | color) if blob is None else (mask | color | blob)
        ys, xs = np.nonzero(region)
        pad = int(round(10.0 / mm_per_px))
        y0, y1 = max(0, ys.min() - pad), min(H, ys.max() + pad + 1)
        x0, x1 = max(0, xs.min() - pad), min(W, xs.max() + pad + 1)
        band_roi = np.zeros((H, W), bool)
        band_roi[y0:y1, x0:x1] = True
        thr = 2.0     # flat, just above the fused-LiDAR noise floor: keeps low parts (shafts, blades) of tall tools
        lidar = (np.nan_to_num(height_raster, nan=0.0) > thr) & band_roi
        # only LiDAR within reach of this tool's own evidence (its colour mask / blob share): a neighbouring
        # tool a few mm away shares the height blob but must not join the footprint
        support0 = mask if blob is None else (mask | blob)
        kr = max(1, int(round(max(4.0, band_mm) / mm_per_px)))
        lidar &= cv2.dilate(support0.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * kr + 1, 2 * kr + 1))) > 0
        num, labels = cv2.connectedComponents(lidar.astype(np.uint8), connectivity=8)
        if num > 1:
            support = mask if blob is None else (mask | blob)
            overlap = np.bincount(labels[support].ravel(), minlength=num)
            overlap[0] = 0
            if overlap.max() == 0:
                overlap = np.bincount(labels[band_roi].ravel(), minlength=num); overlap[0] = 0
            lidar = (overlap > 0)[labels]
            # fill holes in the LiDAR blob (noise) before using its boundary
            lu8 = lidar.astype(np.uint8) * 255
            cs, _ = cv2.findContours(lu8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            lu8 = np.zeros_like(lu8)
            cv2.drawContours(lu8, cs, -1, 255, thickness=cv2.FILLED)
            lidar = lu8 > 0
            # the 2 mm threshold sits low on the smoothed height step, so the blob edge lands up to ~1.5 depth
            # cells outside the wall of a tall tool. Trim the rim to the half-height crossing of the step, with
            # the "top" measured a few cells inward along the boundary normal (omnidirectional windows would
            # see a neighbouring taller part and cut a 7 mm shaft off its 28 mm grip).
            lidar = _trim_rim_half_height(lidar, height_raster, geom.depth_cell_mm / mm_per_px, thr, support0)
            base = _lidar_footprint(lidar, height_raster, nadir_px, geom, thr, homography_after, mm_per_px)
            # colour refinement band around the LiDAR boundary. The colour candidate (silhouette ∩ scaled
            # silhouette) is exact when the tool sits near the camera axis and degrades with the expected
            # perspective displacement h*r/D, so trust it widely when that is small (arc frames), narrowly
            # when large (a far-off-axis tall tool in a single still). Wide bands let colour recover thin
            # features (handle tips, shafts) that the LiDAR grid cannot resolve.
            ys_l, xs_l = np.nonzero(lidar)
            r_mm = float(np.hypot(xs_l.mean() - nadir_px[0], ys_l.mean() - nadir_px[1])) * mm_per_px if xs_l.size else 0.0
            disp = float(height_mm) * r_mm / max(geom.D, 1.0)
            band_eff = max(band_mm, 0.75 * geom.depth_cell_mm, min(8.0, 8.0 - disp))
            kb = max(1, int(round(band_eff / mm_per_px)))
            ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * kb + 1, 2 * kb + 1))
            outer = cv2.dilate(base.astype(np.uint8), ker) > 0
            inner = cv2.erode(base.astype(np.uint8), ker) > 0
            band = outer & ~inner
            if os.environ.get("TC_NO_COLOR_BAND"):
                final = base
            else:
                # colour decides only inside the band AND where the colour mask has an opinion; elsewhere
                # (thin parts SAM skipped, e.g. a screwdriver shaft) the LiDAR footprint stands
                color_nearby = cv2.dilate(color.astype(np.uint8), ker) > 0
                bz = band & color_nearby
                final = (base & ~bz) | (color & bz)
                # far_rim_trim: on edges facing AWAY from the camera foot point the LiDAR footprint can only be too
                # wide (top-edge mixed pixels snapped to the top, gap fill into the floor "shadow" behind the tool),
                # never too narrow, so there the colour outline may trim up to two depth cells instead of the band
                from scipy import ndimage as _ndi
                kt = max(3, int(round(2.0 * geom.depth_cell_mm / mm_per_px)) | 1)
                tall = _ndi.maximum_filter(np.nan_to_num(height_raster, nan=0.0), size=kt) > 12.0   # local top; below the flying-pixel jump there is no such bias
                final &= ~(_far_rim_excess(base, color, nadir_px, 2.0 * geom.depth_cell_mm / mm_per_px) & tall)
                # parts split at the colour/LiDAR hand-over get reconnected (small kernel: a wide closing
                # would fill the inner corners of T- and L-shaped tools), staying inside the LiDAR footprint
                k2 = max(1, int(round(2.0 / mm_per_px)))
                closed = cv2.morphologyEx(final.astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k2 + 1, 2 * k2 + 1))) > 0
                k1 = max(1, int(round(1.0 / mm_per_px)))
                near_base = cv2.dilate(base.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k1 + 1, 2 * k1 + 1))) > 0
                final = final | (closed & near_base)
                final = _keep_connected(base, final, reach_px=12.0 / mm_per_px, min_piece_px=40.0 / mm_per_px ** 2)
            # light open/close to drop pixel-level jaggies before polygonising
            kc = max(1, int(round(1.0 / mm_per_px)))
            kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * kc + 1, 2 * kc + 1))
            f8 = cv2.morphologyEx(final.astype(np.uint8), cv2.MORPH_OPEN, kern)
            f8 = cv2.morphologyEx(f8, cv2.MORPH_CLOSE, kern)
            final = f8 > 0
    if os.environ.get("TC_FP_DEBUG"):
        _FP_DEBUG.append({"mask": mask.copy(), "color": color.copy(), "final": final.copy(),
                          "base": locals().get("base"), "lidar": locals().get("lidar"), "h": height_mm})
    u8 = final.astype(np.uint8) * 255
    contours, _ = cv2.findContours(u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return poly
    cnt = max(contours, key=cv2.contourArea)
    from .geometry import smooth_ring

    pts = cnt.reshape(-1, 2).astype(np.float64) + 0.5
    # the LiDAR grid (mm) and SAM's pixel jaggies carry no shape information below ~1 mm: smooth them out
    sm = smooth_ring(pts, sigma=1.0 / mm_per_px, tol=0.3 / mm_per_px)
    return sm if len(sm) >= 3 else mask_to_polygon(filled_from(cnt, u8.shape), eps_px=eps_px)


def filled_from(cnt, shape):
    m = np.zeros(shape, np.uint8)
    cv2.drawContours(m, [cnt], -1, 255, thickness=cv2.FILLED)
    return m > 0
