"""Register overlapping metric RGB-D floor rasters without ARKit camera poses.

SIFT features are lifted through measured depth to remove tool-top parallax.
A rigid pose graph uses multiple overlaps rather than accumulating shifts.
Floor-only features remain a fallback for callers without source RGB-D.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.optimize import least_squares


def _features(result):
    pitch = result.mm_per_px
    image = result.color_bgr
    scale = min(1., 1200. / max(image.shape[:2]))
    size = (round(image.shape[1] * scale), round(image.shape[0] * scale))
    gray = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), size, interpolation=cv2.INTER_AREA)
    height = result.height_mm
    valid = np.ones(image.shape[:2], np.uint8)
    if height is not None:
        valid = (np.isfinite(height) & (height < 3)).astype(np.uint8)
        # Photo silhouettes extend beyond the orthographic height footprint. A
        # broad exclusion prevents a tool from dragging the floor registration.
        radius = max(2, round(8 / pitch))
        raised = (np.nan_to_num(height, nan=0) >= 3).astype(np.uint8)
        valid &= 1 - cv2.dilate(raised, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2))
    valid = cv2.resize(valid, size, interpolation=cv2.INTER_NEAREST)
    valid = cv2.erode(valid, np.ones((7, 7), np.uint8))
    points, descriptors = cv2.SIFT_create(nfeatures=3500, contrastThreshold=.015).detectAndCompute(gray, valid * 255)
    xy = np.float32([p.pt for p in points]).reshape(-1, 2) * (pitch / scale)
    return xy, descriptors


def _match(a, b):
    pa, da = a; pb, db = b
    if da is None or db is None or min(len(da), len(db)) < 10:
        return None
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    ab = matcher.knnMatch(da, db, k=2)
    ba = matcher.knnMatch(db, da, k=2)
    reverse = {m.queryIdx: m.trainIdx for pair in ba if len(pair) == 2 for m, n in [pair] if m.distance < .8 * n.distance}
    pairs = [(m.queryIdx, m.trainIdx) for pair in ab if len(pair) == 2 for m, n in [pair]
             if m.distance < .8 * n.distance and reverse.get(m.trainIdx) == m.queryIdx]
    if len(pairs) < 10:
        return None
    src = pa[[i for i, _ in pairs]]; dst = pb[[j for _, j in pairs]]
    affine, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=2.0,
                                                maxIters=3000, confidence=.999)
    if affine is None or inliers is None:
        return None
    keep = inliers.ravel().astype(bool)
    if keep.sum() < 10 or keep.mean() < .35:
        return None
    src, dst = src[keep], dst[keep]
    for points in (src, dst):
        # Reject a narrow row of repeated marks or a tiny accidental patch.
        if cv2.contourArea(cv2.convexHull(points.astype(np.float32))) < 1000 or np.linalg.svd(points - points.mean(0), compute_uv=False)[1] / np.sqrt(len(points)) < 8:
            return None
    scale = np.hypot(affine[0, 0], affine[1, 0])
    if abs(scale - 1) > .025:
        return None
    # Similarity is only a hypothesis check. Refine a rigid transform, retaining
    # millimetres and rejecting perspective/scale disagreement between planes.
    u, _, vt = np.linalg.svd((src - src.mean(0)).T @ (dst - dst.mean(0)))
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        return None
    translation = dst.mean(0) - rotation @ src.mean(0)
    error = np.linalg.norm(src @ rotation.T + translation - dst, axis=1)
    if np.median(error) > 1.2 or np.percentile(error, 90) > 2.0:
        return None
    matrix = np.eye(3); matrix[:2, :2] = rotation; matrix[:2, 2] = translation
    sample = np.linspace(0, len(src) - 1, min(100, len(src))).astype(int)
    return matrix, src[sample], dst[sample], int(keep.sum())


def register_rgbd(results, features=None):
    """Return {frame index: local-mm -> anchor-mm rigid transform}, with diagnostics.

    Disconnected or inconsistent frames are excluded, never assigned fake poses.
    """
    if not results:
        return {}, {'matched_pairs': 0}
    features = features if features is not None else [_features(r) for r in results]
    anchor = max(range(len(results)), key=lambda i: (len(results[i].markers_px), len(features[i][0])))
    edges = []
    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            shared = set(results[i].markers_px) & set(results[j].markers_px)
            if len(results) > 40 and j - i > 6 and i % 6 != 0 and not shared:
                continue
            match = _match(features[i], features[j])
            if match is not None:
                matrix, a, b, count = match
                edges.append((i, j, matrix, a, b, count))
    poses = {anchor: np.eye(3)}
    # Strongest links first provide the initial graph; all overlaps constrain the solve.
    while True:
        choices = [e for e in edges if (e[0] in poses) != (e[1] in poses)]
        if not choices:
            break
        i, j, matrix, _, _, _ = max(choices, key=lambda e: e[5])
        if j in poses: poses[i] = poses[j] @ matrix
        else: poses[j] = poses[i] @ np.linalg.inv(matrix)
    if len(poses) < 2:
        return poses, {'matched_pairs': len(edges), 'anchor_frame': anchor}
    nodes = sorted(i for i in poses if i != anchor)
    idx = {i: j for j, i in enumerate(nodes)}
    initial = np.array([[np.arctan2(poses[i][1, 0], poses[i][0, 0]), *poses[i][:2, 2]] for i in nodes]).ravel()
    connected_edges = [e for e in edges if e[0] in poses and e[1] in poses]

    def matrices(x):
        out = {anchor: np.eye(3)}
        for i, values in zip(nodes, x.reshape(-1, 3)):
            th, tx, ty = values
            out[i] = np.array([[np.cos(th), -np.sin(th), tx], [np.sin(th), np.cos(th), ty], [0, 0, 1]])
        return out

    def residual(x):
        transforms = matrices(x)
        return np.concatenate([(a @ transforms[i][:2, :2].T + transforms[i][:2, 2]
                                - b @ transforms[j][:2, :2].T - transforms[j][:2, 2]).ravel()
                               for i, j, _, a, b, _ in connected_edges])

    # Each residual depends only on its two endpoint poses.
    from scipy.sparse import lil_matrix
    sparsity = lil_matrix((sum(2 * len(e[3]) for e in connected_edges), len(initial)), dtype=int)
    row = 0
    for i, j, _, a, _, _ in connected_edges:
        for node in (i, j):
            if node in idx: sparsity[row:row + 2 * len(a), 3 * idx[node]:3 * idx[node] + 3] = 1
        row += 2 * len(a)
    solved = least_squares(residual, initial, loss='soft_l1', f_scale=.7, jac_sparsity=sparsity.tocsr(), max_nfev=80)
    transforms = matrices(solved.x)
    errors = []
    acceptable = []
    for i, j, _, a, b, _ in connected_edges:
        error = np.linalg.norm(a @ transforms[i][:2, :2].T + transforms[i][:2, 2]
                              - b @ transforms[j][:2, :2].T - transforms[j][:2, 2], axis=1)
        med = float(np.median(error))
        if med < 1.5:
            acceptable.append((i, j)); errors.append(med)
    reached = {anchor}
    for _ in nodes:
        for i, j in acceptable:
            if i in reached or j in reached: reached.update((i, j))
    transforms = {i: t for i, t in transforms.items() if i in reached}
    return transforms, {'matched_pairs': len(acceptable), 'anchor_frame': anchor,
                        'median_residual_mm': round(float(np.median(errors)), 3) if errors else None,
                        'solver_evaluations': int(solved.nfev)}


def drawer_corners(results, transforms, *, infer_order=True):
    """A shared drawer rectangle, expressed in every registered source raster.

    Pool marker observations across the sweep, so no individual frame needs to
    see the whole drawer. Preserve the detector's corner order when averaging.
    """
    from . import capture
    observed = {}
    for i, transform in transforms.items():
        result = results[i]
        for mid, corners in result.markers_px.items():
            points = corners * result.mm_per_px
            observed.setdefault(mid, []).append(points @ transform[:2, :2].T + transform[:2, 2])
    markers = {mid: np.median(values, axis=0) for mid, values in observed.items()}
    mapping = capture.corner_order({mid: p.mean(0) for mid, p in markers.items()}) if infer_order else None
    if infer_order and mapping is None:
        return {}, None, None
    markers = capture.relabel_markers(markers, mapping)
    orient = capture._orient_for_markers(markers)[:, :2]
    rectangle = capture.drawer_rectangle_from_markers({mid: p @ orient.T for mid, p in markers.items()})
    if rectangle is None:
        return {}, mapping, None
    rectangle = rectangle @ orient
    sides = np.linalg.norm(np.roll(rectangle, -1, axis=0) - rectangle, axis=1)
    if sides.min() < 50 or max(abs(sides[0] - sides[2]), abs(sides[1] - sides[3])) > 10:
        return {}, mapping, None
    corners = {}
    for i, transform in transforms.items():
        inverse = np.linalg.inv(transform)
        corners[i] = (rectangle @ inverse[:2, :2].T + inverse[:2, 2]) / results[i].mm_per_px
    return corners, mapping, (float((sides[0] + sides[2]) / 2), float((sides[1] + sides[3]) / 2))


def depth_features(image, depth, intrinsics, geometry):
    """Lift RGB features through measured depth, removing tool-top parallax.

    A feature is usable only on a locally continuous, valid depth surface. This
    lets printed tool details contribute where a plain drawer floor has no
    distinctive texture, without treating raised details as floor points.
    """
    scale = min(1., 1600. / max(image.shape[:2]))
    gray = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    keypoints, descriptors = cv2.SIFT_create(nfeatures=5000, contrastThreshold=.02).detectAndCompute(gray, None)
    if descriptors is None:
        return np.empty((0, 2), np.float32), None
    pixels = np.asarray([p.pt for p in keypoints]) / scale
    dx = np.clip(np.rint(pixels[:, 0] * depth.shape[1] / image.shape[1]).astype(int), 0, depth.shape[1] - 1)
    dy = np.clip(np.rint(pixels[:, 1] * depth.shape[0] / image.shape[0]).astype(int), 0, depth.shape[0] - 1)
    valid = np.isfinite(depth) & (depth > .05)
    kernel = np.ones((3, 3), np.uint8)
    low = cv2.erode(np.where(valid, depth, np.inf), kernel)
    high = cv2.dilate(np.where(valid, depth, -np.inf), kernel)
    continuous = cv2.erode(valid.astype(np.uint8), kernel).astype(bool) & (high - low < .006)
    keep = continuous[dy, dx]
    pixels, dx, dy = pixels[keep], dx[keep], dy[keep]
    z = depth[dy, dx] * 1000.
    points = np.column_stack([(pixels[:, 0] - intrinsics[0, 2]) / intrinsics[0, 0] * z,
                              (pixels[:, 1] - intrinsics[1, 2]) / intrinsics[1, 1] * z, z])
    uv, heights = geometry.cam_to_plane(points)
    keep_height = (heights > -5) & (heights < 150)
    xy = geometry.plane_to_raster(uv) * geometry.mm_per_px
    return xy[keep_height].astype(np.float32), descriptors[keep][keep_height]


def register_depth_subset(results, features, infer_order=True):
    """Register depth views even when a capture also contains rear-camera photos.

    Photo-only frames have no metric depth features and must not disable the
    pose graph or participate in it. Return original result indices for callers.
    """
    indices = [i for i, f in enumerate(features) if f is not None]
    subset = [results[i] for i in indices]
    transforms, info = register_rgbd(subset, [features[i] for i in indices])
    corners, mapping, size = drawer_corners(subset, transforms, infer_order=infer_order)
    if "anchor_frame" in info:
        info["anchor_frame"] = indices[info["anchor_frame"]]
    info["depth_frames"] = len(indices)
    return {indices[i]: c for i, c in corners.items()}, mapping, size, info
