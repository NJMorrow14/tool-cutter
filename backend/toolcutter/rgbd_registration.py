"""Register overlapping metric RGB-D floor rasters without ARKit camera poses.

SIFT features are lifted through measured depth to remove tool-top parallax.
A rigid pose graph uses multiple overlaps rather than accumulating shifts.
Floor-only features remain a fallback for callers without source RGB-D.
"""
from __future__ import annotations

import os

PAIR_WINDOW = int(os.environ.get("TC_PAIR_WINDOW", "10"))      # frames either side that are matched
PAIR_ALL_UPTO = int(os.environ.get("TC_PAIR_ALL_UPTO", "30"))  # small captures: match every pair, as before

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
    # Absolute support matters more than the inlier RATIO once relief features are in the pool: on a dark liner the
    # noise features drag the ratio to 20-30 %% while 60+ geometrically consistent inliers sit underneath (2026-10-03:
    # adjacent pairs 2-6 of a sweep were rejected at 29-32 %% with 60-72 inliers each). The rigid residual test
    # below still rejects a bad match.
    if keep.sum() < 10 or (keep.mean() < .35 and keep.sum() < 25):
        return None
    src, dst = src[keep], dst[keep]
    strong = keep.sum() >= 12
    for points in (src, dst):
        # Reject a narrow row of repeated marks or a tiny accidental patch.
        if cv2.contourArea(cv2.convexHull(points.astype(np.float32))) < (600 if strong else 1000) or \
           np.linalg.svd(points - points.mean(0), compute_uv=False)[1] / np.sqrt(len(points)) < (6 if strong else 8):
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
    if np.median(error) > 1.2 or np.percentile(error, 90) > 2.5:
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
    # A sweep is temporally ordered: a frame overlaps its neighbours, not the other end of the drawer. Matching
    # every pair is O(n^2) brute-force knnMatch — 6,903 pairs and 94 s for one 118-frame capture (2026-10-02).
    # Within a window of PAIR_WINDOW it is O(n), and frames that share a marker are always tried so the two ends
    # of a long sweep still connect. Captures of <= PAIR_ALL_UPTO frames keep the exhaustive pairing unchanged.
    n_res = len(results)
    def _pair_ok(i, j):
        if n_res <= PAIR_ALL_UPTO or abs(i - j) <= PAIR_WINDOW:
            return True
        return bool(set(results[i].markers_px) & set(results[j].markers_px))
    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            if not _pair_ok(i, j):
                continue
            shared = set(results[i].markers_px) & set(results[j].markers_px)
            if len(results) > 40 and j - i > 6 and i % 6 != 0 and not shared:
                continue
            match = _match(features[i], features[j])
            if match is not None:
                matrix, a, b, count = match
                edges.append((i, j, matrix, a, b, count))
    components = _solve_components(results, features, edges)
    if not components:
        return {anchor: np.eye(3)}, {'matched_pairs': len(edges), 'anchor_frame': anchor, 'components': []}
    best = max(components, key=lambda c: len(c[0]))
    info = dict(best[1]); info['components'] = [len(c[0]) for c in components]
    return best[0], info


def _solve_components(results, features, edges):
    """Every connected group of matched frames, each solved as its own rigid pose graph with its own anchor.
    The first version solved only the group containing the global anchor: on a 65-frame sweep over a dark liner
    that group was ONE frame while 32 matched pairs sat in other groups — and all of those frames were dropped.
    Returns [(transforms {frame: 3x3 in that group's anchor frame}, info)] sorted biggest first."""
    parent = list(range(len(results)))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for i, j, *_ in edges:
        parent[find(i)] = find(j)
    groups = {}
    for i in range(len(results)):
        groups.setdefault(find(i), []).append(i)
    out = []
    for members in groups.values():
        if len(members) < 2:
            continue
        mem = set(members)
        comp_edges = [e for e in edges if e[0] in mem and e[1] in mem]
        anchor = max(members, key=lambda i: (len(results[i].markers_px), len(features[i][0])))
        poses = {anchor: np.eye(3)}
        while True:
            choices = [e for e in comp_edges if (e[0] in poses) != (e[1] in poses)]
            if not choices:
                break
            i, j, matrix, _, _, _ = max(choices, key=lambda e: e[5])
            if j in poses: poses[i] = poses[j] @ matrix
            else: poses[j] = poses[i] @ np.linalg.inv(matrix)
        if len(poses) < 2:
            continue
        transforms, info = _refine_component(poses, anchor, comp_edges)
        info['anchor_frame'] = anchor
        out.append((transforms, info))
    out.sort(key=lambda c: -len(c[0]))
    return out


def _refine_component(poses, anchor, edges):
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
    return transforms, {'matched_pairs': len(acceptable),
                        'median_residual_mm': round(float(np.median(errors)), 3) if errors else None,
                        'solver_evaluations': int(solved.nfev)}


def register_rgbd_components(results, features=None):
    """All connected groups: [(transforms, info)], biggest first (see _solve_components)."""
    if not results:
        return [], {'matched_pairs': 0}
    features = features if features is not None else [_features(r) for r in results]
    edges = []
    n_res = len(results)
    def _pair_ok(i, j):
        if n_res <= PAIR_ALL_UPTO or abs(i - j) <= PAIR_WINDOW:
            return True
        return bool(set(results[i].markers_px) & set(results[j].markers_px))
    for i in range(n_res):
        for j in range(i + 1, n_res):
            if not _pair_ok(i, j):
                continue
            shared = set(results[i].markers_px) & set(results[j].markers_px)
            if n_res > 40 and j - i > 6 and i % 6 != 0 and not shared:
                continue
            match = _match(features[i], features[j])
            if match is not None:
                matrix, a, b, count = match
                edges.append((i, j, matrix, a, b, count))
    comps = _solve_components(results, features, edges)
    return comps, {'matched_pairs': len(edges), 'components': [len(c[0]) for c in comps]}


def _rigid_fit(src, dst):
    """Kabsch: rotation + translation (no scale) taking src points onto dst points. Returns a 3x3 matrix."""
    sc, dc = src.mean(0), dst.mean(0)
    u, _, vt = np.linalg.svd((src - sc).T @ (dst - dc))
    rot = vt.T @ u.T
    if np.linalg.det(rot) < 0:
        vt[-1] *= -1; rot = vt.T @ u.T
    m = np.eye(3); m[:2, :2] = rot; m[:2, 2] = dc - rot @ sc
    return m


def drawer_corners(results, transforms, *, infer_order=True, reference=None):
    """A shared drawer rectangle, expressed in every registered source raster.

    Pool marker observations across the sweep, so no individual frame needs to
    see the whole drawer. Preserve the detector's corner order when averaging.

    `reference` = (markers_px, drawer_corners_px, mm_per_px) of a frame that saw the whole drawer (the rear
    overview still). When the registered cluster pooled fewer than three markers — a close front-camera sweep
    down a big drawer sees ONE corner marker for a long stretch — the cluster is anchored to the overview's
    rectangle through whatever markers they share (4 corner points per marker, rigid fit), instead of being
    thrown away. On `a237eba87dba` (598 x 800 mm) the cluster saw only marker 0 and all of it was discarded.
    """
    from . import capture
    observed = {}
    for i, transform in transforms.items():
        result = results[i]
        for mid, corners in result.markers_px.items():
            points = corners * result.mm_per_px
            observed.setdefault(mid, []).append(points @ transform[:2, :2].T + transform[:2, 2])
    markers = {mid: np.median(values, axis=0) for mid, values in observed.items()}
    mapping = None
    rectangle = None
    if len(markers) >= 3:
        mapping = capture.corner_order({mid: p.mean(0) for mid, p in markers.items()}) if infer_order else None
        if not (infer_order and mapping is None):
            relabelled = capture.relabel_markers(markers, mapping)
            orient = capture._orient_for_markers(relabelled)[:, :2]
            rect = capture.drawer_rectangle_from_markers({mid: p @ orient.T for mid, p in relabelled.items()})
            if rect is not None:
                rect = rect @ orient
                sides = np.linalg.norm(np.roll(rect, -1, axis=0) - rect, axis=1)
                if sides.min() >= 50 and max(abs(sides[0] - sides[2]), abs(sides[1] - sides[3])) <= 10:
                    rectangle = rect
    anchored_by = None
    if rectangle is None and reference is not None:
        ref_markers, ref_rect_px, ref_mpp = reference
        shared = [mid for mid in markers if mid in ref_markers]
        if shared and ref_rect_px is not None:
            src = np.vstack([markers[mid] for mid in shared])                      # cluster (anchor) mm
            dst = np.vstack([np.asarray(ref_markers[mid]) * ref_mpp for mid in shared])   # overview mm
            to_ref = _rigid_fit(src, dst)
            resid = np.linalg.norm(src @ to_ref[:2, :2].T + to_ref[:2, 2] - dst, axis=1)
            if float(np.median(resid)) <= 4.0:
                inv = np.linalg.inv(to_ref)
                rect_ref = np.asarray(ref_rect_px) * ref_mpp
                rectangle = rect_ref @ inv[:2, :2].T + inv[:2, 2]
                anchored_by = f"{len(shared)} shared marker(s) with the overview, residual {float(np.median(resid)):.1f} mm"
    if rectangle is None:
        return {}, mapping, None
    sides = np.linalg.norm(np.roll(rectangle, -1, axis=0) - rectangle, axis=1)
    corners = {}
    for i, transform in transforms.items():
        inverse = np.linalg.inv(transform)
        corners[i] = (rectangle @ inverse[:2, :2].T + inverse[:2, 2]) / results[i].mm_per_px
    size = (float((sides[0] + sides[2]) / 2), float((sides[1] + sides[3]) / 2))
    if anchored_by:
        import logging
        logging.getLogger("toolcutter").info("TrueDepth: cluster of %d frames anchored to the drawer by %s", len(corners), anchored_by)
    return corners, mapping, size


HEIGHT_CHANNEL_TAG = 1000.0   # appended 129th descriptor dimension: colour features 0, height-map features 1000, so
                              # the matcher never pairs a photo feature with a relief feature (their L2 distance is ~1000)


def _tag(desc, value):
    if desc is None or len(desc) == 0:
        return desc
    return np.hstack([desc, np.full((len(desc), 1), value, desc.dtype)])


def height_features(height_mm, mm_per_px, nfeatures=1500):
    """SIFT on the frame's own ORTHOGRAPHIC height raster: tools are strong relief shapes, there is no parallax to
    remove (the raster is depth-derived), and a dark featureless liner that defeats photo SIFT does not matter.
    Added 2026-10-03 after a 65-frame sweep over a dark liner connected only 8 frames through photo features
    (4-50 keypoints on half the frames). Returns (xy_mm, descriptors) in the raster's mm frame."""
    if height_mm is None:
        return np.empty((0, 2), np.float32), None
    h = np.nan_to_num(np.asarray(height_mm, dtype=np.float32), nan=0.0)
    if not (h > 2.0).any():
        return np.empty((0, 2), np.float32), None
    relief = np.clip(h, 0, 60.0) / 60.0 * 255.0
    relief = cv2.GaussianBlur(relief.astype(np.uint8), (0, 0), 1.0)
    # only around things that stand up: SIFT on the bare liner finds thousands of 0.3 mm-noise "features" that
    # match nothing and bury the real ones (3000-feature saturation on flat frames)
    raised = (h > 1.5).astype(np.uint8)
    k = max(3, int(round(6.0 / mm_per_px)) | 1)
    mask = cv2.dilate(raised, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) * 255
    if not mask.any():
        return np.empty((0, 2), np.float32), None
    keypoints, descriptors = cv2.SIFT_create(nfeatures=nfeatures, contrastThreshold=.03).detectAndCompute(relief, mask)
    if descriptors is None or len(keypoints) == 0:
        return np.empty((0, 2), np.float32), None
    xy = (np.float32([p.pt for p in keypoints]).reshape(-1, 2) * mm_per_px).astype(np.float32)
    return xy, descriptors


def depth_features(image, depth, intrinsics, geometry, height_mm=None):
    """Lift RGB features through measured depth, removing tool-top parallax.

    A feature is usable only on a locally continuous, valid depth surface. This
    lets printed tool details contribute where a plain drawer floor has no
    distinctive texture, without treating raised details as floor points.
    With `height_mm` (the frame's rectified height raster) relief features are
    added as a second, tagged channel — see `height_features`.
    """
    scale = min(1., 1600. / max(image.shape[:2]))
    gray = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    # local contrast normalisation: a dark drawer liner under a phone's torch gives SIFT little to hold on to
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    keypoints, descriptors = cv2.SIFT_create(nfeatures=5000, contrastThreshold=.012).detectAndCompute(gray, None)
    hx, hd = height_features(height_mm, geometry.mm_per_px) if height_mm is not None else (np.empty((0, 2), np.float32), None)
    if descriptors is None:
        return (hx, _tag(hd, HEIGHT_CHANNEL_TAG)) if hd is not None else (np.empty((0, 2), np.float32), None)
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
    xy = (geometry.plane_to_raster(uv) * geometry.mm_per_px)[keep_height].astype(np.float32)
    desc = _tag(descriptors[keep][keep_height], 0.0)
    if hd is not None and len(hd):
        xy = np.vstack([xy, hx]); desc = np.vstack([desc, _tag(hd, HEIGHT_CHANNEL_TAG)])
    return xy, desc


def register_depth_subset(results, features, infer_order=True, reference_index=None):
    """Register depth views even when a capture also contains rear-camera photos.

    Photo-only frames have no metric depth features and must not disable the
    pose graph or participate in it. Return original result indices for callers.
    `reference_index`: a frame that saw the whole drawer (the overview), used to
    anchor a cluster that pooled fewer than three markers.
    """
    indices = [i for i, f in enumerate(features) if f is not None]
    subset = [results[i] for i in indices]
    comps, info = register_rgbd_components(subset, [features[i] for i in indices])
    reference = None
    if reference_index is not None and results[reference_index].drawer_corners is not None:
        ref = results[reference_index]
        reference = (ref.markers_px, ref.drawer_corners, ref.mm_per_px)
    corners_all, mapping_out, size_out = {}, None, None
    anchored, unanchored = [], []
    for transforms, cinfo in comps:
        corners, mapping, size = drawer_corners(subset, transforms, infer_order=infer_order, reference=reference)
        if corners:
            corners_all.update(corners); anchored.append(len(transforms))
            if size_out is None:
                size_out, mapping_out = size, mapping
        else:
            unanchored.append(len(transforms))
    info["anchored_components"] = anchored
    info["unanchored_components"] = unanchored
    info["depth_frames"] = len(indices)
    info["frames_registered"] = len(corners_all)
    # the biggest component's anchor, as an ORIGINAL frame index (callers and the registration test read it)
    if comps and comps[0][1].get("anchor_frame") is not None:
        info["anchor_frame"] = indices[int(comps[0][1]["anchor_frame"])]
    return {indices[i]: c for i, c in corners_all.items()}, mapping_out, size_out, info
