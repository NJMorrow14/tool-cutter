"""Metric registration regressions; no camera or segmentation model required."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import cv2
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from toolcutter.rgbd_registration import _match, register_rgbd, drawer_corners, depth_features


class RegistrationTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(17)
        self.points = rng.uniform(0, 200, (120, 2)).astype(np.float32)
        self.descriptors = rng.normal(size=(120, 128)).astype(np.float32)

    def test_recovers_large_rotation_translation_without_changing_scale(self):
        angle = .65
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        target = self.points @ rotation.T + [35, -90]
        match = _match((self.points, self.descriptors), (target, self.descriptors))
        self.assertIsNotNone(match)
        np.testing.assert_allclose(match[0][:2, :2], rotation, atol=1e-5)
        np.testing.assert_allclose(match[0][:2, 2], [35, -90], atol=1e-4)

    def test_rejects_scale_changes_and_narrow_repeated_features(self):
        self.assertIsNone(_match((self.points, self.descriptors), (self.points * 1.08, self.descriptors)))
        line = self.points.copy(); line[:, 1] *= .01
        self.assertIsNone(_match((line, self.descriptors), (line + 10, self.descriptors)))

    def test_graph_recovers_unmarked_frames_and_excludes_disconnected_frame(self):
        offsets = [np.array([0, 0]), np.array([20, 30]), np.array([80, 40])]
        results = [SimpleNamespace(markers_px={0: None} if i == 0 else {}) for i in range(4)]
        features = [(self.points + delta, self.descriptors) for delta in offsets] + [(np.empty((0, 2)), None)]
        with patch('toolcutter.rgbd_registration._features', side_effect=features):
            poses, info = register_rgbd(results)
        self.assertEqual(set(poses), {0, 1, 2})
        for i, delta in enumerate(offsets):
            np.testing.assert_allclose(poses[i][:2, 2], -delta, atol=.001)
        self.assertLess(info['median_residual_mm'], .001)

    def test_marker_order_is_recovered_across_partial_views(self):
        # Bottom ids 2 and 3 are swapped; neither frame sees three markers.
        square = np.array([[0, 0], [50, 0], [50, 50], [0, 50]], float)
        markers = {0: square, 1: square + [250, 0], 2: square + [0, 350], 3: square + [250, 350]}
        results = [SimpleNamespace(markers_px={k: markers[k] for k in ids}, mm_per_px=1.) for ids in ((0, 1), (2, 3))]
        corners, mapping, size = drawer_corners(results, {0: np.eye(3), 1: np.eye(3)})
        self.assertEqual(mapping, {0: 0, 1: 1, 3: 2, 2: 3})
        np.testing.assert_allclose(size, [300, 400])
        np.testing.assert_allclose(corners[0], [[0, 0], [300, 0], [300, 400], [0, 400]])

    def test_mixed_photo_depth_capture_retains_original_indices(self):
        from toolcutter.rgbd_registration import register_depth_subset
        square = np.array([[0, 0], [50, 0], [50, 50], [0, 50]], float)
        markers = {0: square, 1: square + [250, 0], 2: square + [250, 350], 3: square + [0, 350]}
        depth = SimpleNamespace(markers_px=markers, mm_per_px=1.)
        photo = SimpleNamespace(markers_px={}, mm_per_px=1.)
        features = (self.points, self.descriptors)
        corners, mapping, size, info = register_depth_subset(
            [photo, depth, photo, depth], [None, features, None, features])
        self.assertEqual(set(corners), {1, 3})
        self.assertEqual(info['depth_frames'], 2)
        self.assertEqual(info['anchor_frame'], 1)
        np.testing.assert_allclose(size, [300, 400])

    def test_features_use_measured_tool_depth_and_reject_missing_depth(self):
        image = np.zeros((480, 640, 3), np.uint8)
        depth = np.full((480, 640), .3, np.float32)
        depth[99:102, 99:102] = np.nan
        geometry = SimpleNamespace(mm_per_px=1., cam_to_plane=lambda p: (p[:, :2], 400 - p[:, 2]),
                                   plane_to_raster=lambda uv: uv)
        keypoints = [cv2.KeyPoint(400, 240, 4), cv2.KeyPoint(100, 100, 4)]
        detector = SimpleNamespace(detectAndCompute=lambda *args: (keypoints, self.descriptors[:2]))
        K = np.array([[400, 0, 320], [0, 400, 240], [0, 0, 1]], float)
        with patch('toolcutter.rgbd_registration.cv2.SIFT_create', return_value=detector):
            xy, descriptors = depth_features(image, depth, K, geometry)
        np.testing.assert_allclose(xy, [[60, 0]], atol=.001)  # projecting onto the floor would give 80
        self.assertEqual(len(descriptors), 1)

if __name__ == '__main__': unittest.main()
