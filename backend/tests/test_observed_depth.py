"""Unknown sensor readings must not cut holes through otherwise measured tools."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from toolcutter.capture import _fill_nearest, smooth_observed_height
from toolcutter.calibration import rectify
from app import _marker_mask


class ObservedDepthTests(unittest.TestCase):
    def test_missing_depth_preserves_height_and_remains_unknown(self):
        height = np.full((40, 50), 12., np.float32)
        height[10:30, 20:30] = np.nan
        filled = _fill_nearest(height, max_dist_px=1, unknown_value=np.nan)
        result = smooth_observed_height(filled, 1.5)
        self.assertTrue(np.isnan(result[15:25, 23:27]).all())
        np.testing.assert_allclose(result[np.isfinite(result)], 12, atol=1e-5)
        other_view = np.full(result.shape, 12., np.float32)
        np.testing.assert_allclose(np.nanmedian(np.stack([result, other_view]), axis=0), 12, atol=1e-5)

    def test_missing_depth_warp_does_not_create_zero_height_rim(self):
        height = np.full((40, 50), 12., np.float32)
        height[:, 25:] = np.nan
        image = np.zeros((40, 50, 3), np.uint8)
        corners = np.array([[.4, 0], [50.4, 0], [50.4, 40], [.4, 40]])
        _, _, _, extras = rectify(image, corners, 50, 40, extra=[height], already_ordered=True, ppm_override=1)
        result = extras[0]
        self.assertTrue(np.isnan(result[:, 30:]).all())
        np.testing.assert_allclose(result[np.isfinite(result)], 12, atol=1e-5)

    def test_real_floor_still_separates_adjacent_tools(self):
        height = np.zeros((40, 80), np.float32)
        height[10:30, 10:30] = 20
        height[10:30, 40:60] = 20
        result = smooth_observed_height(height, .5)
        self.assertLess(result[:, 33:37].max(), .01)

    def test_no_refinement_targets_does_not_build_a_depth_stack(self):
        from app import _refine_frame_alignment
        from unittest.mock import patch
        with patch('app._floor_texture', side_effect=AssertionError('No work is needed')):
            self.assertEqual(_refine_frame_alignment([{}], [True], [], 1), [(0., 0., 0.)])

    def test_tiled_fusion_matches_median_without_mutating_observations(self):
        from toolcutter.depth_fusion import fuse_heights
        rng = np.random.default_rng(22)
        heights = rng.uniform(0, 40, (9, 17, 21)).astype(np.float32)
        heights[0, :5] = np.nan
        heights[:, 6, 7] = np.nan
        original = heights.copy()
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            expected = np.nanmedian(heights, axis=0)
        np.testing.assert_allclose(fuse_heights(list(heights), tile_rows=4), expected, equal_nan=True)
        np.testing.assert_array_equal(heights, original)
        self.assertIsNone(fuse_heights([]))

    def test_heightfield_downsampling_preserves_height_and_missing_cells(self):
        import base64
        from app import _heightfield_payload
        height = np.full((8, 8), 20, np.float32)
        height[:, 4:] = np.nan
        height[0, 0] = np.nan
        payload = _heightfield_payload(height, 1, 2)
        values = np.frombuffer(base64.b64decode(payload['heights_b64']), dtype='<f4').reshape(4, 4)
        valid = np.frombuffer(base64.b64decode(payload['valid_b64']), dtype=np.uint8).reshape(4, 4)
        np.testing.assert_allclose(values[:, :2], 20)
        self.assertTrue(valid[:, :2].all())
        self.assertFalse(valid[:, 2:].any())
        empty = _heightfield_payload(np.full((8, 8), np.nan), 1, 2)
        self.assertEqual(empty['max_mm'], 0)

    def test_calibration_mask_uses_registered_corners_only(self):
        frame = dict(markers_px={0: np.array([[10, 10], [30, 10], [30, 30], [10, 30]])}, H=np.array([[1, 0, 20], [0, 1, 0], [0, 0, 1]], float))
        session = SimpleNamespace(rectified=np.zeros((80, 100, 3)), mm_per_px=1., frames=[frame])
        mask = _marker_mask(session)
        self.assertTrue(mask[20, 40])
        self.assertFalse(mask[20, 20])
        self.assertFalse(mask[20, 65])  # a neighbouring tool must remain selectable

if __name__ == '__main__': unittest.main()
