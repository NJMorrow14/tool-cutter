import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from toolcutter.photo_outlines import select_silhouettes

class PhotoOutlineTests(unittest.TestCase):
    def proposal(self, mask):
        return {'segmentation': mask, 'area': int(mask.sum())}

    def test_whole_tool_replaces_nested_details_and_internal_holes(self):
        outer = np.zeros((100, 100), bool); outer[20:70, 20:70] = True
        detail = np.zeros_like(outer); detail[35:50, 35:50] = True
        outer[detail] = False
        masks = select_silhouettes([self.proposal(detail), self.proposal(outer)], 1, min_area_mm2=20)
        self.assertEqual(len(masks), 1)
        self.assertTrue(masks[0][40,40])

    def test_marker_and_background_rejected_but_adjacent_tools_remain(self):
        marker = np.zeros((100,100), bool); marker[:20,:20] = True
        one = np.zeros_like(marker); one[30:50,30:50] = True
        two = np.zeros_like(marker); two[30:50,52:72] = True
        proposals = [self.proposal(m) for m in [marker, one, two, np.ones_like(marker)]]
        masks = select_silhouettes(proposals, 1, marker, 100)
        self.assertEqual(len(masks), 2)
        self.assertFalse(np.any(masks[0] & masks[1]))

    def test_area_filter_uses_physical_units(self):
        mask = np.zeros((100,100), bool); mask[10:30,10:30] = True
        self.assertEqual(len(select_silhouettes([self.proposal(mask)], .5, min_area_mm2=150)), 0)
        self.assertEqual(len(select_silhouettes([self.proposal(mask)], 1, min_area_mm2=150)), 1)


class PhotoEndpointTests(unittest.TestCase):
    def test_photo_outlines_are_not_warped_by_unregistered_depth(self):
        from unittest.mock import patch
        import app
        image = np.zeros((100, 100, 3), np.uint8)
        session = app.STORE.create(source_kind='capture', filename='test.jpg', original=image)
        session.rectified = image
        session.rect_height = np.full((100, 100), 40, np.float32)
        session.mm_per_px = 1
        session.scan_meta = {'sensors': ['rear_photo', 'truedepth_tracked'], 'rgbd_registration': None}
        mask = np.zeros((100,100), bool); mask[20:80, 30:70] = True
        try:
            with patch.object(app.SEGMENTER, 'automatic_masks', return_value=[{'segmentation': mask, 'area': int(mask.sum())}]):
                result = app._detect_photo_tools(session, {})
            self.assertEqual(len(result['tools']), 1)
            tool = result['tools'][0]
            self.assertIsNone(tool['measured_thickness_mm'])
            self.assertEqual(tool['edge_source'], 'photo')
            xs = np.asarray(tool['polygon_px'])[:,0]
            self.assertAlmostEqual(xs.min(), 30.5, delta=1)
            self.assertAlmostEqual(xs.max(), 69.5, delta=1)
        finally:
            with app.STORE._lock:
                app.STORE._items.pop(session.id, None)

class MacDetectionTests(unittest.TestCase):
    def test_metal_failure_retries_on_cpu_and_caches_results(self):
        from types import SimpleNamespace
        from unittest.mock import patch, Mock
        from toolcutter.segmenter import Segmenter
        segmenter = Segmenter()
        segmenter.device = 'mps'
        segmenter._predictor = SimpleNamespace(model=object())
        masks = [{'segmentation': np.ones((10, 10), bool), 'area': 100}]
        generator = Mock()
        generator.generate.side_effect = [TypeError('Metal cannot convert float64'), masks]
        def fallback():
            segmenter.device = 'cpu'
        module = SimpleNamespace(SamAutomaticMaskGenerator=Mock(return_value=generator))
        with patch.dict(sys.modules, {'segment_anything': module}), patch.object(segmenter, '_fallback_to_cpu', side_effect=fallback) as retry:
            image = np.zeros((10,10,3), np.uint8)
            self.assertIs(segmenter.automatic_masks(image, key='one'), masks)
            self.assertIs(segmenter.automatic_masks(image, key='one'), masks)
            retry.assert_called_once()
            self.assertEqual(generator.generate.call_count, 2)

if __name__ == '__main__': unittest.main()
