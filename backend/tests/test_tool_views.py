import sys
import unittest
from pathlib import Path
import cv2
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from toolcutter.tool_views import choose_tool_frame, coverage_report, discovery_windows

class ToolViewTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(22)
        self.color = rng.integers(30, 220, (120, 200, 3), dtype=np.uint8)
        self.mask = np.zeros((120, 200), bool); self.mask[30:90, 40:160] = True

    def frame(self, image=None):
        return dict(color=self.color if image is None else image, valid=np.ones(self.mask.shape, bool),
                    nadir_px=(100,60), camera_height_mm=400, rank=0, use='color')

    def test_complete_sharp_photo_wins_over_blur_and_clipping(self):
        blurry = self.frame(cv2.GaussianBlur(self.color, (21,21), 5))
        sharp = self.frame(); sharp['nadir_px'] = (110,60)
        clipped = self.frame(); clipped['valid'][:, 120:] = False
        self.assertEqual(choose_tool_frame([blurry, sharp, clipped], self.mask, 1), 1)

    def test_tool_spanning_photos_stays_stitched(self):
        left, right = self.frame(), self.frame()
        left['valid'][:, 115:] = False
        right['valid'][:, :85] = False
        self.assertIsNone(choose_tool_frame([left,right], self.mask, 1))
        report = coverage_report([left,right], self.mask.shape)
        self.assertEqual(report['covered_fraction'], 1)
        self.assertAlmostEqual(report['overlap_fraction'], .15)

    def test_depth_only_frame_never_supplies_tool_photo(self):
        frame = self.frame(); frame['use'] = 'depth'
        self.assertIsNone(choose_tool_frame([frame], self.mask, 1))

    def test_wide_windows_overlap_and_cover_both_ends(self):
        windows = discovery_windows(3400,800)
        count = np.zeros((800,3400), np.uint8)
        for x0,y0,x1,y1 in windows: count[y0:y1,x0:x1] += 1
        self.assertTrue((count > 0).all())
        self.assertEqual(windows[-1][2],3400)
        self.assertTrue((count > 1).any())
        for a,b in zip(windows,windows[1:]): self.assertGreaterEqual(a[2]-b[0],300)

    def test_missing_photo_coverage_is_reported(self):
        frame = self.frame(); frame['valid'][:, 100:] = False
        report = coverage_report([frame], self.mask.shape)
        self.assertEqual(report['covered_fraction'],.5)
        self.assertEqual(report['overlap_fraction'],0)

class SeamTests(unittest.TestCase):
    def test_disagreeing_tool_edges_are_not_averaged_without_depth(self):
        from app import _blend_mosaic
        frames = []
        for x0, nadir in [(90, (30,60)), (110, (210,60))]:
            color = np.full((120,240,3),100,np.uint8)
            color[30:90,x0:x0+50] = 220
            frames.append(dict(color=color,valid=np.ones((120,240),bool),nadir_px=nadir,rank=0))
        result = _blend_mosaic(frames,np.zeros((120,240),np.int32),1)
        # The contested boundary uses a source pixel, not a translucent mix.
        self.assertIn(int(result[60,105,0]), (100,220))


class WideDiscoveryTests(unittest.TestCase):
    def test_tile_crossing_tool_is_returned_once_and_crop_is_served(self):
        from unittest.mock import patch
        import app
        image = np.zeros((300,2400,3), np.uint8)
        image[90:200,900:1250] = 255
        s = app.STORE.create(source_kind='capture', filename='wide-test.jpg', original=image)
        s.rectified = image; s.mm_per_px = .5
        def proposals(img, key=None):
            mask = img[:,:,0] > 128
            return [{'segmentation': mask, 'area': int(mask.sum())}] if mask.any() else []
        try:
            with patch.object(app.SEGMENTER, 'automatic_masks', side_effect=proposals) as discover:
                first = app._detect_photo_tools(s, {})
                before_repeat = discover.call_count
                second = app._detect_photo_tools(s, {})
                self.assertEqual(discover.call_count, before_repeat)
            self.assertEqual(len(first['tools']), 1)
            xs = np.asarray(first['tools'][0]['polygon_px'])[:,0]
            self.assertLess(abs(xs.min()-900), 3)
            self.assertLess(abs(xs.max()-1250), 3)
            result = app.app.test_client().get(second['tools'][0]['image_url'])
            self.assertEqual(result.status_code, 200)
            self.assertEqual(result.mimetype, 'image/jpeg')
        finally:
            with app.STORE._lock: app.STORE._items.pop(s.id,None)

class RecalibrationTests(unittest.TestCase):
    def test_photo_grid_tracks_calibration_without_cumulative_warp(self):
        from app import STORE, _apply_calibration
        image = np.full((120,200,3),100,np.uint8)
        s = STORE.create(source_kind='capture', filename='test.jpg', original=image)
        s.rectified = image; s.mm_per_px = 1
        original = dict(color=image,valid=np.ones((120,200),bool),height=None,
                        nadir_px=np.array([100.,60.]),H=np.eye(3),use='color')
        s.frames = [original]
        corners = np.array([[20,10],[180,10],[180,110],[20,110]],np.float32)
        try:
            _apply_calibration(s,corners,160,100)
            first = s.frames[0]['nadir_px'].copy()
            self.assertEqual(s.frames[0]['valid'].shape, s.rectified.shape[:2])
            _apply_calibration(s,corners,160,100)
            np.testing.assert_allclose(s.frames[0]['nadir_px'],first)
            self.assertAlmostEqual(s.scan_meta['photo_coverage']['covered_fraction'],1)
        finally:
            with STORE._lock: STORE._items.pop(s.id,None)

if __name__ == '__main__': unittest.main()
