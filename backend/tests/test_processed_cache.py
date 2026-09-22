import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from toolcutter.sessions import Session, SessionStore
from toolcutter.capture import CaptureGeometry
from toolcutter.processed_cache import save_session, load_session, save_detection, load_detection

class CacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root/'capture.json').write_text(json.dumps({'frames': [], 'form': {}}))
        image = np.zeros((30, 40, 3), np.uint8)
        geom = CaptureGeometry(np.eye(3), np.array([0,0,1]), np.zeros(3), np.array([1,0,0]), np.array([0,1,0]), 400, 0, 0, 1)
        self.s = Session(id='abcdef123456', source_kind='capture', filename='scan.jpg', original=image,
                         rectified=image, rect_height=np.full((30,40), np.nan, np.float32),
                         frames=[{'color':image, 'geom':geom, 'markers_px':{0:np.eye(2)}}], version=1)
    def tearDown(self): self.temp.cleanup()

    def test_round_trip_is_lazy_and_preserves_geometry_nan_and_array_aliases(self):
        save_session(self.root, self.s)
        loaded = load_session(self.root)
        self.assertIsNotNone(loaded)
        self.assertIsInstance(loaded.original, np.memmap)
        self.assertIs(loaded.original, loaded.rectified)
        self.assertTrue(np.isnan(loaded.rect_height).all())
        self.assertIn(0, loaded.frames[0]['markers_px'])
        np.testing.assert_allclose(loaded.frames[0]['geom'].plane_to_cam(np.array([[2,3]])), [[2,3,0]])
        loaded.original[0,0] = 255
        self.assertEqual(load_session(self.root).original[0,0,0], 0)
        store = SessionStore(); self.assertIs(store.restore(loaded), store.get(loaded.id))

    def test_changed_inputs_and_corrupt_snapshots_fall_back(self):
        save_session(self.root, self.s)
        (self.root/'capture.json').write_text(json.dumps({'frames': [], 'form': {'drawer_width_mm': '200'}}))
        self.assertIsNone(load_session(self.root))
        save_session(self.root, self.s)
        meta=json.loads((self.root/'processed.json').read_text())
        next((self.root/meta['directory']).glob('*.npy')).unlink()
        self.assertIsNone(load_session(self.root))

    def test_detection_survives_restart_and_is_invalidated_with_capture(self):
        self.s.photo_result_cache={'key':(1,200), 'masks':{'a':np.ones((3,4),bool)}}
        save_detection(self.root, self.s)
        cached = load_detection(self.root)
        self.assertEqual(cached['key'], (1,200))
        self.assertTrue(cached['masks']['a'].all())
        (self.root/'capture.json').write_text(json.dumps({'frames': [], 'form': {'detail':'high'}}))
        self.assertIsNone(load_detection(self.root))

    def test_cached_scan_opens_without_waiting_for_another_rebuild(self):
        import app
        directory=self.root/self.s.id
        directory.mkdir()
        (directory/'capture.json').write_text((self.root/'capture.json').read_text())
        save_session(directory, self.s)
        with patch.object(app, 'CAPTURE_DIR', self.root), patch.object(app, 'STORE', SessionStore()), \
             patch.object(app, '_REBUILD_LOCK') as lock, patch.object(app, '_rebuild_capture') as rebuild:
            lock.__enter__.side_effect=AssertionError('Cached reads must not queue behind reconstruction')
            self.assertEqual(app._session(self.s.id).id, self.s.id)
            rebuild.assert_not_called()

    def test_detect_result_reuse_keeps_independent_ids_and_masks(self):
        import app
        result={'tools':[{'id':'cached_1','polygon_px':[[0,0],[1,0],[0,1]]}]}
        def discover(s, body):
            s.masks['cached_1']=np.ones((3,4),bool)
            s.tool_image_frames['cached_1']=None
            return result
        with patch.object(app, '_detect_photo_tools_uncached', side_effect=discover) as model:
            first=app._detect_photo_tools(self.s, {'id_prefix':'first_'})
            second=app._detect_photo_tools(self.s, {'id_prefix':'second_'})
            self.assertEqual(model.call_count, 1)
            self.assertEqual(first['tools'][0]['id'], 'first_1')
            self.assertEqual(second['tools'][0]['id'], 'second_1')
            self.s.masks['first_1'][:]=False
            self.assertTrue(self.s.masks['second_1'].all())
            self.s.version += 1
            app._detect_photo_tools(self.s, {})
            self.assertEqual(model.call_count, 2)
