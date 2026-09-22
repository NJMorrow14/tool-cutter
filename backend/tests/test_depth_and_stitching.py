"""Camera calibration, missing-depth and registration regression tests (no model)."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from toolcutter.depth_camera import rectify_lens
from toolcutter.capture import remove_flying_pixels
from toolcutter.geometry import level_height_raster
from toolcutter.registration import floor_alignment, warp_translation
from app import _blend_mosaic, _parse_frame_upload, _build_multi_session, _pose_cv, app


class DepthTests(unittest.TestCase):
    def test_missing_samples_do_not_tilt_the_floor(self):
        yy, xx = np.mgrid[:200, :300]
        plane = 3 + .025 * xx + .008 * yy
        height = np.full(plane.shape, np.nan)
        height[:, 160:] = plane[:, 160:]
        height[50:90, 210:250] += 20
        result, _ = level_height_raster(height)
        self.assertTrue(np.isnan(result[:, :160]).all())
        self.assertLess(np.nanmax(result[:40]), .01)
        self.assertAlmostEqual(float(result[70, 230]), 20, places=3)

    def test_invalid_depth_never_becomes_a_surface(self):
        depth = np.full((40, 40), .5, np.float32)
        depth[12:22, 15:25] = np.nan
        depth[30, 30] = 0
        clean = remove_flying_pixels(depth, nadir_px=(20, 20))
        self.assertTrue(np.isnan(clean[12:22, 15:25]).all())
        self.assertTrue(np.isnan(clean[30, 30]))
        self.assertTrue(np.allclose(clean[np.isfinite(clean)], .5))

    def test_lens_mapping_identity_and_radial_direction(self):
        y, x = np.mgrid[:60, :80]
        image = np.stack([x, y, x], axis=2).astype(np.uint8)
        depth = x.astype(np.float32)
        calibration = {'inverse_lookup': [0, 0], 'center': [40, 30], 'reference': [80, 60]}
        rgb, z = rectify_lens(image, depth, calibration)
        np.testing.assert_array_equal(image, rgb)
        np.testing.assert_array_equal(depth, z)
        calibration['inverse_lookup'] = [.1, .1]
        rgb, z = rectify_lens(image, depth, calibration)
        self.assertEqual(z[30, 60], 62)  # output radius 20 samples input radius 22
        self.assertTrue(np.isnan(z[0, 0]))
        # Same mapping when the depth stream is lower resolution than RGB.
        _, z = rectify_lens(image, depth[::2, ::2], calibration)
        self.assertEqual(z[15, 30], 62)

    def test_manifest_keeps_old_frames_and_rectifies_truedepth(self):
        image = np.zeros((60, 80, 3), np.uint8)
        _, jpg = cv2.imencode('.jpg', image)
        depth = np.full((30, 40), .4, '<f4')
        f = {'image': 'rgb', 'depth': 'z', 'depth_width': 40, 'depth_height': 30,
             'intrinsics': {'fx': 80, 'fy': 80, 'cx': 40, 'cy': 30, 'width': 80, 'height': 60}}
        blobs = {'rgb': jpg.tobytes(), 'z': depth.tobytes()}
        _, z, K = _parse_frame_upload(f, blobs)
        np.testing.assert_array_equal(z, depth)
        f['sensor'] = 'truedepth'
        f['lens_calibration'] = {'inverse_lookup': [0, 0], 'center': [40, 30], 'reference': [80, 60]}
        _, z, K2 = _parse_frame_upload(f, blobs)
        np.testing.assert_array_equal(K2, K)
        np.testing.assert_array_equal(z, depth)

    def test_truedepth_capture_without_arkit_pose_builds_and_detects(self):
        import synth_scene as synth
        scene = synth.scene_small()
        jpg, depth, intr, _ = synth.render(scene)
        frame = dict(image='rgb', depth='z', depth_width=depth.shape[1], depth_height=depth.shape[0],
                     intrinsics=intr, sensor='truedepth',
                     lens_calibration=dict(inverse_lookup=[0, 0], center=[intr['cx'], intr['cy']],
                                           reference=[intr['width'], intr['height']]))
        session = _build_multi_session([frame, frame], {'marker_size_mm': str(synth.MARKER_MM),
                                                       'inset_mm': str(scene.marker_inset)},
                                      {'rgb': jpg, 'z': depth.astype('<f4').tobytes()})
        self.assertEqual(len(session.frames), 2)
        self.assertEqual(session.scan_meta['sensors'], ['truedepth'])
        self.assertAlmostEqual(session.mat_mm[0], scene.drawer_w, delta=2)
        response = app.test_client().post(f'/api/sessions/{session.id}/auto_detect', json={})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.get_json()['tools']), len(scene.tools))

    def test_rear_photo_does_not_disable_depth_registration(self):
        import synth_scene as synth
        scene = synth.scene_small()
        jpg, depth, intr, _ = synth.render(scene)
        frame = dict(image='rgb', depth='z', depth_width=depth.shape[1], depth_height=depth.shape[0],
                     intrinsics=intr, sensor='truedepth', use='depth')
        photo = dict(image='rgb', intrinsics=intr, sensor='rear_photo', use='color')
        session = _build_multi_session([frame, photo, frame], {'marker_size_mm': str(synth.MARKER_MM),
                                      'inset_mm': str(scene.marker_inset)},
                                      {'rgb': jpg, 'z': depth.astype('<f4').tobytes()})
        self.assertEqual(len(session.frames), 3)
        self.assertTrue(session.scan_meta['rgbd_registration']['applied'])
        self.assertEqual(session.scan_meta['rgbd_registration']['frames_registered'], 2)
        self.assertEqual(session.scan_meta['photo_coverage']['photo_count'], 1)
        response = app.test_client().get(f'/api/sessions/{session.id}/heightfield')
        self.assertEqual(response.status_code, 200)
        self.assertIn('valid_b64', response.get_json())

    def test_mirrored_front_camera_sweep_is_unmirrored_and_placed(self):
        """ARKit's FRONT camera delivers a MIRRORED image, which no amount of downstream cleverness survives:
        ArUco cannot decode a reflected marker and ORB cannot match a reflected raster, so every frame of
        Nolan's first tracked capture was dropped and the scan came out empty. Feed the pipeline what the device
        actually sends — mirrored pixels, mirrored depth, and a transform that must be IGNORED because the pose
        does not survive the reflection — and every frame must still land on the drawer."""
        import synth_scene as synth
        scene = synth.scene_small()
        frames = synth.render_arc_frames(scene, n_frames=6, height=420)
        manifest, blobs = [], {}
        for i, (jpg, depth, intr, tf) in enumerate(frames):
            mirrored = cv2.flip(cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR), 1)
            blobs[f'rgb{i}'] = cv2.imencode('.jpg', mirrored)[1].tobytes()
            blobs[f'z{i}'] = np.ascontiguousarray(depth[:, ::-1]).astype('<f4').tobytes()
            manifest.append(dict(image=f'rgb{i}', depth=f'z{i}', depth_width=depth.shape[1],
                                 depth_height=depth.shape[0], intrinsics=intr, sensor='truedepth_tracked',
                                 use='depth', transform=tf, pose_quality='normal'))
        # the pose must be ignored for this sensor, however confidently the device reports it
        self.assertIsNone(_pose_cv(manifest[0]))
        session = _build_multi_session(manifest, {'marker_size_mm': str(synth.MARKER_MM),
                                                 'inset_mm': str(scene.marker_inset)}, blobs)
        self.assertEqual(len(session.frames), len(manifest), session.scan_meta.get('frames_skipped'))
        self.assertAlmostEqual(session.mat_mm[0], scene.drawer_w, delta=3)
        self.assertAlmostEqual(session.mat_mm[1], scene.drawer_h, delta=3)
        response = app.test_client().post(f'/api/sessions/{session.id}/auto_detect', json={})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.get_json()['tools']), len(scene.tools))


class StitchingTests(unittest.TestCase):
    def frame(self, gray):
        return dict(color=cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), valid=np.ones(gray.shape, bool),
                    height=np.zeros(gray.shape, np.float32), rank=0, nadir_px=(gray.shape[1]/2, gray.shape[0]/2), camera_height_mm=400)

    def texture(self):
        rng = np.random.default_rng(17)
        image = np.full((400, 600), 120, np.uint8)
        for _ in range(500):
            x, y = rng.integers([8, 8], [592, 392])
            cv2.circle(image, (int(x), int(y)), int(rng.integers(2, 7)), int(rng.integers(20, 230)), -1)
        return image

    def test_translation_does_not_wrap(self):
        image = np.zeros((20, 20), np.float32)
        image[:, -2:] = 100
        result = warp_translation(image, 4, 0)
        self.assertEqual(float(result.sum()), 0)

    def test_recovers_small_floor_rotation_and_translation(self):
        reference = self.texture()
        transform = cv2.getRotationMatrix2D((300, 200), .65, 1)
        transform[:, 2] += [3, -2]
        source = cv2.warpAffine(reference, transform, (600, 400))
        correction = floor_alignment(self.frame(source), self.frame(reference), .5)
        self.assertIsNotNone(correction)
        expected = np.eye(3); expected[:2] = transform
        np.testing.assert_allclose(correction @ expected, np.eye(3), atol=.2)
        self.assertAlmostEqual(np.linalg.det(correction[:2, :2]), 1, places=5)

    def test_rejects_flat_overlap_and_large_drift(self):
        frame = self.frame(np.full((400, 600), 100, np.uint8))
        self.assertIsNone(floor_alignment(frame, frame, .5))
        reference = self.texture()
        source = warp_translation(reference, 40, 0)
        self.assertIsNone(floor_alignment(self.frame(source), self.frame(reference), .5))

    def test_tool_texture_cannot_vote_as_floor(self):
        reference = self.frame(self.texture())
        source = self.frame(warp_translation(reference['color'][:, :, 0], 4, 0))
        reference['height'][:] = 30
        source['height'][:] = 30
        self.assertIsNone(floor_alignment(source, reference, .5))

    def test_tall_tool_owner_includes_displaced_top(self):
        frames = [self.frame(np.full((180, 240), 100, np.uint8)) for _ in range(2)]
        height = np.zeros((180, 240), np.float32)
        height[65:105, 145:175] = 100
        for f in frames: f['height'] = height.copy()
        frames[0]['nadir_px'] = (60, 85)
        frames[1]['nadir_px'] = (230, 85)
        frames[0]['rank'] = 0
        frames[1]['rank'] = 1
        with patch.dict('os.environ', {'TC_BLEND_DEBUG': '1'}):
            _blend_mosaic(frames, np.zeros(height.shape, np.int32), 1, height)
        from app import _BLEND_DEBUG
        # Far edge x=174 projects to 60 + (174-60) * 400/(400-100) = 212.
        self.assertEqual(_BLEND_DEBUG['solid'][85, 212], 0)

    def test_blurred_closer_view_does_not_win_tool_ownership(self):
        frames = [self.frame(np.full((180, 240), 100, np.uint8)) for _ in range(2)]
        height = np.zeros((180, 240), np.float32)
        height[65:105, 100:140] = 30
        for frame in frames:
            frame['height'] = height.copy()
        frames[0]['quality_penalty_px'] = 60
        frames[1]['nadir_px'] = (150, 90)
        with patch.dict('os.environ', {'TC_BLEND_DEBUG': '1'}):
            _blend_mosaic(frames, np.zeros(height.shape, np.int32), 1, height)
        from app import _BLEND_DEBUG
        self.assertEqual(_BLEND_DEBUG['solid'][85, 120], 1)


if __name__ == '__main__':
    unittest.main()
