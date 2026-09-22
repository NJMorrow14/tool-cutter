"""Metric regression checks for straight and curved foam-pocket boundaries.

Run: .venv/bin/python -m unittest discover -s backend/tests -p 'test_outline_quality.py'
"""
import sys
import unittest
from pathlib import Path

import numpy as np
from shapely import affinity
from shapely.geometry import Point, Polygon, box

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from toolcutter.geometry import level_height_raster, smooth_polygon, subpixel_ring


class OutlineQualityTests(unittest.TestCase):
    def test_floor_tilt_does_not_join_separate_tools(self):
        yy, xx = np.mgrid[:160, :240]
        floor = 2.5 + 0.006 * xx + 0.003 * yy
        height = floor.copy()
        height[30:100, 25:65] += 20
        height[30:100, 90:130] += 30
        levelled, shift = level_height_raster(height)
        self.assertGreater(shift, 1.0)
        self.assertLess(levelled[30:100, 66:89].max(), 0.01)
        self.assertAlmostEqual(float(np.median(levelled[30:100, 25:65])), 20, places=3)
        self.assertAlmostEqual(float(np.median(levelled[30:100, 90:130])), 30, places=3)

    def test_straight_sides_meet_at_corners(self):
        for angle in (0, 17, 63):
            with self.subTest(angle=angle):
                truth = affinity.rotate(box(0, 0, 150, 40), angle)
                result = smooth_polygon(truth, sigma_mm=3, tol_mm=0.15)
                self.assertTrue(result.is_valid)
                self.assertLess(truth.hausdorff_distance(result), 0.2)
                self.assertEqual(len(result.exterior.coords), 5)

    def test_round_tool_does_not_become_faceted(self):
        truth = Point(0, 0).buffer(35, quad_segs=128)
        result = smooth_polygon(truth, sigma_mm=3, tol_mm=0.15)
        self.assertTrue(result.is_valid)
        self.assertLess(truth.hausdorff_distance(result), 0.3)
        self.assertGreater(truth.intersection(result).area / truth.union(result).area, 0.99)

    def test_concave_tool_keeps_its_open_recess(self):
        truth = Polygon([(0, 0), (150, 0), (150, 20), (30, 20), (30, 90), (0, 90)])
        result = smooth_polygon(truth, sigma_mm=3, tol_mm=0.15)
        self.assertTrue(result.is_valid)
        self.assertFalse(result.contains(Point(50, 40)))
        self.assertGreater(truth.intersection(result).area / truth.union(result).area, 0.995)

    def test_subpixel_sampling_uses_pixel_centres(self):
        # An analytic circle with a fractional centre exposes a half-pixel offset
        # between ring coordinates and OpenCV's array-index sampling coordinates.
        yy, xx = np.mgrid[:100, :100]
        centre = np.array([50.3, 49.7])
        signed = 25 - np.hypot(xx + 0.5 - centre[0], yy + 0.5 - centre[1])
        theta = np.linspace(0, 2 * np.pi, 100, endpoint=False)
        ring = centre + 24 * np.column_stack([np.cos(theta), np.sin(theta)])
        refined = subpixel_ring(ring, signed)
        self.assertLess(np.abs(np.linalg.norm(refined - centre, axis=1) - 25).max(), 0.05)


if __name__ == '__main__':
    unittest.main()
