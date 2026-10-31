"""Footprint finishing (scanned tools -> primitives or smooth enclosing outlines) and the fusion support raster
that tells a ghost from a tool (Nolan, 2026-10-03: "the shapes/outlines still look ugly")."""
import os, sys, unittest
import numpy as np
from shapely.geometry import Polygon, box, LineString, Point
from shapely import affinity

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from toolcutter import geometry
from toolcutter.depth_fusion import fuse_heights_with_support


def _noisy(poly, sigma=0.3, step=0.5, seed=1):
    rng = np.random.default_rng(seed)
    ring = np.asarray(poly.exterior.coords)[:-1]
    out = []
    for i in range(len(ring)):
        a, b = ring[i], ring[(i + 1) % len(ring)]
        n = max(1, int(np.linalg.norm(b - a) / step))
        for k in range(n):
            out.append(a + (b - a) * k / n)
    return np.asarray(out) + rng.normal(0, sigma, (len(out), 2))


def _finish(truth, **kw):
    res = geometry.process_outline(_noisy(truth), 1.0, clearance_mm=0.0, smoothing_mm=1.5, **kw)
    return Polygon(res["rings"][0]), res["footprint"]


class FootprintTest(unittest.TestCase):
    def _check(self, truth, kind, min_iou=0.93):
        out, fp = _finish(truth)
        self.assertEqual(fp["kind"], kind, fp)
        enclosed = 1 - truth.difference(out).area / truth.area
        self.assertGreaterEqual(enclosed, 0.999, "a pocket must enclose the tool")
        self.assertGreaterEqual(geometry._iou(truth, out), min_iou)
        self.assertLessEqual(out.area / truth.area - 1, 0.12, "no more than a millimetre generous")
        return out, fp

    def test_rectangle_becomes_sharp_rectangle(self):
        out, fp = self._check(box(0, 0, 100, 40), "rectangle")
        self.assertAlmostEqual(fp["w_mm"], 100, delta=0.6)
        self.assertAlmostEqual(fp["h_mm"], 40, delta=0.6)
        self.assertLessEqual(len(out.exterior.coords), 12)

    def test_rounded_rectangle_keeps_its_radius(self):
        truth = affinity.rotate(box(-40, -25, 40, 25).buffer(-6).buffer(6, resolution=24), 20, origin=(0, 0))
        _, fp = self._check(truth, "rounded_rectangle")
        self.assertAlmostEqual(fp["r_mm"], 6, delta=3.1)
        self.assertAlmostEqual(fp["angle_deg"], 20, delta=1.0)

    def test_thin_capsule(self):
        truth = affinity.rotate(LineString([(-69, 0), (69, 0)]).buffer(6, resolution=24), 35, origin=(0, 0))
        self._check(truth, "capsule", min_iou=0.88)

    def test_circle(self):
        _, fp = self._check(Point(10, 10).buffer(30, resolution=48), "circle")
        self.assertAlmostEqual(fp["diameter_mm"], 60, delta=1.0)

    def test_l_shape_is_not_snapped(self):
        truth = Polygon([(0, 0), (100, 0), (100, 30), (30, 30), (30, 90), (0, 90)])
        out, fp = self._check(truth, "smooth", min_iou=0.97)
        self.assertLessEqual(len(out.exterior.coords), 10, "an L is six corners")

    def test_hammer_is_not_snapped(self):
        truth = Polygon([(0, 0), (120, 0), (120, 30), (70, 30), (70, 45), (20, 45), (20, 30), (0, 30)])
        self._check(truth, "smooth", min_iou=0.96)

    def test_rigid_move_is_a_cache_hit_and_equivalent(self):
        ring = _noisy(box(0, 0, 100, 40))
        a = geometry.process_outline(ring, 1.0, clearance_mm=0.0, smoothing_mm=1.5)
        b = geometry.process_outline(ring, 1.0, clearance_mm=0.0, smoothing_mm=1.5, rotation_deg=30, offset_mm=(10, 5))
        pa, pb = Polygon(a["rings"][0]), Polygon(b["rings"][0])
        back = affinity.translate(affinity.rotate(pb, -30, origin=(a["source_centroid_mm"][0] + 10, a["source_centroid_mm"][1] + 5)), -10, -5)
        self.assertGreater(geometry._iou(pa, back), 0.995)

    def test_exact_and_unfinished_paths_untouched(self):
        ring = _noisy(box(0, 0, 100, 40))
        self.assertIsNone(geometry.process_outline(ring, 1.0, clearance_mm=0.0, smoothing_mm=1.5, exact=True)["footprint"])
        self.assertIsNone(geometry.process_outline(ring, 1.0, clearance_mm=0.0, smoothing_mm=1.5, finish=False)["footprint"])


class SupportTest(unittest.TestCase):
    def test_support_counts_agreeing_frames(self):
        base = np.zeros((20, 20), np.float32)
        base[5:15, 5:15] = 30.0                      # a 30 mm block seen by three frames...
        a, b, c = base.copy(), base.copy(), base.copy()
        d = np.zeros((20, 20), np.float32)           # ...and a fourth, misplaced, frame that puts it elsewhere
        d[:, :] = np.nan
        d[0:4, 0:4] = 44.0
        c[16:20, 16:20] = np.nan                      # unseen corner in one frame
        fused, support = fuse_heights_with_support([a, b, c, d], tol_mm=3.0)
        self.assertEqual(int(support[10, 10]), 3)     # block: three frames agree
        self.assertEqual(int(support[2, 2]), 3)       # the ghost's cells: median is the three floors (0), the 44 disagrees
        self.assertAlmostEqual(float(fused[2, 2]), 0.0)
        self.assertEqual(int(support[18, 18]), 2)     # seen by two frames only (one NaN, one absent)
        # two frames that disagree: the median is nobody's measurement, support 0
        e = np.full((20, 20), np.nan, np.float32); e[1, 1] = 40.0
        f = np.full((20, 20), np.nan, np.float32); f[1, 1] = 0.0
        fused2, support2 = fuse_heights_with_support([e, f], tol_mm=3.0)
        self.assertAlmostEqual(float(fused2[1, 1]), 20.0)
        self.assertEqual(int(support2[1, 1]), 0)
        self.assertEqual(int(support2[5, 5]), 0)      # NaN everywhere: 0


if __name__ == "__main__":
    unittest.main()
