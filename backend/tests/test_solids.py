"""Cleaning a scanned tool's 3D shape: guided filter, symmetry, primitive fits."""
import sys, unittest
from pathlib import Path
import numpy as np
import cv2
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from toolcutter import solids


def _canvas(w=120, h=90):
    return np.zeros((h, w), np.float32), np.zeros((h, w), bool)


class SolidsTest(unittest.TestCase):
    def test_noisy_box_is_fitted_as_a_box(self):
        rng = np.random.default_rng(0)
        h, m = _canvas()
        m[20:70, 30:90] = True
        h[m] = 12.0 + rng.normal(0, 0.35, int(m.sum()))
        out, rep = solids.clean_relief(h, m, 1.0)
        self.assertEqual(rep["solid"], "box", rep)
        self.assertAlmostEqual(rep["top_mm"], 12.0, delta=0.5)
        core = m.copy(); core[:, :] = False; core[25:65, 35:85] = True
        self.assertLess(float(np.std(out[core])), 0.05)          # flat where the box was fitted

    def test_lying_cylinder_is_fitted_with_its_radius(self):
        rng = np.random.default_rng(1)
        h, m = _canvas(120, 90)
        R = 10.0
        for j in range(30, 90):
            x = (j - 60) * 1.0
            if abs(x) < R:
                z = np.sqrt(R * R - x * x)
                m[15:75, j] = True; h[15:75, j] = z + rng.normal(0, 0.25, 60)
        out, rep = solids.clean_relief(h, m, 1.0)
        self.assertEqual(rep["solid"], "cylinder_lying", rep)
        self.assertAlmostEqual(rep["radius_mm"], R, delta=0.8)

    def test_irregular_shape_stays_freeform(self):
        rng = np.random.default_rng(2)
        h, m = _canvas(120, 90)
        yy, xx = np.mgrid[0:90, 0:120]
        m = ((xx - 60) ** 2 / 40 ** 2 + (yy - 45) ** 2 / 25 ** 2) <= 1
        h[m] = 5 + 10 * np.sin(xx[m] / 8.0) + 6 * np.cos(yy[m] / 5.0) + rng.normal(0, 0.2, int(m.sum()))   # wavy, asymmetric
        out, rep = solids.clean_relief(h, m, 1.0, symmetric_hint=False)
        self.assertEqual(rep["solid"], "freeform", rep)
        self.assertNotIn("mirror_symmetry", [s["step"] for s in rep["steps"]])

    def test_symmetric_tool_halves_are_averaged(self):
        rng = np.random.default_rng(3)
        h, m = _canvas(120, 90)
        yy, xx = np.mgrid[0:90, 0:120]
        m = (np.abs(xx - 60) <= 12) & (np.abs(yy - 45) <= 35)
        # a rounded handle: height falls off from the centre line; one half is noisier than the other
        base = 15 * np.sqrt(np.clip(1 - ((xx - 60) / 12.0) ** 2, 0, 1))
        noise = np.where(xx < 60, rng.normal(0, 0.6, h.shape), rng.normal(0, 0.1, h.shape))
        h = np.where(m, base + noise, 0).astype(np.float32)
        out, rep = solids.clean_relief(h, m, 1.0, solid_hint="freeform")
        self.assertIn("mirror_symmetry", [s["step"] for s in rep["steps"]], rep)
        left = out[m & (xx < 58)]; right = out[m & (xx > 62)]
        # after averaging the two halves are equally smooth
        self.assertLess(abs(np.std(left - base[m & (xx < 58)]) - np.std(right - base[m & (xx > 62)])), 0.15)

    def test_semantic_hint_can_force_a_looser_box(self):
        rng = np.random.default_rng(4)
        h, m = _canvas()
        m[20:70, 30:90] = True
        h[m] = 12.0 + rng.normal(0, 0.9, int(m.sum()))          # too noisy for the automatic 0.8 mm rule
        out_auto, rep_auto = solids.clean_relief(h, m, 1.0)
        out_hint, rep_hint = solids.clean_relief(h, m, 1.0, solid_hint="box")
        # the automatic rule may not call it a box (too noisy), though a flat extrusion profile can still pass; the hint makes it a box
        self.assertNotEqual(rep_auto["solid"], "box"); self.assertEqual(rep_hint["solid"], "box")

    def test_guided_filter_keeps_photo_edges(self):
        h, m = _canvas()
        m[10:80, 10:110] = True
        guide = np.full(h.shape, 60, np.uint8); guide[:, 60:] = 200           # a photo edge down the middle
        h[m] = 10.0; h[m & (np.arange(120)[None, :] >= 60)] = 20.0           # matching step in height, plus noise
        h += np.where(m, np.random.default_rng(5).normal(0, 0.4, h.shape), 0).astype(np.float32)
        q = solids.guided_filter(h, guide.astype(np.float32) / 255.0, 3, solids.GUIDE_EPS, mask=m)
        self.assertLess(float(np.std(q[20:70, 20:50])), 0.15)                 # smoothed
        self.assertGreater(float(q[45, 63] - q[45, 57]), 7.0)                 # the step survives
