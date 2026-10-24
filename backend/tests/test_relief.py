"""Form-fit pockets: depth map, STL, 16-bit PNG and G-code from a synthetic layout."""
import io, json, os, sys, unittest, zipfile
from pathlib import Path
import numpy as np
import cv2
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["TC_NO_WARM"] = "1"
import app as A
from toolcutter import relief


def _layout_and_geom():
    """A 300 x 200 mat with a scanned 'block' tool (60 x 30 mm, 12 mm tall with a 20 mm dome rising to 20 mm) placed
    rotated 90 deg and offset, plus a flat 40 x 40 pocket at 10 mm."""
    mpp = 0.5
    mask = np.zeros((200, 300), bool); mask[40:100, 60:180] = True          # 60 x 30 mm at (30, 20) mm
    h = np.zeros((200, 300), np.float32); h[mask] = 12.0
    yy, xx = np.mgrid[0:200, 0:300]
    dome = ((xx - 120) ** 2 + (yy - 70) ** 2) <= 20 ** 2
    h[dome & mask] = 20.0
    poly = [[30, 20], [90, 20], [90, 50], [30, 50]]
    tools = [{"id": "blk", "session_id": "S", "name": "block", "polygon_mm": poly, "include": True, "clearance_mm": 1.0,
              "depth_mm": None, "rotation_deg": 90.0, "offset_mm": {"x": 60.0, "y": 40.0}, "pocket_style": "relief"},
             {"id": "sq", "session_id": "", "name": "square", "polygon_mm": [[200, 100], [240, 100], [240, 140], [200, 140]], "include": True,
              "clearance_mm": 0.0, "depth_mm": 10.0, "rotation_deg": 0.0, "offset_mm": {"x": 0, "y": 0}, "pocket_style": "flat"}]
    body = {"mat": {"width_mm": 300, "height_mm": 200}, "smoothing_mm": 0.0, "default_clearance_mm": 1.0, "tools": tools}
    lay = A._compute_layout(body)
    def geom(raw):
        return (mask, h, mpp) if raw.get("id") == "blk" else None
    return body, lay, geom, mask, h, mpp


class ReliefTest(unittest.TestCase):
    def test_depth_map_follows_the_tool_and_flat_pockets_stay_flat(self):
        body, lay, geom, mask, h, mpp = _layout_and_geom()
        rel = relief.build_depth_map(lay, geom, res_mm=1.0, smooth_mm=0.0, z_clearance_mm=1.0, foam_thickness_mm=30.0)
        D = rel["depth"]
        self.assertEqual(D.shape, (200, 300))
        # the flat square: exactly 10 mm everywhere inside
        sq = D[105:135, 205:235]
        self.assertTrue(np.allclose(sq, 10.0), (sq.min(), sq.max()))
        # the relief block, rotated 90 about its centroid (60, 35) then offset (60, 40): centre lands at (120, 75); the
        # block is now 30 wide x 60 tall; its body should read 13 mm (12 + 1) and the dome 21 mm (20 + 1)
        body_vals = D[50:100, 108:132]
        self.assertGreater((np.abs(body_vals - 13.0) < 0.6).mean(), 0.45, "block body at 13 mm")
        self.assertAlmostEqual(float(D[40:110, 100:140].max()), 21.0, delta=0.6)
        self.assertEqual(float(D[150:190, 20:60].max()), 0.0)        # empty foam untouched
        rep = {r["id"]: r for r in rel["tools"]}
        self.assertEqual(rep["blk"]["style"], "relief"); self.assertEqual(rep["sq"]["style"], "flat")
        # the clearance band (1 mm ring) is as deep as the body next to it, not sloped
        self.assertGreater(float(D[75, 104]), 10.0)

    def test_depth_cap_and_fallback_to_flat(self):
        body, lay, geom, *_ = _layout_and_geom()
        rel = relief.build_depth_map(lay, geom, res_mm=1.0, smooth_mm=0.0, z_clearance_mm=1.0, foam_thickness_mm=12.0, floor_min_mm=2.0)
        self.assertLessEqual(float(rel["depth"].max()), 10.0 + 1e-6)          # never through the floor
        rel2 = relief.build_depth_map(lay, lambda raw: None, res_mm=1.0, foam_thickness_mm=30.0)
        self.assertIn("no scan", {r["id"]: r for r in rel2["tools"]}["blk"]["style"])

    def test_stl_png_gcode_exports(self):
        import trimesh
        body, lay, geom, *_ = _layout_and_geom()
        rel = relief.build_depth_map(lay, geom, res_mm=2.0, smooth_mm=1.0, foam_thickness_mm=30.0)
        stl = relief.depth_to_stl(rel)
        m = trimesh.load(io.BytesIO(stl), file_type="stl")
        self.assertTrue(m.is_watertight, "block mesh must be closed")
        self.assertAlmostEqual(m.bounds[1][2] - m.bounds[0][2], 30.0, delta=0.01)
        self.assertLess(m.volume, 300 * 200 * 30)                    # pockets removed material
        self.assertGreater(m.volume, 300 * 200 * 30 * 0.9)
        png, meta = relief.depth_to_png16(rel)
        from PIL import Image
        img = Image.open(io.BytesIO(png)); self.assertEqual(img.mode, "I;16"); self.assertEqual(img.size, (150, 100))
        self.assertAlmostEqual(meta["max_depth_mm"] / 65535.0, meta["grey_to_mm"], places=5)
        text, stats = relief.depth_to_gcode(rel, cutter_mm=6.0, stepover_mm=2.0, stepdown_mm=8.0, safe_z_mm=5.0)
        self.assertIn("G21 G90", text); self.assertIn("M30", text)
        zs = [float(tok[1:]) for line in text.splitlines() if line.startswith("G1") for tok in line.split() if tok.startswith("Z")]
        self.assertGreaterEqual(min(zs), -float(rel["depth"].max()) - 1e-6)   # never below the deepest modelled point
        xs = [float(tok[1:]) for line in text.splitlines() if line.startswith(("G0", "G1")) for tok in line.split() if tok.startswith("X")]
        self.assertTrue(0 <= min(xs) and max(xs) <= 300)
        self.assertEqual(stats["layers"], 3)                           # 21 mm deep at 8 mm step-down
        # gouge check: the cutter centre never goes below the modelled surface anywhere under the disc
        zc = relief.cutter_floor(rel, 6.0) - rel["thickness_mm"]
        self.assertTrue(np.all(zc >= -rel["depth"] - 1e-6))

    def test_endpoint_grid_and_gcode(self):
        body, lay, geom, mask, h, mpp = _layout_and_geom()
        real = A._geometry_for_tool
        try:
            A._geometry_for_tool = geom
            c = A.app.test_client()
            r = c.post("/api/relief", json={**body, "format": "grid", "relief": {"resolution_mm": 2.0}})
            self.assertEqual(r.status_code, 200, r.get_json())
            d = r.get_json(); self.assertEqual((d["cols"], d["rows"]), (150, 100)); self.assertAlmostEqual(d["max_depth_mm"], 21.0, delta=0.8)
            r = c.post("/api/relief", json={**body, "format": "gcode", "relief": {"resolution_mm": 2.0, "cutter_mm": 6}})
            self.assertEqual(r.status_code, 200); self.assertIn("M3", r.data.decode()); self.assertIn("X-Gcode-Stats", r.headers)
            r = c.post("/api/relief", json={**body, "format": "png", "relief": {"resolution_mm": 2.0}})
            self.assertEqual(r.status_code, 200)
            names = zipfile.ZipFile(io.BytesIO(r.data)).namelist(); self.assertTrue(any(n.endswith("_depth16.png") for n in names))
        finally:
            A._geometry_for_tool = real


if __name__ == "__main__":
    unittest.main()


class PackageTest(unittest.TestCase):
    def test_package_zip_has_everything(self):
        body, lay, geom, *_ = _layout_and_geom()
        real = A._geometry_for_tool
        try:
            A._geometry_for_tool = geom
            c = A.app.test_client()
            r = c.post("/api/relief", json={**body, "format": "package", "relief": {"resolution_mm": 2.0, "cutter_mm": 6}, "export": {"filename": "demo"}})
            self.assertEqual(r.status_code, 200, r.data[:200])
            z = zipfile.ZipFile(io.BytesIO(r.data)); names = set(z.namelist())
            for n in ("demo_relief.nc", "demo_relief.stl", "demo_depth16.png", "demo_depth16.json", "demo.svg", "demo.dxf", "JOB_SHEET.txt"):
                self.assertIn(n, names)
            sheet = z.read("JOB_SHEET.txt").decode()
            self.assertIn("block: relief", sheet); self.assertIn("square: flat", sheet); self.assertIn("Deepest cut", sheet)
        finally:
            A._geometry_for_tool = real


class SmoothWallsTest(unittest.TestCase):
    def test_diagonal_pocket_wall_has_fractional_edge_depths(self):
        """A pocket rotated 30 degrees must not come out as a staircase: edge cells carry intermediate depths."""
        body = {"mat": {"width_mm": 200, "height_mm": 200}, "smoothing_mm": 0.0, "default_clearance_mm": 0.0,
                "tools": [{"id": "r", "session_id": "", "name": "r", "polygon_mm": [[60, 60], [140, 60], [140, 120], [60, 120]], "include": True,
                           "clearance_mm": 0.0, "depth_mm": 10.0, "rotation_deg": 30.0, "offset_mm": {"x": 0, "y": 0}, "pocket_style": "flat"}]}
        lay = A._compute_layout(body)
        rel = relief.build_depth_map(lay, lambda raw: None, res_mm=1.0, foam_thickness_mm=30.0)
        D = rel["depth"]
        frac = (D > 0.5) & (D < 9.5)
        self.assertGreater(int(frac.sum()), 100, "edge cells should carry intermediate (anti-aliased) depths")
        self.assertAlmostEqual(float(D.max()), 10.0, delta=0.01)

    def test_drawn_rounded_rectangle_keeps_its_arcs_in_the_layout(self):
        """A drawn r=4 rounded rectangle must come out of /api/layout with round corners, not chamfers."""
        import math
        w, h, r = 62.3, 89.0, 4.0
        pts = []
        def arc(cx, cy, a0, a1, n=8):
            for i in range(n + 1):
                a = a0 + (a1 - a0) * i / n; pts.append([cx + r * math.cos(a), cy + r * math.sin(a)])
        arc(w - r, r, -math.pi / 2, 0); arc(w - r, h - r, 0, math.pi / 2); arc(r, h - r, math.pi / 2, math.pi); arc(r, r, math.pi, 1.5 * math.pi)
        body = {"mat": {"width_mm": 200, "height_mm": 200}, "smoothing_mm": 1.5, "default_clearance_mm": 1.0,
                "tools": [{"id": "s", "session_id": "", "source": "shape", "shape": {"kind": "rect", "w_mm": w, "h_mm": h, "r_mm": r}, "name": "Rect",
                           "polygon_mm": pts, "include": True, "clearance_mm": 0.0, "depth_mm": 10.0, "rotation_deg": 0.0, "offset_mm": {"x": 20, "y": 20}}]}
        lay = A._compute_layout(body)
        ring = np.asarray(lay["tools"][0]["rings"][0])
        # the true corner arc (centre (20 + w - r, 20 + r), radius r): every ring vertex near that corner must be within 0.25 mm of it
        cx, cy = 20 + w - r, 20 + r
        near = ring[(ring[:, 0] > cx) & (ring[:, 1] < cy)]
        self.assertGreaterEqual(len(near), 3, "the corner must keep several points (an arc), not one chamfer vertex")
        dev = np.abs(np.hypot(near[:, 0] - cx, near[:, 1] - cy) - r)
        self.assertLess(float(dev.max()), 0.25, dev)
