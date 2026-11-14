"""Constraint-based outline cleanup: the trace supplies coordinates, knowledge supplies constraints, and a
constraint the trace does not support is refused rather than forced."""
import sys, unittest
from pathlib import Path
import numpy as np
import cv2
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from toolcutter.cleanup import propose
from shapely.geometry import Polygon, box

rng = np.random.default_rng(3)

def ring_of(poly: Polygon, n=1500, noise=0.25):
    b = poly.exterior; d = np.linspace(0, b.length, n, endpoint=False)
    pts = np.array([b.interpolate(x).coords[0] for x in d])
    return pts + rng.normal(0, noise, pts.shape)

def iou(a, b):
    A, B = Polygon(a).buffer(0), Polygon(b).buffer(0)
    return A.intersection(B).area / A.union(B).area


class CleanupTests(unittest.TestCase):
    def test_noisy_rectangle_becomes_its_rectangle(self):
        truth = box(0, 0, 100, 40)
        out = propose(ring_of(truth))
        self.assertTrue(any(a["type"] == "rectangle" for a in out["applied"]), out)
        self.assertEqual(len(out["polygon_mm"]), 4)
        self.assertGreater(iou(out["polygon_mm"], truth), 0.995)
        rect = next(a for a in out["applied"] if a["type"] == "rectangle")
        w, h = sorted(rect["size_mm"])
        self.assertAlmostEqual(w, 40, delta=0.4); self.assertAlmostEqual(h, 100, delta=0.4)

    def test_l_shape_gets_parallel_edges_and_right_angles_not_a_rectangle(self):
        truth = Polygon([(0, 0), (80, 0), (80, 25), (30, 25), (30, 70), (0, 70)])
        out = propose(ring_of(truth))
        kinds = {a["type"] for a in out["applied"]}
        self.assertNotIn("rectangle", kinds)
        self.assertIn("parallel_edges", kinds, out)
        self.assertIn("right_angles", kinds, out)
        self.assertGreater(iou(out["polygon_mm"], truth), 0.985)
        self.assertLess(len(out["polygon_mm"]), 12)              # six true corners, a little slack

    def test_symmetric_tool_with_one_noisy_side_is_symmetrised(self):
        # a capsule-ish handle: symmetric about its long axis; make one flank noisier
        truth = box(0, 0, 120, 30).buffer(8, join_style=1)
        r = ring_of(truth, noise=0.15)
        r[r[:, 1] > 20] += rng.normal(0, 0.6, r[r[:, 1] > 20].shape)   # the far flank wobbles 4x more
        out = propose(r, {"symmetric_axis": "long", "shape_class": "rounded_rectangle"})
        self.assertTrue(any(a["type"] == "mirror_symmetry" and "refused" not in a for a in out["applied"]), out)
        self.assertGreater(iou(out["polygon_mm"], truth), iou(r, truth))   # closer to truth than the trace

    def test_circle_becomes_a_circle(self):
        truth = Polygon([(30 * np.cos(t), 30 * np.sin(t)) for t in np.linspace(0, 2 * np.pi, 720, endpoint=False)])
        out = propose(ring_of(truth, noise=0.2))
        self.assertTrue(any(a["type"] == "circle" for a in out["applied"]), out)
        self.assertAlmostEqual(next(a for a in out["applied"] if a["type"] == "circle")["radius_mm"], 30, delta=0.15)

    def test_asymmetric_irregular_shape_is_left_alone(self):
        # a hammer-ish silhouette: head one side, handle offset — nothing here should be forced
        truth = Polygon([(0, 0), (60, 0), (60, 18), (38, 18), (38, 140), (24, 140), (24, 18), (0, 18)]).buffer(1.5, join_style=2)
        r = ring_of(truth)
        out = propose(r, {"shape_class": "irregular", "symmetric_axis": "none"})
        self.assertFalse(any(a["type"] in ("rectangle", "circle", "mirror_symmetry") for a in out["applied"]), out)
        self.assertLess(abs(out["area_change_pct"]), 3.0)
        self.assertGreater(iou(out["polygon_mm"], truth), 0.98)

    def test_unsupported_constraint_is_refused_with_a_reason(self):
        truth = Polygon([(0, 0), (80, 0), (80, 25), (30, 25), (30, 70), (0, 70)])   # an L is not a rectangle
        out = propose(ring_of(truth), {"shape_class": "rectangle"})
        self.assertTrue(any(x["type"] == "rectangle" and "refused" in x for x in out["refused"]), out)
        self.assertGreater(iou(out["polygon_mm"], truth), 0.98)      # and the trace is kept, not forced


if __name__ == "__main__":
    unittest.main()


class RecognizeDegradesTest(unittest.TestCase):
    """No credentials, no network, no SDK: the model calls return None and the caller gets empty hints."""

    def test_analyze_returns_none_when_client_fails(self):
        import anthropic
        from toolcutter import recognize
        img = np.zeros((40, 60, 3), np.uint8)
        real = anthropic.Anthropic
        try:
            def boom(*a, **k):
                raise anthropic.AnthropicError("Could not resolve authentication method. Expected one of api_key, auth_token...")
            anthropic.Anthropic = boom
            recognize._UNAVAILABLE = None
            self.assertIsNone(recognize.analyze(img, None, None, name_hint="Tool 1", n_marks=8, mpp=0.19))
            self.assertIsNotNone(recognize.unavailable_reason())
            self.assertIsNone(recognize.verify(img, name_hint="x", n_marks=8, mpp=0.19))
        finally:
            anthropic.Anthropic = real
            recognize._UNAVAILABLE = None
        self.assertEqual(recognize.hints_from(None), {})
        self.assertEqual(recognize.hints_from({"confidence": 0.3, "shape_class": "rectangle"}), {})
        h = recognize.hints_from({"confidence": 0.9, "shape_class": "rectangle", "symmetric_axis": "long",
                                  "straight_edges": True, "right_angles": True, "round_shaft": False})
        self.assertEqual(h["shape_class"], "rectangle")
        self.assertEqual(h["symmetric_axis"], "long")

    def test_schemas_use_only_supported_keywords(self):
        # the API rejected `minimum`/`maximum` on a number with a 400 (2026-10-02); keep every schema to what it accepts
        from toolcutter import recognize
        banned = {"minimum", "maximum", "minLength", "maxLength", "pattern", "format", "minItems", "maxItems"}
        def walk(node):
            if isinstance(node, dict):
                self.assertFalse(banned & set(node.keys()), node.keys())
                for v in node.values(): walk(v)
            elif isinstance(node, list):
                for v in node: walk(v)
        walk(recognize.ANALYSIS_SCHEMA); walk(recognize.VERIFY_SCHEMA)

    def test_clean_edits_drops_bad_marks(self):
        from toolcutter import recognize
        out = recognize._clean_edits([{"from_mark": 3, "to_mark": 5, "kind": "spur", "note": "x"},
                                      {"from_mark": 40, "to_mark": 2, "kind": "spur", "note": "past the end"},
                                      {"from_mark": 1, "to_mark": 2, "kind": "banana", "note": "bad kind"}], 32)
        self.assertEqual(len(out), 1)


def _rect_ring(w=100.0, h=40.0, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    pts = []
    for a, b, n in [((0, 0), (w, 0), 100), ((w, 0), (w, h), 40), ((w, h), (0, h), 100), ((0, h), (0, 0), 40)]:
        t = np.linspace(0, 1, n, endpoint=False)[:, None]
        pts.append(np.array(a) + (np.array(b) - np.array(a)) * t)
    P = np.vstack(pts)
    return P + rng.normal(0, noise, P.shape)


class LocalEditsTest(unittest.TestCase):
    """The model names a run by two marks and says what is wrong there; the geometry measures and applies — or refuses."""

    def _marks_covering(self, ring, marks, x0, x1, y_side):
        """Marks whose ring points lie on the given side (y near y_side) between x0 and x1."""
        pts = ring[marks]
        sel = [k for k, p in enumerate(pts) if abs(p[1] - y_side) < 2.0 and x0 <= p[0] <= x1]
        return min(sel), max(sel)

    def test_spur_is_cut_off_and_notch_is_refused_on_it(self):
        from toolcutter.cleanup import prepare_ring, mark_indices, apply_local_edits
        P = _rect_ring()
        # a 6 mm shadow spur sticking out of the bottom edge (y = 40) between x = 45 and 55
        spur = np.array([[45, 40], [48, 46], [52, 46], [55, 40]], float)
        # the bottom run goes from (100, 40) to (0, 40); index 140 + 45 is x = 55, where the spur begins
        P2 = np.vstack([P[:185], spur[::-1], P[185:]])
        ring = prepare_ring(P2)
        marks = mark_indices(ring, 48)
        a, b = self._marks_covering(ring, marks, 40, 60, 40)
        # widen by one mark either side so the whole bump is inside the run
        a, b = max(0, a - 1), min(len(marks) - 1, b + 1)
        r2, applied, refused = apply_local_edits(ring, marks, [{"from_mark": a, "to_mark": b, "kind": "spur"}])
        self.assertEqual(len(applied), 1, refused)
        self.assertLess(applied[0]["area_change_mm2"], -20)
        self.assertLess(r2[:, 1].max(), 41.0)     # the spur is gone
        # the same run called a 'notch' must be refused: bridging it REMOVES area
        _, applied2, refused2 = apply_local_edits(ring, marks, [{"from_mark": a, "to_mark": b, "kind": "notch"}])
        self.assertEqual(len(applied2), 0); self.assertIn("bulges", refused2[0]["refused"])

    def test_straight_run_is_projected_and_a_curved_one_refused(self):
        from toolcutter.cleanup import prepare_ring, mark_indices, apply_local_edits
        P = _rect_ring(noise=0.4, seed=3)
        ring = prepare_ring(P)
        marks = mark_indices(ring, 32)
        a, b = self._marks_covering(ring, marks, 5, 95, 0)
        r2, applied, refused = apply_local_edits(ring, marks, [{"from_mark": a, "to_mark": b, "kind": "straight"}])
        self.assertEqual(len(applied), 1, refused)
        idx = np.arange(marks[a], marks[b] + 1)[12:-12]        # the ends are tapered into the ring; the body is collinear
        seg = r2[idx]; c = seg.mean(axis=0); d = np.linalg.svd(seg - c)[2][0]
        self.assertLess(np.abs((seg - c) @ np.array([-d[1], d[0]])).max(), 1e-6)
        ends = r2[[marks[a], marks[b]]]; self.assertTrue(np.allclose(ends, ring[[marks[a], marks[b]]]))   # endpoints untouched
        # a semicircle called 'straight' is refused
        th = np.linspace(0, 2 * np.pi, 200, endpoint=False)
        circ = np.column_stack([30 * np.cos(th), 30 * np.sin(th)])
        ring_c = prepare_ring(circ); marks_c = mark_indices(ring_c, 16)
        _, ap, rf = apply_local_edits(ring_c, marks_c, [{"from_mark": 0, "to_mark": 6, "kind": "straight"}])
        self.assertEqual(len(ap), 0); self.assertIn("straight line", rf[0]["refused"])
        # ...but as an 'arc' it is accepted with the right radius
        _, ap2, _ = apply_local_edits(ring_c, marks_c, [{"from_mark": 0, "to_mark": 6, "kind": "arc"}])
        self.assertEqual(len(ap2), 1); self.assertAlmostEqual(ap2[0]["radius_mm"], 30.0, delta=0.2)

    def test_too_tight_run_moves_out_to_the_photo_edge(self):
        from toolcutter.cleanup import prepare_ring, mark_indices, apply_local_edits
        mpp = 0.2
        # photo: a bright 104 x 44 mm tool on a dark liner; the trace is a 100 x 40 rectangle sitting 2 mm INSIDE it
        img = np.full((400, 700, 3), 40, np.uint8)
        origin_mm = np.array([-10.0, -10.0])
        tl = ((np.array([-2.0, -2.0]) - origin_mm) / mpp).astype(int); br = ((np.array([102.0, 42.0]) - origin_mm) / mpp).astype(int)
        cv2.rectangle(img, tuple(tl), tuple(br), (190, 190, 190), -1)
        img = cv2.GaussianBlur(img, (0, 0), 2)
        ring = prepare_ring(_rect_ring()); marks = mark_indices(ring, 32)
        a, b = self._marks_covering(ring, marks, 5, 95, 40)        # the bottom edge
        r2, applied, refused = apply_local_edits(ring, marks, [{"from_mark": a, "to_mark": b, "kind": "too_tight"}],
                                                 photo=img, origin_mm=origin_mm, mpp=mpp)
        self.assertEqual(len(applied), 1, refused)
        self.assertAlmostEqual(applied[0]["moved_mm"], 2.0, delta=0.6)
        mid = ring[(marks[a] + marks[b]) // 2]
        self.assertAlmostEqual(r2[(marks[a] + marks[b]) // 2][1], 42.0, delta=0.7)
        # the other direction has no edge: 'too_loose' on the same run is refused
        _, ap2, rf2 = apply_local_edits(ring, marks, [{"from_mark": a, "to_mark": b, "kind": "too_loose"}],
                                        photo=img, origin_mm=origin_mm, mpp=mpp)
        self.assertEqual(len(ap2), 0); self.assertIn("no clear edge", rf2[0]["refused"])

    def test_run_spanning_most_of_the_ring_is_refused(self):
        from toolcutter.cleanup import prepare_ring, mark_indices, apply_local_edits
        ring = prepare_ring(_rect_ring()); marks = mark_indices(ring, 32)
        _, ap, rf = apply_local_edits(ring, marks, [{"from_mark": 2, "to_mark": 1, "kind": "spur"}])   # 31 of 32 marks
        self.assertEqual(len(ap), 0); self.assertIn("spans", rf[0]["refused"])

    def test_propose_with_local_edits_then_global_constraints(self):
        from toolcutter.cleanup import prepare_ring, mark_indices, propose
        P = _rect_ring(noise=0.3, seed=1)
        spur = np.array([[45, 40], [48, 45], [52, 45], [55, 40]], float)
        P2 = np.vstack([P[:185], spur[::-1], P[185:]])
        ring = prepare_ring(P2); marks = mark_indices(ring, 48)
        pts = ring[marks]
        sel = [k for k, p in enumerate(pts) if abs(p[1] - 40) < 3.0 and 38 <= p[0] <= 62]
        a, b = max(0, min(sel) - 1), min(len(marks) - 1, max(sel) + 1)
        res = propose(P2, {"shape_class": "rectangle", "straight_edges": True, "right_angles": True},
                      ring=ring, marks=marks, local=[{"from_mark": a, "to_mark": b, "kind": "spur", "note": "shadow"}])
        kinds = [x["type"] for x in res["applied"]]
        self.assertIn("local_spur", kinds, res["refused"])
        self.assertIn("rectangle", kinds, res["refused"])
        out = np.asarray(res["polygon_mm"]); self.assertEqual(len(out), 4)
        w, h = sorted(np.ptp(out, axis=0)); self.assertAlmostEqual(w, 40, delta=0.6); self.assertAlmostEqual(h, 100, delta=0.6)


class AnnotateTest(unittest.TestCase):
    def test_annotate_and_preview_draw_without_error(self):
        from toolcutter import recognize
        from toolcutter.cleanup import prepare_ring, mark_indices
        img = np.full((300, 600, 3), 60, np.uint8)
        cv2.rectangle(img, (60, 60), (540, 240), (150, 150, 150), -1)
        h = np.zeros((300, 600), np.float32); h[70:230, 70:530] = 12.0
        ring = prepare_ring(_rect_ring(96, 36)); marks = mark_indices(ring)
        mpp = 0.2
        poly_px = ring / mpp + [60, 60]
        photo, hcrop, (x0, y0, x1, y1) = recognize.make_crops(img, h, poly_px, 50)
        local = poly_px - [x0, y0]
        pm, hm = recognize.annotate(photo, hcrop, local, marks, mpp)
        # small crops are upscaled to MIN_SIDE on the long side, aspect kept, so the marks stay legible
        self.assertEqual(max(pm.shape[:2]), recognize.MIN_SIDE); self.assertIsNotNone(hm)
        self.assertAlmostEqual(pm.shape[0] / pm.shape[1], photo.shape[0] / photo.shape[1], delta=0.01)
        self.assertEqual(hm.shape[:2], pm.shape[:2])
        pv = recognize.render_preview(img[y0:y1, x0:x1], local, local * 0.98 + 5, marks, mpp)
        self.assertEqual(pv.shape[:2], pm.shape[:2])
        out = Path(__file__).parent / "out"; out.mkdir(exist_ok=True)
        cv2.imwrite(str(out / "cleanup_annotate_photo.png"), pm); cv2.imwrite(str(out / "cleanup_annotate_height.png"), hm)
        cv2.imwrite(str(out / "cleanup_preview.png"), pv)


class EndpointOrchestrationTest(unittest.TestCase):
    """The whole cleanup round trip with a MOCK model: analysis with a spur edit, a verify pass that objects to one more
    run, a second round, a preview picture in the response. Exercises everything except the network."""

    def test_cleanup_endpoint_with_mock_model(self):
        import os
        os.environ["TC_NO_WARM"] = "1"
        import app as A
        from toolcutter import recognize
        from toolcutter.cleanup import prepare_ring, mark_indices
        mpp = 0.2
        img = np.full((500, 800, 3), 40, np.uint8)
        cv2.rectangle(img, (100, 100), (600, 300), (170, 170, 170), -1)        # a 100 x 40 mm bright tool
        h = np.zeros((500, 800), np.float32); h[100:300, 100:600] = 10.0

        class Fake:
            rectified = img; rect_height = h; mm_per_px = mpp
        P = _rect_ring() + [20.0, 20.0]
        spur = np.array([[65, 60], [68, 66], [72, 66], [75, 60]], float)
        P2 = np.vstack([P[:185], spur[::-1], P[185:]])
        ring = prepare_ring(P2); marks = mark_indices(ring)
        pts = ring[marks]
        sel = [k for k, p in enumerate(pts) if abs(p[1] - 60) < 3.0 and 58 <= p[0] <= 82]
        a, b = max(0, min(sel) - 1), min(len(marks) - 1, max(sel) + 1)
        top = [k for k, p in enumerate(pts) if abs(p[1] - 20) < 2.0 and 25 <= p[0] <= 115]
        calls = {"analyze": 0, "verify": 0}

        def fake_analyze(pm, hm, ctx, **kw):
            calls["analyze"] += 1
            self.assertEqual(pm.shape[:2], hm.shape[:2]); self.assertIsNotNone(ctx)
            return {"tool_name": "steel block", "description": "a flat rectangular block", "shape_class": "rectangle",
                    "symmetric_axis": "long", "straight_edges": True, "right_angles": True, "round_shaft": False,
                    "trace_quality": "minor_issues", "confidence": 0.9, "notes": "",
                    "edits": [{"from_mark": a, "to_mark": b, "kind": "spur", "note": "shadow"}]}

        def fake_verify(preview, **kw):
            calls["verify"] += 1
            self.assertEqual(preview.ndim, 3)
            return {"follows_edge": False, "better_than_orange": True, "notes": "top edge wobbles",
                    "issues": [{"from_mark": min(top), "to_mark": max(top), "kind": "straight", "note": "ruler edge"}]}

        real = (A._session, A._require_rectified, recognize.analyze, recognize.verify)
        try:
            A._session = lambda sid: Fake(); A._require_rectified = lambda s: s
            recognize.analyze = fake_analyze; recognize.verify = fake_verify
            c = A.app.test_client()
            r = c.post("/api/sessions/abc/cleanup", json={"tools": [{"id": "t1", "polygon_px": (P2 / mpp).tolist(), "name": "Tool 1"}]})
            self.assertEqual(r.status_code, 200, r.get_json())
            d = r.get_json()
            self.assertTrue(d["recognition_available"])
            p = d["proposals"][0]
            kinds = [x["type"] for x in p["applied"]]
            self.assertIn("local_spur", kinds, p["refused"]); self.assertIn("local_straight", kinds, p["refused"])
            self.assertIn("rectangle", kinds, p["refused"])
            self.assertTrue(p["verification"]["second_round"])
            self.assertTrue(p["preview_png"])
            self.assertEqual(p["recognition"]["tool_name"], "steel block")
            self.assertEqual(len(p["polygon_mm"]), 4)
            self.assertEqual(calls, {"analyze": 1, "verify": 1})
        finally:
            A._session, A._require_rectified, recognize.analyze, recognize.verify = real


class DrawerAgentTest(unittest.TestCase):
    """The agent loop with a SCRIPTED model: view the drawer, view a merged tool, measure the valley, split it, name
    the halves, finish. Exercises DrawerTools against the real split/segment code on a synthetic scan."""

    def test_agent_splits_a_merged_pair(self):
        import os, types
        os.environ["TC_NO_WARM"] = "1"
        import app as A
        from toolcutter import agent as agentmod
        from toolcutter.sessions import Session
        mpp = 0.25
        img = np.full((400, 800, 3), 50, np.uint8)
        h = np.zeros((400, 800), np.float32)
        # two 60 x 30 mm blocks 4 mm apart, merged by a low bridge
        cv2.rectangle(img, (100, 100), (340, 220), (170, 170, 170), -1); cv2.rectangle(img, (356, 100), (596, 220), (170, 170, 170), -1)
        h[100:220, 100:340] = 12.0; h[100:220, 356:596] = 12.0; h[150:170, 340:356] = 3.0
        s = Session(id="agenttest", source_kind="capture", filename="synthetic", original=img)
        s.rectified = img; s.rect_height = h; s.mm_per_px = mpp
        merged = np.zeros((400, 800), bool); merged[100:220, 100:596] = True
        s.masks["m1"] = merged
        ring = np.array([[100, 100], [596, 100], [596, 220], [100, 220]], float)
        tool = {"id": "m1", "session_id": "agenttest", "name": "Tool 1", "points": [], "box": None, "polygon_px": ring.tolist(),
                "polygon_mm": (ring * mpp).tolist(), "area_mm2": float(496 * 120 * mpp * mpp), "measured_thickness_mm": 12.0, "height_stats": None}
        dt = agentmod.DrawerTools(s, [tool], {"split_core": A._split_core, "split_core_path": A._split_core_path, "merge_core": A._merge_core,
                                              "segment_topo": A._segment_topo, "depth_cell_mm": A._depth_cell_mm})

        # marks on the top and bottom edge at x = 348 px (the valley): found from the view
        dt.view_tool("m1")
        v = dt.views["m1"]; pts = v["ring"][v["marks"]] / mpp
        top = min(range(len(pts)), key=lambda k: abs(pts[k][0] - 348) + (0 if abs(pts[k][1] - 100) < 3 else 1e6))
        bot = min(range(len(pts)), key=lambda k: abs(pts[k][0] - 348) + (0 if abs(pts[k][1] - 220) < 3 else 1e6))
        dt.views.pop("m1")

        script = [
            [("view_drawer", {})],
            [("view_tool", {"tool_id": "m1"})],
            [("height_profile", {"tool_id": "m1", "from_mark": top, "to_mark": bot})],
            [("split_tool", {"tool_id": "m1", "from_mark": top, "to_mark": bot})],
            None,   # placeholder: rename the halves (ids known only after the split)
            [("finish", {"summary": "Split the merged pair into two blocks."})],
        ]
        calls = {"n": 0}

        class B:  # content blocks
            def __init__(self, **k): self.__dict__.update(k)

        class FakeMessages:
            def create(self, **kw):
                step = script[calls["n"]]; calls["n"] += 1
                if step is None:
                    ids = list(dt.order)
                    step = [("rename_tool", {"tool_id": ids[0], "name": "left block"}), ("rename_tool", {"tool_id": ids[1], "name": "right block"})]
                content = [B(type="tool_use", id=f"tu{calls['n']}_{i}", name=n, input=inp) for i, (n, inp) in enumerate(step)]
                return types.SimpleNamespace(content=content, stop_reason="tool_use", usage=types.SimpleNamespace(input_tokens=100, output_tokens=20))

        fake = types.SimpleNamespace(messages=FakeMessages())
        res = agentmod.run_drawer_agent(dt, client=fake)
        self.assertEqual(res["stopped"], "finish", res["log"])
        self.assertEqual(res["removed"], ["m1"])
        self.assertEqual(len(res["tools"]), 2)
        names = sorted(t["name"] for t in res["tools"]); self.assertEqual(names, ["left block", "right block"])
        for t in res["tools"]:
            self.assertTrue(t["new"] and t["changed"])
            L, S = sorted(np.ptp(np.asarray(t["polygon_mm"]), axis=0), reverse=True)
            self.assertAlmostEqual(L, 60, delta=2.5); self.assertAlmostEqual(S, 30, delta=2.5)
        # undoing the split must take its children away again (the drill was covered twice on a real run)
        parent_hist = dt.history.get("m1"); self.assertTrue(parent_hist)
        dt.derived["m1"] = list(dt.order); dt.tools["m1"] = parent_hist[-1]  # simulate the state right after the split
        msg, _ = dt.undo_tool("m1")
        self.assertIn("removed its split children", msg); self.assertEqual(dt.order, ["m1"])
        profile = next(e for e in res["log"] if e["tool"] == "height_profile")
        self.assertIn("Drops to the floor", profile["summary"])
        self.assertEqual(res["usage"]["input_tokens"], 600)


class ValleySplitTest(unittest.TestCase):
    """Two blocks separated by a DIAGONAL 3 mm gap: a straight cut between the top and bottom of the gap would clip
    both blocks; the valley cut follows the gap and both halves come out whole."""

    def test_valley_path_follows_a_diagonal_gap(self):
        import os
        os.environ["TC_NO_WARM"] = "1"
        import app as A
        from toolcutter import agent as agentmod
        from toolcutter.sessions import Session
        mpp = 0.25
        img = np.full((400, 800, 3), 50, np.uint8); h = np.zeros((400, 800), np.float32)
        # left block: a parallelogram leaning right; right block: the complement, 12 px (3 mm) gap along a 45-degree line
        for y in range(80, 320):
            xg = 300 + (y - 80)          # gap centre x at this row
            h[y, 100:xg - 6] = 12.0; h[y, xg + 6:700] = 12.0
        img[h > 0] = 170
        s = Session(id="valley", source_kind="capture", filename="synthetic", original=img)
        s.rectified = img; s.rect_height = h; s.mm_per_px = mpp
        s.masks["m1"] = h > 0
        s.masks["m1"][80:320, 100:700] = True        # the merged detection bridges the gap
        ring = np.array([[100, 80], [700, 80], [700, 320], [100, 320]], float)
        tool = {"id": "m1", "session_id": "valley", "name": "Pair", "points": [], "box": None, "polygon_px": ring.tolist(),
                "polygon_mm": (ring * mpp).tolist(), "area_mm2": 600 * 240 * mpp * mpp, "measured_thickness_mm": 12.0, "height_stats": None}
        dt = agentmod.DrawerTools(s, [tool], {"split_core": A._split_core, "split_core_path": A._split_core_path, "merge_core": A._merge_core,
                                              "segment_topo": A._segment_topo, "depth_cell_mm": A._depth_cell_mm, "split_at_saddles": A.geometry.split_at_saddles})
        self.assertIn("Likely MERGED", dt.suspects())
        dt.view_tool("m1"); v = dt.views["m1"]; pts = v["ring"][v["marks"]] / mpp
        top = min(range(len(pts)), key=lambda k: abs(pts[k][0] - 300) + (0 if abs(pts[k][1] - 80) < 3 else 1e6))
        bot = min(range(len(pts)), key=lambda k: abs(pts[k][0] - 540) + (0 if abs(pts[k][1] - 320) < 3 else 1e6))
        txt, imgs = dt.split_tool("m1", top, bot)
        self.assertIn("valley", txt, txt)
        self.assertEqual(len(dt.order), 2)
        self.assertGreaterEqual(len(imgs), 2)          # the children's pictures come back with the split
        areas = sorted(dt.tools[t]["area_mm2"] for t in dt.order)
        # each block is 240 rows x ~(200..440) px wide trapezoid; the two sides must each be close to half the solid area
        solid = float((h > 0).sum()) * mpp * mpp
        self.assertAlmostEqual(areas[0] + areas[1], solid, delta=0.08 * solid)
        self.assertGreater(areas[0], 0.35 * solid)      # neither side lost a corner to a straight cut


class SmoothEnclosingTest(unittest.TestCase):
    def test_smoothing_encloses_the_trace_and_cuts_vertices(self):
        from toolcutter.cleanup import smooth_enclosing
        from shapely.geometry import Polygon
        rng = np.random.default_rng(2)
        th = np.linspace(0, 2 * np.pi, 400, endpoint=False)
        # a jittery capsule-ish ring: ellipse 80 x 30 with 0.4 mm noise and a 2 mm tooth
        r = np.column_stack([40 * np.cos(th), 15 * np.sin(th)]) + rng.normal(0, 0.4, (400, 2))
        r[100] += [0, 2.0]
        res = smooth_enclosing(r, sigma_mm=2.0)
        out = np.asarray(res["polygon_mm"])
        self.assertLess(res["vertices_after"], 120, res)
        self.assertLessEqual(res["max_inset_mm"], 0.6, res)          # the pocket is never tight
        self.assertLess(abs(res["area_change_pct"]), 6.0, res)
        cover = Polygon(out).buffer(0.35).contains(Polygon(r).buffer(0)) if Polygon(r).is_valid else True
        self.assertTrue(cover, "smoothed outline must enclose the trace to within 0.35 mm")
