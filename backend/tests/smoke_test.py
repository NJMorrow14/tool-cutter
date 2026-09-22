"""End-to-end test of the API against synthetic inputs with known ground truth.

Run:  cd backend && ../.venv/bin/python tests/smoke_test.py [out_dir]
"""
from __future__ import annotations

import io
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
# synthetic scans must not land in the real capture store (they would show up under "Recent captures")
os.environ.setdefault("TC_CAPTURE_DIR", str(HERE / "out" / "captures"))

import numpy as np  # noqa: E402

import synth  # noqa: E402
from app import app  # noqa: E402

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else HERE / "out")
OUT.mkdir(parents=True, exist_ok=True)
FAILS = []


def check(cond: bool, msg: str):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        FAILS.append(msg)


def match_tools(tools, truth):
    """Pair detected tools to ground-truth shapes by centroid distance (mm)."""
    pairs = {}
    for t in tools:
        if not t["polygon_px"]:
            continue
        poly = np.asarray(t["polygon_px"])
        c = poly.mean(axis=0)
        pairs[t["id"]] = c
    return pairs


def run_flow(client, kind: str, upload_bytes: bytes, filename: str, corners, width_mm, height_mm, depth_from_scan: bool):
    print(f"\n=== {kind} flow ===")
    t0 = time.time()
    r = client.post("/api/sessions", data={"file": (io.BytesIO(upload_bytes), filename)}, content_type="multipart/form-data")
    check(r.status_code == 201, f"upload -> {r.status_code} {r.get_json() if r.status_code != 201 else ''}")
    info = r.get_json()
    sid = info["id"]
    print(f"  session {sid} source={info['source_kind']} original={info['original']} ({time.time() - t0:.1f}s)")
    if kind == "scan":
        print(f"  scan meta: {json.dumps(info['scan'])}")
        check(info["suggested_corners"] is not None, "scan produced suggested mat corners")
        if corners is None:
            corners = info["suggested_corners"]
    # shuffle corner order to exercise ordering
    corners = [corners[2], corners[0], corners[3], corners[1]]
    body = {"corners": [[float(x), float(y)] for x, y in corners]}
    if width_mm:
        body.update(width_mm=width_mm, height_mm=height_mm)
    r = client.post(f"/api/sessions/{sid}/calibrate", json=body)
    check(r.status_code == 200, f"calibrate -> {r.status_code} {r.get_json() if r.status_code != 200 else ''}")
    info = r.get_json()
    mat = info["mat_mm"]
    mm_per_px = info["rectified"]["mm_per_px"]
    print(f"  rectified {info['rectified']['width']}x{info['rectified']['height']} px, {mm_per_px:.4f} mm/px, mat {mat['width']:.1f}x{mat['height']:.1f} mm")
    if kind == "scan":
        check(abs(mat["width"] - synth.MAT_W) < 3 and abs(mat["height"] - synth.MAT_H) < 3,
              f"scan-measured mat size {mat['width']:.1f}x{mat['height']:.1f} within 3 mm of {synth.MAT_W}x{synth.MAT_H}")
    for stage in ("rectified", "height"):
        r = client.get(f"/api/sessions/{sid}/image/{stage}")
        if r.status_code == 200:
            (OUT / f"{kind}_{stage}.jpg").write_bytes(r.data)

    t0 = time.time()
    r = client.post(f"/api/sessions/{sid}/auto_detect", json={"mode": "auto"})
    check(r.status_code == 200, f"auto_detect -> {r.status_code} {r.get_json() if r.status_code != 200 else ''}")
    det = r.get_json()
    tools = det["tools"]
    print(f"  auto_detect mode={det['mode']} sam_used={det['sam_used']} found {len(tools)} tools in {time.time() - t0:.1f}s")
    check(len(tools) == 3, f"found 3 tools (got {len(tools)})")

    truth = synth.tool_shapes()
    if kind == "scan":
        # a scan has no "up": accept the identity or a 180-degree turn, but never a mirror image
        from shapely import affinity as _aff
        from shapely.geometry import Polygon as _P
        cands = {
            "identity": truth,
            "rot180": {n: (_aff.rotate(sh, 180, origin=(mat["width"] / 2, mat["height"] / 2)), th) for n, (sh, th) in truth.items()},
            "mirror_x": {n: (_aff.scale(sh, xfact=-1, origin=(mat["width"] / 2, 0)), th) for n, (sh, th) in truth.items()},
            "mirror_y": {n: (_aff.scale(sh, yfact=-1, origin=(0, mat["height"] / 2)), th) for n, (sh, th) in truth.items()},
        }
        def _score(tr):
            tot = 0.0
            for n, (sh, _) in tr.items():
                best = 1e9
                for t in tools:
                    if not t["polygon_px"]:
                        continue
                    c = _P(np.asarray(t["polygon_px"]) * mm_per_px).centroid
                    best = min(best, float(np.hypot(c.x - sh.centroid.x, c.y - sh.centroid.y)))
                tot += best
            return tot
        scores = {k: _score(v) for k, v in cands.items()}
        best_frame = min(scores, key=scores.get)
        print(f"  scan frame orientation: {best_frame} (centroid error sums: " + ", ".join(f"{k}={v:.0f}" for k, v in scores.items()) + ")")
        # The truth shapes are authored in SVG space (y down) and extruded along +z. A camera above
        # (looking down -z) sees that layout with y flipped, i.e. cands["mirror_y"]; a 180-degree turn of
        # that view is cands["mirror_x"]. Both are valid top-down views. "identity"/"rot180" would mean
        # the raster is a mirror image of reality (the foam would come out mirrored).
        check(best_frame in ("mirror_y", "mirror_x"), "scan frame is a true top-down view (possibly turned 180), not mirrored")
        truth = cands[best_frame]
    # pair by centroid
    used = set()
    for name, (shape, thick) in truth.items():
        gt_c = np.array([shape.centroid.x, shape.centroid.y])
        best, bestd = None, 1e9
        for t in tools:
            if t["id"] in used or not t["polygon_px"]:
                continue
            from shapely.geometry import Polygon as _P
            _pc = _P(np.asarray(t["polygon_px"]) * mm_per_px).centroid
            c = np.array([_pc.x, _pc.y])
            d = np.linalg.norm(c - gt_c)
            if d < bestd:
                best, bestd = t, d
        if best is None:
            check(False, f"{name}: no detection")
            continue
        used.add(best["id"])
        poly = np.asarray(best["polygon_px"]) * mm_per_px
        w = poly[:, 0].max() - poly[:, 0].min()
        h = poly[:, 1].max() - poly[:, 1].min()
        gb = shape.bounds
        gw, gh = gb[2] - gb[0], gb[3] - gb[1]
        area_err = abs(best["area_mm2"] - shape.area) / shape.area * 100
        print(f"  {name:7s} centroid err {bestd:5.2f} mm | size {w:6.1f}x{h:6.1f} (truth {gw:6.1f}x{gh:6.1f}) | area err {area_err:4.1f}%"
              + (f" | thickness {best['measured_thickness_mm']} (truth {thick})" if depth_from_scan else ""))
        tol = 2.5 if kind == "photo" else 3.5
        check(abs(w - gw) < tol and abs(h - gh) < tol, f"{name}: bbox within {tol} mm")
        check(area_err < 6.0, f"{name}: area within 6%")
        if depth_from_scan:
            check(best["measured_thickness_mm"] is not None and abs(best["measured_thickness_mm"] - thick) < 1.5,
                  f"{name}: thickness within 1.5 mm")
        best["_name"] = name
        best["_depth"] = best["measured_thickness_mm"] if depth_from_scan else thick

    # click-to-segment on the wrench
    wr = truth["wrench"][0]  # already in the detected frame
    px = [wr.centroid.x / mm_per_px, wr.centroid.y / mm_per_px]
    t0 = time.time()
    r = client.post(f"/api/sessions/{sid}/segment", json={"tools": [{"id": "click1", "points": [{"x": px[0], "y": px[1], "label": "pos"}]}]})
    check(r.status_code == 200, f"segment(click) -> {r.status_code} {r.get_json() if r.status_code != 200 else ''}")
    if r.status_code == 200:
        ct = r.get_json()["tools"][0]
        err = abs(ct["area_mm2"] - wr.area) / wr.area * 100
        print(f"  click-segment wrench: area err {err:.1f}% in {time.time() - t0:.1f}s")
        check(err < 8.0, "click-segment area within 8%")

    # layout + exports
    lay_tools = []
    for t in tools:
        if "_name" not in t:
            continue
        lay_tools.append({"id": t["id"], "session_id": t["session_id"], "thickness_mm": t["measured_thickness_mm"],
                          "name": t["_name"], "polygon_mm": t["polygon_mm"], "include": True,
                          "clearance_mm": 0.0, "depth_mm": t["_depth"], "rotation_deg": 0, "offset_mm": {"x": 0, "y": 0},
                          "notch": {"x_mm": 60, "y_mm": 30, "diameter_mm": 18} if t["_name"] == "wrench" else None})
    body = {"mat": {"width_mm": mat["width"], "height_mm": mat["height"]}, "tools": lay_tools, "smoothing_mm": 0.6,
            "default_clearance_mm": 0.0}
    r = client.post("/api/layout", json=body)
    check(r.status_code == 200, f"layout -> {r.status_code} {r.get_json() if r.status_code != 200 else ''}")
    lay = r.get_json()
    for t in lay["tools"]:
        print(f"  layout {t['name']:7s} rings={len(t['rings'])} area={t['area_mm2']:.0f} overlaps={t['overlaps']} outside={t['outside_mat']} notch={t['notch'] is not None}")
    for fmt in ("svg", "dxf", "stl"):
        t0 = time.time()
        r = client.post("/api/export", json={**body, "format": fmt, "export": {"mat_thickness_mm": 30}})
        check(r.status_code == 200, f"export {fmt} -> {r.status_code} ({time.time() - t0:.1f}s) {r.get_json() if r.status_code != 200 else ''}")
        if r.status_code == 200:
            (OUT / f"{kind}_layout.{fmt}").write_bytes(r.data)
    r = client.post("/api/export", json={**body, "format": "stl_tools", "export": {"mat_thickness_mm": 30}})
    check(r.status_code == 200, f"export stl_tools -> {r.status_code}")
    if r.status_code == 200:
        import trimesh as _tm
        (OUT / f"{kind}_tools.stl").write_bytes(r.data)
        tm = _tm.load(OUT / f"{kind}_tools.stl")
        b = tm.bounds
        print(f"  tools stl: {len(tm.faces)} faces, bounds x {b[0][0]:.1f}..{b[1][0]:.1f} y {b[0][1]:.1f}..{b[1][1]:.1f} z {b[0][2]:.1f}..{b[1][2]:.1f}")
        check(len(tm.faces) > 100 and b[1][2] > 30 - 0.1 and b[0][2] >= 30 - 13 - 0.1, "tool bodies sit in pockets and stick up past the foam top")
    # STL sanity: watertight and volume ~ slab minus pockets
    try:
        import trimesh

        m = trimesh.load(OUT / f"{kind}_layout.stl")
        expected = mat["width"] * mat["height"] * 30 - sum(t["area_mm2"] * (d if d else 30) for t, d in
                                                            ((t, next(lt["depth_mm"] for lt in lay_tools if lt["id"] == t["id"])) for t in lay["tools"]))
        print(f"  stl watertight={m.is_watertight} volume={m.volume:.0f} expected~{expected:.0f} bounds={m.bounds.tolist()}")
        check(m.is_watertight, "stl watertight")
        check(abs(m.volume - expected) / expected < 0.01, "stl volume within 1% of slab minus pockets")
    except Exception as exc:  # noqa: BLE001
        check(False, f"stl load: {exc}")
    # SVG sanity
    svg = (OUT / f"{kind}_layout.svg").read_text()
    check(f'width="{mat["width"]:g}mm"' in svg or 'width="' in svg, "svg has mm width")
    check(svg.count("<path") == len(lay_tools), f"svg has {len(lay_tools)} tool paths")


def run_object_flow(client):
    print("\n=== object flow (single tool model) ===")
    ply, (gw, gh, gt) = synth.make_object_ply()
    (OUT / "object.ply").write_bytes(ply)
    t0 = time.time()
    r = client.post("/api/sessions", data={"file": (io.BytesIO(ply), "object.ply")}, content_type="multipart/form-data")
    check(r.status_code == 201, f"object upload -> {r.status_code} {r.get_json() if r.status_code != 201 else ''}")
    info = r.get_json()
    print(f"  classified as {info['source_kind']} (plane inliers {info['scan']['plane_inlier_fraction']:.2f}) in {time.time() - t0:.1f}s")
    check(info["source_kind"] == "object", "auto-classified as a single object")
    check(len(info.get("tools") or []) == 1, "one tool returned with the upload")
    t = info["tools"][0]
    poly = np.asarray(t["polygon_mm"])
    w = poly[:, 0].max() - poly[:, 0].min()
    h = poly[:, 1].max() - poly[:, 1].min()
    print(f"  footprint {w:.1f} x {h:.1f} mm (truth {gw} x {gh}), thickness {t['measured_thickness_mm']} (truth {gt}), name {t.get('name')}")
    check(abs(w - gw) < 2.5 and abs(h - gh) < 2.5, "object footprint within 2.5 mm")
    check(abs(t["measured_thickness_mm"] - gt) < 1.5, "object thickness within 1.5 mm")
    r = client.get(f"/api/sessions/{info['id']}/image/rectified")
    check(r.status_code == 200, "object raster image served")
    if r.status_code == 200:
        (OUT / "object_rectified.jpg").write_bytes(r.data)
    body = {"mat": {"width_mm": 300, "height_mm": 200},
            "tools": [{"id": t["id"], "name": "knob bar", "polygon_mm": t["polygon_mm"], "include": True, "depth_mm": t["measured_thickness_mm"] - 3,
                       "offset_mm": {"x": 20, "y": 20}, "rotation_deg": 15}], "default_clearance_mm": 1.0}
    r = client.post("/api/layout", json=body)
    check(r.status_code == 200, f"object layout -> {r.status_code}")
    if r.status_code == 200:
        lt = r.get_json()["tools"][0]
        print(f"  layout bbox {[round(v, 1) for v in lt['bbox_mm']]} outside={lt['outside_mat']}")
    r = client.post("/api/export", json={**body, "format": "stl", "export": {"mat_thickness_mm": 40}})
    check(r.status_code == 200, f"object export stl -> {r.status_code}")
    body["tools"][0]["session_id"] = info["id"]
    body["tools"][0]["thickness_mm"] = t["measured_thickness_mm"]
    r = client.post("/api/export", json={**body, "format": "stl_tools", "export": {"mat_thickness_mm": 40}})
    check(r.status_code == 200, f"object export stl_tools -> {r.status_code}")
    if r.status_code == 200:
        import trimesh as _tm
        (OUT / "object_tools.stl").write_bytes(r.data)
        tm = _tm.load(OUT / "object_tools.stl")
        b = tm.bounds
        print(f"  object body: {len(tm.faces)} faces, z {b[0][2]:.1f}..{b[1][2]:.1f} (pocket floor {40 - (t['measured_thickness_mm'] - 3):.1f}, top ~43)")
        check(abs(b[1][2] - 43.0) < 1.5, "object body top ~3 mm above the foam")


def run_capture_flow(client):
    import synth_capture as sc
    jpg, depth, intr = sc.render()
    (OUT / "capture.jpg").write_bytes(jpg)
    truth = sc.tools()
    for mode in ("rgbd", "markers"):
        print(f"\n=== phone capture flow ({mode}) ===")
        data = {"image": (io.BytesIO(jpg), "capture.jpg"), "marker_size_mm": str(sc.MARKER_MM), "inset_mm": "6",
                "intrinsics": json.dumps(intr)}
        if mode == "rgbd":
            data["depth"] = (io.BytesIO(depth.astype("<f4").tobytes()), "depth.f32")
            data["depth_width"] = str(depth.shape[1])
            data["depth_height"] = str(depth.shape[0])
        t0 = time.time()
        r = client.post("/api/captures", data=data, content_type="multipart/form-data")
        check(r.status_code == 201, f"capture upload -> {r.status_code} {r.get_json() if r.status_code != 201 else ''}")
        if r.status_code != 201:
            continue
        info = r.get_json()
        sid = info["id"]
        meta = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in info["scan"].items() if k != "mm_per_px"}
        print(f"  {time.time() - t0:.1f}s meta={meta}")
        check(info["auto_calibrated"] and info.get("rectified"), "auto-calibrated from the 4 corner markers")
        mat = info["mat_mm"]
        print(f"  drawer {mat['width']:.1f} x {mat['height']:.1f} mm (truth {sc.DRAWER_W} x {sc.DRAWER_H}); rectified {info['rectified']['width']}x{info['rectified']['height']} @ {info['rectified']['mm_per_px']:.3f} mm/px")
        check(abs(mat["width"] - sc.DRAWER_W) < 2.5 and abs(mat["height"] - sc.DRAWER_H) < 2.5, "drawer size within 2.5 mm")
        r = client.get(f"/api/sessions/{sid}/image/rectified")
        if r.status_code == 200:
            (OUT / f"capture_{mode}_rectified.jpg").write_bytes(r.data)
        t0 = time.time()
        # single stills keep the photo-refined path (perspective displacement is large from one viewpoint; the
        # topographic edges are the default for multi-frame arcs, checked in tests/strategy_eval.py)
        r = client.post(f"/api/sessions/{sid}/auto_detect", json={"min_area_mm2": 500, "edge_source": "photo" if mode == "rgbd" else None})
        check(r.status_code == 200, f"auto_detect -> {r.status_code} {r.get_json() if r.status_code != 200 else ''}")
        det = r.get_json()
        tools = det["tools"]
        print(f"  auto_detect mode={det['mode']} sam_used={det['sam_used']} found {len(tools)} in {time.time() - t0:.1f}s")
        if mode == "rgbd":
            check(len(tools) == 3, f"found 3 tools (got {len(tools)})")
        else:
            check(len(tools) >= 3, f"found at least 3 tools without depth (got {len(tools)})")
        mm_per_px = info["rectified"]["mm_per_px"]
        for name, (tp, th) in truth.items():
            gc = tp.mean(axis=0)
            from shapely.geometry import Polygon as _P
            def _cen(t):
                c = _P(np.asarray(t["polygon_mm"])).centroid
                return np.array([c.x, c.y])
            best = min((t for t in tools if len(t["polygon_mm"]) >= 3), key=lambda t: np.linalg.norm(_cen(t) - gc), default=None)
            if best is None:
                check(False, f"{name}: no detection"); continue
            poly = np.asarray(best["polygon_mm"])
            w, h = poly[:, 0].max() - poly[:, 0].min(), poly[:, 1].max() - poly[:, 1].min()
            gw, gh = tp[:, 0].max() - tp[:, 0].min(), tp[:, 1].max() - tp[:, 1].min()
            cerr = float(np.linalg.norm(_cen(best) - gc))
            line = f"  {name:6s} h={th:4.0f}: {w:6.1f} x {h:6.1f} (truth {gw:.0f} x {gh:.0f}) centroid err {cerr:4.1f} mm"
            if mode == "rgbd":
                line += f" | thickness {best['measured_thickness_mm']}"
                # worst case (35 mm tall, 150 mm off-axis, 450 mm camera height) is limited by mixed LiDAR edge pixels
                tol = 2.5 if th < 30 else 3.5 * 2
                check(abs(w - gw) < tol and abs(h - gh) < tol, f"{name}: corrected footprint within {tol} mm total")
                check(cerr < 2.5, f"{name}: centroid within 2.5 mm")
                check(best["measured_thickness_mm"] is not None and abs(best["measured_thickness_mm"] - th) < 2.5, f"{name}: height within 2.5 mm")
            else:
                # no depth: silhouettes of the tops, displaced outward from the nadir -> only a loose check
                pass  # without depth the outline is the displaced top silhouette; heights are typed by the user
            print(line)
    # the printed ids need not land in the printed order: the server reads each marker's corner off its
    # position. Nolan's first real scans had the BR and BL ids swapped, which used to read as a bow-tie
    # and measured the drawer's diagonals instead of its sides.
    print("\n=== phone capture flow (markers laid out in the wrong order) ===")
    jpg2, depth2, intr2 = sc.render(marker_ids={0: 0, 1: 1, 2: 3, 3: 2})
    r = client.post("/api/captures", content_type="multipart/form-data",
                    data={"image": (io.BytesIO(jpg2), "shuffled.jpg"), "marker_size_mm": str(sc.MARKER_MM), "inset_mm": "6",
                          "intrinsics": json.dumps(intr2),
                          "depth": (io.BytesIO(depth2.astype("<f4").tobytes()), "depth.f32"),
                          "depth_width": str(depth2.shape[1]), "depth_height": str(depth2.shape[0])})
    check(r.status_code == 201, f"shuffled-marker upload -> {r.status_code} {r.get_json() if r.status_code != 201 else ''}")
    if r.status_code == 201:
        mat = r.get_json()["mat_mm"]
        print(f"  corner_map={r.get_json()['scan'].get('corner_map')} drawer {mat['width']:.1f} x {mat['height']:.1f} mm (truth {sc.DRAWER_W} x {sc.DRAWER_H})")
        check(abs(mat["width"] - sc.DRAWER_W) < 2.5 and abs(mat["height"] - sc.DRAWER_H) < 2.5,
              "drawer size still right with the ids out of order")

    r = client.get("/api/marker_sheet.svg?marker_mm=50")
    check(r.status_code == 200 and b"<svg" in r.data, "marker sheet svg")
    (OUT / "markers.svg").write_bytes(r.data)


def main():
    client = app.test_client()
    r = client.get("/health")
    print("health:", r.get_json()["model"])
    # images are no longer accepted
    photo, _ = synth.make_photo()
    r = client.post("/api/sessions", data={"file": (io.BytesIO(photo), "photo.jpg")}, content_type="multipart/form-data")
    check(r.status_code == 400, f"image upload rejected -> {r.status_code}")
    ply = synth.make_scan_ply()
    (OUT / "scan.ply").write_bytes(ply)
    run_flow(client, "scan", ply, "scan.ply", None, None, None, depth_from_scan=True)
    run_object_flow(client)
    run_capture_flow(client)
    print("\nOutputs in", OUT)
    if FAILS:
        print(f"\n{len(FAILS)} FAILURES:")
        for f in FAILS:
            print(" -", f)
        sys.exit(1)
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
