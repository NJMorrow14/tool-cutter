"""Accuracy of the phone-capture pipeline on synthetic scenes with complex shapes.

Run: cd backend && ../.venv/bin/python tests/capture_eval.py [scene ...]
Reports, per tool: footprint IoU vs truth, boundary error (Hausdorff, mm), height error, and
writes overlay images to the smoke-test output folder.
"""
from __future__ import annotations

import io
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
# synthetic scans must not land in the real capture store (they would show up under "Recent captures")
os.environ.setdefault("TC_CAPTURE_DIR", str(HERE / "out" / "captures"))
import cv2  # noqa: E402
import numpy as np  # noqa: E402
from shapely.geometry import Polygon  # noqa: E402

import synth_scene as ss  # noqa: E402
from app import STORE, app  # noqa: E402

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 and Path(sys.argv[1]).is_dir() else HERE / "out"
OUT.mkdir(parents=True, exist_ok=True)
names = [a for a in sys.argv[1:] if a in ss.SCENES] or list(ss.SCENES)


def score_session(sid: str, scene, client, label: str):
    """auto_detect on an existing (calibrated) session and score its tools against the scene truth."""
    s = STORE.get(sid)
    info = client.get(f"/api/sessions/{sid}").get_json()
    mat = info["mat_mm"]
    print(f"\n=== {label}: drawer {scene.drawer_w:.0f}x{scene.drawer_h:.0f} -> measured {mat['width']:.1f}x{mat['height']:.1f} ({info['rectified']['mm_per_px']:.3f} mm/px)")
    t0 = time.time()
    det = client.post(f"/api/sessions/{sid}/auto_detect", json={"min_area_mm2": 300}).get_json()
    tools = det["tools"]
    print(f"  auto_detect: {len(tools)} tools (truth {len(scene.tools)}) mode={det['mode']} sam={det['sam_used']} in {time.time() - t0:.1f}s")
    return _score(s, scene, tools, label)


def run(scene_name: str, client):
    scene = ss.SCENES[scene_name]()
    jpg, depth, intr, _ = ss.render(scene)
    (OUT / f"eval_{scene_name}.jpg").write_bytes(jpg)
    data = {"image": (io.BytesIO(jpg), "capture.jpg"), "marker_size_mm": str(ss.MARKER_MM), "inset_mm": str(scene.marker_inset),
            "intrinsics": json.dumps(intr), "depth": (io.BytesIO(depth.astype("<f4").tobytes()), "depth.f32"),
            "depth_width": str(depth.shape[1]), "depth_height": str(depth.shape[0])}
    t0 = time.time()
    r = client.post("/api/captures", data=data, content_type="multipart/form-data")
    info = r.get_json()
    if r.status_code != 201:
        print(f"[{scene_name}] upload failed: {info}"); return
    sid = info["id"]
    if not info.get("mat_mm"):
        print(f"[{scene_name}] not auto-calibrated: scan meta {info.get('scan')}"); return
    mat = info["mat_mm"]
    print(f"\n=== {scene_name}: drawer {scene.drawer_w:.0f}x{scene.drawer_h:.0f} @ cam {scene.cam_height:.0f} mm -> measured {mat['width']:.1f}x{mat['height']:.1f} "
          f"({info['rectified']['mm_per_px']:.3f} mm/px, camera height {info['scan'].get('camera_height_mm', 0):.0f}) in {time.time() - t0:.1f}s")
    t0 = time.time()
    det = client.post(f"/api/sessions/{sid}/auto_detect", json={"min_area_mm2": 300}).get_json()
    tools = det["tools"]
    print(f"  auto_detect: {len(tools)} tools (truth {len(scene.tools)}) sam={det['sam_used']} in {time.time() - t0:.1f}s")
    s = STORE.get(sid)
    return _score(s, scene, tools, scene_name)


def _score(s, scene, tools, label):
    client = app.test_client()
    mpp = s.mm_per_px
    vis = s.rectified.copy()
    print(f"  {'tool':10s} {'h':>5s} {'IoU':>6s} {'hausd':>6s} {'area%':>6s} {'height':>7s} {'note'}")
    rows = []
    used = set()
    for tool in scene.tools:
        truth = tool.footprint
        best, best_iou = None, -1
        for t in tools:
            if t["id"] in used or len(t["polygon_mm"]) < 3:
                continue
            p = Polygon(t["polygon_mm"])
            if not p.is_valid:
                p = p.buffer(0)
            inter = p.intersection(truth).area
            iou = inter / max(1e-6, p.union(truth).area)
            if iou > best_iou:
                best, best_iou = t, iou
        if best is None or best_iou < 0.2:
            print(f"  {tool.name:10s} {tool.height:5.1f}   MISSED"); rows.append((tool.name, None)); continue
        used.add(best["id"])
        p = Polygon(best["polygon_mm"]); p = p if p.is_valid else p.buffer(0)
        haus = p.boundary.hausdorff_distance(truth.boundary)
        # signed boundary error: sample the detected boundary, distance to truth boundary, + outside / - inside
        from shapely.geometry import Point as _Pt
        L = p.exterior.length
        samp = [p.exterior.interpolate(d) for d in np.linspace(0, L, 200, endpoint=False)]
        signed = np.array([(_Pt(q).distance(truth.boundary)) * (1 if not truth.contains(_Pt(q)) else -1) for q in samp])
        sig = f"med {np.median(signed):+.1f} p10 {np.percentile(signed,10):+.1f} p90 {np.percentile(signed,90):+.1f}"
        area_pct = (p.area - truth.area) / truth.area * 100
        hm = best["measured_thickness_mm"]
        herr = (hm - tool.height) if hm is not None else None
        note = ""
        if tool.height <= 3.5: note = "(very thin)"
        print(f"  {tool.name:10s} {tool.height:5.1f} {best_iou:6.3f} {haus:6.1f} {area_pct:+6.1f} {('%+.1f' % herr) if herr is not None else '   n/a':>7s} {sig} {note}")
        rows.append((tool.name, dict(iou=best_iou, haus=haus, area=area_pct, herr=herr)))
        for g in (truth.geoms if hasattr(truth, "geoms") else [truth]):
            cv2.polylines(vis, [np.round(np.asarray(g.exterior.coords) / mpp).astype(np.int32)], True, (0, 255, 0), 2)
        cv2.polylines(vis, [np.round(np.asarray(best["polygon_mm"]) / mpp).astype(np.int32)], True, (0, 0, 255), 2)
    # processed outlines as they would be cut (clearance 0, default smoothing) -> second overlay
    lay = client.post("/api/layout", json={"mat": {"width_mm": scene.drawer_w, "height_mm": scene.drawer_h}, "smoothing_mm": 1.5,
                                            "default_clearance_mm": 0.0,
                                            "tools": [{"id": t["id"], "polygon_mm": t["polygon_mm"], "include": True} for t in tools if len(t["polygon_mm"]) >= 3]}).get_json()
    vis2 = s.rectified.copy()
    for tool in scene.tools:
        for g in (tool.footprint.geoms if hasattr(tool.footprint, "geoms") else [tool.footprint]):
            cv2.polylines(vis2, [np.round(np.asarray(g.exterior.coords) / mpp).astype(np.int32)], True, (0, 255, 0), 2)
    for lt in lay["tools"]:
        for ring in lt["rings"]:
            cv2.polylines(vis2, [np.round(np.asarray(ring) / mpp).astype(np.int32)], True, (0, 0, 255), 2)
    cv2.imwrite(str(OUT / f"eval_{label}_layout_overlay.jpg"), vis2)
    extra = [t for t in tools if t["id"] not in used]
    if extra:
        print(f"  + {len(extra)} spurious detection(s): " + ", ".join(f"{t['area_mm2']/100:.0f} cm2" for t in extra))
        for t in extra:
            cv2.polylines(vis, [np.round(np.asarray(t["polygon_mm"]) / mpp).astype(np.int32)], True, (255, 0, 255), 2)
    cv2.imwrite(str(OUT / f"eval_{label}_overlay.jpg"), vis)
    return rows


if __name__ == "__main__":
    client = app.test_client()
    mesh = [a for a in sys.argv[1:] if a.endswith((".obj", ".ply", ".glb", ".usdz"))]
    if mesh:
        from app import _session_from_mesh_file
        for m in mesh:
            scene = ss.scene_complex_wide()
            sess = _session_from_mesh_file(Path(m), Path(m).name, ss.MARKER_MM, scene.marker_inset)
            print(f"mesh session {sess.id}: auto_calibrated={sess.auto_calibrated} meta={ {k: (round(v,3) if isinstance(v,float) else v) for k,v in sess.scan_meta.items()} }")
            if sess.auto_calibrated:
                score_session(sess.id, scene, client, "photogrammetry_" + Path(m).stem)
            (OUT / f"eval_pg_{Path(m).stem}.jpg").write_bytes(client.get(f"/api/sessions/{sess.id}/image/rectified").data if sess.rectified is not None else client.get(f"/api/sessions/{sess.id}/image/original").data)
    else:
        for n in names:
            run(n, client)
