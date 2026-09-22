"""Compare capture strategies on a long, shallow drawer (900 x 225 mm) with tall tools at both ends.

  single : one still from 800 mm, centred
  multi  : three stills from 700 mm along the drawer, fused (/api/captures/multi)
  sweep  : 36-frame photogrammetry arc (mac/Photogrammetry) -> mesh -> drawer session

Run: cd backend && ../.venv/bin/python tests/strategy_eval.py [out_dir] [single multi sweep]
"""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
# synthetic scans must not land in the real capture store (they would show up under "Recent captures")
os.environ.setdefault("TC_CAPTURE_DIR", str(HERE / "out" / "captures"))
import cv2  # noqa: E402
import numpy as np  # noqa: E402

import synth_scene as ss  # noqa: E402
from app import STORE, app, _session_from_mesh_file  # noqa: E402
from capture_eval import score_session  # noqa: E402
from toolcutter import photogrammetry as pgm  # noqa: E402

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 and Path(sys.argv[1]).is_dir() else HERE / "out"
OUT.mkdir(parents=True, exist_ok=True)
which = [a for a in sys.argv[1:] if a in ("single", "multi", "arc", "sweep", "human", "human0")] or ["single", "multi", "arc", "sweep"]
scene = ss.SCENES[os.environ.get("SCENE", "long_shallow")]()
client = app.test_client()
summary = {}


def summarize(label, rows):
    ok = [r[1] for r in rows if r[1]]
    missed = sum(1 for r in rows if r[1] is None)
    if ok:
        summary[label] = dict(n=len(rows), missed=missed, iou=np.mean([r["iou"] for r in ok]), haus=np.mean([r["haus"] for r in ok]),
                              area=np.mean([abs(r["area"]) for r in ok]), herr=np.mean([abs(r["herr"]) for r in ok if r["herr"] is not None] or [np.nan]))


def frame_files(stills):
    files, manifest = {}, []
    for i, (jpg, depth, intr) in enumerate(stills):
        files[f"frame_{i}.jpg"] = (io.BytesIO(jpg), f"frame_{i}.jpg")
        files[f"depth_{i}.f32"] = (io.BytesIO(depth.astype("<f4").tobytes()), f"depth_{i}.f32")
        manifest.append({"image": f"frame_{i}.jpg", "depth": f"depth_{i}.f32", "depth_width": depth.shape[1], "depth_height": depth.shape[0], "intrinsics": intr})
    return files, manifest


if "single" in which:
    jpg, depth, intr = ss.render_still(scene, (scene.drawer_w / 2 + 10, scene.drawer_h / 2 + 5), 800, tilt_toward=(20, 10))
    (OUT / "long_single.jpg").write_bytes(jpg)
    data = {"image": (io.BytesIO(jpg), "c.jpg"), "marker_size_mm": str(ss.MARKER_MM), "inset_mm": str(scene.marker_inset), "intrinsics": json.dumps(intr),
            "depth": (io.BytesIO(depth.astype("<f4").tobytes()), "d.f32"), "depth_width": str(depth.shape[1]), "depth_height": str(depth.shape[0])}
    r = client.post("/api/captures", data=data, content_type="multipart/form-data")
    info = r.get_json()
    if r.status_code == 201 and info.get("mat_mm"):
        summarize("single still (800 mm)", score_session(info["id"], scene, client, "long_single"))
    else:
        print("single: failed", info)

if "multi" in which:
    # realistic sequence: one overview from 800 mm, then a close-up of each end from 600 mm
    stills = [ss.render_still(scene, (scene.drawer_w / 2 + 10, scene.drawer_h / 2 + 5), 800, tilt_toward=(20, 10), seed=0),
              ss.render_still(scene, (200, scene.drawer_h / 2), 600, tilt_toward=(15, 8), seed=1),
              ss.render_still(scene, (700, scene.drawer_h / 2), 600, tilt_toward=(-15, 8), seed=2)]
    for i, (jpg, *_) in enumerate(stills):
        (OUT / f"long_multi_{i}.jpg").write_bytes(jpg)
    files, manifest = frame_files(stills)
    data = {**files, "manifest": (io.BytesIO(json.dumps({"frames": manifest}).encode()), "manifest.json"),
            "marker_size_mm": str(ss.MARKER_MM), "inset_mm": str(scene.marker_inset)}
    r = client.post("/api/captures/multi", data=data, content_type="multipart/form-data")
    info = r.get_json()
    if r.status_code == 201:
        print("multi meta:", {k: v for k, v in info["scan"].items() if k in ("frames_used", "frames_skipped", "camera_heights_mm", "drawer_sizes_mm")})
        summarize("overview + 2 close-ups fused", score_session(info["id"], scene, client, "long_multi"))
    else:
        print("multi: failed", info)

if "arc" in which:
    # LiDAR arc: 12 close frames (550 mm) along the drawer, registered through ARKit poses; only the frames
    # that happen to see two markers anchor the drawer rectangle (size known from the overview capture)
    frames = ss.render_arc_frames(scene, n_frames=12, height=float(os.environ.get("ARC_H", "550")))
    files, manifest = {}, []
    for i, (jpg, depth, intr, tf) in enumerate(frames):
        files[f"frame_{i}.jpg"] = (io.BytesIO(jpg), f"frame_{i}.jpg")
        files[f"depth_{i}.f32"] = (io.BytesIO(depth.astype("<f4").tobytes()), f"depth_{i}.f32")
        manifest.append({"image": f"frame_{i}.jpg", "depth": f"depth_{i}.f32", "depth_width": depth.shape[1], "depth_height": depth.shape[0],
                         "intrinsics": intr, "transform": tf})
    data = {**files, "manifest": (io.BytesIO(json.dumps({"frames": manifest}).encode()), "manifest.json"),
            "marker_size_mm": str(ss.MARKER_MM), "inset_mm": str(scene.marker_inset),
            "drawer_width_mm": str(scene.drawer_w), "drawer_height_mm": str(scene.drawer_h)}
    r = client.post("/api/captures/multi", data=data, content_type="multipart/form-data")
    info = r.get_json()
    if r.status_code == 201:
        print("arc meta:", {k: v for k, v in info["scan"].items() if k in ("frames_used", "frames_skipped")})
        summarize("LiDAR arc, 12 frames @550 mm", score_session(info["id"], scene, client, "long_arc"))
    else:
        print("arc: failed", info)

def post_frames(frames, drawer_size=None):
    files, manifest = {}, []
    for i, (jpg, depth, intr, tf) in enumerate(frames):
        files[f"frame_{i}.jpg"] = (io.BytesIO(jpg), f"frame_{i}.jpg")
        files[f"depth_{i}.f32"] = (io.BytesIO(depth.astype("<f4").tobytes()), f"depth_{i}.f32")
        manifest.append({"image": f"frame_{i}.jpg", "depth": f"depth_{i}.f32", "depth_width": depth.shape[1], "depth_height": depth.shape[0],
                         "intrinsics": intr, "transform": tf})
    data = {**files, "manifest": (io.BytesIO(json.dumps({"frames": manifest}).encode()), "manifest.json"),
            "marker_size_mm": str(ss.MARKER_MM), "inset_mm": str(scene.marker_inset)}
    if drawer_size:
        data["drawer_width_mm"] = str(drawer_size[0]); data["drawer_height_mm"] = str(drawer_size[1])
    return client.post("/api/captures/multi", data=data, content_type="multipart/form-data")


if "human" in which:
    # hand-held glides of different lengths, drawer size NOT given (must come from markers seen along the way)
    for n_frames, seed in ((6, 1), (10, 2), (16, 3), (24, 4), (16, 5), (10, 6)):
        frames = ss.render_human_arc(scene, n_frames, seed=seed)
        r = post_frames(frames)
        info = r.get_json()
        label = f"hand glide, {n_frames} frames (seed {seed})"
        if r.status_code != 201:
            print(f"\n=== {label}: FAILED {info.get('error')}"); summary[label] = dict(n=8, missed=8, iou=0, haus=0, area=0, herr=0); continue
        meta = info["scan"]
        print(f"\n--- {label}: frames used {meta['frames_used']}, skipped {len(meta['frames_skipped'])}, drawer measured {info['mat_mm']['width']} x {info['mat_mm']['height']}")
        summarize(label, score_session(info["id"], scene, client, f"long_human_{n_frames}_{seed}"))

if "human0" in which:
    # same hand motion, but perfect poses: isolates how much pose error costs
    for n_frames, seed in ((10, 2), (16, 5)):
        frames = ss.render_human_arc(scene, n_frames, seed=seed, pose_noise_mm=0.0, pose_noise_deg=0.0)
        r = post_frames(frames)
        info = r.get_json()
        label = f"hand glide, perfect poses, {n_frames}f (seed {seed})"
        if r.status_code != 201:
            print(f"\n=== {label}: FAILED {info.get('error')}"); continue
        summarize(label, score_session(info["id"], scene, client, f"long_human0_{n_frames}_{seed}"))

if "sweep" in which:
    sweep_dir = ss.render_sweep_path(scene, OUT / "long_sweep", n_frames=36, rx=380, ry=120, height=650)
    out_obj = OUT / "long_sweep_model.obj"
    for f in OUT.glob("long_sweep_model.*"):
        f.unlink()
    t0 = time.time()
    try:
        pgm.run_photogrammetry(sweep_dir, out_obj, detail="reduced")
        print(f"photogrammetry done in {time.time() - t0:.0f}s")
        sess = _session_from_mesh_file(out_obj, out_obj.name, ss.MARKER_MM, scene.marker_inset,
                                       drawer_size_mm=(scene.drawer_w, scene.drawer_h))   # as the app would pass from the still capture
        print("sweep meta:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in sess.scan_meta.items()})
        if sess.auto_calibrated:
            summarize("photogrammetry sweep", score_session(sess.id, scene, client, "long_sweep"))
        else:
            print("sweep: not calibrated")
    except Exception as exc:  # noqa: BLE001
        print("sweep failed:", exc)

print("\n=== summary (means over detected tools) ===")
print(f"{'strategy':26s} {'found':>6s} {'IoU':>6s} {'hausd':>6s} {'|area%|':>8s} {'|dh| mm':>8s}")
for k, v in summary.items():
    print(f"{k:26s} {v['n'] - v['missed']:>3d}/{v['n']:<2d} {v['iou']:6.3f} {v['haus']:6.1f} {v['area']:8.1f} {v['herr']:8.1f}")
