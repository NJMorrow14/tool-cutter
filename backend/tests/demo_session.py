"""Post a simulated hand-held LiDAR glide of the complex-tools drawer to a RUNNING backend and print the session id.

Sessions live in the backend's memory, so run this after every backend restart to get a drawer to play with:
    cd backend && ../.venv/bin/python tests/demo_session.py [--api http://localhost:8000] [--frames 16] [--seed 5] [--open]
Then open http://localhost:3000/?session=<id> (the --open flag does it).
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
import synth_scene as ss  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--api", default="http://localhost:8000")
ap.add_argument("--admin", default="http://localhost:3000")
ap.add_argument("--scene", default="complex_tools", choices=sorted(ss.SCENES))
ap.add_argument("--frames", type=int, default=16)
ap.add_argument("--seed", type=int, default=5)
ap.add_argument("--open", action="store_true", help="open the session in the default browser")
args = ap.parse_args()

scene = ss.SCENES[args.scene]()
frames = ss.render_human_arc(scene, args.frames, seed=args.seed)
files, manifest = [], []
for i, (jpg, depth, intr, tf) in enumerate(frames):
    files.append((f"frame_{i}.jpg", (f"frame_{i}.jpg", jpg, "image/jpeg")))
    files.append((f"depth_{i}.f32", (f"depth_{i}.f32", depth.astype("<f4").tobytes(), "application/octet-stream")))
    manifest.append({"image": f"frame_{i}.jpg", "depth": f"depth_{i}.f32", "depth_width": depth.shape[1],
                     "depth_height": depth.shape[0], "intrinsics": intr, "transform": tf})
files.append(("manifest", ("manifest.json", json.dumps({"frames": manifest}).encode(), "application/json")))
r = requests.post(f"{args.api}/api/captures/multi", files=files,
                  data={"marker_size_mm": str(ss.MARKER_MM), "inset_mm": str(scene.marker_inset)}, timeout=600)
r.raise_for_status()
info = r.json()
url = f"{args.admin}/?session={info['id']}"
print(f"drawer {info['mat_mm']['width']} x {info['mat_mm']['height']} mm, {info['scan']['frames_used']} frames")
print("SESSION", info["id"])
print(url)
if args.open:
    subprocess.run(["open", url], check=False)
