# ToolFoam Pro

On-site industrial tool organization: we come to the plant, capture each drawer with the iPhone (LiDAR + corner
markers), and cut foam inserts from the result. This repo holds the whole business:

| Part | Where | Run |
|---|---|---|
| Landing site (brochure, walkthrough request form, QR landing) | `apps/web` | `npm run dev:web` → :3001 |
| HQ / admin: foam workflow, analytics, walkthrough requests, brochure + QR | `apps/admin` | `npm run dev:admin` → :3000 |
| Foam API (Flask, OpenCV, HQ-SAM) | `backend` | `cd backend && ../.venv/bin/python app.py --preload` → :8000 |
| iPhone capture app / Mac photogrammetry CLI | `ios`, `mac` | see below |
| Postgres schema + seed | `db/migrations`, `scripts` | `npm run migrate && npm run seed` |

Setup: `docker compose -f compose.dev.yml up -d` (local Postgres), copy `.env.example` values into `apps/web/.env.local` and `apps/admin/.env.local`,
`npm install`, `npm run migrate && npm run seed`. Deployment: `DEPLOYMENT.md` (prepared, not yet deployed).

## Landing site + admin

- The landing copy and photos come from `brand/toolfoam-brochure.pdf`. `/brochure` serves the current PDF
  (uploaded from the admin, stored in Postgres; falls back to the file in `brand/`).
- Every page view, QR scan (`?src=brochure-qr`), CTA click, brochure open, form start and request is an `events`
  row. Visitors are a daily anonymous hash of address + agent; no IP is stored.
- The brochure PDF is generated from the admin's print template (`/brochure/print`) by `npm run brochure`, so its QR code
  always matches `NEXT_PUBLIC_WEB_URL`. Edit copy/photos there, re-run, done.
- Admin → *Brochure & QR* renders the code for `NEXT_PUBLIC_WEB_URL/?src=brochure-qr` (SVG/PNG download) and
  previews/replaces the PDF. *Analytics* shows visits, scans, sources, referrers, which button converts and a
  source → request funnel over 7/30/90 days. *Walkthrough requests* is the lead list with status + notes + CSV.

Drawers of tools → foam-insert cutting files. Three kinds of input, mixable in one layout:

- a **phone capture** from the iOS app (photo + LiDAR depth of the drawer with four printed
  corner markers), which gives a to-scale top-down map, per-tool outlines and heights;
- a **3D scan of the whole layout** (tools lying on the foam mat / in the drawer);
- **single tool models** (scan one tool at a time, or download a model), which are laid flat
  automatically and dropped onto the mat for you to arrange.

Either way you get back:

- an **SVG exactly the size of the mat / drawer** (mm), one closed path per tool, positioned as
  laid out, stroke color = pocket depth, plus an optional label/legend layer;
- a **DXF** with one layer per pocket depth;
- an **STL** of the foam block with every pocket cut to depth (for CNC / CAM), plus an in-browser
  3D preview that shows the block empty or with the tool bodies sitting in their pockets
  (`format=stl_tools` returns those bodies, built from each tool's scanned height map).

Status: local tool, runs on this Mac. Not deployed.

## iPhone app (ios/)

`ios/ToolCutter.xcodeproj` (sources also listed in `ios/project.yml`) offers two scanners:

- **Front TrueDepth** (default on supported phones): synchronized RGB and metric depth through
  AVFoundation, without face tracking. Tap Start, then point the screen-side camera at the tools
  about 20–50 cm away. Pause briefly at each view, overlap by at least half, and use bright, even light.
  Include the corner markers at both ends; a steady overview with three markers helps. Markerless
  close-ups are aligned through depth-corrected image features. Unreliable matches are excluded.
  A vibration marks every five accepted frames.
- **Rear LiDAR / camera**: ARKit depth plus camera poses for a slow glide about 50 cm above the drawer.
  Rear LiDAR rejects low-confidence depth; non-LiDAR phones retain the marker/photo path.

Print the four corner markers at 100%, upright, with their outer corners touching the drawer corners.
TrueDepth records up to three frames per second; rear capture records two. Both stop at 120 frames;
at least five are required. TrueDepth waits for 0.25 seconds of low rotation and acceleration.
Rapid motion is skipped, and Stop drains the encoder before upload so its final frame is included.
TrueDepth selects a compatible high-resolution depth format, sends lens calibration, and keeps both
streams unmirrored in native sensor orientation. Frame timing uses exact supported CMTime ranges:
rounding 1/30 second to microseconds can exceed a 30 fps limit and crash AVFoundation.
The server rectifies lens distortion before measuring.
TrueDepth's actual accuracy and usable range need verification on your phone; the software does not
assume it is always more accurate than rear LiDAR. Implementation follows Apple's
[TrueDepth capture sample](https://developer.apple.com/documentation/avfoundation/streaming-depth-data-from-the-truedepth-camera).

The Mac (`POST /api/captures/multi`) places frames on a metric drawer grid using markers and,
where available, camera poses. Bounded visual corrections use floor features against marker-anchored
frames, rejecting changes that lack agreement or improve neither alignment nor texture residual.
TrueDepth jointly optimizes overlapping metric views, using measured depth to remove parallax
from feature locations on tool surfaces, and recovers corner-marker order across partial views.
Unalignable partial sweeps return capture guidance instead of a misleading mosaic.
Translations never wrap at image edges. The display mosaic
keeps each tall tool and its displaced visible top in one frame, feathering the surrounding floor.
TrueDepth mosaics penalize blurred views when choosing a source.
Depth-based outlines remain separate from the display blending. The app opens the review UI for
outline refinement, layout and export.

Build: open the project in Xcode, set your team under Signing, and run on the phone. The server address
is editable in Settings. Requires iOS 17. Build checks cover simulator and physical-device targets;
a physical TrueDepth scan is still required to validate sensor behavior. Updated TrueDepth uploads
require the server's `truedepth_lens_v1` capability; restart an older server before scanning.

### Hand-held arcs, simulated imperfectly

`tests/strategy_eval.py human` glides over the 900 × 225 mm drawer with uneven speed, ±25 mm height
wobble and drift, ±30 mm lateral wander, random tilt and roll, motion blur on a third of the frames
and ARKit-like pose error (2 mm, 0.3°). All runs found all 8 tools; frame count barely matters:

| glide | frames | mean IoU | mean boundary error | mean height error |
| --- | --- | --- | --- | --- |
| seed 1 | 6 | 0.90 | 4.1 mm | 0.5 mm |
| seed 2 | 10 | 0.95 | 2.5 mm | 0.4 mm |
| seed 3 | 16 | 0.90 | 3.6 mm | 0.3 mm |
| seed 4 | 24 | 0.94 | 2.6 mm | 0.4 mm |
| seed 5 | 16 | 0.90 | 3.7 mm | 0.4 mm |
| seed 6 | 10 | 0.83 | 6.4 mm | 0.4 mm (pose drift cost 8 mm of drawer width) |

### 3D scan and 3D layout

The arc capture is a full height map of the drawer, not just a photo. *Detect* and *Refine* have a **3D scan** toggle:
the fused LiDAR surface with the photo draped over it, orbit/zoom/pan, outlines drawn on the surface, height
exaggeration ×1–4, click a tool to select it. *Layout* opens in **3D** by default: the foam block with the scanned tool
bodies sitting in their pockets (pocket floor dark, through-cuts orange, conflicts red). Drag a tool to move it, `R` to
rotate; the sheet view, exports and the exact-pocket STL preview are unchanged.

### Drawn shapes

*Layout* has a drawing toolbar over the sheet: pick rectangle, slot, circle, hexagon or polygon (keys 1–5, V to go
back to select) and draw straight on the mat — drag corner to corner (Shift for a square), drag a circle out from
its centre (Alt for corner to corner), or click polygon points and close on the first one. A live size readout
follows the drag. **+ Shape** in the tools panel does the same by typed numbers. Drawn pockets behave like any
other tool — move, rotate, auto layout, depth rule, export — and their dimensions stay editable in the tool panel.

### Splitting a merged tool, and Auto layout

If two tools came out as one outline, select it in *Detect*, press **✂ Split tool** (or hold ⌘/Ctrl) and drag a line
across the join: each side becomes its own tool with borders re-read from the scan. In *Layout*, **⊞ Auto layout**
packs every included tool onto the mat — biggest first, long sides squared up, the chosen gap of foam between
pockets and 10 mm from the edge — and anything that does not fit is named so you can shrink the gap or drop it.
You can still drag, nudge and rotate afterwards.

### Refine outlines (hand correction)

Between *Detect tools* and *Layout* there is a **Refine outlines** step: the rectified drawer image with every
outline on top. No tool modes — the gestures are: drag a handle to move it, click on the line to add a handle,
drag empty space to pan, wheel to zoom. Hold **Shift** for the push brush (a circle that shoves the line out of
its way; Shift+wheel sets its size). Hold **Alt** and drag a box to select several handles, then drag them, nudge
with the arrows, rotate with `R` / `[` / `]`, or `Delete` them. Whole-outline *Smooth 1 / 2.5 mm*, *Fewer / More
handles*, undo/redo (⌘Z) and *Reset to detected* live in the panel; the dashed amber line is the detected outline.
Edits are stored on the tool in mm, so the layout, SVG/DXF and STL follow them. *Detect* uses the same idea:
click the mat for a new tool, click a tool to select it, Shift/Alt-click add include/exclude points, drag a box
to prompt inside it.

### Automatic tool outlines

For phone captures, **Detect tools** now defaults to **Follow visible tool edges** in Settings.
HQ-SAM discovers complete objects directly in the displayed photo, filters calibration markers and nested
parts, then lightly smooths the silhouette in physical units. This avoids using inaccurate depth blobs as
photo prompts or pulling a good photo boundary back onto a noisy depth edge. Detection reuses the image
proposals when rerun on the same session version. On Macs, unsupported Metal operations fall back to CPU.

Review the tool list after detection: background regions or clipped objects can still be proposed. The visible
photo silhouette is not a guaranteed perspective-correct physical footprint. Unregistered rear-photo/front-depth
captures do not receive inferred pocket depths from mismatched pixels; set those depths in the layout panel.

Turn off the photo option for depth-only outlines, or choose Height mode explicitly. Mesh layout scans retain
depth-based detection by default. Existing outlines are preserved until you run detection again, and the previous
list can be restored with **Undo list change**.

### Complex shapes (`SCENE=complex_tools`)

Pliers (4–10 mm tapered handles, 8 mm gap), an adjustable wrench with its jaw notch 4 mm from a
utility knife, a hammer, a screwdriver with a 7 mm shaft on a 28 mm grip, a 6 mm hex key, a tape
measure and two sockets. Ideal 12-frame arc: 9/9 found, mean IoU 0.91, mean boundary error 2.0 mm,
heights within 0.6 mm; hand-held glides: 9/9 in every run, 2.2–3.3 mm. Fidelity measures that made
this work: HQ-SAM runs on an upsampled crop around each tool (not the whole drawer), colour edges are
trusted more widely when the tool sits near the camera axis, a per-boundary-point perspective
correction (near edges kept, far edges pulled back) instead of intersecting scaled silhouettes so thin
parts keep their width, a flat 2 mm LiDAR threshold so low parts of tall tools survive (its rim trimmed back to 35 % of the local top,
measured inward along the edge normal so a shaft is not judged against its grip), mixed depth pixels snapped
to the tool only when mostly covered, camera-facing wall samples kept as measured, and merged neighbours split
by re-seeding whatever SAM leaves behind. The colour outline may never cut the LiDAR footprint into pieces. Residual: ±2.5 mm on 4 mm-wide handle
tips (the LiDAR grid is 2.9 mm at 55 cm).

### Hand-held glides: how frames are registered

Every frame that sees a marker is placed by that marker (two adjacent markers fix the frame exactly; one
marker fixes its position and takes the axes from the marker's edges). ARKit poses are used only to place
frames that see no marker, and to triangulate a first guess of the drawer size when no frame sees three
markers. Frames without a marker are then shifted to agree with the marker-placed height maps (phase
correlation); frames at the far end of the drawer are correlated too, but only to re-measure the drawer size
(their agreed shift is the size error), never moved — correlation across viewpoints carries a 1–2 mm bias.
Result on the 760 × 480 complex scene, six simulated hand glides of 6–24 frames: 9/9 tools every time,
mean IoU 0.81–0.90, mean boundary error 2.2–3.3 mm, drawer size within 3 mm.

### Arc (LiDAR fusion) — best accuracy for long drawers

In the capture screen choose **Arc**: start, glide the phone about 50 cm above the drawer from one
end to the other, stop. Each frame (photo + LiDAR depth + ARKit camera pose) is registered onto one
metric drawer grid on the Mac (`POST /api/captures/multi`): frames that see two or more corner markers
anchor the rectangle, the rest register through their pose; heights are the per-cell median across
frames; every tool is outlined in the frame whose camera was closest to it, so perspective displacement
stays small however long the drawer is. Result in seconds, no texture needed.

On the synthetic 900 × 225 mm test drawer (tall tools at both ends, 55 and 60 mm):

| strategy | tools found | mean IoU | mean boundary error | mean area error | mean height error |
| --- | --- | --- | --- | --- | --- |
| one still from 80 cm | 8/8 | 0.89 | 4.5 mm | 5.1 % | 1.4 mm |
| **LiDAR arc, 12 frames @ 55 cm** | 8/8 | **0.95** | **1.9 mm** | 2.7 % | **0.3 mm** |
| photogrammetry sweep (reduced detail) | 7/8 | 0.72 | 12 mm | 15 % | 9 mm |

(Numbers after fixing a marker-inset bug that had compressed every capture's short axis by up to 3.6 %.)

### 3D sweep (photogrammetry via the Mac)

In the app's capture screen switch to **3D sweep**: start, move the phone in a slow arc over the
drawer for ~20 s, stop. The app records a frame every 0.6 s (photo + LiDAR depth + gravity) and
posts them to `POST /api/sweeps`. The Mac runs Apple's PhotogrammetrySession
(`mac/Photogrammetry`, build once with `cd mac/Photogrammetry && swift build -c release`) and turns
the textured mesh into a drawer session: markers give orientation and the drawer rectangle,
heights come from the mesh, and the 3D preview shows the real sculpted tools. Progress is polled at
`GET /api/jobs/<id>`. A medium-detail reconstruction of 36 frames takes a few minutes on an M1 Pro.

Which to use: **Still** for small drawers (one shot, seconds); **Arc** for anything long or with tall
tools (seconds, best outlines); **3D sweep** when you want the real sculpted tool bodies in the 3D
preview. On synthetic scenes photogrammetry outlines were the coarsest of the three; real,
well-textured tools should do better, and its scale is corrected from the known drawer size.

## How it works

| Step | Phone capture (photo + depth + markers) | Layout scan (mat + tools) | Single tool model |
| --- | --- | --- | --- |
| Upload | Plane from depth, photo re-projected onto it, markers → drawer rectangle, auto-calibrated | RANSAC plane fit of the mat → top-down height + color raster; auto-classified | Resting pose from convex-hull facets → top-down footprint + height; auto-classified |
| Calibrate | done by the markers (adjustable) | Corners are pre-detected, mat size measured from the scan (override with a tape value); rotate 90° to pick the top-left | not needed, already to scale |
| Detect tools | Height blobs seeded, HQ-SAM outlines, perspective-corrected footprints | Height-above-mat blobs (0.5 contour of the "above mat" fraction), optional HQ-SAM refinement or click-to-outline on the color raster | footprint is the tool |
| Depth | LiDAR height per tool (95th percentile) | Measured per tool (95th-percentile height) | Measured (99th-percentile height) |
| Layout | Clearance offset, smoothing, finger notches, drag/rotate, overlap warnings, add more tool models | same | same |
| Export | SVG / DXF / STL + in-browser 3D preview (foam / tools toggles) | same | same |

Accuracy on synthetic ground truth (`backend/tests/smoke_test.py`, `backend/tests/capture_eval.py`):
phone still + LiDAR on a 700 × 450 mm drawer from 78 cm, eight tools including a sphere, a lying
cylinder, an L-bracket, a 3 mm ruler and a 45 mm block: median boundary error within ±1 mm for every
tool, 10th/90th percentiles within about ±3–4 mm, areas within ±4 %, heights within 2 mm; drawer size
within 3 mm even with one corner marker hidden by a tall tool (three markers suffice). Layout-scan
outlines within ~1 mm and thickness within ~0.6 mm; single-model footprints within 0.3 mm. Use
1.5–2 mm clearance for LiDAR captures. Real phone LiDAR is
much noisier (~5 mm outlines); photogrammetry / object-capture scans are far sharper.

## Stack

- Backend: Python 3.12, Flask, OpenCV, shapely, trimesh + manifold3d (STL booleans), ezdxf,
  PyTorch + vendored [sam-hq](https://github.com/SysCV/sam-hq) (HQ-SAM).
- Frontend: Next.js 14 (app router), React 18, TypeScript, three.js (3D preview).

## Layout

| Path | What |
| --- | --- |
| `backend/app.py` | Flask API (sessions, calibrate, auto_detect, segment, layout, export) |
| `backend/toolcutter/` | `imaging` (encode/colormap), `calibration` (homography), `capture` (ArUco markers, RGB-D / markers-only rectification, perspective footprint correction), `scan` (mesh → height map, layout/object classifier, resting pose), `segmenter` (HQ-SAM), `geometry` (masks → mm outlines, offsets, notches), `exporters` (SVG/DXF/STL + tool bodies), `sessions` |
| `ios/` | SwiftUI iPhone app: arc capture (ARKit + LiDAR + camera poses), upload, review in a web view |
| `mac/Photogrammetry/` | Swift CLI around Apple PhotogrammetrySession: frames (+depth, gravity) → USDZ → OBJ with texture |
| `backend/tests/` | `synth.py` (fake layout scan + tool model), `synth_capture.py` / `synth_scene.py` (mesh-rendered drawers with depth, markers, multi-frame sweeps), `smoke_test.py` runs every flow end-to-end, `capture_eval.py` scores footprints against truth (IoU, Hausdorff, signed error), `strategy_eval.py` compares still / multi-still / LiDAR arc / photogrammetry on a long drawer |
| `backend/sam-hq/` | Vendored upstream HQ-SAM — do not modify |
| `apps/admin/app/page.tsx` | Step state machine (Upload → Calibrate → Detect → Refine outlines → Layout & export) |
| `apps/admin/components/` | `UploadStep`, `CalibrateStep`, `DetectStep`, `RefineStep`, `LayoutStep`, `ThreeViewer`, `ImageStage`, `Stepper` |
| `apps/admin/lib/` | `api.ts` (client + depth rule), `types.ts`, `geom.ts` |

## Run locally

Backend (port 8000):

```
cd backend
../.venv/bin/python -m pip install -r requirements.txt   # first time
../.venv/bin/python app.py --host 0.0.0.0 --preload
```

HQ-SAM is optional: it refines outlines on textured layout scans and powers click-to-outline.
`sam_hq_vit_b.pth` (380 MB) is in `backend/` (gitignored); the largest `*.pth` present is picked
automatically, or set `HQSAM_CKPT`. Without a checkpoint everything height-based still works.

Frontend (port 3000):

```
cd apps/admin
npm install
npm run dev
```

Open http://localhost:3000. From a phone on the same Wi-Fi use `http://<mac-ip>:3000` and set
`NEXT_PUBLIC_API_BASE_URL=http://<mac-ip>:8000` (the admin defaults to `http://localhost:8000`).

Smoke test (synthetic ground truth, exercises the model):

```
cd backend && ../.venv/bin/python tests/smoke_test.py
```

Outline geometry regression tests (no model required, run from the repository root):

```
.venv/bin/python -m unittest discover -s backend/tests -p 'test_outline_quality.py'
```

Outline cleanup fits straight sides away from rounded corner samples, joins nearby straight
sides at their intersection, and retains curved sections rather than flattening them into facets.
Multi-frame captures also re-reference a residual floor tilt or offset larger than 1 mm before
detecting tools, matching the single-capture path. The regression tests check floor separation,
metric corner error, circular edges, concave recesses, and subpixel coordinates.

## Scanning tips

- Whole layout: Polycam / Scaniverse / RealityScan export (PLY or GLB with color is ideal).
  Include plenty of bare mat or drawer floor so the plane fit locks onto it.
- Single tools: object-capture / photogrammetry mode, or any downloaded model. Units are guessed
  from the size (meters for phone scans, mm for CAD); override on upload if a tool comes out
  1000× off.
- Depth rule defaults to *tool thickness − 3 mm* so tools sit slightly proud for grabbing;
  "through" cuts everything out (typical for laser-cut layered foam).

## Deploy

Not deployed — local tool.

Scanner regression checks (from the repository root):

```sh
.venv/bin/python -m unittest discover -s backend/tests -p 'test_*.py'
swiftc ios/ToolCutter/CaptureTiming.swift ios/tests/capture_timing_test.swift -o /tmp/tool-cutter-timing-test
/tmp/tool-cutter-timing-test
```

Replaying saved TrueDepth capture `7197636cc28f` recovered 23/25 frames (previously 10/25),
with 48 accepted overlaps and a median internal feature residual of 0.77 mm. This is
registration consistency, not measured cutting accuracy; motion blur and thin/reflective tool
surfaces still limit that capture. Comparison images are in `output/scan-quality/`.

TrueDepth outline continuity: unknown depth now remains unknown during per-frame gap filling,
smoothing, and resampling, allowing other views to contribute instead of voting for zero-height
floor. Measured floor still separates tools. Automatic height detection excludes observed marker
squares and the 25% paper quiet border printed by the app; manual outlining remains available.
On capture `e2a2e7ca6ff8`, the first replay reduced 12 detections to 7, with a connected central ruler
and fewer marker-paper false positives. An edge artifact and noisy thin-tool boundaries remain;
this is not a physical fit certification. The three-tool synthetic check remained within 0.004 IoU
per tool of the previous result. Regression suite: 25 tests. Reopen the saved capture and run
Detect tools to regenerate existing outlines after restarting the backend.

### Wide drawers and per-tool overhead views

Use an overview with the corner markers, followed by level, overlapping rear-camera stills across the drawer.
The photo step estimates overlap after each shot and warns below 45%; it does not claim that the entire drawer
has been covered. The server’s **Photo coverage** panel maps actual photo support after upload (green overlap,
amber single views, red missing or incomplete coverage). At least half-frame overlap is the capture target.

Mosaics prefer complete, locally sharp views for raised tools and avoid blending conflicting edges into ghosts.
Automatic discovery uses overlapping local tiles on wide drawers, with a whole-drawer pass to retain tools that
span tiles. Each detected tool is refined from a complete photo when that result agrees with the mosaic;
otherwise it retains its stitched silhouette. **Tool overhead image** shows the chosen source and original
outline. Unknown or unregistered depth is not used to invent a perspective correction: inspect tall tools,
visible seams, and incomplete coverage before cutting. Individual refinements are cached within the session.

Phone changes require rebuilding/installing the iOS app; refreshing the web editor only updates server/web features.


### Measured 3D surfaces

Mixed TrueDepth + rear-photo captures now jointly register the depth views using
measured-depth feature positions. Rear photos stay on the marker-calibrated color
path; they no longer disable the depth pose graph. Disconnected depth views are
excluded when the joint registration succeeds, rather than forced into the scan.

Fusion processes bounded row tiles, preserving the median while avoiding a full multi-gigabyte frame stack.
Unknown depth survives fusion and weighted display resampling. The 3D mesh skips
triangles with unmeasured vertices, and the viewer reports depth coverage. Missing
returns are not flat floor or invented half-height tool surfaces. This remains a
measured overhead height field, not a closed 3D model of tool undersides. Server
changes apply to saved captures when rebuilt; these fixes require no phone update.

### Fast reopening and repeated detection

Processed phone captures are saved beside the raw frames as versioned JSON and
NumPy snapshots. Arrays load lazily, so reopening a scan after a restart does not
repeat reconstruction or eagerly read every source photo. Changed raw inputs or
an incompatible cache version cause a rebuild; partial/corrupt snapshots safely
fall back to raw frames. The first opening of an older capture creates its cache.
Startup restores available snapshots and does not reconstruct unrelated captures
in the background while someone is detecting tools.

Photo detection keeps its full result, source-frame choices, and masks both in
memory and on disk. Repeated detection reuses these results with fresh tool IDs;
changes to calibration or minimum tool area trigger fresh detection. Metal
inference uses float32 point prompts and larger GPU batches without reducing the
point grid or changing the outline thresholds. Raw captures and cache files stay
in the existing ignored `backend/captures` directory.
