# tool-cutter

3D scans of tools → foam-insert cutting files. Two kinds of input, mixable in one layout:

- a **scan of the whole layout** (tools lying on the foam mat / in the drawer), or
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

## How it works

| Step | Layout scan (mat + tools) | Single tool model |
| --- | --- | --- |
| Upload (PLY / OBJ / GLB / STL) | RANSAC plane fit of the mat → top-down height + color raster; auto-classified | Resting pose from convex-hull facets → top-down footprint + height; auto-classified |
| Calibrate | Corners are pre-detected, mat size measured from the scan (override with a tape value); rotate 90° to pick the top-left | not needed, already to scale |
| Detect tools | Height-above-mat blobs (0.5 contour of the "above mat" fraction), optional HQ-SAM refinement or click-to-outline on the color raster | footprint is the tool |
| Depth | Measured per tool (95th-percentile height) | Measured (99th-percentile height) |
| Layout | Clearance offset, smoothing, finger notches, drag/rotate, overlap warnings, add more tool models | same |
| Export | SVG / DXF / STL + in-browser 3D preview | same |

Accuracy on synthetic ground truth (`backend/tests/smoke_test.py`): layout-scan outlines within
~1 mm and thickness within ~0.6 mm; single-model footprints within 0.3 mm. Real phone LiDAR is
much noisier (~5 mm outlines); photogrammetry / object-capture scans are far sharper.

## Stack

- Backend: Python 3.12, Flask, OpenCV, shapely, trimesh + manifold3d (STL booleans), ezdxf,
  PyTorch + vendored [sam-hq](https://github.com/SysCV/sam-hq) (HQ-SAM).
- Frontend: Next.js 14 (app router), React 18, TypeScript, three.js (3D preview).

## Layout

| Path | What |
| --- | --- |
| `backend/app.py` | Flask API (sessions, calibrate, auto_detect, segment, layout, export) |
| `backend/toolcutter/` | `imaging` (encode/colormap), `calibration` (homography), `scan` (mesh → height map, layout/object classifier, resting pose), `segmenter` (HQ-SAM), `geometry` (masks → mm outlines, offsets, notches), `exporters` (SVG/DXF/STL), `sessions` |
| `backend/tests/` | `synth.py` builds a fake layout scan and a fake tool model with known sizes; `smoke_test.py` runs both flows end-to-end |
| `backend/sam-hq/` | Vendored upstream HQ-SAM — do not modify |
| `frontend/app/page.tsx` | Step state machine (Upload → Calibrate → Detect → Layout & export) |
| `frontend/components/` | `UploadStep`, `CalibrateStep`, `DetectStep`, `LayoutStep`, `ThreeViewer`, `ImageStage`, `Stepper` |
| `frontend/lib/` | `api.ts` (client + depth rule), `types.ts`, `geom.ts` |

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
cd frontend
npm install
npm run dev
```

Open http://localhost:3000. From a phone on the same Wi-Fi use `http://<mac-ip>:3000` and set
`NEXT_PUBLIC_API_BASE_URL=http://<mac-ip>:8000` (the frontend defaults to `http://localhost:8000`).

Smoke test (synthetic ground truth, exercises the model):

```
cd backend && ../.venv/bin/python tests/smoke_test.py
```

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
