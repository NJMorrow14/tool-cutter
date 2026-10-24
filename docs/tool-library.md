# Tool library — per-tool capture (2026-10-13)

Nolan: "take pictures of each tool in the app one by one and they automatically outline. Users can use their
fingers to retrace incorrect portions of the outline and the algorithm will recalculate and correct. Those outlines
are then transferred to the website where they can be added to the toolboard and positioned."
Decisions (Nolan, 2026-10-13): photo only, thickness typed per tool, flat pockets; scale from the LiDAR depth +
plane fit (no markers); outlining and correction ON THE PHONE (Vision foreground mask + Core ML SAM 2.1 Small);
tools go into a LIBRARY per job/customer that the website lists and adds to any board.

## Storage (backend)
`backend/library/<job_id>/job.json` and `backend/library/<job_id>/<tool_id>/{tool.json, photo.jpg, thumb.jpg[, mask.png]}`
(`TC_LIBRARY_DIR` overrides; gitignored). Ids are 12 lowercase hex (`[0-9a-f]{12}`); the PHONE generates tool ids so
retries are idempotent (an upload with an existing id replaces that tool).

## ToolRecord (JSON, the one shape used by phone, server and website)
```
{
  "id": "0123456789ab", "job_id": "fedcba987654", "name": "Claw hammer",
  "thickness_mm": 28.0,                       // typed on the phone; may be null if skipped
  "polygon_mm": [[x, y], ...],                // mm, y-DOWN, NORMALISED so min x = min y = 0; closed ring without a repeated last point
  "area_mm2": 13200.5, "bbox_mm": [w, h],      // derived on the server from polygon_mm
  "captured_at": "2026-10-13T15:04:05Z",
  "source": "phone_photo",
  "scale": {"method": "lidar_plane", "mm_per_px": 0.21, "distance_mm": 412.0, "tilt_deg": 2.1, "plane_inliers": 0.93},
  "edited": true,                             // the person retraced at least one stroke
  "notes": "",
  "photo_url": "/api/library/tools/<id>/photo", "thumb_url": "/api/library/tools/<id>/thumb",
  "created": 1760367845.1, "updated": 1760367845.1
}
```
Server validation: >= 3 finite points, shapely-valid after buffer(0) (largest part kept), area >= 25 mm2,
thickness null or 0 < t <= 300, name <= 80 chars (default "Tool N").

## Routes (Flask, backend/app.py, `/api/library/...`, CORS already covers /api/*)
- `GET  /api/library/jobs`                       -> {jobs: [{id, name, created, updated, tool_count}]} newest first
- `POST /api/library/jobs` {name}                -> job, 201 (name required, <= 80 chars)
- `GET  /api/library/jobs/<job_id>`              -> job (+ tool_count)
- `PATCH /api/library/jobs/<job_id>` {name}      -> job
- `DELETE /api/library/jobs/<job_id>`            -> 204 (removes its tools too)
- `GET  /api/library/jobs/<job_id>/tools`        -> {tools: [ToolRecord]} newest first
- `POST /api/library/jobs/<job_id>/tools`        multipart: `tool` (JSON string: id?, name, thickness_mm, polygon_mm,
                                                   captured_at, scale, edited, notes), `photo` (JPEG, required),
                                                   `mask` (PNG, optional) -> ToolRecord, 201 (200 when replacing an id)
- `GET  /api/library/tools/<tool_id>`            -> ToolRecord
- `PATCH /api/library/tools/<tool_id>` JSON      (name, thickness_mm, polygon_mm, notes, job_id to move) -> ToolRecord
- `DELETE /api/library/tools/<tool_id>`          -> 204
- `GET  /api/library/tools/<tool_id>/photo`      -> image/jpeg (Cache-Control: no-store)
- `GET  /api/library/tools/<tool_id>/thumb`      -> image/jpeg, longest side 320 px, made at upload
Errors: `{"error": msg}` with 400 / 404 via ApiError (existing pattern).

## Website (apps/admin, Design insert)
"Tool library" section in the Tools panel: job picker (last choice remembered), tool rows with thumbnail, name,
w x h mm, thickness, "Add" / "Add all"; polls every 5 s while open so tools appear as the phone captures them.
Adding makes a `Tool` with `source: 'library'`, `session_id: ''`, `library_tool_id`, `polygon_mm` (as stored, origin
top-left), `offset_mm` from `placementOffset`, `measured_thickness_mm`, `pocket_style: 'flat'`, `image_url` = thumb_url.
Library tools survive a new scan/session like 'object' and 'shape' tools. `?job=<id>` opens Design insert with that job.

## Phone (ios/ToolCutter)
Capture tools one by one: aim (level gauge) -> shutter (full-res still + the live frame's LiDAR depth + intrinsics)
-> automatic outline (Vision foreground instance mask) -> finger retrace (one finger draws a correction stroke, two
fingers pan/zoom; the stroke replaces the nearest run of the outline and SAM 2 re-segments inside the stroke's
corridor; undo) -> name + thickness -> save (upload to the current job, or queue offline) -> next tool.
Scale: fit the resting plane to the depth outside the mask; project outline pixels through the still's intrinsics
onto that plane raised by the typed thickness; express in the plane's basis (mm, y-down) and normalise.
Models: `ios/ToolCutter/Models/SAM2_1Small{ImageEncoder,PromptEncoder,MaskDecoder}FLOAT16.mlpackage` (gitignored,
`ios/scripts/fetch_sam2_models.sh` downloads them from huggingface.co/apple/coreml-sam2.1-small, Apache-2.0).
