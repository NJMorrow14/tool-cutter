# CLAUDE.md

ToolFoam Pro — on-site industrial tool organization (custom-cut foam drawer inserts). Monorepo:

- `apps/web/` — public landing site (Next.js 14, port 3001): brochure content, `/walkthrough` request form,
  `/brochure` (current PDF), `/api/events` analytics beacon. Design follows `brand/toolfoam-brochure.pdf`
  (charcoal `#1f2326`, orange `#f26a1b`, light uppercase display type, Inter).
- `apps/admin/` — HQ site (Next.js 14, port 3000): the foam workflow UI (capture → detect → refine → layout →
  SVG/DXF/STL) at `/`, plus `/analytics`, `/leads` (walkthrough requests), `/brochure` (PDF upload + QR code).
  Password wall via `middleware.ts` + `lib/session.ts` (off when `ADMIN_PASSWORD` is empty); Cloudflare Access on top in prod.
- `backend/` — Flask + HQ-SAM foam API (port 8000). `ios/` SwiftUI capture app. `mac/Photogrammetry` Swift CLI.
- `db/migrations/*.sql` + `scripts/migrate.mjs` / `scripts/seed.mjs` (idempotent; seed loads the brochure PDF).
  Postgres tables: `events` (page views, qr_scan, cta_click, brochure_download, form_started, walkthrough_submitted;
  `src` carries `?src=` attribution, `brochure-qr` is the printed code), `walkthrough_requests`, `assets` (brochure bytes).
- Logo: `components/Mark.tsx` (identical copies in apps/web and apps/admin) — foam tile + orange wrench/socket pockets;
  wordmark `TOOLFOAM<em>PRO</em>` (PRO orange, bold). Favicons `apps/*/app/icon.svg`, standalone `brand/logo-mark.svg`.
- `brand/` — `toolfoam-brochure.pdf` is GENERATED: `npm run brochure` renders `apps/admin/app/brochure/print` (2 × US Letter,
  photos from `apps/admin/public/brochure/`, QR from `/api/qr` for `NEXT_PUBLIC_WEB_URL/?src=brochure-qr`) with headless Chrome
  and uploads it to the admin's brochure slot. Re-run after the domain is set. `toolfoam-brochure-v1-original.pdf` is
  Nolan's original design; `brand/photos/` the full-size photos.
- Deploy: **not deployed** — `DEPLOYMENT.md`, `compose.prod.yml`, `deploy.sh` are prepared for the on-prem
  pattern (prod-apps 8086/8087/8088, Cloudflare tunnel + Access). Never suggest Vercel/Netlify/Pages.

- Run: `docker compose -f compose.dev.yml up -d` (Postgres 16 on :5432, toolfoam/toolfoam — needs Docker Desktop running);
  `cd backend && ../.venv/bin/python app.py --host 0.0.0.0 --preload` (8000); `npm run dev:admin` (3000); `npm run dev:web` (3001).
  Env in `apps/*/.env.local` (see `.env.example`); first time `npm run migrate && npm run seed` with that DATABASE_URL.
  Sessions are in-memory: after a backend restart `cd backend && ../.venv/bin/python tests/demo_session.py --open` makes a demo drawer.
  macOS 27 (2026-09-20): no Rosetta, so the Intel Homebrew under /usr/local (its Postgres 14, psql, pg_isready) is dead and
  /opt/homebrew's brew 4.3 refuses the OS — do not rely on either; the scratchpad is wiped on reboot, keep scripts in the repo.
- Test: `cd backend && ../.venv/bin/python tests/smoke_test.py` — must end with ALL CHECKS PASSED.
  `npx tsc --noEmit -p apps/web` / `-p apps/admin` for the sites. Headless-Chrome walkthroughs live in the scratchpad.
- Accuracy harness: `cd backend && ../.venv/bin/python tests/capture_eval.py [scene…|mesh.obj]`
- iOS: `cd ios && xcodegen generate` (project.yml is the source of truth; re-run after adding Swift files) —
  but xcodegen is an Intel binary and macOS 27 here has no Rosetta ("bad CPU type"), so on this Mac a new
  Swift file must be added to `ToolCutter.xcodeproj/project.pbxproj` by hand instead: a PBXBuildFile line, a
  PBXFileReference line, an entry in the ToolCutter PBXGroup and one in PBXSourcesBuildPhase (copy the
  AppSettings.swift lines, new 24-hex ids). Keep project.yml right anyway for when xcodegen runs again. Build with
  `xcodebuild -project ios/ToolCutter.xcodeproj -scheme ToolCutter -sdk iphonesimulator -destination 'platform=iOS Simulator,name=iPhone 17 Pro' CODE_SIGNING_ALLOWED=NO build`;
  camera/ARKit only work on a real device.
- Gotchas:
  - `backend/sam-hq/` is vendored upstream code (SysCV/sam-hq) — do not modify or reformat it;
    it makes the repo large, so keep searches scoped when possible. It is imported as
    `segment_anything` (the HQ fork), added to `sys.path` in `toolcutter/segmenter.py`.
  - Model checkpoints (`*.pth`) are gitignored; the largest `sam_hq_vit_*.pth` in `backend/` is
    auto-selected, or set `HQSAM_CKPT`. Without one, click-to-segment is disabled.
  - The repo `.venv/bin/pip` shim points at a stale path; use `.venv/bin/python -m pip`.
  - Tools carry `polygon_mm` (mm, y-down, origin = mat top-left) so `/api/layout` and `/api/export`
    are session-independent and tools from several uploads mix. `polygon_px` is only for drawing on
    that upload's raster. DXF and STL flip y so they match the SVG when viewed from above.
  - Capture footprints (`capture.footprint_from_mask`): LiDAR is primary, colour refines. Depth is cleaned
    first (`remove_flying_pixels`: mixed top/floor edge pixels snap to the near surface, wall pixels are dropped),
    scattered orthographically onto the floor plane, gap-filled by nearest sample only within 0.75 depth cell
    (farther = floor), then the SAM silhouette ∩ its (D−h)/D-scaled copy is applied inside a ±max(2 mm, 0.75 cell)
    band. Lessons: averaging gap fills and OpenCV's chamfer distance transform both shifted footprints by
    several mm; a photo shows tool TOPS displaced from the nadir by h·r/D; height-blob masks are already
    orthographic and must never be perspective-corrected. `tests/capture_eval.py` is the accuracy harness.
  - Multi-frame captures (`POST /api/captures/multi`): all frames rectified onto ONE grid (`ppm_override`),
    validity masks per frame (warps replicate edges outside the view: never fuse without them), nanmedian heights,
    nearest-foot-point colour mosaic, SAM + footprint correction per tool in the frame nearest to it
    (`s.frames`, `_nearest_frame`). Frame registration: >=3 markers, or 2 adjacent markers + known drawer size
    (`rectangle_from_two_markers`), or an ARKit pose relative to an anchor frame (`_pose_cv`). Drawer corners are
    picked per marker ID in the oriented frame (`outer_corner`) — the "farthest from the marker centroid" rule is
    ambiguous with two markers and cost a day.
  - Strategy findings (tests/strategy_eval.py, 900x225 drawer): LiDAR arc >> single still > photogrammetry sweep.
    Hand-held glides (`human`) work from 5 frames up. Frames without markers register by ARKit pose; marker corners
    are triangulated across frames (`create_multi_capture`); `_refine_frame_alignment` phase-correlates height maps to
    remove pose drift. Marker inset MUST expand along the rectangle's axes (`capture.expand_rectangle`), never along
    the diagonal (that compressed long drawers' short axis by 3.6 %).
  - iOS app is arc-only by request (2026-09-17); /api/captures (single still) and /api/sweeps (photogrammetry) remain
    as backend endpoints.
  - **Markers 25 mm, shapes editable everywhere (2026-10-02, Nolan: "markers like 1/4 the size"; "adding shapes is weird.
    I cant delete them and cant figure out how to edit their nodes"; "make the options for adding shapes and editing
    nodes/vectors available in the Layout and export view").**
    * Marker default 50 -> **25 mm** (a quarter of the area; `AppSettings.markerSizeMm`, the sheet is `?marker_mm=`
      and already parametric). The sheet URL now formats the size with `%g` — `Int()` would have printed 12.5 as 12
      and silently mismatched the app's setting. Preflight JPEG 800 -> 1200 px so a 25 mm marker at 60 cm is ~43 px
      (ArUco wants >= 20). Cost of smaller markers: one-marker frames take their axes from a marker edge, so +-0.5 deg
      at 50 mm becomes +-1 deg at 25 — front frames are placed by RGB-D overlap anyway, and the overview still is
      what sets the size. 12.5 mm would work for detection but doubles that again; it is one Settings field away.
    * ROOT CAUSE of the shape complaints in Outlines: shapes were listed but never SELECTABLE — `selected` came from
      `mine` (session-filtered) and shapes have `session_id ''`, so a click on one did nothing, Delete did nothing,
      and ScanViewer excluded `source === 'shape'` from editing. Now `allTools = mine + shapes` (shapes get a derived
      `polygon_px` = placed ring / mpp), `selected` searches it, the shapes list rows select on click, a shape has its
      own compact panel (name, nodes, area, size, Remove), and the FIRST HANDLE DRAG BAKES the primitive into a free
      polygon in mat coordinates (`commit`: absolute mm, offset 0, `shape: undefined`) — until then its size stays
      editable in Layout. ScanViewer grabs/drags on `placed(t)` (a shape's polygon_mm is local to its offset).
      Second bug found by the headless test: click-select raycast the MESH only, so clicking a shape on the bare margin
      returned no hit and DESELECTED — the same miss that once made drawing on the margin do nothing. Click and grab
      now fall back to the mat plane (`localAt(e, null) ?? localAt(e, 0)`). `__scan3d.vertexPos(id, i)` added for tests.
    * Layout & export: node handles on the single selected tool, computed client-side with the /api/layout convention
      (rotate about the AREA centroid, then offset_mm — the sheet draws the server's clearance rings, so handles come
      from the tool's own outline), drag rotated back into the tool's frame by -rotation_deg and run through the same
      `softDragRing` as Outlines (`Pull` field, 25 mm). A session tool's `polygon_px` is kept in step by the bbox
      scale ratio; a shape baked this way drops its `shape` spec. "+ Shape" (the sized dialog) added beside the draw
      bar, which Layout already had. Verified headless: Outlines shape click -> panel, node drag 77.8 -> 76.0 cm2 and
      "free outline", Delete removes; Layout: 20 handles on a selected tool, node drag changes the path.
  - **THE WHITE BORDER IS PART OF THE MARKER (2026-10-02, Nolan cut the new 25 mm markers to the black and "the app
    seems to have problems seeing" them).** ArUco finds a marker by the edge between its black border and a LIGHT
    surround; on a dark liner, cut to the black there is no edge. Measured on a real liner patch from b1e99fbbb404
    (textured, std 64) at real pixel scales: with the 25 %% quiet zone or even a 1 mm sliver of white, detected at
    every scale; with NO white, detected at none of the three real scales (36-px preflight passed only because blur
    merged it). A first synthetic test with a FLAT grey liner said the opposite — do not trust a uniform-background
    test for detection questions; paste onto a real frame. The sheet now labels every square "✂ cut here · keep this
    white border" and the header says why. The 1.5x `_marker_mask` zone has always assumed the border is there.
    Marker size lives in THREE places that must agree — `AppSettings.markerSizeMm` (25), `MARKER_SHEET_URL` in
    apps/admin/lib/api.ts (25; it was still 50 and is where Nolan printed from), and the server default in
    `marker_sheet()` (25) — one source of truth (admin/app asking the server) is the proper fix. Pixel budget for
    25 mm: rear overview at 60 cm ~125 px, front frame at 30 cm ~200 px, 1200-px preflight ~36 px (ArUco floor ~20).
    `/api/markers` logs the ids it saw per call so a phone that cannot see markers is diagnosable from the server log.
  - **SPEED (2026-10-02, Nolan: "sending the frames and building the drawer on the mac take a while").** Profiled on
    `b1e99fbbb404` (118 front frames + 1 rear photo): **262 s** build, 205 MB upload (67 MB JPEG at 4224x2376 +
    138 MB float32 depth). Where it went: RGB-D registration 126 s (of which `_match` 94 s — it matched EVERY PAIR,
    6,903 brute-force knnMatches), `rectify_rgbd` 69 s serial on a 10-core machine, fuse 17, _register 15.
    Done: (1) per-frame parse+rectify+depth_features on a ThreadPoolExecutor (`TC_FRAME_WORKERS`, 8): 262 -> 192 s,
    result bit-identical; (2) `register_rgbd` pairs only within `PAIR_WINDOW` (10) frames in time, plus any pair
    sharing a marker, exhaustive below `PAIR_ALL_UPTO` (30) frames so existing tests are unchanged: 6,903 -> 777
    pairs, **192 -> 142 s**, tools within 0.5 cm2 / ~1 mm (one 26x59 mm tool moved 4.7 mm in length — fewer
    graph constraints; raise the window if that matters); (3) phone: front frames **6 -> 3 /s** (`FRAME_INTERVAL_HZ`
    in FaceDepthController; ~80 %% overlap remains), colour **half size** (server rescales K; 0.19 mm/px raster is
    already exceeded by 2112 px over 400 mm; ArUco keeps ~120 px per marker) and depth as **uint16 millimetres**
    (`SweepRecorder.depthPayloadU16`, manifest `depth_dtype: "u2mm"`, parsed in `_parse_frame_upload`; round-trip vs
    float32: height p99 0.62 mm, tool areas within 0.3 cm2). Expected per drawer: ~60 frames, ~45 MB, build ~70 s.
    (4) `_prune_processed` keeps snapshots for the newest `TC_KEEP_PROCESSED` (3) captures — one snapshot is ~3 GB.
    NOT done and why: a gate skipping RGB-D registration when markers place 80 %% of depth frames NEVER FIRES on real
    front sweeps (0 of 118 frames see two markers at 30 cm — the earlier "118 placed by two markers" was a LABEL BUG,
    now split out as `placed_by.rgbd_overlap`) and broke `test_rear_photo_does_not_disable_depth_registration`; removed.
    Still worth doing for FELT speed: return a job id from /api/captures/multi and poll progress, so the phone shows
    "rectifying 40/60" instead of a spinner for a minute.
  - **BACK TO THE FRONT TRUEDEPTH CAMERA, ONE SCREEN (2026-10-02, Nolan: "move back to using the front TrueDepth
    camera. But keep the app flow simple").** The rear LiDAR's 256x144 depth could not see a 3 mm rule at any usable
    height (see below); the front camera read it at 24.7 vs a caliper 25.0 and the tape at 89.5 = caliper. Flow is
    still one button: REAR camera while aiming (marker preflight gates Start on 3+ markers) -> Start takes ONE
    full-res rear OVERVIEW still (`controller.stillFrame()`, sensor "rear_photo", use "color": it measures the drawer
    exactly, the thing every failed rear scan lacked) -> rear ARSession paused, `FaceDepthController` started (one
    ARSession at a time) -> 3-2-1 ticks + haptics while the phone is turned SCREEN-DOWN -> front sweep (frames
    `truedepth_tracked`, use "depth", 6/s, buzz every 5) -> Stop -> upload overview + depth together -> review.
    `Phase` gained `.overview`; `cancelToReady` / `stopAndUpload` always restart the rear session. No live map on the
    front pass (screen faces the drawer); the recording screen shows a giant frame counter for the record.
    `FaceDepthController.swift` is back in the build from `ios/attic/` (pbxproj by hand); `TrueDepthController`
    (plain AVFoundation) stays in the attic. Verified end to end under today's server (nearest-anchor, near-plane
    guard): `b59fc5fdd2d1` (83 front + 1 rear) -> 78 frames placed by two markers, tape **89.5 (+0.0 mm)**.
  - **FIRST REAR-LIDAR-ONLY SCANS (2026-10-01) — the scan-height tension, measured.** Two real scans of the same drawer:
    * `fed17f4780dc`, **42 cm** high, 19 frames: registration fine (13 frames with 2+ markers, 0 skipped) but the
      3 mm steel rule read **0.0 mm in all 19 frames** and the combination square's blade was missing. The rear LiDAR
      depth map is **256x144 px**; at 42 cm that is ~1.9 mm/px (fused cell 2.4 mm) — a 3 mm-tall tool is below what
      the sensor returns. The photo sees it plainly (brightness 140 vs floor 28). This is a HARDWARE limit; the front
      TrueDepth (640x480, used at 20-50 cm) read the same rule at 2.6 mm / 24.7 mm wide. Not fixable in software
      except by a photo-assisted fallback for low tools (not built; Nolan asked for topography-only edges).
    * `3130f123ed91`, **18 cm** high, 94 frames: the rule appears (2.8 mm, in 15 of 37 frames that see it) and floor
      noise drops 4x (p90 0.62 vs 2.28 mm) — but NO frame sees two markers, 70 of 94 are pose-only, and the tools come
      out SHREDDED: the hammer head's centroid landed **45 mm apart** between frames. Diagnosis by marker re-projection
      (solvePnP + ARKit pose -> world): markers 2 and 3 cluster within **1 mm** across frames, markers 0 and 1 scatter
      **11 and 32 mm** — ARKit DRIFTED during one stretch of the sweep over featureless dark liner at 18 cm.
    Fix that measured clean: **pose-only frames anchor to the NEAREST marker-placed frame in time** (`_register`;
    was one global anchor, so every pose frame inherited the whole sweep's drift). Head spread 45 -> 33 mm at the
    25 mm floor gate, the hammer is one body again; arc 0.906 -> 0.907, all glide seeds 8/8, tests OK.
    Tried and NOT adopted, each measured on 3130f123ed91: a 12 mm floor gate (`TC_POSE_FLOOR_TOL_MM`, spread 19.6 but
    35 of 94 frames dropped and floor noise 4x worse); raising `TC_APPLY_MAX_MM` to 40 so phase correlation could
    correct big drift (no change — its shifts were rejected for "overlap 0 px", not for size); dropping poses entirely
    (spread 10.7 mm but 68 frames skipped — ORB neighbour matching recovers only 9 on this liner); pinning the drawer
    size (no change). **Depth-scale calibration (`TC_DEPTH_CAL=1`) stays OFF**: the LiDAR measures +3.1 % (tight,
    n=17), and the cal did pull the pose-triangulated drawer height 425.6 -> 417.6 toward truth, but on the
    MARKER-placed scan it shrank 296.8 x 414.9 (matching Nolan's typed 297 x 417) to 293.3 x 410.0 and doubled floor
    noise — rescaling mm_per_px of a marker-placed frame corrupts the in-plane size the markers already fix. If it is
    ever enabled it must apply to heights and to pose/matching-placed frames ONLY.
    * `a7af39bd92eb`, **30 cm** (the recommended band), 63 frames: ARKit drift down to <= 8 mm, nothing skipped —
      and STILL shredded (floor p90 7.3 mm, head placement 22 mm apart). Two causes, both diagnosed by measurement:
      (1) the sweep started at 31 cm seeing ONE marker and the only two-marker frames saw DIAGONAL pairs (1,2 and
      0,3 — confirmed diagonals from ARKit world distances: 0-1 256, 2-3 252, 0-2 369, 1-3 352, 0-3 446, 1-2 436 mm,
      and the server's corner labelling was CORRECT), so the drawer size fell back to pose triangulation: **280 mm
      for a ~297 mm drawer**, which stretches all 44 one-marker frames 6 %%. On a bare patch (photo brightness 28,
      fused p50 0.00) individual frames put 17-32 mm of tool. (2) Frame 45's floor-plane fit locked onto the ROOM
      floor (D = 960 mm with the drawer at 306; neighbour frame 44 chose right) — RANSAC is bistable when the LiDAR
      view runs past the drawer edge, which 15 of 63 frames did. `rectify_rgbd` now refits on near points when the
      fitted D exceeds 1.3x the median point depth (`capture.py`, "plane fit grabbed a far plane" in the log). It
      was the right fix but NOT this scan's main damage: dropping frame 45 alone changed nothing measurable.
      THE LEVER IS THE OVERVIEW FRAME. `POST /api/markers` (one JPEG -> {ids, count, adjacent_pair, ok}) is a
      preflight; the app polls it every 1.2 s while waiting to start (`CaptureController.previewJPEG`, 800 px) and
      the Start button is DISABLED until the server sees 3+ markers — the view the drawer is measured exactly from.
      "Start high" as text was ignored three scans running; a gate is not.
      A probe that misled for a while: decomposing a diagonal pair along "the marker's first edge" gave 288 x 278
      and 14 x 417 — the first edge of a relabelled marker is NOT the drawer axis. Use ARKit world distances or
      an overview frame to establish layout; do not infer axes from one marker's corner order.
    The remaining lever is the APP: scan at **~30 cm** (readout `cameraHeightMm` from the LiDAR centre pixel, green
    in 25-38 cm, "↑ higher / ↓ lower" outside it) where two markers stay in view AND the LiDAR resolves ~1.3 mm/px.
    Preview: `feedLiveMap` now runs from `CaptureController.onFrame` at 10 Hz (was the 0.5 s recorder timer only),
    cells 5 -> 2.5 mm (600x600 over 1.5 m), `.interpolation(.medium)` — Nolan: "the preview on the phone app is
    quite bad". A `processed*` snapshot in `captures/<id>/` is built with the code of its time: after a registration
    change, `rm -rf captures/<id>/processed*` (raw frames stay) and restart, or the UI keeps showing the old build.
  - **Live coverage map on the phone (2026-09-29, Nolan: "can we make the live preview on the phone?")**: during
    the REAR LiDAR depth pass a top-down height map builds in the top-right corner as you sweep (`LiveMap.swift`
    maths, `LiveMapView.swift` picture, fed from `SweepRecorder`'s timer with every recorded frame). Floor is dark
    slate, raised things run blue -> red with height, unseen cells are transparent — so the SHAPE of the picture is
    the coverage and a hole means "not scanned yet" (the first depth pass ever taken missed half the drawer and nobody
    knew until the Mac). It is a 5 mm preview, not the model: nothing in it feeds the outlines. Rear pass ONLY by
    physics — on the front TrueDepth pass the screen faces the drawer, and its ARKit poses are unusable anyway.
    Geometry: depth pixel (u, v) -> camera ((u-cx)/fx*d, -(v-cy)/fy*d, -d) (ARKit camera looks along -z, image v
    grows DOWN) -> world by `camera.transform`; world y is up (`.gravity`); the map is on (x, z); the floor is the
    15th percentile of world y (a drawer is mostly floor). Depth intrinsics = colour intrinsics scaled by the
    depth/colour size ratio. `LiveMap.swift` is ARKit/UIKit-free so `ios/tests/run_live_map_test.sh` checks it on
    the Mac with a ray-marched synthetic block in four camera poses (straight down, tilted, yawed 90, both):
    height 30.0/30.0 mm, footprint 100 x 40 within 5 mm, floor 0.000, and the MIRROR position reads floor — the
    handedness check that matters. New Swift files were added to project.pbxproj by hand (4 places each).
  - **ONE-SCREEN CAPTURE (2026-09-29, Nolan: "make it super simple ... remove fields for adding drawer dimensions ...
    just have rear lidar")**: `CaptureView` is a single screen with a `Phase` state machine — ready -> countdown(3-2-1)
    -> recording (live map top-right, Stop button shows the running count) -> uploading -> done (review opens by
    itself; "Scan another drawer" behind it). No form: no name (named by the clock, "Drawer · <date time>"), no
    width/depth (the server measures the drawer from the markers; `uploadLidarArc(drawerSize: nil)`), no sensor
    picker. Rear LiDAR only: its frames carry photo + depth + a usable pose, so ONE sweep is the whole capture; the
    separate photo pass only ever existed for the front camera. `FaceDepthController.swift` and
    `TrueDepthController.swift` are out of the build in `ios/attic/` (kept for reference; the backend still accepts
    "truedepth"/"truedepth_tracked" frames and replays old captures). `LensCalibration` moved to SweepRecorder.swift
    because `SweepFrame.lens` is part of the wire format. minFrames 5 -> 8. `HomeView` copy already described this
    flow. The Bullseye level note below still applies.
  - **Bullseye level (2026-09-20, Nolan: "the circle in the middle only moves 1 direction")**: `tiltDegrees` was a
    scalar (the ANGLE off straight down) and `LevelGauge` offset the dot along +y only, so it slid down whichever way
    the phone leaned. `CaptureController` now publishes `tiltOffset: CGSize` as well — a 2D vector in screen axes whose
    length is the angle — from `LevelMath.tiltVector(view:)`, which takes `ARCamera.viewMatrix(for: .portrait)` (the
    app is portrait-only; the view matrix carries the interface rotation, so view space is already +x screen-right,
    +y screen-up, -z the way the camera looks), transforms world gravity into it, and returns `(g.x, -g.y)` normalised
    × the angle. Smoothed with a 0.2 EMA on the vector (a handheld phone jitters); `tiltDegrees` is then its length.
    The dot is a TARGET, not a floating bubble: it marks straight down, so it sits on the LOW side and you tilt toward
    it (lift the right edge -> the camera aims right -> the dot goes LEFT). That is the opposite of a spirit level's
    bubble; negate `planar` to swap. The gauge caption says "tilt toward the dot" so it is not a guess.
    `LevelMath.swift` is deliberately free of ARKit/UIKit so `ios/tests/run_level_math_test.sh` can check every
    direction with `swiftc` on the Mac — no device, no simulator. That test caught a sign error in my own reasoning;
    do not change the signs without running it.
  - **Edge source (2026-09-18, Nolan: "determine borders from the topography of the scan, not HQ-SAM")**: default is
    `edge_source: "topo"` whenever a height map exists — `geometry.topo_footprint` keeps a pixel when its height is ≥ 50 %
    of the nearest *flat* pixel's height (gradient < 0.8 mm/mm, i.e. its own ridge/plateau, so a shaft is not judged
    against its grip); pixels > 1.5 cells from any flat pixel use 50 % of the blob top; features thinner than 2.5 cells
    use 35 % (they never reach true height on the grid). `geometry.split_at_saddles` separates merged neighbours by
    watershed from ≥ 300 mm² plateaus when the saddle is < 60 % of the lower top (never for blobs < 2.5 cells tall).
    `_segment_topo` gives click / box / exclude-point outlining without SAM. Runs in ~0.8 s per drawer. Accuracy on
    complex_tools arc: IoU 0.84, mean edge 3.2 mm (photo path: 0.90 / 2.1) — thin round shafts read ~12 % narrow,
    hammer inner corners round by ~2 mm. Saddle rule (p50 < 0.55·top and p90 < 0.75·top along the shared boundary)
    was measured: a 4 mm gap reads 0.46–0.48 / 0.57–0.63, a pliers pivot 0.71 / 0.82 (good capture) or 0.43 / 0.65
    (6-frame glide) — so sparse captures may split a pivot; merged neighbours are the worse failure and always split.
    Watershed must NOT have a background marker (it wins the low ramps of low tools and leaves them unassigned).
    `TC_SADDLE_DEBUG=1` prints every split decision; `TC_TOPO_FRAC` / `TC_TOPO_SLOPE` override the level rule. `edge_source: "photo"` (checkbox in Detect) is the old HQ-SAM + LiDAR-band
    path; single stills in the smoke test use it because one viewpoint's far-side gap fill biases topo edges outward.
  - Fidelity rules in `footprint_from_mask` (photo path): colour candidate is built per boundary point (near edges kept,
    far edges scaled) — never `mask & scaled(mask)` (erodes thin parts); LiDAR threshold is a flat 2 mm (a
    fraction of tool height drops shafts/blades); band width = max(2, 0.75 cell, 8 − h·r/D); colour decides only
    inside the band where it has an opinion, LiDAR stands elsewhere; the LiDAR base is scoped to the tool's own
    mask/blob so 4 mm neighbours don't merge; reconnection closing must stay ≤ 2 mm (a wide one fills T/L inner
    corners). `_sam_mask` crops + upsamples around each tool. `auto_detect` re-seeds blob remainders when SAM
    returns a part (merged neighbours).
  - **Mosaic quality (2026-09-20, Nolan: "a lot of these combined photos are[n't] connected well")**. Three
    separate causes, each measured on his real desk scans (`captures/c66ff5b09912`, `d85032972dc8`):
    1. *Phase correlation false-locking on a repeating subject.* The keyboard's key grid gave confident shifts
       of exactly one key pitch (14-19 mm, response 0.11-0.27) which were applied and tore the keyboard into
       offset strips. `_shift_gain` (residual before / after the move) does NOT separate them — the false locks
       scored 1.7-2.1, BETTER than the genuine 1.02-1.08 corrections. Only the size does: real pose drift over
       one glide is under 5.5 mm on every synthetic glide. So `APPLY_SHIFT_MAX_MM = 8.0` caps what may be
       applied, and `MIN_SHIFT_GAIN = 1.0` additionally drops shifts that fit worse than standing still
       (`TC_APPLY_MAX_MM`, `TC_MIN_GAIN`). Tuning history: a gain threshold of 1.08 cost seed 5 (0.848 -> 0.807)
       and a "move one-marker frames when grossly off" rule cost seed 6 and made its drawer 8 mm too wide —
       both dropped. Also tried and reverted: letting every processed frame seed the reference to close the
       "overlap 0 px" hole mid-glide (frames then chase drifted pose placements; two seeds lost 0.04 IoU).
    2. *Exposure and specular differences.* 13% brightness spread across one glide; hard cuts showed every join.
       `_blend_mosaic` matches a per-frame gain to the consensus, then cross-fades joins over `SEAM_FEATHER_MM`.
    3. *Parallax.* The mosaic is a floor projection, so a raised top appears at `nadir + (1 + h/D)(p - nadir)` —
       a different place in every frame (11 mm for 25 mm of height 200 mm off-nadir). A join crossing a tool
       steps, and once cross-faded it ghosts. Fix: each raised blob (`TALL_OBJECT_MM`, connected components) is
       taken whole from the one frame that saw it most nearly overhead, and a frame may not supply colour where
       it would paint a tool onto ground the consensus calls floor — computed from that frame's OWN height map
       (per-pixel, because one scan had a 25 mm keyboard beside a 110 mm upright can, and a single global height
       threw most of the mosaic away; and because a MISPLACED frame carries its tool with it, which is how the
       ghost of a keyboard on bare desk was traced to frames 16/17 whose own depth put the keyboard there).
       That halo must be masked OFF the consensus-tall area: barring a frame from a real tool's own footprint
       leaves nobody able to paint it and shreds it into wedges (seen and fixed). `TC_BLEND_DEBUG=1` stashes
       `pick`/`solid`/`halos` in `app._BLEND_DEBUG`.
    Result on the real scans: seam visibility (mean |Laplacian| on the joins / elsewhere) 1.47-1.60 -> 0.84-0.89,
    i.e. joins are now smoother than the picture. Synthetic accuracy unchanged or better (arc 0.844/3.2 as
    before; glide seed 5 0.848 -> 0.859). Geometry is untouched: outlines come from the fused height map and the
    per-frame images the tool outliner uses are not modified.
  - **Outline smoothness (2026-09-20, Nolan: outlines are "very rough")**. The traced edge wobbles by ~0.3-0.7 mm
    rms at roughly the depth-cell wavelength, which reads as a zigzag and would make the cutter chatter.
    `topo_polygon` now takes `sigma_mm` and `_topo_tool_result` passes `TOPO_SMOOTH_CELLS * _depth_cell_mm(s)`
    (`TC_TOPO_SMOOTH`, default **1.0 cell** — the smoothing has to scale with the SENSOR, not the raster: the old
    fixed 1.0 mm was ~0.35 cells). `straighten_ring` then snaps the straight runs back, so corners come from
    intersecting runs rather than being rounded off. Measured on `captures/d85032972dc8`: vertex count halved
    (386 -> 196 on the keyboard) and perimeter/convex-hull 1.13/1.33/1.22 -> 1.10/1.22/1.14. Accuracy: single
    still clearly better (mean IoU 0.740 -> 0.764, Hausdorff 8.50 -> 6.89 mm); multi-frame unchanged within
    0.005 IoU (arc 0.844/3.2 -> 0.843/3.5), so the cost is ~0.3 mm of worst-case edge for a much cleaner path.
    Sweeps: single still keeps improving to 1.2 cells, the arc's Hausdorff starts slipping past 0.8, hence 1.0.
    Two things were tried and REJECTED, both kept behind flags so they are not re-attempted blind:
      * *Sub-pixel iso-contour* (`geometry.subpixel_ring`, `topo_footprint(field_out=...)` returns the
        height-minus-keep-level field, `TC_SUBPIXEL=1`): sliding each vertex onto the zero crossing sounds
        obviously right, but it undoes the morphological cleanup the mask went through (speckle, pinholes) and
        pulls the outline back onto the raw noisy threshold — arc IoU 0.844 -> 0.829. It also barely moved the
        roughness, which is the measurement that proved the jaggedness was NOT whole-pixel quantisation.
      * Blaming the LiDAR's coarse cell for a blocky height field: measured, the field's constant-value runs are
        1 px, and synthetic edge noise (0.35 mm) matches real (0.32-0.66 mm) — so the harness is representative
        and there is no real-vs-synthetic noise gap to chase.
    Beware: turn-angle-per-vertex is a MISLEADING roughness metric (a well-simplified outline has few vertices
    and large honest turns at its corners); use perimeter / convex-hull perimeter and rms deviation from a
    3 mm-smoothed ring, as the sweeps above did.
  - **Two-pass capture (2026-09-21, Nolan's idea)**: sweep the drawer with the FRONT TrueDepth camera for the height
    map, then shoot sharp stills with the REAR camera, each triggered by hand so none is taken mid-movement. Each
    manifest frame may carry `"use": "depth" | "color" | "both"` (default `both`, so existing captures are
    bit-identical — arc stays 0.842/3.5). A `depth` frame registers off its markers and feeds the height fusion but
    is kept out of the colour mosaic (`paint` in `_build_multi_session`); a `color` frame has no depth and so was
    already excluded from the height fusion. Verified end to end on a synthetic mixed manifest (12 depth + 4 colour):
    drawer measured 760.0 x 480.0 against a truth of 760 x 480.
    Why it should help, from measurements on `captures/e2a2e7ca6ff8`: several frames there are visibly motion-blurred
    and the mosaic is smeared around the steel rule, and TrueDepth resolves far finer than the rear LiDAR at 20-50 cm,
    which is exactly the range that matters for a drawer. THE KEY POINT is that both passes register off the ArUco
    markers, so they land on the same grid without any pose chain between them — do not try to relate the two
    sessions through ARKit poses, they are separate sessions with unrelated origins. On a drawer small enough for one
    still to contain 3+ markers, every photo registers exactly and drift stops being a factor at all.
    iOS side (`CaptureView`): a `Pass` enum runs the capture as FOUR numbered steps, each its own view
    (`setupStep` / `depthStep` / `photoStep` / `sendStep`) with Back at every stage —
    **1 Drawer** (name, optional width/depth, sensor picker) -> **2 Depth** (front TrueDepth, continuous) ->
    **3 Photos** (rear camera, ONE still per tap via `CaptureController.stillFrame()`, so nothing is shot
    mid-move, with Undo) -> **4 Send** (a summary of what is about to go, then upload `depthFrames + photoFrames`).
    The level gauge and the front camera only run during the steps that need them, so the setup form is not
    covered by a bubble or backed by a selfie preview.
    Starting the depth scan runs a 3-2-1 countdown first (`startCountdown`), with a tick sound and a rigid haptic
    each second and a distinct chime plus a success haptic on "go" — on the depth pass the SCREEN FACES THE DRAWER,
    so the big number is only useful for the rear-camera case and the tick has to carry it. Cancelling bumps
    `countdownRun`, which strands the in-flight task: without that token, cancel-then-restart leaves the old counter
    alive and it starts recording early.
    TEXT COLOUR TRAP: the width/depth fields used to live in a `DisclosureGroup` carrying
    `.foregroundStyle(.white)`, which propagated into the `.roundedBorder` fields — white text on a white field,
    invisible while typing (Nolan hit this). They now take `.primary`, which is readable in light AND dark mode.
    Do not "fix" such a field by hard-coding `.black`: that just moves the bug to dark mode.
    `SweepFrame.use` carries the role into the manifest. Size with NO drawer dimensions typed: 759.9 x 479.9 mm
    against a truth of 760 x 480, `drawer_from: triangulated` — i.e. the markers in the stills settle the size, which
    is what the photo pass is for.
    **A TrueDepth sweep has NO POSE AND SEES NO MARKERS** (2026-09-21, Nolan: "the truedepth frames missed the whole
    bottom half of the drawer"). AVFoundation gives no world transform, and at 20-50 cm the camera's view is far too
    narrow to keep a corner marker in shot. Measured on `captures/b35103e40780`: of 44 depth frames, **35 saw no
    marker at all** and 9 saw exactly one; none saw two. A frame with no markers and no pose cannot be placed, so
    only 15 of 50 frames survived — and the missing half was simply where no corner was visible.
    Fix: `_chain_unanchored` places such a frame by matching it to a neighbour that IS placed. Each frame is already
    rectified to a metric top-down raster, so two frames differ only by a rigid move in the plane: ORB features,
    ratio test, `estimateAffinePartial2D` with RANSAC, reject unless >= 15 inliers and the recovered scale is within
    6 % of 1 (both rasters are metric, so scale is a free check, not a free parameter), then carry the drawer
    rectangle across. Anchors spread outward from the frames the markers did place, iterating until nothing more can
    be added. Result on that capture: 15 -> **39 of 50 frames**, and colour coverage 100 % in every band down the
    drawer (depth coverage 68-100 %; the low bands are the black tape measure, which TrueDepth struggles to see).
    It only runs when a frame has neither markers nor a pose, so marker or pose captures are untouched — the
    synthetic harness is unchanged (arc 0.842/3.5). It costs real time: that rebuild took ~95 s for 50 frames.
    **Tracked TrueDepth (`FaceDepthController.swift`, 2026-09-21)** is the proper fix and is now BUILT but UNTESTED
    ON A DEVICE. `ARFaceTrackingConfiguration` + `isWorldTrackingEnabled` runs the front camera through ARKit, so
    `frame.capturedDepthData` gives the same TrueDepth map while `frame.camera.transform` gives a tracked,
    gravity-aligned world pose — every depth frame can then be placed by pose, the way the rear LiDAR arc already
    is, with no marker needed and no feature matching. Compiling against the iOS SDK confirms all four APIs exist
    (`supportsWorldTracking`, `isWorldTrackingEnabled`, `capturedDepthData`, `capturedDepthDataTimestamp`).
    Notes: depth arrives ~15 Hz against 60 Hz video, so `capturedDepthData` is nil on most ARFrames — take a frame
    only when `capturedDepthDataTimestamp` CHANGES, never reuse the previous map. `ARFrame` is not Sendable and is
    valid only inside the delegate call, so everything is copied out before hopping to the main actor.
    Sensor picker gains "Front · tracked" (`captureSensor == "truedepth_tracked"`, sensor string
    `truedepth_tracked`) and warns in-line when a device cannot world-track the front camera.
    FIRST DEVICE RUN CAPTURED NOTHING. Causes addressed, and the reason each was a candidate:
      * `maximumNumberOfTrackedFaces = 0` — asking ARKit to track no faces may leave the TrueDepth stream off
        entirely. Back to 1 (the default).
      * rejecting on `depthDataAccuracy != .absolute` — ARKit may report `relative` for the same map AVFoundation
        calls absolute, and that guard silently discarded every frame. It now notes the accuracy and uses the depth.
      * a refused configuration produced no frames and no message — `session(_:didFailWithError:)` and
        `sessionWasInterrupted` now report it.
    The controller publishes `diag` ("frames N · depth M · absolute · pose ON", plus the commonest rejection
    reason), shown live under the depth step, so a failure says WHERE it stopped instead of just "no frames".
    SETTLED ON DEVICE: **depth and pose DO arrive together** — `isWorldTrackingEnabled` does not cost the depth
    stream, so the approach is sound. (The auto-fallback that restarts without world tracking after 90 depth-less
    frames stays in as a safety net for other devices.)
    Second device run then stalled on the QUALITY GATE, "mostly wrong distance or too little depth" — one message
    covering two unrelated causes, which is why it could not be acted on. Now split, with the live numbers on
    screen: `diag` reads "frames N · depth M · cover P% · D.DD m · pose ON". Thresholds were inherited from the
    AVFoundation stream and are wrong for ARKit's face-tracking depth, which is a different and sparser map:
    coverage 0.30 -> 0.10, distance window 0.15-0.65 m -> 0.12-0.90 m.
    That still said "too far" with the drawer plainly in view, because the gate tested the MEDIAN depth: hold the
    phone 30 cm over a drawer and everything past the drawer's edge is room, metres away, which drags the median
    over any sane limit. `FaceDepthController.depthStats` now reports coverage, the distance to the NEAREST surface
    (10th percentile, robust to stray close samples) and the share of the map actually AT working distance, and the
    gate asks for `cover >= 8%` and `in-range >= 6%`. Checked against simulated depth maps: a drawer at 30 cm
    filling only a quarter of the view is accepted (the old rule called it "too far"), while a phone aimed at a wall,
    lying flat on the desk, or with a covered lens is still rejected with the right reason. `diag` now reads
    "frames N · depth M · cover P% · near D.DD m · in-range Q% · pose ON".
    NEVER gate a top-down capture on median or mean depth — the subject is the NEAR surface, the rest is background.
    The CoreMotion gate is GONE from this path entirely (it then said "mostly: moving too fast"). It was a second,
    stricter opinion on a question ARKit already answers: ARKit drops out of `.normal` tracking by itself when
    motion, light or featurelessness is genuinely too much, and it is far better calibrated than a hand-picked gyro
    threshold. The controller now reports WHICH limitation ARKit reports — excessive motion, insufficient features,
    initializing, relocalizing — so the next stall names its own cause. Rate limit 4/s -> 6/s, no dwell. It existed because the old path needed sharp photos that also had to register
    by appearance; here the pose registers and these frames never paint the mosaic (`use: "depth"`), so only gross
    motion matters and the sweep can be continuous — which is what Nolan wanted from "continuously scan".
    **THE REAL CAUSE OF "0 depth frames" WAS A DISPATCH BUG, NOT THE SENSOR (2026-09-21).** `startCountdown` did
    `if useTrueDepth { trueDepth.beginRecording() } else { recorder.start(session: controller.session, …) }` —
    a TWO-way branch over THREE sensors, so the tracked front path was handed to the photogrammetry recorder and
    `FaceDepthController.beginRecording()` was never called. `recording` stayed false, `ingest` appended nothing,
    and stop reported 0 frames. Four device rounds were spent tuning quality gates while nothing was recording:
    the status line publishes from the PREVIEW path whether or not recording is on, so "too far" / "moving too
    fast" / "still starting up" were all live and all irrelevant. LESSON: when a capture yields ZERO output,
    verify the recorder was started before touching a single threshold — no gate can explain 0 when the same gate
    is visibly passing in the preview. The start and stop dispatches are now exhaustive `switch depthSource`
    statements with NO default, so a fourth sensor is a compile error rather than an empty capture, and the Stop
    button always shows the running count ("Stop · 0 frames (need 12+)") because "Stop (need 12+)" hid it.
    **A POSE IS OPTIONAL, NEVER A PRECONDITION (2026-09-21, four device rounds ending in "It seems to be stuck
    saying ARKit is still starting up").** Front-camera world tracking sat in `.limited(.initializing)` FOREVER,
    and panning the room first did not rescue it. That is the expected outcome, not a bug to tune: with
    `isWorldTrackingEnabled` the world frame is built from the FRONT camera, which is 20-50 cm from a dark, flat
    drawer liner — no parallax, no features, nothing to initialise against. Each of my three attempts to gate on
    it made the app *less* usable (first "moving too fast", then a hard stop, then a "waiting for tracking" wall),
    because the premise was wrong: the capture NEVER needed a pose on every frame. The server already places
    pose-less frames by matching them to a placed neighbour (`_chain_unanchored`) or by depth registration.
    So `ingest` no longer returns early on tracking state — it attaches `transform` only when tracking is `.normal`
    (or `.limited(.insufficientFeatures)` AFTER a `.normal` was once seen, which is normal once you are down over
    the drawer and the IMU is carrying an established frame) and sends the frame either way. `SweepFrame.poseQuality`
    ("normal" / "limited" / "none") rides along in the manifest as `pose_quality`. The Start button is never
    disabled by tracking; a quiet line says frames will be lined up by overlap instead, so OVERLAP EACH VIEW BY HALF.
    `diag` reports `pose OK (N kept)` / `no pose yet`, which is how to tell whether this path earns its keep at all.
    `finishRecording()` relabels a sweep where NOT ONE frame got a pose from `truedepth_tracked` to `truedepth`,
    because the backend's depth-only joint registration (`register_rgbd`) only runs when every frame is plain and
    pose-less — an honest label gets the better path instead of pretending a tracked scan half-worked.
    Backend: `truedepth` and `truedepth_tracked` ARE THE SAME SENSOR (same sparse map, same holes), so
    `_build_multi_session` keys every depth decision on `is_truedepth(f)` (`startswith("truedepth")`) —
    `preserve_unknown`, `depth_features` and the register_rgbd gate. Keying on the exact string silently gave
    tracked frames LiDAR treatment. Regression test: `tests/test_depth_and_stitching.py`
    `test_tracked_truedepth_sweep_with_only_some_poses` (8 frames, only the last 4 posed — all 8 placed, drawer
    within 3 mm, all tools found), because a mixed session is now the NORMAL case, not an edge case.
    HARNESS ARTEFACT, do not chase: every synthetic multi-frame run logs "N markers do not form a rectangle in
    world coordinates" with all markers sharing one flattened axis. `_corner_map_for_frames` flattens world marker
    centres to `(x, z)` because ARKit is Y-UP, while `synth_scene`'s world is Z-UP — so the pose pre-pass always
    fails there and falls back to the frame seeing the most markers. Real ARKit data is fine (it is what fixed
    `efb7ad666414`). Consequence worth knowing: the synthetic tests DO NOT exercise pose-based corner ordering.
    Do not "fix" the flatten to match synth — that would break real captures; make synth emit ARKit-convention
    poses, or pick the two axes the markers actually span while keeping `u × v = -up` so the cycle is not mirrored.
    **`pkill -f "python app.py"` KILLS NOTHING (2026-10-02).** The venv's python execs the framework binary and the
    process shows as `Python app.py` (capital P); `pkill -f` is case-sensitive, so three "restarts" in a row started
    servers that failed to bind while the OLD one kept serving — exactly the stale-server failure below, self-inflicted.
    Use `pkill -fi "app.py --host"` or `lsof -nP -iTCP:8000 -sTCP:LISTEN` for the PID, and check the PID CHANGED.
    **A RUNNING BACKEND HIDES EVERY BACKEND FIX (2026-09-21, Nolan: "i still cant see the scan in the outlines
    view" — after the mirror fix was verified).** Sessions live in memory, so a server started before an edit
    keeps serving the session it built with the OLD code, forever: `/api/sessions/<id>` still read
    `frames_used: 6, has_height: false` and `/heightfield` still 404'd, which looks exactly like "the fix did
    not work". Restarting is not optional after touching `app.py` or `toolcutter/`, and the in-memory session
    must be gone (a restart does that; `DELETE /api/sessions/<id>` would also delete the saved frames, so do NOT
    use it for this). Check the fix landed with
    `curl -s localhost:8000/api/sessions/<id> | python3 -c "import sys,json;print(json.load(sys.stdin)['scan'])"`
    and look at `frames_used` / `has_height`, not at the UI.
    Rebuilding a 108-frame capture from disk costs ~220 s (~2 s/frame), and it happens on the FIRST request that
    touches the capture, so after a restart the admin sits on an empty scan for four minutes. `main()` now warms
    saved captures (newest first) on a daemon thread at startup — same work, off the request path — with
    `TC_NO_WARM=1` to skip it for tests and harnesses, which point `TC_CAPTURE_DIR` at their own output.
    **SETTLED (2026-09-21, capture `136c77400c3f`, 102 front depth frames + 6 rear photos): THE ARKIT FRONT
    CAMERA DELIVERS A MIRRORED IMAGE, AND ITS POSE IS UNUSABLE.** This was the "mirroring is the plausible
    failure" risk flagged below, and it is real. Symptom: the scan built from the 6 photos only and the 3D view
    was empty; all 102 depth frames were skipped as "no camera pose on any marker-registered frame to anchor to".
    Diagnosis, in order, each step measured rather than assumed:
      1. `frames_used = 6` in `capture.json` said the depth pass was being thrown away, not mis-fused.
      2. The skip reason named the anchor rule: placing a pose-only frame needs one frame with markers AND depth
         geometry AND a pose. The two-pass capture HAS NO SUCH FRAME BY CONSTRUCTION — the front sweep is too
         close (20-50 cm) to see a marker, and the rear photos that see markers carry no depth, so no geometry.
      3. Widening the appearance-matching fallback (`_chain_unanchored`) to run whenever poses cannot be
         anchored placed ZERO frames: 500-850 ORB features per frame but only 7-18 matches and 4 inliers.
      4. Dumping a rectified front frame and LOOKING at it showed an ArUco marker plainly in shot that the
         detector had not found, and mirrored ruler digits. `detect_markers` on the raw jpeg: 0 markers as
         delivered, 1 marker after `cv2.flip(img, 1)`, on every front frame tested; rear photos the reverse.
    Fix: `_parse_frame_upload` un-mirrors `truedepth_tracked` frames — image, depth, and `K[0,2] -> W-1-cx` —
    AFTER the lens model (whose centre is measured in the mirrored frame, so it must be applied first). Plain
    `truedepth` (AVFoundation) frames are NOT mirrored (`b35103e40780` decodes markers as delivered), so this is
    keyed to the ARKit path exactly. Result on `136c77400c3f`: 6 -> 85 of 108 frames.
    The POSE does not survive the reflection either. Honouring it placed 4 frames and got 23 REJECTED as "pose
    inconsistent with the floor plane" (30-132 mm). `NO_POSE_SENSORS = {"truedepth_tracked"}` makes `_pose_cv`
    return None for them, and markers + neighbour matching then place **108 of 108, nothing skipped**, with an
    equally clean height map (fused 300.4 x 408.5 mm from markers, 13 tools, ~220 s to rebuild).
    `placed_by` for that capture: markers_2plus 3, one_marker 78, pose_only 0, matched_to_neighbour 27 — i.e. on
    a drawer this size a close sweep usually has ONE marker in view, which is enough. The transforms are still
    sent and stored, so the convention can be recovered later from saved captures; nothing depends on them now.
    WHY THIS COST A WHOLE DEBUGGING ROUND: I reasoned about registration maths for three steps before looking at
    the picture. A mirrored raster is invisible to every numeric check (feature counts, coverage, depth stats all
    look healthy) and obvious in one glance. WHEN MARKERS ARE NOT FOUND IN A FRAME THAT PLAINLY CONTAINS ONE,
    DUMP THE IMAGE AND LOOK AT IT, and test the flip — it is two lines.
    Synthetic tests must therefore feed MIRRORED input for this sensor or they are testing nothing:
    `test_mirrored_front_camera_sweep_is_unmirrored_and_placed` mirrors its rendered frames and asserts
    `_pose_cv` returns None. An unmirrored synthetic frame labelled `truedepth_tracked` is now WRONG input and
    the pipeline rightly fails on it.
    `scan_meta.placed_by` reports `{markers_2plus, one_marker, pose_only, matched_to_neighbour}` per capture,
    which is what made each step above measurable. On the untracked capture
    `b35103e40780` it reads markers_2plus 4, one_marker 11, matched_to_neighbour 24, pose_only 0.
    STILL TO DO — drawer size from the DRAWER'S OWN EDGES in the depth pass. It cannot be done from the captures that
    exist: the rectified grid is cropped to the marker rectangle, so the walls are outside it entirely (border strips
    of `captures/e2a2e7ca6ff8` read ~0 mm except where a tool happens to sit at the edge). It needs wall detection on
    the RAW per-frame depth before rectification, and it needs a real TrueDepth sweep that actually includes the walls
    to develop against. Do not build it blind — that is exactly how `snap_to_base` came to pass on synthetic cones and
    over-expand three real tools by up to +85 %.
  - Multi-frame registration (`/api/captures/multi`): markers place frames (2 adjacent = exact, 1 = position +
    marker-edge axes); poses only place marker-less frames and seed the drawer size. Never use pose-triangulated
    corners as a frame's rectangle (they were 5–8 mm off and broke everything anchored to them). Height-map phase
    correlation has a 1–2 mm viewpoint bias: apply its shifts only to marker-less frames; far-end shifts re-measure
    the size only when they agree and exceed 3 mm. Floor-texture correlation is a fallback for flat overlaps only —
    mixed in, it swamps the height signal.
  - `remove_flying_pixels`: mixed pixels go to the top only when ≥ 35 % covered (`MIXED_NEAR_FRAC`); intermediate
    pixels on the edge facing the nadir are wall samples and stay as measured (snapping them to the top slides them
    h·r/D outside). `_trim_rim_half_height` (35 % of the local top, sampled inward along the normal) fixes the
    outward bias of the flat 2 mm threshold. `_far_rim_excess`: colour may trim up to 2 depth cells on edges facing
    away from the nadir for parts > 12 mm tall. `_keep_connected`: colour can never split the LiDAR footprint.
    Do not try an occlusion/"shadow" test on the raster: under-the-top and in-shadow cells look identical.
    Also tried and rejected: preferring two-marker frames for SAM/mosaic (`RANK_PENALTY_MM`, fixed the pliers
    in one glide, lost the knife/wrench in others — left at 0), aligning the colour outline to the LiDAR base
    by phase correlation (moved correct colour onto biased LiDAR), pose-chain axes for one-marker frames
    (worse than the marker's own edges). Known outlier: a glide where thin tools are covered mostly by
    one-marker frames can still fatten 6 mm handles (+35 % area on the pliers in seed-4/24-frame run).
  - Mesh rasters: the floor normal sign comes from mesh face normals (`_load_points_full`), with a marker-based
    re-rasterize fallback; the height raster is re-levelled to the visible floor after calibration
    (`geometry.level_height_raster`) because photogrammetry floors are bowed/offset.
  - Photogrammetry scale drifts ~1-2 %: `_session_from_mesh_file(drawer_size_mm=…)` rescales from marker spacing.
  - Photogrammetry worker: `mac/Photogrammetry` (Swift, `swift build -c release`). PhotogrammetrySession only
    writes USDZ; the CLI converts to OBJ via Model I/O and `photogrammetry.fix_obj_texture` pulls the diffuse
    texture out of the USDZ. Synthetic multi-view test scenes must use a right-handed world (`synth_scene.render_from_pose`
    flips z) or the reconstruction comes out mirrored.
  - Uploads are classified layout vs. single object by `scan.classify_scan` (plane inlier fraction +
    whether off-plane points are stacked over plane points); form field `scan_kind` overrides.
  - Scan outlines use the "fraction of points above the mat" raster (`above_frac`), not the max
    height, to avoid growing thin tools by half a cell.
  - 3D views (added 2026-09-18 after Nolan said the workflow "feels like an overhead image"): `components/ScanViewer.tsx` renders
    the fused LiDAR height map (`GET /api/sessions/<id>/heightfield?step_mm=2`, base64 float32 grid) as a photo-draped surface
    with draped outlines, click-to-select (Detect + Refine "3D scan" toggle). `components/Layout3D.tsx` is the foam block with
    per-tool scanned bodies (`GET /api/sessions/<id>/tools/<tid>/heightfield`, 1 mm) sitting in pockets from the computed layout;
    drag = move (commits offset_mm on release), same keyboard nudges as the sheet; R = 90° (Shift −90°), [ ] = ±5° (Nolan asked for 90° on R) ("Sheet | 3D" toggle, 3D default).
    Frame convention in `lib/three-util.ts`: the image frame (x right, y DOWN, z up) is LEFT-handed, so `makeScene` uses
    root.rotation.x = −π/2 **plus root.scale.y = −1** (a reflection of the local image-y axis; scale.z would flip the heights) — a pure rotation shows the drawer mirrored vs the 2D
    view (Nolan noticed). heightfieldGeometry reverses PlaneGeometry winding after re-mapping rows to y-down. Test hook `window.__layout3d.{ids,screenPos}` for headless drags.
    Hand-edited tools (`edited`) fall back to an extruded polygon body (their mask no longer matches). Tool bodies drop the
    crop rectangle's flat (≤ 0.2 mm) triangles — otherwise the margin shows as a coloured skirt outside the pocket when a
    tool is rotated or sits near the mat edge (seen after Auto layout).
  - Split / auto layout (2026-09-18): `POST /api/sessions/<id>/split` {tool_id, line px} divides the tool's stored mask by
    the infinite line (1.5 mm seam), re-derives each side topographically, returns two tools (Detect: ✂ Split button or
    ⌘/Ctrl+drag across the join). `POST /api/autolayout` packs polygons onto the mat on a 2 mm occupancy grid: biggest first,
    long side horizontal or vertical (minAreaRect), bottom-left fill, `gap_mm` between pockets (default 8), 10 mm margin;
    returns rotation_deg + offset_mm in the /api/layout convention (rotate about the outline's own area centroid, then
    translate). `direction`: "columns" (default — Nolan: "make the auto-layout vertical" = tools stand upright, long side
    vertical, placed side by side across the drawer, wrapping to a band below) or "rows" (lying, stacked top→bottom,
    then the next column). The preferred orientation is tried first at every spot; the other only if it fits nowhere.
    Layout toolbar: "⊞ Auto layout" + gap field + "↕ upright, across / ↔ lying, stacked"; unplaced tools are listed.
  - Multi-frame sessions: `original` IS the fused mosaic, so `corners`/`suggested_corners` must be the mosaic rectangle
    (was frame-0 source-raster corners → Calibrate showed handles off-image and re-calibrating re-cropped to nonsense;
    Nolan hit this 2026-09-18). CalibrateStep shows a banner for marker-calibrated captures.
  - Drawn shapes (2026-09-18, Nolan: "add simple shapes to the board"): `Tool.source === 'shape'` with `Tool.shape = {kind:
    rect|slot|circle|hex, w_mm, h_mm, r_mm}`; polygon from `geom.shapePolygon`, name from `shapeName`, added via
    `components/AddShape.tsx` ("+ Shape" in the Layout tools panel). Size stays editable in the selected-tool panel
    (`updateShape` regenerates the polygon). Shapes have no session/mask: Layout3D extrudes them, STL export extrudes
    them, `page.tsx` keeps them (like 'object' tools) when a new capture session loads. Canvas drawing (Nolan: "like a
    traditional canvas app"): segmented tool bar on the sheet — ↖ select (V), ▭ rect (1, drag corner→corner, Shift = square,
    radius field), ⬭ slot (2), ○ circle (3, drag from centre, Alt = corner→corner), ⬡ hex (4, from centre), ⬠ polygon (5,
    click points; click first point / double-click / Enter closes, Backspace undoes, Esc cancels). `geom.shapeFromDrag`,
    `shapeFromPoints`; kind 'poly' stores `points` and has no w/h editor.
  - Foam workflow steps (apps/admin, `app/page.tsx`): Upload → Calibrate → **Outlines** → Layout & export.
    **Detect and Refine were merged into `components/OutlineStep.tsx` on 2026-09-20** (Nolan: "it feels like theres a lot of
    overlap between detect tools, refine outlines, and layout/export"; he picked merging the two over a shared-canvas
    variant). They had drawn the same picture, listed the same tools and carried the same height-map / 2D-3D toggles, differing
    only in the right-hand panel. The merged step keeps Refine's zoomable SVG canvas (the better one) and folds Detect's
    auto-detect panel, click-to-outline, box prompt and ✂ split into it. Still NO modes — the gesture follows from what is
    under the cursor and whether a tool is selected:
      * nothing selected: click the mat outlines a new tool there, Alt+drag box-prompts one, click a tool selects it;
      * a tool selected: its handles are live (drag = move, click the line = add a handle), Shift+DRAG = push brush while
        Shift+CLICK = include hint, Alt+DRAG = marquee while Alt+CLICK = exclude hint, ⌘/Ctrl+drag = split;
      * always: drag empty space pans, wheel zooms (Shift+wheel = brush size).
    Click-vs-drag is what separates the Shift/Alt pairs, the same distinction the rest of the canvas already used.
    A click on bare mat DESELECTS first and only outlines a new tool on the next click, so a stray click while editing
    cannot spawn a tool. Selecting from the tool list zooms to that tool; clicking it on the canvas does not (you are
    already looking at it). The brush slider is always rendered (disabled when nothing is selected) because letting it
    appear re-flowed the toolbar and resized the canvas under the cursor. Auto-detect collapses itself after a successful
    run. Undo history per tool, `Tool.auto_polygon_px` keeps the detected outline for reset, `Tool.edited` flags hand edits.
    `StepId` is now `upload | calibrate | outlines | layout`.
    **Simple shapes (2026-09-20, Nolan: "have it do simple shapes only to outline the objects it finds")**: `geom.fitShapes`
    builds every candidate primitive around the tool's own `minAreaRect` (so it follows the TOOL's angle, not the drawer's)
    — sharp rectangle, three rounded radii, capsule, and for a roughly square blob (long/short < 1.35) an area-matched
    circle and two hexagon orientations — scores each against the traced outline and returns them best-first. Near-ties
    (< 0.015) go to the more canonical shape, so a rounded bar reads "Capsule", not "Rounded rectangle". `bestShape` refuses
    below `minIou` (default 0.82) and the tool keeps its traced line — an L-shape scores 0.51 and is correctly left alone.
    UI: "Outline as simple shapes" + a "fit at least" slider under Auto-detect (applied to fresh detections), "◻ Shapes" by
    the tool list for all of them, "◻ Simple shape" / "Other shapes…" per tool. Always reversible: the traced ring stays in
    `auto_polygon_px`, so "Reset to detected" restores it exactly.
    **Two performance traps, both hit and fixed the same day:**
      * `fitShapes` was called in the RENDER path (for live "Rectangle 97%" buttons). Every pointermove of a vertex drag
        re-scored eight candidates against a several-hundred-vertex ring — Nolan: "editing points has gotten very slow".
        Candidates are now computed only on demand (`shapeFits` state, cleared when the selection changes). NEVER put a
        fit, an IoU or anything else superlinear in the render path of this component; the drag commits on every move.
      * IoU by sampling a grid and testing each point against every edge is O(n² · V): 900 ms for a 340-vertex outline.
        `rasterMask` scanline-fills each polygon into an n×n Uint8Array instead (O(n · V + area)) and `fitShapes`
        rasterises the traced outline ONCE on a grid shared with all candidates → **7.3 ms, 123× faster, and exact rather
        than sampled** (scores went 96-98% → 100% on clean test shapes). Measured worst case after the fix: a 1664-handle
        outline drags at 62 fps.
    **Proportional ("soft") dragging (2026-09-20, Nolan: a vertex moves "but it doesnt actually reshape the shape …
    I may want to pull out one edge of a circle, or move the whole left side of a ruler")**: dragging a handle now
    carries its neighbours with a raised-cosine falloff, measured ALONG the outline (arc length), not across it —
    Euclidean falloff would drag the right edge of a narrow ruler when you pull the left. `Pull` slider in the toolbar
    (default 25 mm, 0 = the old single-point move), and the WHEEL resizes it mid-drag, Blender style, since zooming
    then is useless anyway; a green dashed ring on the hovered/dragged handle shows the reach. The drag keeps the ring
    as it was at grab time (`Drag.poly0` + `arc`) and rebuilds from the total delta, so changing the radius mid-drag
    does not make the shape creep, and the radius is capped at half the perimeter. An explicit Alt-marquee selection
    still drags RIGIDLY (`Drag.rigid`) — that is the "move exactly these" case. Measured on a 38-point capsule:
    pull 0 → 1 point moves, 10 mm → 2, 40 mm → 8; drag still runs at 62 fps on a 1664-handle outline.
    Related fix: a fitted primitive came out of `shapePolygon` at ~1 mm arc resolution (608 points for a capsule),
    which hid the handles entirely (`showVertices` needs ≥ 5 px spacing) and made a nonsense of "simple shape" —
    `shapeRing` now runs `simplifyRing(0.25 mm)` over it, giving 38 points for the same capsule.
    **Editing in the 3D view (2026-09-20, Nolan: "allow me to edit outlines while in 3D scan mode")**: `ScanViewer`
    takes `onEdit` / `onEditStart` / `softMm` and draws the selected tool's vertices as screen-sized `THREE.Points`
    (`sizeAttenuation: false`, `depthTest: false`) draped on the scan, so they stay grabbable at any zoom. A grab picks
    the nearest vertex within a tolerance that scales with camera distance; the drag then raycasts a HORIZONTAL plane at
    the grabbed vertex's height rather than the mesh, so it keeps working when the pointer leaves the surface. The
    pointerdown listener is registered with `capture: true` and sets `controls.enabled = false` — OrbitControls checks
    that flag first thing, which is how the orbit is suppressed without fighting its listeners. It emits polygon_px so
    `commit` is reused unchanged, and the same `geom.softDragRing` drives it, so 2D and 3D reshape identically.
    Measured: 60 fps dragging a 1664-handle outline in 3D (`rebuildOutlines` runs per commit; fine at these counts).
    **Print an outline 1:1 (2026-09-20, Nolan: "print the outline of an item via a piece of paper to size, so i can
    make sure its actually the right size/shape")**: `lib/print.ts` builds a self-contained HTML document and opens it
    in a new tab with `window.print()`. Scale comes from `@page { size: <w>mm <h>mm; margin: 0 }` plus an SVG whose
    `width`/`height` in mm equal its viewBox, so ONE USER UNIT IS ONE MILLIMETRE — verified in a browser: the 100 mm
    ruler renders at 377.95 px and a Letter page at 816 px, i.e. exactly 100.00 mm and 215.90 mm at 96 dpi.
    Every sheet carries that ruler ("must measure exactly 100 mm"), because the one thing that silently ruins this is
    a print dialog left on "fit to page". Anything bigger than a sheet is tiled with a 10 mm shared band of outline and
    a dashed trim line on each continuing edge; sheets are labelled "sheet 3 of 4 (column 1, row 2)". Letter/A4 and
    landscape are options — landscape takes a 660 mm ruler from 4 sheets to 3. Buttons: "🖨 Print 1:1" (with the sheet
    count) in the selected-tool panel and "🖨 All" by the tool list.
    **3D-only canvas (2026-09-20, Nolan: "remove the Height map and 2D view … add a button to straighten the 3D view
    to directly overhead")**: the Outlines step has ONE canvas now, `ScanViewer`. The 2D SVG editor, the 2D/3D
    segmented toggle and the Height-map overlay are gone (OutlineStep 810 → 625 lines), and `⤓ Overhead` snaps the
    camera straight down — that view is what the 2D one looked like, photo draped and all, so nothing was lost by
    dropping it. Every gesture the 2D canvas owned was ported into `ScanViewer` first, via the `EditHooks` prop
    (`onEditStart/onEdit/onCreate/onHint/onSplit`): click a tool selects, click bare mat outlines a new one (after a
    first click lets go of the selection), drag a handle reshapes with the Pull falloff, Shift/Alt click add
    include/exclude hints, ⌘/Ctrl-drag draws a red line across a join and splits on release. NOT ported, because the
    soft drag supersedes them: the Shift push brush and the Alt marquee group-select.
    **"+ Shape" moved from Layout to Outlines** on the same request. Drawn primitives have `session_id: ''` and are
    positioned by `offset_mm` on the MAT, not on the scan, so they cannot be edited on the scan canvas: Outlines is
    where they are created and it lists them under "Added shapes" with a note that they are placed in Layout. Layout
    keeps its own canvas drawing toolbar (↖▭⬭○⬡⬠) — that one draws shapes in place on the foam sheet, which is
    layout work, and was left alone.
    **Snap to base (2026-09-20, Nolan: "the vertexes seem to start on top of the object and i have to spread them out
    so they are around it … they all just seem to need to be pushed out just a little")**: `geometry.snap_to_base`
    walks every vertex along the outline normal to the LAST point that still has something above `floor_mm` (1.5) under
    it, capped at `max_mm` (12) so a vertex over a gap cannot run to the next tool, then smooths (a per-vertex march on
    a noisy height map is jittery). A vertex already over bare mat walks INWARD instead, so the one pass also tightens
    an outline that is too generous. `POST /api/sessions/<id>/snap_base` {polygon_px} -> {polygon_px, polygon_mm};
    "⤢ Snap to base" in the selected-tool panel; reversible through undo / reset.
    Why it is needed: `topo_footprint` cuts at half the wall height, which is exact for a VERTICAL wall but lands
    part-way up anything that slopes, so the outline hugs the top of the tool. Validated on synthetic solids: a
    truncated cone (base r=60 px) snaps to 57.6 px from an outline on the top face (r=35), from the half-height
    contour (r=47.5) AND from a too-wide r=72 — all three converge, residual −1.07 mm which is just the 1.5 mm floor
    threshold on a sloped wall; a vertical-walled block moves −1.8 % (i.e. stays put); a two-tier "mushroom" with a
    brim at r=55 snaps to 54.9. Real mouse outline: 78.6 → 90.4 cm².
    NOTE the normal orientation: `(t_y, -t_x)` is already OUTWARD for a positively-signed ring — flip it for the
    other winding. Getting that backwards collapses every outline to a point, which is how it was caught.
    **Drag to draw a shape**: a compact ↖▭⬭○⬡ bar in the Outlines toolbar arms a kind; `ScanViewer` then takes a drag
    on the mat (`drawArmed`, `EditHooks.onDrawShape`), rubber-bands a box and calls `shapeFromDrag`. The kind stays
    armed so several can be drawn in a row; ↖ goes back to editing.
    The rubber band must be raycast against the MAT PLANE (`localAt(e, 0)`), not the scan mesh. It first required a
    hit on the mesh, and since a drawer is a narrow strip in a big grey viewport, any drag that began off the scan
    silently did nothing — Nolan: "dragged shapes are not being added in the outlines view". Synthetic pointer events
    at the centre of the canvas hid this completely; it only showed up when the test dragged at 18 % and 85 % of the
    canvas height. When testing a canvas gesture, DRAG SOMEWHERE OTHER THAN THE MIDDLE.
    A second, separate reason the same complaint came back ("the shape still doesnt show on the outline view"):
    the shape was in the list but INVISIBLE ON THE CANVAS. `ScanViewer` was given `tools={mine}`, which filters on
    `session_id`, and a drawn shape has `session_id: ''` — so it never reached the viewer at all; and
    `rebuildOutlines` drew `t.polygon_mm` raw, while a shape keeps a shape-local ring plus `offset_mm`, so even once
    passed it would have drawn in the drawer's top-left corner. Both fixed: the viewer gets `[...mine, ...shapes]`
    and everything goes through `placed(t)` (ring + offset) for drawing, for handles and for the click hit-test.
    Shapes are deliberately NOT handle-editable there (`editable` skips `source === 'shape'`) — they are sized and
    moved in Layout. Lesson: "was it added?" is not the same question as "can he see it?"; assert on the CANVAS
    (screenshot or scene contents), not just on the side panel.
    **Low tools (2026-09-21, Nolan: outlines "still come out a bit rough and don't totally match the shape")**.
    Diagnosed on his real drawer (`captures/e2a2e7ca6ff8`, 0.19 mm/px, 0.59 mm cell): the worst offender was a 3 mm
    steel rule whose outline was 34 mm wide and wandered a centimetre across a dead-straight object. The HEIGHT MAP
    WAS FINE — its >1 mm band measures 26.0 mm with a 24-27 mm spread. The extraction threw that away: a FIXED
    `threshold_mm = 2.0` cuts a 3 mm tool at two-thirds of its height, keeping a ragged 60 % of it. Two fixes:
      * `topo_footprint` now scales the threshold to the blob's own height — `clip(LOW_TOOL_FRAC * p90(blob), noise,
        threshold_mm)` with `LOW_TOOL_FRAC = 0.35` and a 0.8 mm floor (`TC_LOW_FRAC`, `TC_NOISE_FLOOR`). Bare mat on
        that scan reads p90 = 0.08 mm, so 0.8 mm is well clear of it. The ceiling is the caller's value, so tall
        tools are bit-for-bit unchanged.
      * `auto_detect` re-fits any blob under `REFIT_LOW_MM` (8 mm, `TC_REFIT_LOW_MM`) once more, seeded by its own
        result. The first pass inherits a too-generous fixed-threshold seed; the second converges. For anything
        taller the second pass is a fixed point, so it is only paid for where it earns its keep.
      Real drawer, 7 tools: the 3 mm rule 65.6 -> 75.8 cm², perimeter/hull 1.15 -> 1.09; every tool above 8 mm
      unchanged to 3 significant figures. Synthetic harness identical to baseline (arc 0.842/3.5, glide seeds within
      0.002) — these changes buy real low tools and cost nothing.
    REJECTED, measured worse, do not re-try blind: snapping vertices to the strongest photo gradient within a few mm
    of the normal. A drawer liner is textured and tools carry printed scales and labels, so the strongest nearby edge
    is often not the boundary — perimeter/hull got WORSE on 5 of 7 tools (e.g. the rule 1.15 -> 1.32). The code was
    written, measured and deleted.
    HARNESS GOTCHA: `strategy_eval.py human` and `strategy_eval.py arc human` give DIFFERENT numbers for the same
    seeds (seed 5: 0.793 vs 0.854), because rendering the arc first perturbs what the glide scenes come out as.
    Each invocation is internally deterministic (RANSAC is seeded). Only ever compare runs invoked identically —
    several "regressions" chased during this work were that artefact, not the code.
    **Outlines are drawn ON THE MAT in the 3D view (2026-09-23, Nolan: "still very rocky" after the 2D shape was
    fixed).** The line used to be draped on the scan surface, i.e. along the FOOT OF A CLIFF, where the 2 mm display
    grid turns 0.3 mm of lateral wobble into centimetres of vertical zigzag. Measured on the tape: bilinear drape
    2.2 mm step-to-step, floor-side sampling 1.95 mm with a WORSE worst jump (9.0 vs 5.6) — no sampling scheme
    follows a cliff on that grid, so the draping approach was dropped, not tuned. A footprint is where the cutter
    goes; it is drawn planar at z=0.8 with depthTest off (so the far side is not swallowed by the body), handles
    on the same plane, drag plane matched. The Relief slider does not move it. The remaining sub-mm roughness
    was then REAL polygon content: +-0.5 mm alternation at ~8 mm wavelength on long flanks, which no hand tool
    has and the sensor's ~2 mm edge response cannot resolve — `clean_ring` smooths free-curve runs by tier
    (`TC_CURVE_SMOOTH_MM` 1.5 for 12-30 mm runs, `TC_CURVE_SMOOTH_LONG_MM` 2.5 for >= 30 mm; runs < 12 mm keep the
    light blur because a fixed 1 mm blur ate 5 mm shaft tips). Verified by LOOKING (screwdriver handle at 0.06
    mm/px, before/after): sawtooth gone, shaft edge one line. Known-truth IoU 0.768 -> 0.770, arc 0.910 -> 0.906,
    vertices on the real drawer 361 -> 220, areas within 2 %. Turn-angle "wiggle" metrics went UP while the picture
    got obviously better — on a sparse polyline they count legitimate vertices; do not steer by them.
    `TC_CLEAN_DEBUG=1` prints each ring's corners and how every run was classified (line / arc / other).
    **Edge-bias trim (2026-09-23, fitted to calipers).** `geometry.trim_edge_bias` pulls each outline vertex inward
    along its normal by `EDGE_TRIM_MM` (1.3, `TC_EDGE_TRIM_MM`; 0 disables) scaled by the LOCAL wall height sampled
    1-6 mm inside (0 below 3 mm, full above 15 mm, smoothed along the ring, capped at a third of the local width).
    Rationale: the IR edge ramp is a roughly constant-width blur, so the outward bias is ~constant on tall walls and
    absent on thin tools — the rule already read 24.7 vs 25.0. Result: tape 92.1 -> **89.5 (caliper 89.5)**; rule
    untouched. Synthetic complex_tools: IoU 0.770 -> 0.765, Hausdorff 6.9 -> 6.7, signed area +18.3 %% -> +6.4 %% —
    the renderer's ramp is narrower than the real TrueDepth's, so a real-fitted trim over-trims the synthetic hammer
    (-10 %%); the calipers win. GATED TO TRUEDEPTH CAPTURES (`scan_meta.sensors`): applied to the rear-LiDAR arc it cost
    0.906 -> 0.870 and added a smoke failure — a different sensor has a different ramp. NEXT: make the trim self-calibrating — measure the 20-80 %% rise distance along each
    vertex's normal and trim by a fixed fraction of THAT, which fits both sensors without a constant.
    **The "razor tooth" (2026-09-23, screwdriver knob) — four causes were found and fixed in one pass, and a fifth
    turned out not to be one.** (1) `clean_ring` was classifying pairs of spurious corners 3-5 mm apart as tiny
    "lines" and so PRESERVING a wiggle as geometry: opposite-turning corners closer than `TOOTH_MM` (6) with an
    excursion under `TOOTH_DEPTH_MM` (3) are now pruned as one tooth before runs are classified. (2) A Gaussian only
    halves a 3 mm-wide impulse, so free-curve runs now do outlier rejection: samples more than `SPIKE_MM` (1.2; trace
    noise is 0.3 rms) off the smoothed curve are replaced and the run re-smoothed. (3) `straighten_ring` must NOT run
    before `clean_ring`: it projects a run's INTERIOR onto a fitted line but leaves the run's endpoints, so two
    straightened runs meeting at a vertex leave a 1-3 mm Z-jog — clean_ring fits its own lines, so the old pass is
    now legacy-only (TC_CLEAN_TOL < 0). (4) `geometry.remove_hairs` (open+close with a `HAIR_MM` = 1.6 mm disc,
    refused if it would change > 5 % of the mask) runs on the mask in `_topo_tool_result` before tracing: a
    1-2 px-wide hair several mm long is THIN IN WIDTH BUT LONG ALONG THE CONTOUR (out and back), so no along-the-curve
    filter can remove it — only mask morphology can. (5) NOT a tooth: the -121 deg vertex at the shaft's TIP. A
    tapered screwdriver tip is supposed to turn sharply; a "sharpest vertex" search kept steering to it. Judge teeth
    by LOOKING at the region the user pointed at, not by the global max turn. Verified on the knob render: both
    flanks single strokes; known-truth IoU 0.770 unchanged, arc 0.906 unchanged, tests OK.
    **Marker mask must not erase tools (2026-09-23, Nolan: "why is the head of the hammer shaped so weird").** The
    hammer's striking face had a clean RECTANGULAR notch cut from its corner — axis-aligned, so not noise. Cause:
    `_marker_mask` excludes 1.5x each marker (the paper's quiet border) from detection, and the hammer head sat
    inside the top-left marker's zone: 4.8 cm2 of a 28 mm-tall head zeroed. Paper reads ~0 mm (folded, 2-4 mm), so
    the mask now keeps only pixels under `MARKER_MASK_MAX_MM` (5.0, `TC_MARKER_MASK_MAX_MM`). Hammer 129.4 ->
    133.1 cm2 with its corner back; synthetic 9/9 at IoU 0.768 unchanged. Side effect: whatever tall thing the mask
    used to hide near a marker now shows — on b59fc5fdd2d1 a 13.6 x 69 mm, 24 mm-tall sliver hugging the top edge
    appeared (almost certainly the drawer wall). WHEN AN OUTLINE HAS A STRAIGHT, AXIS-ALIGNED DEFECT, LOOK FOR A
    MASK OR CROP RECTANGLE, NOT A SENSOR EXPLANATION.
    **GROUND TRUTH, FINALLY (2026-09-23, Nolan's calipers).** Steel rule **25.0 mm** wide. Black tape measure:
    **87.5 mm across its base, 89.5 mm at its widest** (a pocket must pass the widest part, so 89.5 is the target).
    Measured against them, on capture `b59fc5fdd2d1` (78 frames, ALL placed by two markers + the typed 297 x 417
    drawer — registration exact, so this isolates the sensor + extraction):
      * tape outline **93.3 mm -> +3.8 mm** over the true widest; **raising height_threshold_mm 2 -> 12 moves it only
        to 92.5** — `height_threshold_mm` is the blob SEED, not the level; `TC_TOPO_FRAC` (the level) DOES move it (0.5 -> 92.1, 0.8 -> 90.2, 0.9 -> 87.9) but opens sockets' bores as holes and costs IoU. So the level is not the RIGHT lever; The excess is ~1.9 mm per side, which is the sensor's measured
        20-80 %% edge rise (1.9-2.6 mm on this scan) — IR depth blurs the wall outward, and topo_footprint's "50 %% of
        the nearest FLAT pixel" lands on the ramp's shoulder. In the height map the tape reads 89.5 mm at 20 mm up,
        90.4 at 12, 91.5 at 5, 94.5 at 0.5: the sensor SEES the true girth at mid-height; only the low levels are fat.
      * on the earlier capture `136c77400c3f` the rule read 24.2-24.7 (-0.3 to -0.8 mm) in one build and 27.1 in
        another of the SAME frames — that build was placing frames a few mm apart (78 one-marker frames), so on
        one-marker captures registration error dominates and no extraction rule can fix it.
    Two things measured and NOT the cause: the marker print (50 mm markers read 50.9 / 52.1 mm; a fit-to-page shrink
    would read UNDER 50), and depth scale (+1.7 %% / +4.2 %% — real, but in-plane size comes from the MARKERS when
    frames are marker-placed, so calibrating it moved the tape 0.5 mm). `_build_multi_session` has a marker-based
    depth-scale calibration behind `TC_DEPTH_CAL=1`, OFF by default, reported as `scan_meta.depth_scale`; it only
    matters for heights and for pose/matching-placed frames.
    NEXT LEVER (not yet built): on tall tools take the footprint at mid-height (where the sensor is right) rather than
    at the ramp's shoulder — or trim the traced edge inward by half the measured edge rise (~1 mm/side), which
    `_trim_rim_half_height` already does for the photo path. Score any attempt on: tape = 89.5 +- 0.5, rule = 25.0
    +- 0.5. Synthetic complex_tools area errors are +19 %% in the same direction, so a fix should improve both.
    **Outline cleaning (2026-09-23, Nolan: outlines "not smooth, too many vertexes, and not understanding the overall
    shape")**: `geometry.clean_ring(pts_mm, tol)` now finishes every topographic outline (hooked into `smooth_polygon`
    via `clean_tol_mm`, and into `topo_polygon`'s fallback path, which used to skip all cleaning — two real tools kept
    every vertex because of that). It re-expresses the trace as the shape a person would draw: corners found on a
    2 mm-smoothed copy (signed turn summed over +-2 mm, peaks >= 38 deg, >= 3 mm apart, so noise cancels while a real
    corner accumulates); between corners each run becomes a straight line (2 points), a circular arc (Kasa fit,
    sampled by sagitta), or a smoothed curve + Douglas-Peucker — all within `tol` = min(0.5, max(0.35, 0.6*sigma)) mm
    (`TC_CLEAN_TOL`; a NEGATIVE value disables and restores the old 0.15 mm approxPolyDP, which is why there were so
    many vertices: 0.15 mm is finer than the sensor's ~2 mm edge response, so every pixel wiggle survived as geometry).
    Two rules that mattered: (1) STRAIGHTNESS IS JUDGED ON THE BOW, NOT THE WOBBLE — low-pass the signed chord
    deviation over ~3 mm; a wall's wobble cancels, an arc's sagitta does not; accept a line when the bow is within tol
    even if raw wobble is 3x that (same idea for arcs' radial residual). Before this, `straighten_ring`'s 0.18 mm p95
    guard refused every real wall. (2) where two lines meet, the corner is the INTERSECTION of the fitted lines — the
    smoothed apex sits ~0.3 mm inside, which on a rectangle is a 0.3 mm inset of every side. Fitting smoothness scales
    with tol (a fixed 1 mm ate 5 mm shaft tips: hex key -0.06 IoU). Measured: synthetic 100x40 rect -> 5 vertices,
    100.06 x 40.02 mm; L-shape -> its 6 vertices, IoU 0.998; circle r=30 -> 28 points, radius to 0.01 mm.
    complex_tools known-truth mean IoU 0.765 -> 0.768 (no tool worse than -0.003), arc harness 0.911 -> 0.910; real
    drawer 136c77400c3f: 824 -> 337 vertices with every area within +-0.4 %. What it correctly does NOT hide: the steel
    rule keeps ~110 points because its fused edge genuinely wanders ~1 mm in slow waves over 330 mm — that is
    one-marker frame rotation error upstream (registration), not tracing noise; for a known-straight tool the Simple
    shape -> Rectangle fit is the right tool.
    **What the residual roughness actually IS (2026-09-21, measured, worth reading before "improving" it again).**
    High-frequency wobble is ALREADY SMALL: true perpendicular deviation from a 4 mm-smoothed ring is 0.11-0.28 mm
    rms across all seven tools on the real drawer. So generic smoothing and primitive fitting have nothing left to
    remove — `regularizeRing` (break the ring at corners, fit a line and a circle to each run, replace where it fits
    within 0.8 mm) was written, measured at 0.23 -> 0.22 mm mean wobble, and DELETED. What remains is medium-scale
    (2-5 mm) SHAPE error: outlines sitting outside a domed body, a spur on the socket. That is the height map's
    reading of soft/rounded edges, not noise in the tracing.
    Also beware two misleading metrics used earlier in this work: perimeter/convex-hull punishes GENUINE concavity
    (the combination square reads 1.22 because it is L-shaped, and its wobble is mid-pack), and a point-to-ring
    distance against a coarse smoothed ring inflates wobble 2-4x versus true perpendicular distance.
    **`snap_to_base` over-expands on real data.** Validated against synthetic cones it was exact, but on the real
    drawer it grew three tools by +42 %, +45 % and +85 %, the last having crawled onto an ArUco marker: bare mat
    there reaches 3.4 mm at p99, well above the 1.5 mm floor, so "keep walking while above the floor" never stops.
    It now measures the mat in a 4-12 mm annulus around each tool and uses p97 + 0.5 mm as the floor, which brings
    those to +18 %, +45 %, +64 %. STILL NOT SAFE unsupervised — treat it as a suggestion, check against a 1:1 print.
    Missing ingredient for any further tuning: GROUND TRUTH. Every rule here is tuned against proxies (perimeter/hull,
    wobble, synthetic scenes). Caliper measurements of a few real tools would let these thresholds be fitted and
    verified instead of guessed.
    Headless UI tests for all of this live in the scratchpad (`shape_test.mjs`, `shape_test2.mjs`, `perf2.mjs`) and drive
    puppeteer-core from `apps/admin/node_modules`; note panel titles are CSS-uppercased, so match their text case-insensitively,
    and the Upload step contains an "auto-detect" option that a naive text wait will match. Layout & export stays separate: it is different work (mat,
    clearance, pocket depth, export) and only shares a tool list, which means something else there. Outline
    maths live in `lib/geom.ts` (`resampleRing`, `simplifyRing`, `smoothRing`, `pointSegment`). polygon_mm = polygon_px ×
    rectified mm_per_px, so edits are committed in both. `_geometry_for_tool` rasterises an edited polygon for the STL body.
  - First real-device scans (2026-09-20, iPhone → Mac over LAN). Two bugs found: (1) the app's review WebView loads the
    admin from `http://<mac-ip>:3000`, but the page called the API at `localhost:8000` = the phone → "backend offline";
    `lib/api.ts` `resolveApiBase()` now swaps a localhost API base for the page's own host when the page is not local.
    (2) sessions lived only in memory. Phone captures are now saved raw to `backend/captures/<id>/` (`capture.json` + frames,
    gitignored, `TC_CAPTURE_DIR`), `_session()` rebuilds a missing one from disk under the SAME id (~1 s/frame), and
    `GET /api/sessions` lists memory + disk → "Recent captures" panel on the Upload step (polls every 5 s). Saved captures
    are real test data: replay with `_rebuild_capture(id)`. Refused uploads now log their reason (`_handle_api_error`).
    `DELETE /api/sessions/<id>` (Nolan asked for it 2026-09-20) drops the session AND `rm -rf`s its saved frames —
    irreversible, id must match `[0-9a-f]{12}` (no path traversal), taken under `_REBUILD_LOCK` so it cannot race a
    rebuild. `RecentCaptures` has a ✕ per row with an inline two-step confirm naming the frame count. A rebuild also
    refreshes `mat_mm` / `frames_used` in `capture.json`, since the listing reads that cache and newer code can measure
    the drawer differently (the marker-order fix moved one saved scan from 870 × 910 to 922 × 265 mm).
    Those first real scans logged many frames "skipped (overlap 0 px)", every phase-correlation shift rejected, and a
    drawer of ~860 × 900 mm. DIAGNOSED from the saved frames (`captures/efb7ad666414`, `3f5f669dbb44`, a desk test): the
    markers were placed out of order — ids 0 & 2 side by side (~220 mm) at one end, 1 & 3 at the other (~890 mm away) — so
    the TL,TR,BR,BL quad was a bow-tie and the server read the two diagonals as width and height. ARKit poses themselves are
    good (marker centres re-project within 1–4 mm across frames; transform is column-major, metres), marker size checks out
    at 50–53 mm against LiDAR. Also seen: a finger over the lens blurs the right third of every photo, and legs/floor are in view.
  - **Marker order is read off the positions (2026-09-20, Nolan: "have the server work out the marker order from their
    positions")**: `capture.corner_order(centres)` maps id -> corner slot (0 TL, 1 TR, 2 BR, 3 BL) by sorting the markers
    clockwise about their centroid (the frame is x-right/y-DOWN, where a growing atan2 angle turns clockwise) and choosing
    the start of the cycle that agrees with the printed sheet best — a correct sheet maps to itself and nothing moves. With
    3 markers the 4th is completed first (the right-angled corner is the one between the other two); a non-convex quad or a
    triangle >20° out of square returns None = leave the ids alone. `capture.relabel_markers` re-keys, and every consumer
    keeps its old slot semantics, so `outer_corner` / `_orient_for_markers` / `rectangle_from_*` are unchanged.
    Applied in `rectify_rgbd` / `rectify_markers_only` (auto from that frame's plane corners, or an explicit `corner_map=`)
    and in `_session_from_mesh_file`. Multi-frame needs ONE global answer before rectification (a glide frame sees two
    markers at most, and the per-frame raster orientation depends on the answer): `app._corner_map_for_frames` does a cheap
    pre-pass — reduced-size decode, ArUco, `solvePnP(SOLVEPNP_IPPE_SQUARE)` + the ARKit pose -> marker centres in world mm,
    flattened to (x, z) because ARKit world is y-up and right-handed, so (x, z) is the top-down un-mirrored frame — and
    falls back to the frame that sees the most markers when there are no poses. Session `scan_meta.marker_corners` /
    `markers_reordered` drive a line in the CalibrateStep banner. `_orient_for_markers` also accepts a DIAGONAL pair now
    (0->2 always points within 45° of 45°, whatever the aspect ratio), which a glide sees often once the ids are ordered.
    `capture.corner_markers()` drops ids outside 0-3 everywhere: ArUco read 4X4 codes off keyboard keys (ids 16/17/28/37)
    in the real scans, and one bogus marker rotates a frame or wrecks the markers-only homography (it assumes marker_size_mm).
    Effect on the real desk scan `efb7ad666414`: 870 × 910 mm and many frames skipped -> 922 × 265 mm, 28/28 frames used
    (truth ~934 × 270 from the marker centres + one marker size). Smoke test has a shuffled-sheet case
    (`synth_capture.render(marker_ids=...)`, BR/BL swapped -> still 400.0 × 300.0); arc/glide accuracy unchanged
    (arc IoU 0.844 / 3.2 mm). The marker sheet SVG now says the order does not matter.
    The eval harnesses set `TC_CAPTURE_DIR` to `tests/out/captures` so synthetic runs stay out of "Recent captures".
  - **GHOSTS, PHOTO-FIRST FOOTPRINTS AND PRIMITIVE SNAPPING (2026-10-03, Nolan: "the shapes/outlines still look
    ugly").** Diagnosed on the desk drawer `a237eba87dba` by drawing the detected outlines over the photo AND over the
    height map (do that first; the layout rings alone cannot tell a bad trace from a bad detection). The ugliness was
    not smoothing: (a) 7 of 11 height blobs were GHOSTS — a tall sliver beside the power bank, a fragment beside the
    pin box, a wedge beside the case — left by ONE misplaced pose-chained frame, or the midpoint of two frames that
    disagreed (nanmedian of [44, 0] = 22, a height nobody measured); (b) objects the photo had outlined cleanly kept
    their half-covered height blob (the case's bottom half returns no depth) because the old hybrid rule dropped any
    silhouette overlapping a blob. Fixes, each measured:
    * `depth_fusion.fuse_heights_with_support` -> `Session.rect_support` (uint8: frames agreeing with the fused value
      within `TC_SUPPORT_TOL_MM` 3 mm; `RECONSTRUCTION_VERSION` 6, carried through re-rectification as a third
      `extra`). Real tools had median support 2-7 per cell, every ghost 0-1. `_auto_detect_impl` drops a topo BLOB
      whose median support < `TC_MIN_SUPPORT` (2) — judged per blob, not per cell, so a tool seen once at its edge
      is not nibbled (the cube was seen by exactly 2 frames and survives). `_trusted_height` NaNs raised cells with
      support < 2 for the relief, so a ghost inside a silhouette cannot become a pocket depth. Floor-level
      readings are left alone (seen once is fine for 0 mm).
    * `_detect_hybrid` is PHOTO-FIRST where both see an object: a height blob >= 30 %% inside a photo silhouette is
      absorbed (`found_by: "both"`, the silhouette is the footprint, the blob's cells remain the depth); a
      silhouette >= 60 %% inside a bigger one is a part (a panel of the case) and is dropped; photo-only tools keep
      the tall-or-big rule; topo-only blobs stay as they are. Result: 15 tools -> 8, all real, 0 ghosts.
    * `geometry.finish_footprint` (via `process_outline(finish=True)`, default; `finish_footprints: false` in the
      layout body restores the old path): a scanned footprint that IS a primitive (`fit_footprint_primitive`,
      IoU >= `TC_PRIMITIVE_IOU` 0.93; an L scores ~0.5, a hammer ~0.7, a bottle ~0.85) becomes that exact
      rectangle / rounded rectangle / capsule / circle at the tool's own angle, grown <= 1 mm to ENCLOSE the trace
      (p90 of the overhang); otherwise `cleanup.smooth_enclosing`. The box is sized BEFORE the kinds are compared
      (minAreaRect hugs the noise ~0.6 mm; on a 12 mm pencil that alone costs 0.1 IoU and made a fat sharp box lose
      to a rounded one); near-ties (< 0.002) go to the more canonical kind. Finishing runs in the tool's own frame
      before rotation/offset behind a 512-entry LRU, so a drag is a cache hit: layout 118 ms cold, 8 ms warm for 8
      tools. Layout entries carry `footprint {kind, iou, w_mm, h_mm, r_mm | diameter_mm | vertices, grown_mm}` and
      the Design-insert panel prints it ("Footprint: rounded rectangle 91.6 x 159.6 mm, r 8 (fits the scan 95 %)").
      Desk drawer: drive/pin box/power bank/case -> rounded rectangles & a rectangle, USB stick -> capsule, bottle
      -> smooth (neck), cube -> smooth 59.6 x 53.9 (true 57: its top-left was seen by 1-2 frames; registration, not
      tracing). Tests: `tests/test_footprints.py` (9).
    * Relief on dark material: floor-level readings INSIDE a tall tool are sensor failures, not geometry (the black
      neoprene case: 45 %% of cells read the 22-25 mm top, 42 %% read 0-4 mm, and three frames AGREED on those — frame
      support cannot help; the IR pattern vanishes on that material). `_place_height` marks cells under
      max(1.5, `TC_RELIEF_FAIL_FRAC` 0.10 x p90) as holes when holes + near-floor cells are >= 10 %% of the footprint,
      and fills holes from a 12 mm measured NEIGHBOURHOOD (a constant p75 fill left 2-3 mm steps). Case pocket floor:
      0.41 -> 0.19 mm rms, p5 15 -> 21 mm. A tool with < `TC_RELIEF_LOW_COVERAGE` 25 %% measured cells is carved FLAT
      at the typed thickness (style "flat (depth coverage N %%)") — with the ghost excluded the power bank's 10 %% of
      cells read 3 mm on a 57 mm object. `GUIDE_EPS` 0.02^2 -> 0.08^2 (texture transfer; small effect, kept).
    * KNOWN AT HEAD, NOT FROM THIS WORK: `tests/smoke_test.py` fails 5 single-still RGB-D checks ("corrected
      footprint" bar 195 x 42 vs 180 x 35, centroid 5.9 mm) identically on the committed `capture-pipeline-baseline`
      code (verified by running the smoke test on `git archive HEAD`). The arc/glide/multi-frame suites pass.
    * NOT DONE: the pencil (depth on its lower half only) and the screwdriver (black, thin: no depth, no silhouette)
      — nothing sees the screwdriver; there is no click-to-add in the Design-insert flow any more. Iterated SAM
      growth from a blob was tried for the cube and pencil: SAM returns the same partial region however the box is
      grown, so it was not adopted.
  - **AI outline cleanup (2026-10-02, Nolan: "Do you think we could use AI to clean up the outlines?" → "okay, lets add
    it" → "Make it more advanced. It can take longer, look at the height map, and the picture.")**
    Design rule: THE MODEL SUPPLIES KNOWLEDGE AND JUDGEMENT, THE GEOMETRY SUPPLIES COORDINATES. The model never
    returns a number in millimetres or pixels; it names PLACES by numbered marks and the geometry measures there.
    * `toolcutter/cleanup.py`: `prepare_ring` (0.5 mm resample, 1 mm pre-smooth when the perimeter exceeds the hull's
      by 15 %) is THE working ring; `mark_indices` puts ~32 numbered marks around it (indices into that ring — compute
      it once and pass it everywhere, the marks on the pictures must be the marks the edits refer to).
      `apply_local_edits(ring, marks, edits, photo, origin_mm, mpp)` applies the model's place-by-place verdicts:
      `straight` (project the run onto its fitted line; refused if any point is > 2.5 mm from it), `arc` (Kasa fit,
      refused > 2 mm p95), `spur` / `notch` (replace the run by its chord; a 'spur' that would ADD area, or a 'notch'
      that would remove it, is refused — the model had the direction wrong), `too_tight` / `too_loose` (move the run
      out / in to the PHOTO's strongest edge along each point's normal, ROBUST median across the run, needs an edge at
      half the points, capped 5 mm, tapered over 3 mm at both ends so the ring does not step; "no clear edge" is a
      refusal), `merged_neighbour` / `missing_part` (advice only, shown, nothing moved). A run spanning > 60 % of the
      ring is refused (a range read backwards). Then the global constraints as before — parallel edges (±4°), right
      angles (±4°), mirror symmetry (area-moment axes), circle, rectangle from the FITTED edges — and the area cap is
      8 % (20 % when the model asked for local edits: cutting a shadow spur off a small tool). `_result` returns a
      0.2 mm Douglas-Peucker pass (`SIMPLIFY_MM`) of the 0.5 mm working ring; the first real run returned 57 → 1446
      points, i.e. the SAMPLING instead of the shape.
    * `toolcutter/recognize.py`: `make_crops` isolates ONE tool (outside its outline dimmed to 40 %, height masked —
      a plain bounding-box crop of a crowded drawer showed the hammer with three screwdrivers and two pliers, and the
      model could not know which was meant); `annotate` draws the trace in orange + numbered white marks + a 20 mm
      scale bar on the FULL-RESOLUTION photo crop (`MAX_SIDE` 1400) and on the turbo height crop ("colour tops out at
      N mm"); `context_image` is the whole drawer with this tool outlined. `analyze` sends all three with the facts
      (size, height, mm/px, mark count) to `claude-opus-5` at `TC_RECOGNIZE_EFFORT` (high; thinking is on by default
      on Opus 5) with structured output (`output_config.format` json_schema): `tool_name`, `description`, `shape_class`,
      `symmetric_axis`, `straight_edges`, `right_angles`, `round_shaft`, `trace_quality`, `edits[{from_mark, to_mark,
      kind, note}]`, `confidence`. `render_preview` draws the proposal GREEN over the photo with the measured trace
      faded and the ORIGINAL marks; `verify` shows it back and asks "does the green line follow the tool everywhere?"
      → `follows_edge`, `issues[]` (same edit shape, same marks), `better_than_orange`. `_clean_edits` drops marks
      out of range or unknown kinds. Timeout 180 s, 2 retries; it may take a minute per tool and that is accepted.
      LOOKED AT on the real 805 x 434 drawer (`6b9814466751`, 0.45 mm/px mosaic): the hammer picture shows the
      claw's stray spike at marks 27-29 and the shadow bulge at 0-1 plainly, in photo and height alike — this is
      the kind of thing the model is for. But a 70 x 30 mm screwdriver was a 168 x 344 px crop on which 32 marks
      overlapped each other and the line, so marks now follow the PERIMETER (`MARK_SPACING_MM` 8, 10..40 marks) and
      small crops are UPSCALED to `MIN_SIDE` 700 px (cubic) before anything is drawn (`_upscale`; the preview too).
      Next lever if small tools still read poorly: the per-frame source JPEG behind `tool.image_url` is far sharper
      than the mosaic crop — it needs that frame's homography to place the ring, which the image endpoint has.
    * `POST /api/sessions/<sid>/cleanup` {tools: [{id, polygon_px, name}], recognize?, verify?}: per tool (4 threads,
      `TC_CLEANUP_WORKERS`) ring → marks → crops → analyze → propose(local edits + hints) → preview → verify → if it
      has issues, ONE more propose with `local + issues` (same marks, deterministic) → preview as `preview_png`
      (base64, ≤ 900 px) in the proposal. A verification that says "not better than the trace" with no issues
      empties the proposal. Returns PROPOSALS ONLY with `recognition`, `verification`, `applied`, `refused`,
      `max_move_mm`, area deltas. `TC_CLEANUP_VERIFY=0` skips the second pass. `TC_RECOGNIZE_DUMP=<dir>` writes every
      annotated photo / height / context picture the model is shown — LOOK AT THEM before blaming the model.
    * Degrades: with no `ANTHROPIC_API_KEY` — put it in `backend/.env` (gitignored; `main()` loads it via `_load_env_file`,
      `backend/.env.example` shows the shape) or export it in the shell that starts `app.py` — otherwise
      `analyze` / `verify` return None (logged ONCE per process, `recognize._UNAVAILABLE`), the endpoint returns
      `recognition_reason`, and only trace-proven constraints apply (rect IoU ≥ 0.95, symmetry cover ≥ 0.95) — measured
      on the real drawers (`a7af39bd92eb`, `b1e99fbbb404`, 13 tools): every move ≤ 1.0 mm, every area within 0.2 %, the
      tape measure 90.4 × 89.5 stays 89.5 (= caliper). THE KNOWLEDGE HAS TO COME FROM RECOGNITION; the strict auto mode
      is a safety floor, not the feature. Never route customer drawer photos through anything but the Anthropic API.
    * UI (Outlines): "✨ Clean up" in the selected-tool panel, "✨ Clean up all" by the tool list — the latter runs in
      BATCHES of 4 tools per request and shows "Inspecting 8 / 24…" as proposals arrive. Proposals are drawn GREEN on
      the canvas (`ScanViewer` `ghosts` prop) over the measured line faded to 35 %. Each card: "looks like <tool> ·
      N % · trace minor issues", the model's description, the PREVIEW PICTURE (green proposal, faded trace, marks),
      every applied edit in words with its marks and millimetres ("Marks 12–14: spur cut off (0.8 cm², up to 4.1 mm)
      — shadow of the handle"), every refusal with its reason, the model's check, largest move, area before → after,
      Accept (through `commit`: undo / Reset to detected work) / Reject / "Name it …". Nothing applies silently. The
      status line is a hint under "Proposed clean-ups", not the red error box; cards use theme variables (hard-coded
      light greens were unreadable on Nolan's dark theme).
    * FIRST REAL MODEL RUN (2026-10-02, Nolan's key in backend/.env, hammer on `6b9814466751`, 63 s for two tools
      incl. verify, ~4.1k input / 2.1k output tokens per analyze): "claw hammer 0.86, T_shape, trace poor" — and the
      reason it said poor was RIGHT: the trace had swallowed the tape measure lying above the head
      (`merged_neighbour` 37-6), the claw tip was clipped (`too_tight` 33-36, moved 2.25 mm from the photo edge), the
      shaft edges were straight (9-13, 29-32). The verify pass agreed and said the merge is "inherited from the
      scanner" — i.e. a CLEANUP cannot fix a merged detection; it needs a split and two re-detections, which is the
      case for an agent over the whole drawer (Nolan: "What if we took a more agentic/overall approach?").
      Three things broke before the first answer, all now guarded: (1) structured output REJECTS `minimum`/`maximum`
      on a number (400) — `test_schemas_use_only_supported_keywords`; (2) the server died of "Too many open files":
      the processed-capture snapshot memory-maps ~1,150 .npy files per warm drawer and each holds an fd —
      `_raise_fd_limit()` in `main()` lifts the soft limit (the proper fix is one big array file per snapshot);
      (3) a reply with NO text block (the budget spent on thinking) raised a StopIteration that logged as an EMPTY
      warning — now logged with stop_reason and usage, max_tokens 16000 / 8000. Local straight/arc edits are
      TAPERED at both ends (`_taper`) — a hard projection left a jog where the shaft run met the head.
    * Tests: `tests/test_cleanup.py` (16): global constraints (noisy rect → exact 4-pt rectangle, L-shape stays an L,
      hinted symmetry, circle ±0.15 mm, irregular untouched, hinted rectangle on an L refused), local edits (spur cut
      and the same run refused as a notch; straight run projected and a semicircle refused as 'straight' but accepted
      as 'arc' with r = 30.0; `too_tight` moved 2.0 mm out to a synthetic photo edge and `too_loose` refused where
      there is none; a 31-of-32-mark run refused; spur + rectangle hint end to end → 4 points, 100 × 40), the
      annotated pictures render (saved to `tests/out/cleanup_*.png` — look at them), and recognition degrades offline.
      Inserting a synthetic spur into `_rect_ring` must happen at the index where that edge passes x = 55
      (`P[:185]`), not at the start of the edge — the first version retraced the edge and tested a mess.
  - **THE DRAWER AGENT (2026-10-02, Nolan: "What if we took a more agentic/overall approach?" → "Tes")**. One model
    session per drawer with TOOLS, in `toolcutter/agent.py`: `DrawerTools` is the geometry side (a working copy of the
    tool list + the operations), `run_drawer_agent` the loop (manual tool-use loop over `client.messages.create`, so
    the budget and the log are ours; `TC_AGENT_MAX_CALLS` 60, `TC_AGENT_MAX_S` 900, effort `TC_AGENT_EFFORT` high).
    Tools the model may call: `list_tools`, `view_drawer` (all outlines + id labels; orange = as detected, green =
    changed), `view_tool` (the annotated photo + height crops with NUMBERED MARKS — marks are per VIEWED VERSION and go
    stale on any change, the tool results say so), `height_profile` (heights along the chord between two marks, numbers
    + a plot: confirms a valley between merged objects), `split_tool` (line through two marks → `_split_core`),
    `merge_tools` (→ `_merge_core`), `redetect_tool` (`_segment_topo` seeded at the tool's own centre inside its box
    at a chosen `height_threshold_mm`), `edit_outline` (`cleanup.propose` with local edits by marks + optional hints),
    `fit_shape`, `rename_tool`, `remove_tool`, `undo_tool`, `finish`. Tool results carry pictures (<= 1000 px) back
    into the conversation; every call is logged {step, tool, input, summary, t}. The model still never produces a
    coordinate. `_split_core` / `_merge_core` were extracted from the /split and /merge routes so the agent and the UI
    run the same code. `POST /api/sessions/<sid>/agent` {tools, instructions?} → {job_id} (thread);
    `GET /api/agent/<job_id>?since=N` → {status, log[since:], result} — the result is a PROPOSAL: the full tool list
    with `changed` / `new` flags + `removed` ids + per-tool before/after `preview_png` + summary/usage. The server's
    masks ARE updated by splits/merges (new ids) — harmless if rejected. UI (Outlines): "🤖 Agent" by the tool list
    runs it on every scanned tool, the log streams into a panel every 2 s ("Looked at claw hammer", "Split Tool 5
    along marks 21–8 → ..."), then "Agent proposal · N changed, M removed" with cards (preview, status) and "Accept
    all changes" (replaces/removes/adds tools, keeping include/clearance/placement of existing ids) / Reject; an
    "Instructions" field feeds the next run (caliper facts, "leave the sockets alone"). Changed outlines draw green
    on the canvas; outlines for NEW ids (a split) are drawn standalone by `ScanViewer`. Test: `DrawerAgentTest` runs
    the loop with a SCRIPTED client (view → view → height_profile → split → rename → finish) against the real split
    code on a synthetic merged pair: two 60 x 30 blocks, named, `m1` removed.
    FIRST REAL RUN (2026-10-02, 24-tool drawer `6b9814466751`, 59 calls, 11 min, then STOPPED BY "credit balance too
    low" on Nolan's Anthropic account). The judgement was excellent and every action was right: measured the valley
    with `height_profile` BEFORE splitting; hammer | tape measure (348 x 129 / 129 x 55); pry bar | hand plane; the
    drill lump -> cordless drill + battery charger + charger cord + battery pack (three splits and a merge of a
    mis-split piece); socket bit set | long drill bit; named everything; tried `too_loose` on the tape's shadow edge
    twice (refused: "no clear edge" — the photo-edge finder looks <= 5 mm inward at gradient >= 6; the shadow strip
    may need the HEIGHT map as the edge source), re-detected the tape at 6 mm, judged it worse and UNDID it.
    Two bugs it exposed, both fixed: (1) ID COLLISION — a merge and a split in the same second both made
    `a58388_1`; ids are now `{letter}{seq}x{time}` (`_new_id`), `_add` refuses to overwrite, `result()` skips ids it
    no longer has, and the loop returns PARTIAL WORK on any failure instead of raising (11 minutes were lost).
    (2) COST — every turn re-sends the whole history including every picture, so a long run is quadratic in
    pictures: `_prune_images` keeps pictures only in the last `TC_AGENT_KEEP_IMAGES` (4) tool-result turns and
    `_mark_cache` puts a prompt-cache breakpoint on the system prompt, the tool list and the last user turn.
    `usage` now reports cache_read/cache_write tokens; check them on the next run.
    JOBS OUTLIVE THE TAB (Nolan: "I lose the agent if I change tabs or drawers"): `OutlineStep` unmounts on a step or
    drawer change (`key={session.id}`), so its polling died with it while the thread kept working. Jobs are server
    truth: `GET /api/sessions/<sid>/agent` lists a drawer's runs (newest first), `POST /api/agent/<id>/dismiss` marks
    one accepted/rejected, finished jobs are written to `backend/agent_jobs/<id>.json` (gitignored) and reloaded at
    startup (a job that was "running" when the process died comes back as an error that says so). On mount the step
    lists the jobs and FOLLOWS a running one or SHOWS the newest finished undismissed one (`followAgent`, full log from
    `since=0`); Accept/Reject dismiss on the server. Starting a second run on a drawer that has one running is a 409.
    SECOND REAL RUN (same drawer, 60 calls = the whole budget, 787 s, usage: cache_read 319k, cache_write 1.19M, fresh
    input 80, output 41k incl. thinking). Right again on every split (hammer | tape, pry bar | plane, drill into three,
    drill bit | bit set, two screwdriver handles) — but it hit the call budget mid-inspection with NO names and NO
    summary, and it had UNDONE a split of the drill lump while BOTH CHILDREN STAYED, so the drill was covered twice.
    Fixed: (1) every tool result ends with "[budget: N calls and S s left]" and at <= 10 calls "WRAP UP NOW"; when the
    budget still ends without `finish`, one tool-less low-effort turn asks for the summary; `MAX_CALLS` 60 -> 80.
    (2) `undo_tool` on a split parent drops `derived[parent]` (its children); on a merged tool it removes it and
    restores `merged_from[new]`. (3) Caching barely helped: `_prune_images` changed one old message EVERY turn, so
    the cached prefix broke there every turn and the last four picture-turns were re-written each time (writes 4x
    reads). Pruning now runs every `TC_AGENT_PRUNE_EVERY` (8) calls, so the prefix is stable for 8 turns at a time.
    Measure cache_read vs cache_write on the next run; reads should dominate. It also re-merged the pry bar and
    plane it had split (264 x 137) — whether that lump is one tool or two is genuinely unclear from above.
    LIVE OUTLINES (Nolan: "i want to see the changes it makes to the outlines as its making them"): `DrawerTools.live()`
    snapshots the working tool list (polygons, status, changed/new, removed) under `dt.lock` — the loop holds that lock
    around every tool call — and `GET /api/agent/<id>` includes it as `live` while the job runs (`_AGENT_LIVE` keeps
    the DrawerTools out of the JSON job dict). The Outlines step feeds it to the canvas every 2 s: changed/new
    outlines draw GREEN over the faded originals, removed ones (split parents, merged originals) draw at 12 %
    (`ScanViewer` `removedIds`), the panel lists "name · status · area" as each change lands and the header counts
    "N changed so far". Nothing is applied until Accept; Reject clears it all.
    THIRD REAL RUN (after the fixes above): FINISHED BY ITSELF — 79 calls, 683 s, every tool named (30 renames), a
    readable summary with a "left as measured, check by hand" section, and caching finally working: cache_read 942k
    vs cache_write 160k, fresh input 70, output 37k. Splits: hammer | tape; pry bar | saw blade (| rubber knob);
    screwdriver 5 | 6; bit holder | long drill bit; folding knife | key tag; drill | battery pack. Merge: ratchet head +
    shaft. It kept the drill + charger + cable as ONE pocket on purpose ("no straight cut separates them without
    clipping the battery foot") — a split along a POLYLINE, not a line, is the next tool to give it. It reverted a
    higher-level re-detect that merged the six screwdriver handles and said so. The live counter rose 0 -> 4 -> 8 ->
    30 during the run, so the canvas showed the work as it happened.
    PERFORMANCE PASS (Nolan: "Can you improve the agents performance?" + "Make sure it uses common sense ... If it
    sees a hammer, it should make sure the outline looks like a hammer ... nice smooth outlines that will create a
    professional looking toolboard"):
    * `split_tool` FOLLOWS THE VALLEY by default (`follow_valley`): `agent.valley_path` runs Dijkstra on the height
      map (2 mm cells, cost = height + a little per step, inside the tool's mask dilated 4 mm) from mark to mark, and
      `app._split_core_path` cuts along that polyline (1.5 mm seam; the two biggest remaining components are the
      sides, crumbs join the nearer). Used only when the valley's p95 height is >= 0.5 mm lower than the straight
      line's max — the result text says which and the heights. This is what the drill + charger + cable needed
      ("no straight cut separates them without clipping the battery foot"). `ValleySplitTest`: a diagonal 3 mm gap.
    * FEWER CALLS: `split_tool`, `merge_tools` and `redetect_tool` return the new tools' annotated pictures with
      fresh marks (`_views_of`), so the model no longer spends a `view_tool` per child (12+ calls on a 24-tool
      drawer); the intro lists SUSPECTS (`DrawerTools.suspects`: a tool whose mask at half its own p90 height has
      >= 2 plateaus of >= 3 cm2 is "likely merged"; blobs < 3 cm2 are "debris or marker corner?") and says several
      tools may be called per turn.
    * COMMON SENSE + POLISH in the system prompt: once it knows the tool, the outline must look like that tool;
      never cut INTO the tool (a pocket may be a little generous, never tight); after the detections, a POLISH PASS
      with `smooth_outline` on everything not already reshaped and symmetry/straight hints where obvious.
    * `smooth_outline(tool_ids, strength_mm)` -> `cleanup.smooth_enclosing`: the trace is blurred 1 mm into its
      BODY (kills +-0.4 mm jitter and one-sample teeth), `geometry.clean_ring(tol 0.4, curve_sigma_mm=strength)`
      smooths PIECEWISE between the corners (new `curve_sigma_mm` parameter: a whole-ring Gaussian rounded every
      corner by ~0.6 sigma and the enclosure step then grew tools a millimetre a side), then if the result dips more
      than 0.3 mm inside the body the WHOLE line is offset outward by (dip - 0.2) with round joins (pushing single
      points back out hands the jitter straight back — measured 61 bumpy vertices), then Douglas-Peucker 0.2.
      Reports vertices before/after, area change and `max_inset_mm`. Synthetic: noisy 80 x 30 capsule -> 22 vertices,
      81.4 x 31.4, inset 0.21; noisy 100 x 40 rectangle -> 12 vertices, 101.2 x 40.7; a T stays a T (22 vertices).
      The ~+0.3 mm per side is the price of "never tight" on a +-0.3 mm noisy trace; measure against the calipers.
    FOURTH REAL RUN (with all of the above, Opus 5 high): finished by itself, 80 calls, 658 s, cache_read 842k /
    cache_write 232k / output 38k. Valley cuts fired on the socket block | drill bit, the hammer | stud finder and
    the screwdriver pair ("cut runs through ground <= 11.8 mm; a straight line would have crossed 15.6"), straight
    where the valley was no lower. It did a polish pass over all 27 outlines (1.5 mm small / 3 mm big; the smoother
    reported every one still enclosing its trace) and wrote a "check by hand" list: the screwdriver SHAFTS are
    outside the detected regions (thin steel, below the depth threshold — a `missing_part` the engine cannot grant
    without a detection at a lower level in that spot) and the drill keeps a dark shadow area the trim refused.
    27 of the 80 calls were single `rename_tool`s -> `rename_tool` now takes `renames: [...]` for the whole drawer.
    MODEL COMPARISON (same drawer, same tools, `TC_AGENT_MODEL=claude-sonnet-5`): Sonnet 5 took LONGER (736 vs 658 s),
    used MORE tokens (cache_read 1.93M vs 0.84M, output 55k vs 38k — it flailed: 16 split calls and 7 undos on the
    drill) and MISSED what Opus found: left the hammer merged with the tape/stud finder (206 cm2 "claw hammer"),
    left the two screwdriver handles merged, left the knife + key tag merged, and left tool 4 whole after reading
    its own height profile as "one continuous plateau". It did split the socket block | drill bit, merge the ratchet,
    name everything (one batched call) and polish all outlines. VERDICT: stay on Opus 5 for the agent — on this job
    the smaller model is not cheaper, because judgement errors cost calls. Its one useful lesson is baked in:
    `split_tool` now refuses two marks on the SAME edge (<= n/8 marks apart) with an explanation of where to put them.
    The "common sense" rule about lobes/forks (split off and reassign, do not just mention) landed in the prompt after
    run 4 left "screwdriver 3" with a fork of its neighbour and only wrote it in the summary. Split children are named
    `<base> (n)` with the base stripped of earlier "(n)" suffixes ("Tool 1 (1) (2) (2) (1)" happened).
  - **FORM-FIT 3D POCKETS (2026-10-03, Nolan: "What if i made this more of a 3D workflow? Taking the height map,
    smoothing it, and cutting those shapes directly into the foam via CNC.")** `toolcutter/relief.py` turns the
    computed layout into ONE depth map of the foam block, D(x, y) = mm to remove below the top, on a `res_mm` grid:
    for each tool with `pocket_style: relief` and a scan in memory (`_geometry_for_tool`), the tool's height raster is
    smoothed INSIDE its mask (normalised Gaussian, `smooth_mm`, so the edge does not bleed into the floor), placed
    through the layout transform (inverse rotate about `source_centroid_mm`, offset, mirror) with `cv2.remap`, and
    depth = height + `z_clearance_mm`, capped at the tool's `depth_mm` if set and always at thickness - `floor_min_mm`;
    the clearance band (layout ring minus scanned footprint) takes the nearest scanned depth so the wall is vertical at
    the ring. Flat tools / shapes / unscanned tools are flat at their depth. Exports from that one array:
    `depth_to_stl` (closed grid mesh: top surface with steep cells as walls + slab + sides, watertight in the test),
    `depth_to_png16` (16-bit PNG, 0 = top, 65535 = deepest, JSON sidecar with grey_to_mm, zipped — Aspire / VCarve /
    Carbide Create Pro import these as reliefs), `depth_to_gcode` (GRBL 3-axis raster finishing for a FLAT end mill:
    `cutter_floor` = T - minimum_filter(D, disc of the cutter radius) is the gouge-free cutter-centre height; step-down
    layers, alternating X runs every `stepover_mm`, Z emitted only where it changes > 0.05 mm so a flat floor is one
    G1; Z = 0 at the FOAM TOP, origin near-left; header/footer G21 G90 G17 G94 / M3 S / M5 M30; stats in
    `X-Gcode-Stats`). `POST /api/relief` = the /api/layout body + `relief{...}` + `format` grid | stl | png | gcode;
    `grid` returns the carved surface as a heightfield payload with z <= 0 for the 3D preview.
    UI (Design insert): "Pocket style" select in Pocket fit (flat | form-fit 3D) with smoothing / depth clearance /
    grid fields; per-tool "Pocket style" override in the selected-tool panel (scanned tools only); the 3D view
    (`Layout3D` `reliefGrid` prop) draws the carved surface + slab + outer walls instead of the flat block and keeps
    only the pocket edge loops; Export panel gains "Form-fit 3D cutting files" with cutter / stepover / step-down /
    feed / plunge / safe Z / spindle and the three downloads, plus the G-code stats line. `LayoutSettings` grew
    `pocket_style`, `relief_*`, `cnc_*`; `Tool.pocket_style`. Tests: `tests/test_relief.py` (depth map follows a
    rotated+offset scanned block to 13 / 21 mm with the flat square at exactly 10; depth cap and no-scan fallback;
    STL watertight with the right volume; PNG mode I;16; G-code never below the modelled depth, inside the block,
    3 layers at 8 mm step-down; endpoint grid/gcode/png). FIRST REAL RUN on `6b9814466751` (24 tools, 805 x 434,
    50 mm foam, 1 mm grid, 6 mm cutter, 2 mm stepover, 8 mm step-down): the depth map built in ~3 s, G-code 1.6 MB /
    98k lines / 6 layers / 5,736 runs / 221 m of cutting / ~4.4 h at 1500 mm/min — a full form-fit of a whole drawer
    is a long job; a bigger stepover or a bigger cutter for the deep flat bodies is the lever. GOTCHA fixed on that
    run: the layout's depth RULE (measured minus 3 mm) made every tool taller than the foam a THROUGH cut in relief
    mode; relief pockets now honour only `depth_override_mm` (the per-tool override the UI sends separately) and
    otherwise cap at thickness - floor_min. Not done: a ball-nose cutter model (the gouge rule is the
    flat-mill one), roughing passes (every layer rasters the whole pocket), tool-change / multiple cutters.
  - **ONE-CLICK FLOW (2026-10-03, Nolan: "How make this workflow easier and improve the UI so its easier, faster, and
    better" / "Right now the flow seems to be the same as before")**. The admin's steps are unchanged (phone captures
    already land on Outlines); what changed is how many clicks a drawer takes:
    * Outlines: "⚡ Prepare drawer" (top of the Tools panel, only for scans with height data) = detect if nothing is
      detected -> run the agent -> ACCEPT its proposal -> open Design insert, hands-off (`prepareDrawer`, with
      `autoAcceptRef` consumed in `followAgent`; `applyAgentResult(res, jobId)` is the accept logic, dismissing the
      job on the server). `?prepare=1` on the URL runs it on arrival, so the phone can open a fresh capture straight
      into the pipeline. Undo still works afterwards.
    * Design insert: `pocket_style` defaults to **relief** (form-fit 3D) and the view opens in 3D when it is; the
      Export panel puts "Form-fit 3D cutting files" FIRST with ONE primary button, "Download CNC package (zip)" =
      `POST /api/relief format=package`: `<name>_relief.nc`, `<name>_relief.stl`, `<name>_depth16.png/.json`, the
      flat `<name>.svg/.dxf`, and `JOB_SHEET.txt` (block, pocket count by style, deepest cut, every CNC setting,
      the G-code stats, Z/origin convention, per-tool depth ranges). The single-file buttons remain underneath; the
      standalone "Preview 3D" (`fetchStl`) loads the CARVED STL when form-fit is on. `PackageTest` checks the zip.
    Felt-speed item still open: the phone upload/build has no progress (job id + "rectifying 40/60").
  - **THE FLOW IS NOW SCAN -> PLACE -> NEGATIVE (2026-10-03, Nolan: "the outlining and agent steps are now unnecessary.
    Please remove them. We are simply taking the height map, finding tools (by seeing what is extruding from floor),
    and placing them. Then taking the negative to determine what the foam should look like.")** `app/page.tsx`:
    a calibrated capture (or a photo after Calibration) goes through `findTools` — `autoDetect` in `height` mode,
    topo edges, no SAM, 2 mm threshold, min 2 cm2 — every tool gets `pocket_style: 'relief'` and the page lands on
    **Design insert**. The "Trace & refine" nav button, the Outlines step render and the "Refine outline on scan"
    link are GONE from the UI; `components/OutlineStep.tsx`, `ScanViewer.tsx`, the cleanup/agent endpoints and their
    tests are still in the repo (unused by the UI; delete when sure). Design insert has "↻ Re-find tools from scan"
    (`onRedetect`) and hides the flat "Pocket depth" rule section in form-fit mode. The tool outline is only the
    FOOTPRINT for placement, clearance and the flat fallback; the foam comes from the height-map negative
    (`relief.py`). Merged blobs (hammer + tape) are one movable item — that is fine for a negative; Ctrl-drag split
    on the sheet still separates them if they must move apart.
  - **"ONLY 30 OF THE FRAMES WORKED" (2026-10-03, capture `a237eba87dba`, 65 front frames + 1 overview) — THREE causes,
    found in order by instrumenting `register_depth_subset` offline (`processed_cache.load_session = lambda d: None`
    bypasses the snapshot; `TC_NO_WARM=1 python diag.py`):**
    1. THE PHONE DECLARED 50 mm MARKERS, THE DRAWER HAD THE NEW 25 mm ONES. Depth frames measured them at 25.1 mm.
       The overview photo (markers-only rectification) was therefore built at 2x scale and the drawer came out
       598 x 800 for a ~299 x 400 drawer; nothing metric could line up with it (the cluster-to-overview fit had a
       17 mm residual, 0.4 mm after the fix). `_build_multi_session` now measures the marker size from the depth
       frames and, when it disagrees with the declared size by > 15 %%, logs a WARNING, snaps to 25 / 50 and
       RE-RECTIFIES the photo-only frames with it (`scan_meta.marker_size_mm` / `marker_size_declared_mm`). The
       upload default in `app.py` is still 50 — the iOS setting is what sends it; check the app's `markerSizeMm`.
    2. `drawer_corners` needed >= 3 pooled markers to accept a registered cluster. A close sweep down a big drawer
       sees ONE marker for a long stretch, so a perfectly good 9-frame cluster was discarded whole. Now a cluster with
       fewer markers is anchored to the OVERVIEW's rectangle through whatever markers they share (rigid Kabsch fit on
       the 4 corners per marker, accepted at median residual <= 4 mm; `reference_index` = the frame with
       drawer_corners). And `register_rgbd` solved only the component containing the global anchor — ONE frame here
       while 32 matched pairs sat in other groups; `_solve_components` now solves every connected group with its own
       anchor and `register_depth_subset` anchors each (info: components / anchored_components / unanchored_components).
       Frames the registration cannot place are no longer SKIPPED: they fall back to markers / neighbour matching
       (`_register` used to skip every depth frame outside `visual_corners` once any registration "applied").
    3. Matching was starved and then over-strict: photo SIFT found 4-50 features on half the frames (dark liner) —
       CLAHE before SIFT, contrast .012, plus RELIEF FEATURES from the frame's orthographic height raster masked to
       raised areas (`height_features`, tagged in a 129th descriptor dimension so colour and relief never cross-match);
       and `_match` rejected pairs with 60-72 RANSAC inliers because noise features dragged the inlier RATIO under
       35 %% — absolute support (>= 25 inliers) now suffices, hull/spread relaxed at >= 12 inliers, p90 2.0 -> 2.5 mm.
    Result on that capture: drawer 598 x 800 -> 299 x 400 (correct), depth-registered frames 0 -> 23 (10 components,
    4 anchored), adjacent-pair matches 14 -> 21, frames used 30 -> 32. The rest are bare liner / featureless smooth
    objects (a zipper case, a black box, a pencil — frames 35 and 50 LOOKED AT) with no pose: unplaceable without
    ARKit, and carrying no tool information anyway; 7 unplaced frames hold >= 15 cm2 of raised object. Stitching
    tests (15) unchanged. The stale `processed-*` snapshot of that capture was deleted so the server rebuilds it.
  - **"PLEASE IMPROVE THE RESULT ON THE LATEST SCAN" (2026-10-03, `a237eba87dba`, a DESK drawer: Rubik's cube, pill
    bottle, Anker power bank, zipper case, USB stick, pencil, pin box, WD drive, small screwdriver).** After the
    three registration fixes the mosaic was right but the HEIGHT MAP covered only the upper-left objects. Two more
    causes, both measured:
    1. 34 of 65 depth frames had no markers and no photo match (bare liner / smooth black things) — but every frame
       has an ARKit pose. The ABSOLUTE front-camera pose is unusable (mirrored image, drifting world), yet the
       RELATIVE step between adjacent frames matched the registered placements to a 2.7 mm median over 19 pairs once
       x is MIRRORED and the world yaw fitted. `_chain_by_relative_pose` fits mirror + yaw per capture from frames
       placed by depth registration or 2+ markers ONLY (`placed_by_rgbd` / markers — including one-marker and
       photo-matched frames took the fit from 2.7 to 11-15 mm), refuses to chain if the fit is > 8 mm, then places
       each unplaced frame from its nearest placed neighbour (<= 6 frames) with rank 2 so the height refinement may
       still correct it. Result: 66 / 66 frames placed (fit 2.0 mm median, p90 5 mm). `_pose_raw` reads the ARKit
       matrix for every sensor; `NO_POSE_SENSORS` still blocks the absolute path. Rank counting was also wrong since
       the depth registration landed (`if (visual_corners or ...)` made every frame rank 0) — fixed, `placed_by`
       now reports `pose_only` honestly.
    2. TRUEDEPTH RETURNS NO DEPTH ON BLACK / GLOSSY SURFACES: 22 %% of this drawer's cells had none, exactly the WD
       drive, the Anker and the zipper case; the matte drive even reads at FLOOR height where it does return.
       Height-only detection found 8 of 9 objects and missed the drive. New detection mode **`hybrid`**
       (`_detect_hybrid`): the topo height blobs PLUS HQ-SAM photo silhouettes that overlap them < 30 %% and are
       either measurably tall (p95 >= 1.5 mm) or >= 8 cm2 (a big flat-looking silhouette is a dark object the sensor
       did not see); every tool carries `depth_coverage`. The new flow (`findTools`) uses `hybrid` when the SAM model
       is available. Relief (`_place_height`) fills a tool's unknown cells from its measured parts (p75) when
       >= 200 cells or 3 %% are measured (10 %% left the Anker at 9.8 %% as a 1 mm scratch), and a tool whose
       measured footprint is < 10 %% raised falls back to the TYPED thickness + clearance (else the flat rule) and is
       reported as "flat (no usable depth)" — Design insert warns when `depth_coverage` < 50 %% and asks for the
       thickness. On this drawer: 8 -> 15 detections (4 from the photo: drive, power bank, case, pin box), and the
       depth map now shows every object. Lesson: on a desk drawer the limiting sensor is TrueDepth's IR on black,
       not registration — the photo has to carry those objects and the person types a thickness when the sensor
       read nothing.
  - **CHAMFERED CORNERS AND STAIRCASE WALLS (2026-10-03, Nolan: "Even when I draw a square, the corners aren't
    perfectly rounded ... the cutouts in the foam are quantized and don't look continuous").** Two separate causes.
    (1) `geometry.process_outline` ran EVERY tool — drawn primitives included — through morphological smoothing, an
    along-contour Gaussian and the LEGACY `straighten_ring` (coarse tolerance 1.5 mm): a 4 mm-radius corner has a bow
    of 1.17 mm, so it was "straight" and became a CHAMFER; the same rule flattened every gentle arc on scanned tools
    into chords ("making my other objects look weird"). Now `exact=True` (sent as `source: 'shape'` / `shape` by
    `buildLayoutBody`) skips all of that for drawn shapes — only the clearance offset (round joins, resolution 24)
    and a 0.1 mm simplify — and scanned tools go through `clean_ring` (bow-judged lines/arcs at 0.3-0.5 mm) instead
    of straighten_ring. Test: a drawn r = 4 rounded rect comes out of /api/layout within 0.25 mm of the true arc.
    (2) `relief._rasterise` was a binary mask on the grid, so every pocket wall was a staircase along the grid axes
    and `depth_to_stl` took the MAX of neighbouring cells for corner heights, which made the steps vertical. The
    outline is now rasterised at 4x and averaged (`SUPERSAMPLE`): edge cells carry fractional coverage and get
    `depth * coverage` (flat pockets and the no-depth fallback) or the nearest inside depth * coverage (relief), and
    mesh corners are the MEAN of the four cells — a one-cell ramp that follows the outline exactly, continuous at
    any angle. The 3D preview grid is 1 mm on inserts <= 0.15 m2 (1.5 to 0.3 m2, else 2). Test: a 30-degree pocket
    has hundreds of intermediate-depth edge cells and max depth exactly its depth.
    (3) Switching the layout to `clean_ring` exposed a third: its straightness test judged the WHOLE run, and the
    ends of a run carry the rounded corner (clearance offsets round by their radius), so a 60 mm square side with
    2.5 mm-round ends had a 0.9 mm bow, read as "other" and came out as a 3-point bowed curve. Straightness is now
    judged on the run's CORE (up to `CORNER_ROUND_MM` = 3 mm trimmed at each end of runs >= 12 mm); the line is fitted
    to the core and neighbouring lines meet at their intersection — a smoothed, offset 60 x 60 square now returns
    exactly 4 vertices at 62.00 x 62.00. Checked with `TC_CLEAN_DEBUG=1` ("runs: line 60mm->1pt" x 4).
  - **RUGGED POCKET WALLS (2026-10-03, Nolan: "there are still just so many shapes that are rugged and not clean",
    with a 3D screenshot of terraced pocket walls).** Two causes. (1) A relief pocket's rim took the SCANNED MASK's
    edge depth cell by cell, so the sensor's noisy edge ramp became horizontal terraces along every wall.
    `relief._relief_pocket_depth` now gives each pocket a WALL PROFILE: at each rim point (every ~1 mm along the
    pocket contour) the p75 of the tool's relief within `wall_band_mm` (5) + 3 mm inside, gap-filled and smoothed
    along the contour (sigma 8 mm, circular), assigned to every cell within 5 mm of the rim and blended into the
    interior relief over 3 mm — one smooth vertical wall that follows the tool's local height (a hammer head's wall
    is deeper than its handle's, smoothly). Interior holes and the clearance ring fill from the nearest valid cell.
    (2) The 3D preview was FLAT-SHADED: the relief grid carried a `valid` mask, and `heightfieldGeometry` then drops
    triangles and goes non-indexed, so every grid cell rendered as its own facet. `getReliefGrid` strips `valid`;
    the carved block is one indexed, smooth-normal surface. Tests unchanged (relief 7 OK).
  - **CLEANING THE 3D SHAPES (2026-10-03, Nolan: "i think i may need to use some sort of AI model to 'clean' the 3D
    shapes of the scanner objects/tools" → "okay do it").** Agreed order: deterministic, measurable steps first, the
    model only NAMING the solid; a learned depth refiner later for black objects; no image-to-3D generation (plausible
    is not millimetre-true). `toolcutter/solids.py` `clean_relief(h, inside, res_mm, guide, solid_hint, symmetric_hint)`
    runs on a tool's placed height raster before the wall profile: (1) a GUIDED FILTER (He et al., own numpy/cv2
    implementation — OpenCV here has no ximgproc) with the rectified PHOTO as guide, normalised to the footprint
    (`_place_height(..., guide_src)` warps the grey photo through the same remap): flat where the photo is flat, crisp
    at photo edges; (2) MIRROR SYMMETRY about the long axis (minAreaRect frame) when the halves agree within
    `SYM_TOL_MM` 0.8 over >= 85 %%; (3) PRIMITIVES fitted on the footprint eroded by 2 mm — box (p85 top), lying
    cylinder (least-squares circle through the column-median profile, 3-150 mm radius), extrusion (the column-median
    profile itself) — accepted only when p90 |h - fit| <= `FIT_TOL_MM` 0.8 (box > cylinder > extrusion), or
    <= `SEMANTIC_TOL_MM` 1.5 when the model asked for that solid; the primitive replaces the eroded core, the sensor's
    edge ramp stays so the wall profile still finds the rim. Report per tool: `solid`, `solid_residual_mm`,
    `clean_steps` (shown in the Pocket fit hint: "Fitted as solids: 3 box, 1 cylinder lying").
    `recognize.solid_class` (one light structured call: box | cylinder_lying | cylinder_upright | extrusion | freeform
    + symmetric) is cached per (session, tool, mask area) in `app._SOLID_CACHE`; `_solid_hint_for_tool` builds the
    crop; off when no key. Settings: `relief.clean_solids` / `relief.semantic` (UI: "Clean solids", "Let the model
    name each solid", both default on). Tests `tests/test_solids.py` (6): noisy box -> box (top within 0.5, core
    std < 0.05), lying cylinder -> radius within 0.8, wavy asymmetric -> freeform with no symmetry, symmetric handle
    halves averaged, hinted box beats the automatic rule on 0.9 mm noise, guided filter keeps a photo step while
    smoothing. A flat extrusion profile can pass as "extrusion" on a box — same geometry, different label.
    FIRST REAL RUN (desk drawer, 14 model calls, 58 s sequential -> ~20 s with the 4-thread prefetch in `relief_export`):
    the model named every object right (Rubik's cube box 0.95, pencil cylinder 0.90, Anker box, pin box box 0.82,
    pill bottle cylinder lying) but a flat 1.5 mm semantic tolerance refused most of them — their depth is hole-filled
    sensor garbage (pin box fit residual 9.3 mm, bottle 36 mm). Two rules added: `semantic_tolerance(confidence)`
    (4.0 mm at >= 0.85, 2.5 at >= 0.7, else 1.5) and CONSTRUCTED primitives — a confident (>= 0.8) box / upright
    cylinder with a box-shaped footprint (fill >= 0.75) becomes a slab at the measured p85 top even when no fit passes;
    a confident lying cylinder with a bar-shaped footprint (fill >= 0.7, aspect >= 1.4) becomes a half-cylinder of
    radius = half the footprint width, centre at or below the floor. Reported as `by: model (constructed)`. Desk
    drawer now: cube, power bank, pin box -> box; pencil, bottle -> lying cylinder; case, pouch fragments -> freeform.
    NEXT (option 2): Depth Anything V2 / Marigold relative depth, scale+offset fitted per tool to the metric map,
    to fill what TrueDepth cannot see on black objects.
  - Admin API base is `NEXT_PUBLIC_API_BASE_URL` (default `http://localhost:8000`), see `apps/admin/lib/api.ts`.
    Analytics must never break the page: `trackEvent` swallows errors; the client beacon is fire-and-forget.
    The `frontend/` folder was renamed to `apps/admin/` on 2026-09-18 — old paths in notes mean that.
