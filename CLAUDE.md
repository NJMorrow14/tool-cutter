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
  - Admin API base is `NEXT_PUBLIC_API_BASE_URL` (default `http://localhost:8000`), see `apps/admin/lib/api.ts`.
    Analytics must never break the page: `trackEvent` swallows errors; the client beacon is fire-and-forget.
    The `frontend/` folder was renamed to `apps/admin/` on 2026-09-18 — old paths in notes mean that.
