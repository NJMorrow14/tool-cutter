# CLAUDE.md

3D scans of laid-out tools (or single tool models) → foam insert cutting files (SVG sized to the
mat, DXF by depth, STL block with pockets). Flask backend, Next.js frontend. Images are NOT accepted
(removed on request 2026-09-16).

- Stack: Python 3.12 / Flask / OpenCV / shapely / trimesh+manifold3d / ezdxf / PyTorch (backend);
  Next.js 14 + TypeScript + three.js (frontend)
- Run: `cd backend && ../.venv/bin/python app.py --host 0.0.0.0 --preload` (port 8000);
  `cd frontend && npm run dev` (port 3000)
- Test: `cd backend && ../.venv/bin/python tests/smoke_test.py` — synthetic layout scan + tool model
  with known sizes through the whole API; must end with ALL CHECKS PASSED
- Deploy: not deployed — local tool
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
  - Uploads are classified layout vs. single object by `scan.classify_scan` (plane inlier fraction +
    whether off-plane points are stacked over plane points); form field `scan_kind` overrides.
  - Scan outlines use the "fraction of points above the mat" raster (`above_frac`), not the max
    height, to avoid growing thin tools by half a cell.
  - Frontend API base is `NEXT_PUBLIC_API_BASE_URL` (default `http://localhost:8000`), see
    `frontend/lib/api.ts`.
