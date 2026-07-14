# CLAUDE.md

Tool-outline-to-SVG app: Flask + HQ-SAM backend, Next.js frontend.

- Stack: Python/Flask + PyTorch (backend), Next.js 14 + TypeScript (frontend)
- Run: `cd backend && python3 app.py --host 0.0.0.0` (port 8000); `cd frontend && npm run dev`
- Deploy: not deployed — local tool
- Gotchas:
  - `backend/sam-hq/` is vendored upstream code (SysCV/sam-hq) — do not modify or
    reformat it; it makes the repo large, so keep searches scoped when possible.
  - Model checkpoints (`*.pth`) are gitignored and must be downloaded separately;
    backend reads `HQSAM_CKPT` (default model type `vit_h`).
  - Frontend API base is `NEXT_PUBLIC_API_BASE_URL` (defaults to `http://localhost:8000`),
    see `frontend/lib/api.ts`.
