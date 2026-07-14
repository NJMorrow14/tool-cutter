# tool-cutter

Turns overhead photos of tools into SVG outlines (e.g. for cutting foam shadow-board
inserts). A Flask backend wraps Meta's HQ-SAM segmentation to mask the tool and trace
it to an SVG; a small Next.js frontend handles upload, mask preview, and SVG export.
Status: prototype, local tool.

## Stack

- Backend: Python 3, Flask, PyTorch + vendored [sam-hq](https://github.com/SysCV/sam-hq) (HQ-SAM)
- Frontend: Next.js 14 (app router), React 18, TypeScript

## Layout

| Path | What |
| --- | --- |
| `backend/` | Flask API (`app.py`), SVG tracing (`tool_image_to_svg.py`), param sweep script |
| `backend/sam-hq/` | Vendored upstream HQ-SAM repo — do not modify |
| `frontend/` | Next.js UI; API client in `lib/api.ts` |

## Run locally

Backend (serves on port 8000 by default):

```
cd backend
python3 app.py --host 0.0.0.0
```

Requires a HQ-SAM checkpoint (`*.pth`, gitignored — download separately, e.g.
`sam_hq_vit_h.pth`). Point to it with the `HQSAM_CKPT` env var or per-request
`hqsam_checkpoint`. Python deps: `pip install -r requirements.txt` (a `.venv`
exists at the repo root).

Frontend:

```
cd frontend
npm install
npm run dev
```

The frontend calls `http://localhost:8000` by default; override with
`NEXT_PUBLIC_API_BASE_URL`.

## Deploy

Not deployed — local tool.
