# ToolFoam Pro — Deployment

Same pattern as every project on the ThinkCentre: **one command, builds on the
server, verifies.** See `~/Projects/infrastructure/thinkcentre vms/`.
**Status: not deployed yet.** Everything below is prepared; nothing has been run against prod.

## What & where

| Service | Compose | Port (prod-apps) | Public URL |
|---------|---------|------------------|------------|
| Landing site (`apps/web`) | `tf-web` | 8086 | *(domain pending — e.g. toolfoampro.com)* |
| Admin / HQ (`apps/admin`) | `tf-admin` | 8087 | admin.\<domain\> *(Cloudflare Access-gated)* |
| Foam API (`backend/`, Flask + HQ-SAM) | `tf-api` | 8088 | api.\<domain\> *(Cloudflare Access-gated, admin-only)* |
| Postgres 16 | `tf-db` | internal | — |

- **Runs on:** `prod-apps` (192.168.1.61), app tier. Postgres co-located for MVP
  (volume `tf-pgdata`); promote to `prod-data` when the lead volume justifies it.
- The admin site is the whole foam workflow (capture → outlines → cutting files) plus
  analytics, walkthrough requests and the brochure/QR page. It talks to the Flask API
  from the browser, so the API hostname must also be reachable through the tunnel and
  behind the same Access policy.
- The iPhone app and the Mac photogrammetry CLI keep pointing at the API host.
- **HQ-SAM on the server is CPU-only** (the VM has no GPU): the vit_b checkpoint takes
  ~10–20 s per click there vs ~2 s on the Mac. Acceptable for occasional use; the Mac
  can keep running the backend locally for heavy sessions.

## First-time setup

1. On `prod-apps`: `mkdir -p ~/deploy/tool-cutter/{env,models}` and create
   `env/web.env` + `env/admin.env` from `.env.example`:
   - both: `DATABASE_URL=postgres://toolfoam:toolfoam@tf-db:5432/toolfoam`
   - admin: `ADMIN_PASSWORD=…`, `SESSION_SECRET=…` (long random)
   - web: `SESSION_SECRET=…` (salts the anonymous visitor hash)
   `deploy.sh` never rsyncs `env/*.env` or `models/`.
2. Copy the checkpoint: `scp backend/sam_hq_vit_b.pth prod-apps:~/deploy/tool-cutter/models/`.
3. Set the public URLs once in `~/deploy/tool-cutter/.env` on the server
   (`PUBLIC_WEB_URL=https://<domain>`, `PUBLIC_API_URL=https://api.<domain>`) — they are
   baked into the Next builds (QR target, API base).
4. Cloudflare (once a domain is chosen): tunnel ingress on prod-data `config.yml` →
   apex/www to `http://192.168.1.61:8086`, admin.\<domain\> → `:8087`,
   api.\<domain\> → `:8088`; proxied CNAMEs on the tunnel.
   **Create the Cloudflare Access applications for admin.\<domain\> and api.\<domain\>
   (allow NJMorrow14@gmail.com — reusable "Allow Nolan" policy) and verify the login
   wall BEFORE creating those DNS records.** The admin app has its own password wall;
   Access is the house rule on top. (The iPhone app will need an Access service token
   or a WARP login to reach api.\<domain\> — decide before printing anything.)
5. Print the QR code from the admin *Brochure & QR* page only after
   `PUBLIC_WEB_URL` is the final domain — the code encodes the URL literally.

## Deploy

```sh
./deploy.sh
```

Rsyncs source → builds the three images on the server → restarts → runs
`scripts/migrate.mjs` + `scripts/seed.mjs` (idempotent; seed loads
`brand/toolfoam-brochure.pdf` into the database if no brochure exists) → curl-verifies
all three ports.

## Verify / rollback

```sh
ssh prod-apps 'cd ~/deploy/tool-cutter && docker compose -f compose.prod.yml ps'
git checkout <sha> && ./deploy.sh   # rollback = redeploy older tree
```

## Local dev

```sh
docker compose -f compose.dev.yml up -d            # Postgres 16 on :5432 (toolfoam/toolfoam)
npm install                                        # workspaces: web + admin
DATABASE_URL=postgres://toolfoam:toolfoam@localhost:5432/toolfoam npm run migrate && npm run seed
cd backend && ../.venv/bin/python app.py --host 0.0.0.0 --preload   # :8000
npm run dev:admin                                  # :3000  (reads apps/admin/.env.local)
npm run dev:web                                    # :3001  (reads apps/web/.env.local)
```

With `ADMIN_PASSWORD` empty the admin login wall is off (local only).
