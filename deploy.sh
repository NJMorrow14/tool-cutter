#!/bin/zsh
# One-command deploy, same pattern as every project on the ThinkCentre:
# rsync source to prod-apps, build there, restart, migrate, verify. No local Docker.
# NOT RUN YET — see DEPLOYMENT.md for the one-time setup (env files, model checkpoint, Cloudflare Access + tunnel).
set -euo pipefail

HOST=prod-apps
DEST='~/deploy/tool-cutter'
WEB_PORT=8086
ADMIN_PORT=8087
API_PORT=8088

echo "==> rsync source to $HOST"
rsync -az --delete \
  --exclude '.git' --exclude 'node_modules' --exclude '.next' --exclude '.venv' --exclude '__pycache__' \
  --exclude 'backend/*.pth' --exclude 'backend/tests/out' --exclude 'ios' --exclude 'mac/Photogrammetry/.build' \
  --exclude 'env/*.env' --exclude 'models' \
  ./ "$HOST:$DEST/"

echo "==> build + restart on $HOST"
ssh "$HOST" "cd $DEST && docker compose -f compose.prod.yml up -d --build"

echo "==> run migrations + seed"
ssh "$HOST" "cd $DEST && docker compose -f compose.prod.yml exec -T tf-web \
  sh -c 'node scripts/migrate.mjs && node scripts/seed.mjs'"

echo "==> verify"
ssh "$HOST" "curl -sf -o /dev/null -w 'web:   HTTP %{http_code}\n' http://localhost:$WEB_PORT/ && \
             curl -sf -o /dev/null -w 'admin: HTTP %{http_code}\n' http://localhost:$ADMIN_PORT/login && \
             curl -sf -o /dev/null -w 'api:   HTTP %{http_code}\n' http://localhost:$API_PORT/health"
echo "deployed."
