#!/usr/bin/env bash
# Connect the public Vercel site to FinBot running on THIS Mac (local qwen3:8b). Keeps the Mac awake while the API runs.
# Stop with: pkill -f "uvicorn finbot.api"; pkill -f "cloudflared tunnel --no-autoupdate"
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; . ./.env; set +a
mkdir -p results
pkill -f "uvicorn finbot.api" 2>/dev/null || true
pkill -f "cloudflared tunnel --no-autoupdate --url http://127.0.0.1:8080" 2>/dev/null || true
sleep 1
nohup caffeinate -dimsu uv run --offline uvicorn finbot.api:app --host 127.0.0.1 --port 8080 > results/api.log 2>&1 &
for _ in $(seq 1 60); do curl -sf localhost:8080/api/health >/dev/null && break; sleep 1; done
curl -sf localhost:8080/api/health >/dev/null || { echo "API did not start; see results/api.log"; exit 1; }
nohup cloudflared tunnel --no-autoupdate --url http://127.0.0.1:8080 > results/tunnel.log 2>&1 &
URL=""
for _ in $(seq 1 60); do URL=$(grep -oE "https://[a-z0-9-]+\.trycloudflare\.com" results/tunnel.log | head -1 || true); [ -n "$URL" ] && break; sleep 1; done
[ -n "$URL" ] || { echo "Tunnel did not start; see results/tunnel.log"; exit 1; }
echo "tunnel: $URL"
cd web
npx --yes vercel env rm FINBOT_BACKEND_URL production --yes >/dev/null 2>&1 || true
printf '%s' "$URL" | npx --yes vercel env add FINBOT_BACKEND_URL production >/dev/null
npx --yes vercel deploy --prod --yes 2>&1 | grep -oE "Production URL: https://[^ \"]+" | head -1
