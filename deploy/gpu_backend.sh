#!/usr/bin/env bash
# Deploy the FinBot API on the Lambda GPU box (SPEC 7.2). Run from the Mac:  deploy/gpu_backend.sh <ip>
# Needs: vLLM already serving on the box (~/start_vllm.sh), FINBOT_API_TOKEN in .env.
# Only redistributable documents are in kb/index (SPEC 7.2); private books must never be indexed before running this.
set -euo pipefail
IP=${1:?usage: deploy/gpu_backend.sh <instance-ip>}
cd "$(dirname "$0")/.."
set -a; . ./.env; set +a
SSH="ssh -o BatchMode=yes -o ConnectTimeout=20 ubuntu@$IP"

echo "== copy code and the public index =="
rsync -az --delete --exclude .venv --exclude .git --exclude 'kb/raw' --exclude results --exclude web --exclude .env --exclude 'train/out*' --exclude '*.gguf' \
      --exclude '__pycache__' --exclude '.pytest_cache' ./ "ubuntu@$IP:~/finbot/"

echo "== remote setup =="
$SSH "FINBOT_API_TOKEN='$FINBOT_API_TOKEN' bash -s" <<'REMOTE'
set -e
export PATH=$HOME/.local/bin:$PATH
cd ~/finbot && uv sync -q --python 3.12
# Embeddings (bge-m3) on the CPU so the GPU stays with the main model.
if ! command -v ollama >/dev/null; then curl -fsSL https://ollama.com/install.sh | sh >/dev/null 2>&1; fi
sudo mkdir -p /etc/systemd/system/ollama.service.d
printf '[Service]\nEnvironment="CUDA_VISIBLE_DEVICES="\n' | sudo tee /etc/systemd/system/ollama.service.d/cpu.conf >/dev/null
sudo systemctl daemon-reload && sudo systemctl restart ollama
for i in $(seq 1 30); do curl -sf localhost:11434/api/version >/dev/null && break; sleep 1; done
ollama pull bge-m3 >/dev/null 2>&1
# The API: localhost only, GPU model, no local fallback on this box.
pkill -f "uvicorn finbot.api" 2>/dev/null || true
cat > ~/start_api.sh <<EOS
#!/usr/bin/env bash
cd ~/finbot
export FINBOT_BACKEND=openai FINBOT_BASE_URL=http://127.0.0.1:8000/v1 FINBOT_MODEL=finbot FINBOT_FALLBACK_MODEL=none FINBOT_CHATLOG=0 \
       FINBOT_EMBED_URL=http://127.0.0.1:11434 FINBOT_API_TOKEN=$FINBOT_API_TOKEN FINBOT_THINKING=off
exec ~/.local/bin/uv run uvicorn finbot.api:app --host 127.0.0.1 --port 8080 --workers 1
EOS
chmod 700 ~/start_api.sh
nohup ~/start_api.sh > ~/api.log 2>&1 < /dev/null &
for i in $(seq 1 60); do curl -sf localhost:8080/api/health >/dev/null && break; sleep 1; done
echo "api health: $(curl -s localhost:8080/api/health)"
# HTTPS address via a Cloudflare quick tunnel (no open ports on the instance).
if ! command -v cloudflared >/dev/null; then
  curl -fsSL -o /tmp/cloudflared.deb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
  sudo dpkg -i /tmp/cloudflared.deb >/dev/null
fi
pkill -f "[c]loudflared tunnel --no-autoupdate.*--url" 2>/dev/null || true
: > ~/cloudflared-empty.yml   # Lambda images run their own cloudflared whose /etc/cloudflared/config.yml would 404 everything
nohup cloudflared tunnel --no-autoupdate --config ~/cloudflared-empty.yml --url http://127.0.0.1:8080 > ~/tunnel.log 2>&1 < /dev/null &
for i in $(seq 1 40); do URL=$(grep -oE "https://[a-z0-9-]+\.trycloudflare\.com" ~/tunnel.log | head -1); [ -n "$URL" ] && break; sleep 1; done
echo "TUNNEL_URL=$URL"
REMOTE
