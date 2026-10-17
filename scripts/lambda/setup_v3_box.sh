#!/usr/bin/env bash
# finbot-local on Lambda: finbot-8b-v3 on Ollama (GPU) + the FinBot API + a quick tunnel. Run on the box (A10) after the
# code (~/finbot) and the GGUF (~/models/finbot-8b-v3.Q5_K_M.gguf) and ~/Modelfile.v3 are in place.
# usage: FINBOT_API_TOKEN=... bash setup_v3_box.sh
set -e
export PATH=$HOME/.local/bin:$PATH
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
cd ~/finbot && uv sync -q --python 3.12
# Ollama on the GPU: 6 parallel chats, models kept loaded
if ! command -v ollama >/dev/null; then curl -fsSL https://ollama.com/install.sh | sh >/dev/null 2>&1; fi
sudo mkdir -p /etc/systemd/system/ollama.service.d
printf '[Service]\nEnvironment="OLLAMA_NUM_PARALLEL=6" "OLLAMA_KEEP_ALIVE=-1" "OLLAMA_MAX_LOADED_MODELS=2"\n' | sudo tee /etc/systemd/system/ollama.service.d/finbot.conf >/dev/null
sudo systemctl daemon-reload && sudo systemctl restart ollama
for i in $(seq 1 30); do curl -sf localhost:11434/api/version >/dev/null && break; sleep 1; done
ollama pull bge-m3 >/dev/null 2>&1
sudo mkdir -p /usr/share/ollama/models-src && sudo cp ~/models/finbot-8b-v3.Q5_K_M.gguf ~/Modelfile.v3 /usr/share/ollama/models-src/
sed "s|^FROM .*|FROM /usr/share/ollama/models-src/finbot-8b-v3.Q5_K_M.gguf|" ~/Modelfile.v3 > ~/Modelfile.v3.box
ollama create finbot-8b-v3 -f ~/Modelfile.v3.box 2>&1 | tail -1
curl -s localhost:11434/api/generate -d '{"model":"finbot-8b-v3","keep_alive":-1}' >/dev/null
# the API (chat logs kept for v4 training data)
pkill -f "[u]vicorn finbot.api" 2>/dev/null || true
mkdir -p ~/chatlog
cat > ~/start_api.sh <<EOS
#!/usr/bin/env bash
cd ~/finbot
export FINBOT_BACKEND=ollama FINBOT_BASE_URL=http://127.0.0.1:11434 FINBOT_MODEL=finbot-8b-v3 FINBOT_COMPACT_TOOLS=on \
       FINBOT_THINKING=off FINBOT_EMBED_URL=http://127.0.0.1:11434 FINBOT_HOST=cloud FINBOT_CHATLOG_DIR=\$HOME/chatlog \
       FINBOT_API_TOKEN=$FINBOT_API_TOKEN
exec ~/.local/bin/uv run --no-sync uvicorn finbot.api:app --host 127.0.0.1 --port 8080 --workers 1
EOS
chmod 700 ~/start_api.sh
setsid nohup ~/start_api.sh > ~/api.log 2>&1 < /dev/null &
for i in $(seq 1 90); do curl -sf localhost:8080/api/health >/dev/null && break; sleep 1; done
echo "api health: $(curl -s localhost:8080/api/health)"
# HTTPS address via a Cloudflare quick tunnel (empty config: Lambda images ship their own cloudflared config)
if ! command -v cloudflared >/dev/null; then
  curl -fsSL -o /tmp/cloudflared.deb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
  sudo dpkg -i /tmp/cloudflared.deb >/dev/null
fi
pkill -f "[c]loudflared tunnel --no-autoupdate.*--url" 2>/dev/null || true
: > ~/cloudflared-empty.yml
setsid nohup cloudflared tunnel --no-autoupdate --config ~/cloudflared-empty.yml --url http://127.0.0.1:8080 > ~/tunnel.log 2>&1 < /dev/null &
for i in $(seq 1 40); do URL=$(grep -oE "https://[a-z0-9-]+\.trycloudflare\.com" ~/tunnel.log | head -1); [ -n "$URL" ] && break; sleep 1; done
echo "TUNNEL_URL=$URL"
