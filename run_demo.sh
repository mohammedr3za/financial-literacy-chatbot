#!/usr/bin/env bash
# Start FinBot for the booth. Works offline once models and dependencies are installed.
set -euo pipefail
cd "$(dirname "$0")"
MODEL=$(grep -E '^FINBOT_MODEL=' .env 2>/dev/null | cut -d= -f2- || true)
MODEL=${MODEL:-$(uv run --offline python -c "from finbot.config import get_settings; print(get_settings().model)")}
CTX=$(uv run --offline python -c "from finbot.config import get_settings; print(get_settings().num_ctx)")

if ! curl -sf localhost:11434/api/version >/dev/null; then
  echo "Starting Ollama…"; brew services start ollama >/dev/null
  for _ in $(seq 1 30); do curl -sf localhost:11434/api/version >/dev/null && break; sleep 1; done
fi
echo "Warming up $MODEL (context $CTX) and the embedding model…"
curl -sf localhost:11434/api/generate -d "{\"model\":\"$MODEL\",\"prompt\":\"hi\",\"stream\":false,\"keep_alive\":\"3h\",\"options\":{\"num_ctx\":$CTX,\"num_predict\":1}}" >/dev/null
curl -sf localhost:11434/api/embed -d '{"model":"bge-m3","input":"warm up","keep_alive":"3h"}' >/dev/null
echo "Opening FinBot…"
exec uv run --offline streamlit run app.py --server.address localhost --browser.gatherUsageStats false
