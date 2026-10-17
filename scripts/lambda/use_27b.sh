#!/usr/bin/env bash
# finbot-blond back on the 27B (vLLM, port 8000). Stops the GPU Ollama with finbot-8b-v3 (the model stays registered).
set -u
pkill -u "$(id -un)" -f "[o]llama serve" && echo "stopped GPU Ollama (v3)"
sleep 3
setsid nohup ~/start_vllm.sh > ~/vllm.log 2>&1 < /dev/null &
echo "loading the 27B (about 3-5 minutes)..."
for i in $(seq 1 120); do curl -sf localhost:8000/version >/dev/null && break; sleep 5; done
cp ~/start_api_27b.sh ~/start_api.sh
pkill -f "[u]vicorn finbot.api"; sleep 2
setsid nohup ~/start_api.sh > ~/api.log 2>&1 < /dev/null &
for i in $(seq 1 90); do curl -sf localhost:8080/api/health >/dev/null && break; sleep 1; done
echo "api: $(curl -s localhost:8080/api/health)"
