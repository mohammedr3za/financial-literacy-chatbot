#!/usr/bin/env bash
# finbot-blond on the fine-tuned finbot-8b-v3 (Ollama on the GPU, port 11435). The 27B's vLLM is stopped; its weights
# stay in the Hugging Face cache, so ~/use_27b.sh brings it back. Embeddings stay on the CPU Ollama (port 11434).
set -u
pkill -f "[v]llm serve" && echo "stopped vLLM (27B)"
for i in $(seq 1 60); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -lt 2000 ] && break; sleep 2; done
if ! curl -sf localhost:11435/api/version >/dev/null; then
  setsid nohup ~/start_ollama_gpu.sh > ~/ollama_gpu.log 2>&1 < /dev/null &
  for i in $(seq 1 30); do curl -sf localhost:11435/api/version >/dev/null && break; sleep 1; done
fi
export OLLAMA_HOST=127.0.0.1:11435
ollama list | grep -q "^finbot-8b-v3" || ollama create finbot-8b-v3 -f ~/Modelfile.v3 2>&1 | tail -1
curl -s localhost:11435/api/generate -d '{"model":"finbot-8b-v3","keep_alive":-1}' >/dev/null   # load it now
cp ~/start_api_v3.sh ~/start_api.sh
pkill -f "[u]vicorn finbot.api"; sleep 2
setsid nohup ~/start_api.sh > ~/api.log 2>&1 < /dev/null &
for i in $(seq 1 90); do curl -sf localhost:8080/api/health >/dev/null && break; sleep 1; done
echo "api: $(curl -s localhost:8080/api/health)"
