#!/usr/bin/env bash
# M6 training on the Lambda A100, from the Mac:  train/run_finetune.sh <ip>
# 1) filter the generated data against every test set, 2) stop vLLM to free the GPU, 3) LoRA-train Qwen3-8B with Unsloth,
# 4) export GGUF Q4_K_M, 5) download it and register it with Ollama as finbot-8b.
set -euo pipefail
IP=${1:?usage: train/run_finetune.sh <instance-ip>}
cd "$(dirname "$0")/.."
SSH="ssh -o BatchMode=yes -o ServerAliveInterval=30 ubuntu@$IP"
uv run python train/filter_dataset.py train/data/sft.jsonl train/data/sft_clean.jsonl --external /Users/reza/MS3610/mcgraw/dataset/problems.jsonl
rsync -az train/finetune.py train/data/sft_clean.jsonl "ubuntu@$IP:~/finbot-train/"
$SSH 'pkill -f "vllm serve" || true; sleep 8; nvidia-smi --query-gpu=memory.used --format=csv,noheader'
$SSH 'cd ~/finbot-train && export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:$LD_LIBRARY_PATH HF_HUB_ENABLE_HF_TRANSFER=1 && \
      ~/train-env/bin/python finetune.py --base Qwen/Qwen3-8B --data sft_clean.jsonl --out out --epochs 2 --max-seq 4096 2>&1 | tee train.log | \
      grep -E --line-buffered "loss|eval|Saved|Error|Traceback|train=" '
mkdir -p train/out/gguf
rsync -az --include "*Q4_K_M*.gguf" --include "*/" --exclude "*" "ubuntu@$IP:~/finbot-train/out/" train/out/
GGUF=$(find train/out -name "*Q4_K_M*.gguf" | head -1)
[ -n "$GGUF" ] || { echo "No GGUF found; check ~/finbot-train/train.log on the instance"; exit 1; }
ollama show qwen3:8b --modelfile | sed "s|^FROM .*|FROM $PWD/$GGUF|" > train/Modelfile
ollama create finbot-8b -f train/Modelfile && echo "registered finbot-8b with Ollama"
