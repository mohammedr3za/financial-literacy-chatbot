#!/usr/bin/env bash
export VLLM_USE_FLASHINFER_SAMPLER=0 PATH=$HOME/vllm-env/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:$LD_LIBRARY_PATH HF_HUB_ENABLE_HF_TRANSFER=1
exec ~/vllm-env/bin/vllm serve Qwen/Qwen3.6-27B-FP8 --host 127.0.0.1 --port 8000 --served-model-name qwen3.6-27b \
  --max-model-len 16384 --max-num-seqs 16 --max-num-batched-tokens 4096 --gpu-memory-utilization 0.92 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 --language-model-only
