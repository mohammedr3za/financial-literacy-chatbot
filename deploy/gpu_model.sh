#!/usr/bin/env bash
# Set up and start the FinBot GPU model on a fresh Lambda A100 (40 GB). Run from the Mac:  deploy/gpu_model.sh <ip>
# Lessons from 2026-10-16, all handled here:
#  - latest vLLM wheels need CUDA 13; Lambda's driver is CUDA 12.8 -> install NVIDIA's cuda-compat-13-0 and use its libraries
#  - default 256 concurrent sequences OOM with Qwen3.6's attention -> --max-num-seqs 16, --max-model-len 16384
#  - FlashInfer's sampler JIT-compiles with ninja/nvcc on first use -> VLLM_USE_FLASHINFER_SAMPLER=0 (PyTorch sampler)
set -euo pipefail
IP=${1:?usage: deploy/gpu_model.sh <instance-ip>}
ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new ubuntu@"$IP" 'bash -s' <<'REMOTE'
set -e
export PATH=$HOME/.local/bin:$PATH
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
[ -d ~/vllm-env ] || uv venv -q ~/vllm-env --python 3.12
. ~/vllm-env/bin/activate && uv pip install -q vllm hf_transfer ninja
if [ ! -d /usr/local/cuda-13.0/compat ]; then
  cd /tmp && wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
  sudo dpkg -i cuda-keyring_1.1-1_all.deb >/dev/null && sudo apt-get update -qq >/dev/null 2>&1
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq cuda-compat-13-0 >/dev/null 2>&1
fi
cat > ~/start_vllm.sh <<'EOS'
#!/usr/bin/env bash
export VLLM_USE_FLASHINFER_SAMPLER=0 PATH=$HOME/vllm-env/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:$LD_LIBRARY_PATH HF_HUB_ENABLE_HF_TRANSFER=1
exec ~/vllm-env/bin/vllm serve Qwen/Qwen3.6-27B-FP8 --host 127.0.0.1 --port 8000 --served-model-name qwen3.6-27b \
  --max-model-len 16384 --max-num-seqs 16 --max-num-batched-tokens 4096 --gpu-memory-utilization 0.92 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 --language-model-only
EOS
chmod +x ~/start_vllm.sh
pkill -f "vllm serve" 2>/dev/null || true
nohup ~/start_vllm.sh > ~/vllm.log 2>&1 &
until grep -qE "Application startup complete|Engine core initialization failed" ~/vllm.log; do sleep 10; done
grep -E "Application startup complete|Engine core initialization failed" ~/vllm.log | tail -1
REMOTE
