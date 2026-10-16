#!/usr/bin/env bash
# Fine-tune v2 on a fresh Lambda A100, from the Mac:  train/run_finetune_v2.sh <ip> setup|train|fetch
#   setup : install uv + an Unsloth training env on the box (about 5-10 min)
#   train : upload finetune.py + train/data/sft_v2_final.jsonl and start training DETACHED (nohup; survives ssh drops)
#   fetch : download the GGUFs (Q5_K_M, Q4_K_M), LoRA and log, and register finbot-8b-v2 with Ollama (Q5_K_M)
# Settings live in train/finetune.py (v2 defaults: 12k tokens, LoRA r32/a32, lr 1e-4, best checkpoint by eval loss).
set -euo pipefail
IP=${1:?usage: train/run_finetune_v2.sh <ip> setup|train|fetch}
STEP=${2:?setup|train|fetch}
cd "$(dirname "$0")/.."
SSH="ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 ubuntu@$IP"
case "$STEP" in
setup)
  $SSH 'bash -s' <<'REMOTE'
set -e
export PATH=$HOME/.local/bin:$PATH
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
[ -d ~/train-env ] || uv venv -q ~/train-env --python 3.12
. ~/train-env/bin/activate && uv pip install -q unsloth unsloth_zoo trl datasets hf_transfer
# Latest torch wheels are built for CUDA 13; Lambda's driver is CUDA 12.8 -> NVIDIA's forward-compat libraries
if [ ! -d /usr/local/cuda-13.0/compat ]; then
  cd /tmp && wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
  sudo dpkg -i cuda-keyring_1.1-1_all.deb >/dev/null && sudo apt-get update -qq >/dev/null 2>&1
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq cuda-compat-13-0 >/dev/null 2>&1
fi
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}
python -c "import unsloth, torch; print('unsloth ok, torch', torch.__version__, 'cuda', torch.cuda.is_available())"
mkdir -p ~/finbot-train
REMOTE
  ;;
train)
  [ -s train/data/sft_v2_final.jsonl ] || { echo "train/data/sft_v2_final.jsonl missing: run prepare_sft first"; exit 1; }
  rsync -az train/finetune.py train/data/sft_v2_final.jsonl "ubuntu@$IP:~/finbot-train/"
  $SSH 'cd ~/finbot-train && cat > run_v2.sh <<EOS
#!/usr/bin/env bash
export HF_HUB_ENABLE_HF_TRANSFER=1 LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:\${LD_LIBRARY_PATH:-}
cd ~/finbot-train
~/train-env/bin/python finetune.py --base Qwen/Qwen3-8B --data sft_v2_final.jsonl --out out_v2
echo EXITED \$?
EOS
chmod +x run_v2.sh && nohup ./run_v2.sh > train_v2.log 2>&1 < /dev/null & echo started'
  ;;
fetch)
  mkdir -p train/out_v2
  rsync -az --include "*Q5_K_M*.gguf" --include "*Q4_K_M*.gguf" --include "lora/***" --include "*/" --exclude "*" \
        "ubuntu@$IP:~/finbot-train/out_v2/" train/out_v2/
  rsync -az "ubuntu@$IP:~/finbot-train/train_v2.log" train/out_v2/
  GGUF=$(find train/out_v2 -name "*Q5_K_M*.gguf" | head -1)
  [ -n "$GGUF" ] || { echo "No Q5_K_M GGUF found; check train_v2.log"; exit 1; }
  ollama show qwen3:8b --modelfile | sed "s|^FROM .*|FROM $PWD/$GGUF|" > train/out_v2/Modelfile
  ollama create finbot-8b-v2 -f train/out_v2/Modelfile && echo "registered finbot-8b-v2"
  ;;
esac
