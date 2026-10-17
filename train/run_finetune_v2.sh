#!/usr/bin/env bash
# Fine-tune on a fresh Lambda A100, from the Mac:  [V=v3] train/run_finetune_v2.sh <ip> setup|train|fetch
#   setup : install uv + an Unsloth training env on the box (about 5-10 min)
#   train : upload finetune.py + train/data/sft_v2_final.jsonl and start training DETACHED (nohup; survives ssh drops)
#   fetch : download the GGUFs (Q5_K_M, Q4_K_M), LoRA and log, and register finbot-8b-v2 with Ollama (Q5_K_M)
# Settings live in train/finetune.py (v2 defaults: 12k tokens, LoRA r32/a32, lr 1e-4, best checkpoint by eval loss).
set -euo pipefail
IP=${1:?usage: train/run_finetune_v2.sh <ip> setup|train|fetch}
STEP=${2:?setup|train|fetch}
V=${V:-v2}                                   # V=v3: out_v3, train_v3.log, finbot-8b-v3
DATA=${DATA:-train/data/sft_${V}_final.jsonl}
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
  [ -s "$DATA" ] || { echo "$DATA missing: run prepare_sft first"; exit 1; }
  RUN=$(mktemp)
  cat > "$RUN" <<EOS
#!/usr/bin/env bash
export HF_HUB_ENABLE_HF_TRANSFER=1 LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:\${LD_LIBRARY_PATH:-}
cd ~/finbot-train
~/train-env/bin/python finetune.py --base Qwen/Qwen3-8B --data $(basename "$DATA") --out out_$V
echo EXITED \$?
EOS
  rsync -az train/finetune.py train/sft_format.py "$DATA" "ubuntu@$IP:~/finbot-train/"
  rsync -az "$RUN" "ubuntu@$IP:~/finbot-train/run_$V.sh" && rm -f "$RUN"
  $SSH "cd ~/finbot-train && chmod +x run_$V.sh && { nohup ./run_$V.sh > train_$V.log 2>&1 < /dev/null & } && echo started"
  ;;
fetch)
  mkdir -p train/out_$V
  rsync -az --partial --include "*Q5_K_M*.gguf" --include "lora/***" --include "*/" --exclude "*" \
        "ubuntu@$IP:~/finbot-train/out_$V/" train/out_$V/
  rsync -az "ubuntu@$IP:~/finbot-train/train_$V.log" train/out_$V/
  GGUF=$(find train/out_$V -name "*Q5_K_M*.gguf" | head -1)
  [ -n "$GGUF" ] || { echo "No Q5_K_M GGUF found; check train_$V.log"; exit 1; }
  ollama show qwen3:8b --modelfile | sed "s|^FROM .*|FROM $PWD/$GGUF|" > train/out_$V/Modelfile
  ollama create finbot-8b-$V -f train/out_$V/Modelfile && echo "registered finbot-8b-$V"
  ;;
esac
