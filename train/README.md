# M6 runbook: fine-tune on a rented GPU (optional)

The fine-tune teaches FinBot's *habits*: calling calculators, citing sources, refusing guarantees, and referring personal decisions to a SEBI-registered adviser. Facts still come from document search. The fine-tuned model ships only if it matches or beats the base model on the benchmark (SPEC section 9).

## 1. Rent a GPU (user action)
- One GPU with **at least 48 GB**, such as an A100 80 GB, H100, or L40S. It needs room for the 35B teacher in FP8, then a 9B bf16 LoRA (about 22 GB).
- Providers include RunPod, Vast.ai, and Jarvislabs (Indian, INR billing). Pick an Ubuntu + CUDA image. Budget about 4–6 hours, and check the current hourly rate.
- Share the SSH host and port. Use a key, not a password.

## 2. Copy the project (from the Mac)
```bash
rsync -av --exclude .venv --exclude kb/raw --exclude results ./ user@gpu:/workspace/finbot/
```
`kb/index/` holds the chunk text the generator needs. It stays private and is deleted from the GPU box at the end.

## 3. Serve the teacher with vLLM (on the GPU box)
```bash
pip install -U uv && cd /workspace/finbot && uv sync && uv add langchain-openai
pip install -U vllm --pre    # Qwen3.5/3.6 need a recent or nightly vLLM
vllm serve Qwen/Qwen3.6-35B-A3B-FP8 --port 8000 --max-model-len 32768 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 &
```

## 4. Generate the training data
```bash
FINBOT_BACKEND=openai FINBOT_BASE_URL=http://localhost:8000/v1 FINBOT_MODEL=Qwen/Qwen3.6-35B-A3B-FP8 \
  uv run python train/make_dataset.py --n-kb 1000 --n-calc 500
```
Search embeddings come from Ollama (`FINBOT_EMBED_URL`, default `http://localhost:11434`). Install Ollama on the box and run `ollama pull bge-m3` before this step.

## 5. Train (stop vLLM first to free GPU memory)
```bash
pkill -f "vllm serve"
pip install -U unsloth unsloth_zoo "transformers>=5"
python train/finetune.py --base Qwen/Qwen3.5-9B --data train/data/sft.jsonl --out train/out
```
If the comparison in SPEC picked `qwen3:8b` instead, use `--base Qwen/Qwen3-8B`.

## 6. Bring it back to the Mac and register it with Ollama
```bash
rsync -av user@gpu:/workspace/finbot/train/out/gguf/ train/out/gguf/
ollama show qwen3.5:9b-q4_K_M --modelfile | sed "s|^FROM .*|FROM $(ls train/out/gguf/*Q4_K_M*.gguf)|" > train/Modelfile
ollama create finbot-9b -f train/Modelfile
```
Copying the base model's Modelfile keeps the chat and tool-calling template identical to training.

## 7. Compare and decide
```bash
uv run python bench/run_finbot.py --config rag_ft
```
Compare it with the `rag` results on calculation accuracy and the scored sample. Record both results in `PLAN.md`. Then delete `kb/index/` and `train/data/` from the GPU box and stop the instance.
