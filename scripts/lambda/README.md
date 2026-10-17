# Lambda scripts (finbot-blond and finbot-local on Lambda GPUs)

- `use_27b.sh` / `use_v3.sh` (on the A100 `finbot-serve`): switch finbot-blond between Qwen3.6-27B (vLLM, `start_vllm.sh`)
  and the fine-tuned finbot-8b-v3 (GPU Ollama on port 11435, `start_ollama_gpu.sh`). The other model stays on disk.
- `setup_v3_box.sh` (run on a fresh GPU box, e.g. the A10 `finbot-local-v3`): finbot-8b-v3 on Ollama, the FinBot API
  (chat logs on, for v4 training data) and a Cloudflare quick tunnel. Needs `~/finbot`, `~/models/finbot-8b-v3.Q5_K_M.gguf`,
  `~/Modelfile.v3`; pass `FINBOT_API_TOKEN` in the environment (never stored here).
- `eval_sets.py`, `run_all.sh`, `finish.sh`, `analyze.py` (in `~/evals` on the box): run the test sets (test bank, CA
  Foundation, McGraw, question bank, held-out, benchmark grid) against the served model with at most 3 requests at a time,
  then write `~/evals/v3_report.md` (scores, every failure with tools, review codes and tool errors, failure patterns).
  The question sets themselves are private test data and are not in the repo.
