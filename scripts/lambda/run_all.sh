#!/usr/bin/env bash
# finbot-8b-v3 evaluation on the GPU box. At most 5 requests at a time, so at least 3 of Ollama's 8 slots stay free for
# finbot-blond users. Progress: ~/evals/status.log; report: ~/evals/v3_report.md.
cd ~/finbot
export PATH=$HOME/.local/bin:$PATH FINBOT_CHATLOG=0 FINBOT_BACKEND=ollama FINBOT_BASE_URL=http://127.0.0.1:11434 \
       FINBOT_MODEL=finbot-8b-v3 FINBOT_COMPACT_TOOLS=on FINBOT_THINKING=off FINBOT_EMBED_URL=http://127.0.0.1:11434 FINBOT_HOST=cloud
E=$HOME/evals; mkdir -p $E/results ~/finbot/results
log() { echo "[$(date -u +%H:%M)Z] $*" >> $E/status.log; }
PY="uv run --no-sync python"
log "start (phase 1: test bank, CA, McGraw)"
$PY $E/eval_sets.py $E/sets/testbank.jsonl $E/results/testbank.jsonl 1 > $E/results/testbank.log 2>&1 &
$PY $E/eval_sets.py $E/sets/ca.jsonl $E/results/ca.jsonl 1 > $E/results/ca.log 2>&1 &
$PY bench/run_external.py $E/sets/mcgraw_problems.jsonl --label v3-mcgraw > $E/results/mcgraw.log 2>&1 &
wait
log "phase 1 done; phase 2: team question bank, held-out, benchmark grid"
$PY $E/eval_sets.py $E/sets/qbank.jsonl $E/results/qbank.jsonl 1 > $E/results/qbank.log 2>&1 &
$PY bench/run_finbot.py --config rag --set heldout --out $E/results/heldout.jsonl > $E/results/heldout.log 2>&1 &
$PY bench/run_finbot.py --config rag --set questions --out $E/results/grid.jsonl > $E/results/grid.log 2>&1 &
wait
log "phase 2 done; analysing"
$PY $E/analyze.py > $E/results/analyze.log 2>&1
log "ALL DONE (report: ~/evals/v3_report.md)"
