#!/usr/bin/env bash
# Finish the v3 evaluation quickly: after McGraw, a second test-bank worker runs the remaining questions in reverse
# order (3 requests at a time, 3 of 6 slots free for users); phase 2 (question bank, held-out, grid) is skipped because
# the H100 run already measured those for v3. Then the report.
cd ~/finbot
export PATH=$HOME/.local/bin:$PATH FINBOT_CHATLOG=0 FINBOT_BACKEND=ollama FINBOT_BASE_URL=http://127.0.0.1:11434 \
       FINBOT_MODEL=finbot-8b-v3 FINBOT_COMPACT_TOOLS=on FINBOT_THINKING=off FINBOT_EMBED_URL=http://127.0.0.1:11434 FINBOT_HOST=cloud
E=$HOME/evals
log() { echo "[$(date -u +%H:%M)Z] $*" >> $E/status.log; }
pkill -f "[r]un_all.sh"; log "finisher: phase 2 skipped (already measured on the H100)"
while pgrep -f "[r]un_external.py" >/dev/null; do sleep 10; done
python3 - <<'PY'
import json
from pathlib import Path
E = Path.home() / "evals"
done = {json.loads(l)["id"] for l in (E / "results/testbank.jsonl").read_text().splitlines() if l.strip()}
qs = [json.loads(l) for l in (E / "sets/testbank.jsonl").read_text().splitlines() if l.strip()]
rest = [q for q in qs if q["id"] not in done][::-1]
(E / "sets/testbank_rev.jsonl").write_text("".join(json.dumps(q, ensure_ascii=False) + "\n" for q in rest))
PY
uv run --no-sync python $E/eval_sets.py $E/sets/testbank_rev.jsonl $E/results/testbank_b.jsonl 1 > $E/results/testbank_b.log 2>&1 &
log "second test-bank worker started"
covered() { cat $E/results/testbank.jsonl $E/results/testbank_b.jsonl 2>/dev/null | python3 -c "import sys,json; print(len({json.loads(l)['id'] for l in sys.stdin if l.strip()}))"; }
until [ "$(covered)" -ge 159 ]; do sleep 10; done
pkill -f "[e]val_sets.py .*testbank"
while pgrep -f "[e]val_sets.py .*ca.jsonl" >/dev/null; do sleep 10; done
python3 - <<'PY'
from pathlib import Path
p = Path.home() / "evals/analyze.py"; s = p.read_text()
old = 'sets = {"testbank": load(R / "testbank.jsonl"),'
new = 'sets = {"testbank": list({r["id"]: r for r in load(R / "testbank.jsonl") + load(R / "testbank_b.jsonl")}.values()),'
if old in s:
    p.write_text(s.replace(old, new))
PY
uv run --no-sync python $E/analyze.py > $E/results/analyze.log 2>&1
log "ALL DONE (report: ~/evals/v3_report.md)"
