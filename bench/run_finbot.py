"""Run the benchmark prompts through FinBot.

uv run python bench/run_finbot.py --config rag            # questions.yaml + calc_problems.yaml
uv run python bench/run_finbot.py --config base --set calc
uv run python bench/run_finbot.py --config rag --limit 5  # quick smoke run
Configs: base = no document search, rag = with search, rag_ft = with search on the fine-tuned model (FINBOT_MODEL=finbot-8b).
Re-running with --out <file> resumes and skips prompts already done.
"""
import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bench.scoring import calc_correct  # noqa: E402


def prompts(which: str):
    if which in ("questions", "all"):
        q = yaml.safe_load((ROOT / "bench/questions.yaml").read_text())
        for item in q["questions"]:
            for persona, framings in item["variants"].items():
                for framing, text in framings.items():
                    yield {"id": f"{item['id']}:{persona}:{framing}", "kind": "question", "question_id": item["id"],
                           "persona": persona, "framing": framing, "question": text, "qset_version": q["version"]}
    if which in ("calc", "all", "heldout"):
        c = yaml.safe_load((ROOT / ("bench/heldout_problems.yaml" if which == "heldout" else "bench/calc_problems.yaml")).read_text())
        for p in c["problems"]:
            yield {"id": p["id"], "kind": "calc", "question_id": p["id"], "persona": None, "framing": None,
                   "question": p["question"], "expected": p["expected"], "unit": p["unit"], "tolerance_pct": c["tolerance_pct"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=["base", "rag", "rag_ft"], required=True)
    ap.add_argument("--set", choices=["questions", "calc", "all", "heldout"], default="all")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.config == "rag_ft":
        os.environ.setdefault("FINBOT_MODEL", "finbot-8b")
    from finbot.graph import FinBot
    bot = FinBot()
    out = Path(a.out) if a.out else ROOT / f"results/finbot-{a.config}-{dt.datetime.now():%Y%m%d-%H%M}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(line)["id"] for line in out.read_text().splitlines()} if out.exists() else set()
    todo = [p for p in prompts(a.set) if p["id"] not in done][: a.limit]
    print(f"model={bot.settings.model} config={a.config} todo={len(todo)} (skipping {len(done)} done) -> {out}")
    t_start, lat, calc_ok, calc_n = time.time(), [], 0, 0
    with out.open("a", encoding="utf-8") as f:
        for i, p in enumerate(todo, 1):
            try:
                t = bot.ask(p["question"], thread_id=f"bench-{p['id']}-{time.time_ns()}", use_rag=a.config != "base")
                rec = {**p, "bot": "FinBot", "config": a.config, "model": bot.settings.model, "answer": t["answer"],
                       "tools": t["tools"], "sources": t["sources"], "review": t["review"], "revisions": t["revisions"],
                       "flags": t["flags"], "latency_s": t["latency_s"], "error": None}
            except Exception as exc:  # record and continue; a long run must not die on one prompt
                rec = {**p, "bot": "FinBot", "config": a.config, "model": bot.settings.model, "answer": "", "error": repr(exc)}
            if p["kind"] == "calc":
                ok, matched = calc_correct(rec["answer"], p["question"], p["expected"], p["tolerance_pct"])
                rec["calc_correct"], rec["calc_matched"] = ok, matched
                calc_ok += ok
                calc_n += 1
            rec["run_at"] = dt.datetime.now().isoformat(timespec="seconds")
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            lat.append(rec.get("latency_s") or 0)
            status = "ERR" if rec["error"] else ("ok" if p["kind"] != "calc" else ("✓" if rec["calc_correct"] else "✗"))
            print(f"[{i}/{len(todo)}] {status} {rec.get('latency_s', 0):5.1f}s {p['id']}", flush=True)
    if lat:
        lat.sort()
        print(f"\nmedian latency {lat[len(lat) // 2]:.1f}s, total {time.time() - t_start:.0f}s"
              + (f", calc accuracy {calc_ok}/{calc_n}" if calc_n else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
