"""Evaluate FinBot on an external, private problem set (e.g. the McGraw practice problems). The set stays outside this repo.

uv run python bench/run_external.py /Users/reza/MS3610/mcgraw/dataset/problems.jsonl --label mcgraw
Each record needs: id, question, answers[] with value (string) and unit. Results go to results/ (git-ignored).
Grading: a numeric part is correct if any number in the reply is within the tolerance of the gold value (numbers merely
copied from the question do not count); a choice part (unit null) is correct if its word appears in the reply.
"""
import argparse
import datetime as dt
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bench.scoring import calc_correct  # noqa: E402


def gold_number(value: str) -> float | None:
    v = value.replace(",", "").replace("$", "").replace("%", "").strip()
    neg = v.startswith("(") and v.endswith(")")
    try:
        x = float(v.strip("()"))
    except ValueError:
        return None
    return -x if neg else x


def grade(answer: str, question: str, parts: list[dict], tol: float) -> list[bool]:
    out = []
    for p in parts:
        g = gold_number(str(p["value"]))
        if p.get("unit") is None or g is None:
            out.append(bool(re.search(rf"(?i)\b{re.escape(str(p['value']).strip())}\b", answer)))
        else:
            out.append(calc_correct(answer, question, g, tol)[0])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--label", default="external")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    from finbot.graph import FinBot
    bot = FinBot()
    rows = [json.loads(line) for line in Path(a.path).read_text(encoding="utf-8").splitlines() if line.strip()][: a.limit]
    out = ROOT / f"results/{a.label}-{bot.settings.model.replace(':', '_').replace('/', '_')}-{dt.datetime.now():%Y%m%d-%H%M}.jsonl"
    out.parent.mkdir(exist_ok=True)
    tot = {"parts": 0, "ok05": 0, "ok1": 0, "problems_all_ok1": 0}
    with out.open("w", encoding="utf-8") as f:
        for i, r in enumerate(rows, 1):
            parts = r["answers"] if isinstance(r["answers"], list) else json.loads(r["answers"])
            try:
                t = bot.ask(r["question"], thread_id=f"{a.label}-{r['id']}-{time.time_ns()}")
                answer, err = t["answer"], None
            except Exception as exc:
                t, answer, err = {}, "", repr(exc)
            g05, g1 = grade(answer, r["question"], parts, 0.5), grade(answer, r["question"], parts, 1.0)
            tot["parts"] += len(parts)
            tot["ok05"] += sum(g05)
            tot["ok1"] += sum(g1)
            tot["problems_all_ok1"] += all(g1)
            f.write(json.dumps({"id": r["id"], "topic": r.get("topic"), "parts": len(parts), "correct_0.5pct": g05, "correct_1pct": g1,
                                "tools": [x["name"] for x in t.get("tools", [])], "latency_s": t.get("latency_s"),
                                "revisions": t.get("revisions"), "error": err, "answer": answer}, ensure_ascii=False) + "\n")
            print(f"[{i}/{len(rows)}] {r['id']} {sum(g1)}/{len(parts)} parts (1%)  {t.get('latency_s', 0) or 0:5.1f}s  tools={[x['name'] for x in t.get('tools', [])]}", flush=True)
    print(f"\n{a.label} on {bot.settings.model}: parts correct {tot['ok05']}/{tot['parts']} at 0.5%, {tot['ok1']}/{tot['parts']} at 1%; "
          f"problems fully correct {tot['problems_all_ok1']}/{len(rows)} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
