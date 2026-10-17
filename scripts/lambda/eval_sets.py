"""Run one question set through FinBot and grade it (runs on the GPU box). usage: eval_sets.py <set.jsonl> <out.jsonl> <workers>
Sets: qbank ("q", numeric "expect"/"expect_any"), testbank and CA ("q", "expect" numbers, "expect_any" keywords, "must_not").
Graded: numeric -> every expect value within tol %; keywords -> any present and no must_not phrase; otherwise ungraded (None)."""
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path.home() / "finbot"))
from bench.scoring import calc_correct  # noqa: E402
from finbot.graph import FinBot  # noqa: E402

qpath, out, workers = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
qs = [json.loads(line) for line in qpath.open(encoding="utf-8") if line.strip()]
done = {json.loads(line)["id"] for line in out.read_text().splitlines()} if out.exists() else set()
todo = [q for q in qs if q["id"] not in done]
local, lock = threading.local(), threading.Lock()


def text(q: dict) -> str:
    return q.get("q") or q.get("question") or ""


def gradable(q: dict) -> bool:
    return bool(q.get("expect") or q.get("expect_any") or q.get("must_not"))


def grade(q: dict, ans: str) -> bool:
    low = ans.lower()
    if q.get("expect"):
        return all(calc_correct(ans, text(q), v, q.get("tol", 1.0))[0] for v in q["expect"])
    anyof = q.get("expect_any") or []
    if anyof and all(isinstance(x, (int, float)) for x in anyof):
        return any(calc_correct(ans, text(q), v, q.get("tol", 1.0))[0] for v in anyof)
    ok = not any(p.lower() in low for p in q.get("must_not") or [])
    return ok and (not anyof or any(str(k).lower() in low for k in anyof))


def run(q: dict) -> dict:
    if not hasattr(local, "bot"):
        local.bot = FinBot()
    t0 = time.time()
    try:
        t = local.bot.ask(text(q), thread_id=f"eval-{qpath.stem}-{q['id']}-{time.time_ns()}")
        tool_errors = [x["name"] + ": " + str(x["output"].get("error", ""))[:160] for x in t["tools"]
                       if isinstance(x.get("output"), dict) and x["output"].get("error")]
        rec = {"answer": t["answer"], "tools": [x["name"] for x in t["tools"]], "tool_errors": tool_errors,
               "review": t["review"], "revisions": t["revisions"], "flags": t.get("flags"), "latency_s": t["latency_s"],
               "error": None}
    except Exception as exc:  # noqa: BLE001
        rec = {"answer": "", "tools": [], "tool_errors": [], "error": repr(exc)[:300], "latency_s": round(time.time() - t0, 1)}
    rec["correct"] = (not rec["error"] and grade(q, rec["answer"])) if gradable(q) else None
    return {**q, **rec}


with out.open("a", encoding="utf-8") as f, ThreadPoolExecutor(workers) as pool:
    for i, r in enumerate(pool.map(run, todo), 1):
        with lock:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            mark = {True: "✓", False: "✗", None: "·"}[r["correct"]]
            print(f"[{i}/{len(todo)}] {r['id']} {mark} {r['latency_s']}s tools={r['tools']} err={r['error'] and r['error'][:60]}", flush=True)
