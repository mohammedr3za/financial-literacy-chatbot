"""Summarise the v3 evaluation (runs on the GPU box after run_all.sh): scores per set and topic, every failure with tools,
review codes and tool errors, and the failure patterns that point to what v4 needs. Writes ~/evals/v3_report.md."""
import collections
import glob
import json
import statistics
import time
from pathlib import Path

E = Path.home() / "evals"
R = E / "results"
FALLBACK = ("could not compute this reliably", "could not complete this answer reliably")


def load(path) -> list[dict]:
    p = Path(path)
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()] if p.exists() else []


def heldout_ok(r: dict):
    v = r.get("calc_correct", r.get("correct"))
    return all(v) if isinstance(v, list) else v


sets = {"testbank": load(R / "testbank.jsonl"), "ca": load(R / "ca.jsonl"), "qbank": load(R / "qbank.jsonl")}
mc_files = sorted(glob.glob(str(Path.home() / "finbot/results/v3-mcgraw-*.jsonl")))
mcgraw = load(mc_files[-1]) if mc_files else []
heldout = [r for r in load(R / "heldout.jsonl") if r.get("kind") != "summary"]
grid = load(R / "grid.jsonl")

lines = [f"# finbot-8b-v3 evaluation ({time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())})", "",
         "Model finbot-8b-v3 (Ollama Q5_K_M on the Lambda A100), FinBot production code, compact tools, thinking off.", "",
         "## Scores", "", "| Set | Graded | Correct | Accuracy | Median latency | Errors |", "|---|---|---|---|---|---|"]


def row(name, rows, ok_fn):
    graded = [r for r in rows if ok_fn(r) is not None]
    ok = sum(bool(ok_fn(r)) for r in graded)
    lat = [r.get("latency_s") for r in rows if r.get("latency_s")]
    errs = sum(bool(r.get("error")) for r in rows)
    acc = f"{100 * ok / len(graded):.0f}%" if graded else "-"
    lines.append(f"| {name} | {len(graded)} | {ok} | {acc} | {statistics.median(lat):.1f}s | {errs} |" if lat else f"| {name} | 0 | 0 | - | - | {errs} |")


for name, rows in sets.items():
    row(name, rows, lambda r: r.get("correct"))
row("mcgraw (problems fully correct)", mcgraw, lambda r: all(r.get("correct_1pct") or [False]))
row("heldout", heldout, heldout_ok)
lines.append(f"| benchmark grid (ungraded, for human scoring) | {len(grid)} answers | | | | {sum(bool(r.get('error')) for r in grid)} |")
if mcgraw:
    parts = sum(sum(r.get("correct_1pct") or []) for r in mcgraw); total = sum(r.get("parts", 0) for r in mcgraw)
    lines += ["", f"McGraw parts correct: {parts}/{total}."]

lines += ["", "## By kind / topic", ""]
for name, rows in sets.items():
    by = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        if r.get("correct") is None:
            continue
        k = r.get("kind") or r.get("topic") or r.get("id", "").split("-")[0]
        by[k][0] += bool(r["correct"]); by[k][1] += 1
    if by:
        lines.append(f"- **{name}**: " + ", ".join(f"{k} {a}/{b}" for k, (a, b) in sorted(by.items())))
    if name == "ca":
        bt = collections.defaultdict(lambda: [0, 0])
        for r in rows:
            if r.get("correct") is not None:
                bt[r.get("topic", "?")][0] += bool(r["correct"]); bt[r.get("topic", "?")][1] += 1
        lines.append("  - CA by topic: " + ", ".join(f"{k} {a}/{b}" for k, (a, b) in sorted(bt.items())))

# failure patterns across the graded sets
fails = [(n, r) for n, rows in sets.items() for r in rows if r.get("correct") is False]
fails += [("mcgraw", r) for r in mcgraw if not all(r.get("correct_1pct") or [False])]
fails += [("heldout", r) for r in heldout if heldout_ok(r) is False]
codes = collections.Counter(c for _, r in fails for c in ((r.get("review") or {}).get("codes") or []))
first = collections.Counter(c for _, r in fails for c in ((r.get("review") or {}).get("first_draft_codes") or []))
tools = collections.Counter(t if isinstance(t, str) else t.get("name") for _, r in fails for t in (r.get("tools") or []))
terr = collections.Counter(e.split(":")[0] + ": " + e.split(":", 1)[1][:70] for _, r in fails for e in (r.get("tool_errors") or []))
no_tool = sum(1 for _, r in fails if not r.get("tools"))
blank = sum(1 for _, r in fails if any(f in (r.get("answer") or "").lower() for f in FALLBACK))
all_rows = [r for rows in sets.values() for r in rows] + grid
review_all = collections.Counter(c for r in all_rows if not (r.get("review") or {}).get("passed")
                                 for c in ((r.get("review") or {}).get("codes") or []))
passed = sum(bool((r.get("review") or {}).get("passed")) for r in all_rows)
lines += ["", "## Failure patterns", "",
          f"- Failures: {len(fails)} (graded sets). With no tool call: {no_tool}. Ending in the blank fallback: {blank}.",
          f"- Final review codes on failures: {dict(codes.most_common())}",
          f"- First-draft review codes on failures: {dict(first.most_common())}",
          f"- Tools used in failed answers: {dict(tools.most_common(12))}",
          f"- Tool errors in failed answers: {dict(terr.most_common(10))}",
          f"- Review passed on all answered questions (incl. ungraded): {passed}/{len(all_rows)}; failing codes overall: {dict(review_all.most_common())}",
          "", "## Every failure", ""]
for name, r in fails:
    q = (r.get("q") or r.get("question") or "")[:220].replace("\n", " ")
    exp = r.get("expect") or r.get("expect_any") or r.get("expected") or [a.get("value") for a in r.get("answers", [])][:6]
    ans = (r.get("answer") or "")[:260].replace("\n", " ")
    lines += [f"- **{name} {r.get('id')}** ({r.get('kind') or r.get('topic') or ''}) tools={r.get('tools')} "
              f"codes={(r.get('review') or {}).get('codes')} errors={r.get('tool_errors') or r.get('error') or ''}",
              f"  - Q: {q}", f"  - expected: {exp}", f"  - A: {ans}"]
(E / "v3_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines[:20]))
