"""Parse FinBot Markdown chat logs (SPEC 7.4) and list the failed or weak answers, for fixes and v4 training prompts.
usage: uv run python scripts/chatlog_failures.py <chatlog.md> [...] [--model qwen3.6-27b] [--jsonl out.jsonl]
A turn counts as failed when its answer is the number-free fallback, an error or empty, or when review did not pass."""
import argparse
import collections
import json
import re
import sys
from pathlib import Path

FALLBACK = ("could not compute this reliably", "could not complete this answer reliably", "temporarily unavailable",
            "something went wrong", "too long for me")


def parse(path: Path) -> list[dict]:
    turns = []
    for block in re.split(r"\n(?=## \d{4}-\d{2}-\d{2} )", path.read_text(encoding="utf-8")):
        m = re.search(r"<!-- meta (\{.*?\}) -->", block, re.S)
        if not m:
            continue
        try:
            meta = json.loads(m.group(1))
        except ValueError:
            continue
        q = re.search(r"### Question\s*```text\n(.*?)\n```", block, re.S)
        a = re.search(r"### Answer\s*```markdown\n(.*?)\n```", block, re.S)
        meta["question"] = q.group(1).strip() if q else ""
        meta["answer"] = a.group(1).strip() if a else ""
        meta["file"] = path.name
        turns.append(meta)
    return turns


def failed(t: dict) -> str | None:
    ans = (t.get("answer") or "").lower()
    if not ans:
        return "empty"
    if any(f in ans for f in FALLBACK):
        return "fallback"
    if t.get("error"):
        return "error"
    if t.get("review_passed") is False:
        return "review:" + ",".join(t.get("review_codes") or [])
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--model")
    ap.add_argument("--jsonl")
    a = ap.parse_args()
    turns = [t for p in a.logs for t in parse(Path(p)) if not a.model or t.get("model") == a.model]
    by_model = collections.Counter(t.get("model") for t in turns)
    fails = [(failed(t), t) for t in turns if failed(t)]
    kinds = collections.Counter(k.split(":")[0] for k, _ in fails)
    codes = collections.Counter(c for k, _ in fails if k.startswith("review") for c in k.split(":")[1].split(","))
    print(f"turns {len(turns)} by model {dict(by_model)}; failed {len(fails)} {dict(kinds)}; review codes {dict(codes)}")
    lat = sorted(t.get("latency_s") or 0 for t in turns)
    if lat:
        print(f"latency median {lat[len(lat) // 2]:.1f}s, 90th {lat[int(len(lat) * 0.9)]:.1f}s, max {lat[-1]:.1f}s")
    for k, t in fails:
        print(f"- [{k}] {t.get('model')} {t.get('latency_s')}s turn {t.get('turn')} | Q: {t['question'][:160]!r}")
        print(f"    A: {t['answer'][:200]!r}")
    if a.jsonl:
        Path(a.jsonl).write_text("".join(json.dumps({"kind": k, **t}, ensure_ascii=False) + "\n" for k, t in fails), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
