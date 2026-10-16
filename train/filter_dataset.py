"""Drop training examples whose question resembles any test question (benchmark, held-out, McGraw). Run before training.

uv run python train/filter_dataset.py train/data/sft.jsonl train/data/sft_clean.jsonl --external /Users/reza/MS3610/mcgraw/dataset/problems.jsonl
"""
import argparse
import difflib
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_questions(external: list[str]) -> list[str]:
    qs = yaml.safe_load((ROOT / "bench/questions.yaml").read_text())
    out = [t for item in qs["questions"] for p in item["variants"].values() for t in p.values()]
    for f in ("bench/calc_problems.yaml", "bench/heldout_problems.yaml"):
        out += [p["question"] for p in yaml.safe_load((ROOT / f).read_text())["problems"]]
    for path in external:
        out += [json.loads(line)["question"] for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    return [q.lower() for q in out]


def question_of(rec: dict) -> str:
    user = next(m["content"] for m in rec["messages"] if m["role"] == "user")
    return user.split("User question:\n", 1)[-1].strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--external", nargs="*", default=[])
    ap.add_argument("--threshold", type=float, default=0.8)
    a = ap.parse_args()
    tests = test_questions(a.external)
    kept = dropped = 0
    with open(a.dst, "w", encoding="utf-8") as out:
        for line in Path(a.src).read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            q = question_of(rec).lower()
            if any(difflib.SequenceMatcher(None, q, t).ratio() >= a.threshold for t in tests):
                dropped += 1
                continue
            out.write(line + "\n")
            kept += 1
    print(f"kept {kept}, dropped {dropped} as too similar to {len(tests)} test questions -> {a.dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
