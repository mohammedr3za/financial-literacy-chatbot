"""Build blind score sheets from FinBot results and bank-bot responses.

uv run python bench/make_score_sheet.py results/finbot-rag-*.jsonl [results/finbot-base-*.jsonl]
Writes bench/sheets/score_sheet_member1.csv, score_sheet_member2.csv (shuffled, no bot names)
and bench/sheets/score_key.csv (who wrote what; keep this away from scorers until scoring is done).
About 25% of rows go to both scorers so agreement can be measured.
"""
import csv
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CRITERIA = ["accuracy", "risk_disclosure", "flags_uncertainty", "cites_source", "resists_leading", "info_not_advice"]
SEED = 3610


def load_rows(jsonl_paths: list[str]) -> list[dict]:
    rows = []
    for path in jsonl_paths:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r.get("kind") != "question" or r.get("error"):
                continue
            rows.append({"bot": f"FinBot ({r['config']})", "question_id": r["question_id"], "persona": r["persona"],
                         "framing": r["framing"], "question": r["question"], "response": r["answer"]})
    bank = ROOT / "bench/bank_responses.csv"
    with bank.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["question_id"].startswith("q") and r["response_text"].strip():
                rows.append({"bot": r["bot"], "question_id": r["question_id"], "persona": r["persona"],
                             "framing": r["framing"], "question": r["question"], "response": r["response_text"]})
    return rows


def main() -> int:
    rows = load_rows(sys.argv[1:])
    if not rows:
        print("No responses found. Pass FinBot result files and/or fill bench/bank_responses.csv.")
        return 1
    rng = random.Random(SEED)
    rng.shuffle(rows)
    n_double = round(len(rows) * 0.25)
    out = ROOT / "bench/sheets"
    out.mkdir(parents=True, exist_ok=True)
    sheets = {"member1": [], "member2": []}
    with (out / "score_key.csv").open("w", newline="", encoding="utf-8") as kf:
        key = csv.writer(kf)
        key.writerow(["row_id", "bot", "question_id", "persona", "framing", "double_scored"])
        for i, r in enumerate(rows):
            rid = f"R{i + 1:04d}"
            double = i < n_double
            key.writerow([rid, r["bot"], r["question_id"], r["persona"], r["framing"], double])
            targets = ["member1", "member2"] if double else ["member1" if i % 2 else "member2"]
            for t in targets:
                sheets[t].append([rid, r["question"], r["response"]] + [""] * len(CRITERIA) + [""])
    for name, data in sheets.items():
        data.sort(key=lambda row: rng.random())
        with (out / f"score_sheet_{name}.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["row_id", "question", "response"] + CRITERIA + ["notes"])
            w.writerows(data)
    print(f"{len(rows)} responses -> member1: {len(sheets['member1'])} rows, member2: {len(sheets['member2'])} rows, "
          f"{n_double} double-scored. Rubric: bench/RUBRIC.md. Sheets in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
