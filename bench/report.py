"""Merge filled score sheets with the key and write bench/report/report.md plus charts.

uv run python bench/report.py [results/finbot-*.jsonl ...]   (result files add the calculation accuracy table)
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bench.make_score_sheet import CRITERIA  # noqa: E402
from bench.scoring import calc_correct  # noqa: E402


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def kappa(a: list[int], b: list[int]) -> float | None:
    if not a:
        return None
    cats = sorted(set(a) | set(b))
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def main() -> int:
    sheets_dir, out = ROOT / "bench/sheets", ROOT / "bench/report"
    out.mkdir(parents=True, exist_ok=True)
    key = {r["row_id"]: r for r in read_csv(sheets_dir / "score_key.csv")}
    scores = defaultdict(dict)  # row_id -> scorer -> {criterion: int}
    for scorer in ("member1", "member2"):
        for r in read_csv(sheets_dir / f"score_sheet_{scorer}.csv"):
            vals = {c: r[c].strip() for c in CRITERIA}
            if all(v in ("0", "1", "2") for v in vals.values()):
                scores[r["row_id"]][scorer] = {c: int(v) for c, v in vals.items()}
    lines = ["# Benchmark report", ""]
    # agreement
    lines += ["## Scorer agreement (double-scored rows)", "", "| Criterion | Rows | Exact agreement | Cohen's kappa |", "|---|---|---|---|"]
    for c in CRITERIA:
        pairs = [(s["member1"][c], s["member2"][c]) for s in scores.values() if "member1" in s and "member2" in s]
        if pairs:
            a, b = zip(*pairs)
            k = kappa(list(a), list(b))
            lines.append(f"| {c} | {len(pairs)} | {sum(x == y for x, y in pairs) / len(pairs):.0%} | {k:.2f} |")
    # merged mean per row
    merged = {}
    for rid, by in scores.items():
        merged[rid] = {c: sum(v[c] for v in by.values()) / len(by) for c in CRITERIA}
    by_bot, by_bot_framing, by_bot_persona = defaultdict(list), defaultdict(list), defaultdict(list)
    for rid, sc in merged.items():
        k = key[rid]
        by_bot[k["bot"]].append(sc)
        by_bot_framing[(k["bot"], k["framing"])].append(sum(sc.values()))
        by_bot_persona[(k["bot"], k["persona"])].append(sum(sc.values()))
    lines += ["", "## Mean score per criterion (0–2)", "", "| Bot | n | " + " | ".join(CRITERIA) + " | Total /12 |",
              "|---|---|" + "---|" * (len(CRITERIA) + 1)]
    for bot, rows in sorted(by_bot.items()):
        means = [sum(r[c] for r in rows) / len(rows) for c in CRITERIA]
        lines.append(f"| {bot} | {len(rows)} | " + " | ".join(f"{m:.2f}" for m in means) + f" | {sum(means):.1f} |")
    for title, table, col in (("framing", by_bot_framing, ["neutral", "eager", "leading"]),
                              ("persona", by_bot_persona, ["novice", "greedy_literate", "anxious_saver"])):
        lines += ["", f"## Total score (/12) by {title}", "", "| Bot | " + " | ".join(col) + " |", "|---|" + "---|" * len(col)]
        for bot in sorted(by_bot):
            cells = [table.get((bot, x)) for x in col]
            lines.append(f"| {bot} | " + " | ".join(f"{sum(v) / len(v):.1f}" if v else "–" for v in cells) + " |")
    # calculation accuracy
    calc = defaultdict(lambda: [0, 0])
    for path in sys.argv[1:]:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r.get("kind") == "calc":
                calc[f"FinBot ({r['config']})"][0] += bool(r.get("calc_correct"))
                calc[f"FinBot ({r['config']})"][1] += 1
    import yaml
    problems = {p["id"]: p for p in yaml.safe_load((ROOT / "bench/calc_problems.yaml").read_text())["problems"]}
    for r in read_csv(ROOT / "bench/bank_responses.csv"):
        p = problems.get(r["question_id"])
        if p and r["response_text"].strip():
            calc[r["bot"]][0] += calc_correct(r["response_text"], p["question"], p["expected"])[0]
            calc[r["bot"]][1] += 1
    if calc:
        lines += ["", "## Calculation accuracy (within 0.5%)", "", "| Bot | Correct | Asked |", "|---|---|---|"]
        lines += [f"| {b} | {ok} | {n} |" for b, (ok, n) in sorted(calc.items())]
    # chart
    if by_bot_framing:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        bots, framings = sorted(by_bot), ["neutral", "eager", "leading"]
        fig, ax = plt.subplots(figsize=(8, 4))
        w = 0.8 / len(framings)
        for j, fr in enumerate(framings):
            vals = [sum(v) / len(v) if (v := by_bot_framing.get((b, fr))) else 0 for b in bots]
            ax.bar([i + j * w for i in range(len(bots))], vals, w, label=fr)
        ax.set_xticks([i + w for i in range(len(bots))], bots, rotation=15, ha="right")
        ax.set_ylabel("Mean total score (out of 12)")
        ax.set_ylim(0, 12)
        ax.set_title("Trust score by question framing")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "score_by_framing.png", dpi=160)
        lines += ["", "![Score by framing](score_by_framing.png)"]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {out / 'report.md'} ({len(merged)} scored responses)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
