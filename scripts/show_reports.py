"""List 'Report a problem' submissions for triage (SPEC 7.3).

uv run python scripts/show_reports.py [--last 10] [--path results/reports/reports.jsonl]
"""
import argparse
import json
from pathlib import Path

from finbot.reports import reports_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--last", type=int, default=10)
    ap.add_argument("--path", default=str(reports_path()))
    a = ap.parse_args()
    path = Path(a.path)
    if not path.exists():
        print(f"No reports yet ({path}).")
        return
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"{len(rows)} reports in {path}\n")
    for r in rows[-a.last:]:
        last = r["conversation"][-1]
        print(f"#{r['report_id']}  {r['created_at']}  [{r['channel']}]  turns={r['turns']}  model={last.get('served_by') or '?'}")
        print(f"  note:     {r['note'] or '-'}")
        print(f"  question: {last['question'][:200]}")
        print(f"  answer:   {last['answer'][:300].replace(chr(10), ' ')}")
        tools = [(t['name'], t.get('args')) for t in last.get('tools', [])]
        print(f"  tools:    {tools or '-'}  | review: {last.get('review')}  | sources: {[s['title'][:40] for s in last.get('sources', [])]}\n")


if __name__ == "__main__":
    main()
