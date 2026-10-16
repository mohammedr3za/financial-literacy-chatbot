"""Fact-check teacher answers against the official answer before training (fine-tune v2).

uv run python train/judge_sft.py train/data/sft_v2.jsonl --attempts train/data/sft_v2.jsonl.attempts.jsonl \\
    --questions train/data/questions_v2/questions.jsonl --out train/data/sft_v2_checked.jsonl
For every kept transcript whose question has an official text answer (OpenStax answer keys, Income Tax / RBI / SEBI /
NPS / EPFO FAQs), the teacher model (FINBOT_BASE_URL, an OpenAI-compatible vLLM server) is asked whether the bot's
answer contradicts the official one. Contradictions are dropped; transcripts without an official answer pass through.
Numeric questions are skipped here: their figures were already matched to the gold value.
"""
import argparse
import json
import os
import re
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

PROMPT = """You check a finance chatbot's answer against an official answer.
Question: {q}
Official answer: {gold}
Chatbot answer: {ans}

Does the chatbot answer state anything that CONTRADICTS the official answer (a wrong fact, number, rule, limit or yes/no)?
Missing details are fine; extra correct details are fine. Reply with JSON only: {{"contradicts": true|false, "issue": "<short>"}}"""


def ask(prompt: str) -> dict:
    body = json.dumps({"model": os.getenv("FINBOT_MODEL", "qwen3.6-27b"), "temperature": 0, "max_tokens": 120,
                       "messages": [{"role": "user", "content": prompt}],
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request(os.getenv("FINBOT_BASE_URL", "http://127.0.0.1:8001/v1").rstrip("/") + "/chat/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    text = json.loads(urllib.request.urlopen(req, timeout=120).read())["choices"][0]["message"]["content"]
    m = re.search(r"\{.*\}", text, re.S)
    return json.loads(m.group(0)) if m else {"contradicts": None, "issue": text[:200]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("sft")
    ap.add_argument("--attempts", required=True)
    ap.add_argument("--questions", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    gold = {}
    for path in a.questions:
        for line in open(path, encoding="utf-8"):
            q = json.loads(line)
            if q.get("type") != "numeric" and q.get("gold_answer"):
                gold[q["id"]] = (q["question"], q["gold_answer"])
    answers = {}
    for line in open(a.attempts, encoding="utf-8"):
        r = json.loads(line)
        answers[r["id"]] = r.get("answer", "")
    recs = [json.loads(line) for line in open(a.sft, encoding="utf-8")]

    def check(rec):
        qid = rec.get("qid")
        if qid not in gold:
            return rec, {"checked": False}
        q, g = gold[qid]
        try:
            v = ask(PROMPT.format(q=q, gold=str(g)[:2500], ans=answers.get(qid, rec["messages"][-1]["content"])[:3000]))
        except Exception as exc:
            v = {"contradicts": None, "issue": repr(exc)[:200]}
        return rec, {"checked": True, **v}

    kept = dropped = unchecked = 0
    with open(a.out, "w", encoding="utf-8") as f, open(a.out + ".verdicts.jsonl", "w", encoding="utf-8") as fv, \
            ThreadPoolExecutor(a.workers) as pool:
        for rec, v in pool.map(check, recs):
            fv.write(json.dumps({"qid": rec.get("qid"), **v}, ensure_ascii=False) + "\n")
            if v.get("contradicts") is True:
                dropped += 1
                continue
            unchecked += not v.get("checked")
            kept += 1
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"kept {kept} (of which {unchecked} had no official answer), dropped {dropped} contradicting -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
