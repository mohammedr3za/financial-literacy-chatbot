"""M6 step 1: build fine-tuning data by running FinBot's own graph with a stronger open "teacher" model.

Run on the rented GPU box, with vLLM serving the teacher (see train/README.md):
  FINBOT_BACKEND=openai FINBOT_BASE_URL=http://localhost:8000/v1 FINBOT_MODEL=Qwen/Qwen3.6-35B-A3B \
  uv run python train/make_dataset.py --n-kb 1000 --n-calc 500
Smoke test on the Mac with the local model as a stand-in teacher:
  uv run python train/make_dataset.py --n-kb 3 --n-calc 3 --out train/data/smoke.jsonl

Only transcripts that pass every review check on the first draft are kept. Calculation transcripts must
also call the right tool with the right inputs. Questions too similar to the benchmark are dropped.
"""
import argparse
import difflib
import json
import random
import re
import sys
from pathlib import Path

import yaml
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PERSONAS = ["novice", "greedy_literate", "anxious_saver"]
FRAMINGS = ["neutral", "eager", "leading"]
QGEN_PROMPT = """Below is an excerpt from "{title}". Write {n} different questions an Indian retail investor might ask
that this excerpt helps answer. Persona: {persona}. Framing: {framing} (neutral = plain request; eager = excited,
wants a particular answer; leading = builds a false or one-sided assumption into the question and asks you to agree).
Return only a JSON list of strings.

Excerpt:
{text}"""


def calc_templates(rng: random.Random):
    """Programmatic calculation questions with the tool call we expect."""
    lakh = rng.choice([5, 8, 10, 15, 20, 25, 30, 40, 50, 75])
    rate = rng.choice([7.5, 8, 8.25, 8.5, 8.75, 9, 9.5, 10.5, 11, 12, 14])
    yrs = rng.choice([3, 5, 7, 10, 15, 20, 25])
    sip = rng.choice([1000, 2000, 2500, 5000, 7500, 10000, 15000, 25000])
    ret = rng.choice([8, 10, 11, 12, 14, 15])
    return rng.choice([
        (f"What EMI will I pay on a ₹{lakh} lakh loan at {rate}% for {yrs} years?", "loan_emi",
         {"principal": lakh * 1e5, "annual_rate_pct": rate, "months": yrs * 12}),
        (f"If I invest ₹{sip:,} every month at {ret}% a year for {yrs} years, how much will I have?", "sip_future_value",
         {"monthly": sip, "annual_return_pct": ret, "years": yrs}),
        (f"I need ₹{lakh} lakh in {yrs} years. How much should I invest today at {rate}% a year?", "tvm",
         {"solve_for": "pv", "annual_rate_pct": rate, "years": yrs, "fv": lakh * 1e5}),
        (f"My investment grew from ₹{sip:,} to ₹{sip * rng.choice([2, 3, 4]):,} in {yrs} years. What is the CAGR?", "tvm",
         {"solve_for": "annual_rate_pct", "years": yrs}),
    ])


def similar_to_benchmark(q: str, bench: list[str], threshold: float = 0.85) -> bool:
    ql = q.lower()
    return any(difflib.SequenceMatcher(None, ql, b).ratio() >= threshold for b in bench)


def benchmark_questions() -> list[str]:
    qs = yaml.safe_load((ROOT / "bench/questions.yaml").read_text())
    calc = yaml.safe_load((ROOT / "bench/calc_problems.yaml").read_text())
    out = [t for item in qs["questions"] for p in item["variants"].values() for t in p.values()]
    return [x.lower() for x in out + [p["question"] for p in calc["problems"]]]


def to_record(state: dict) -> dict:
    """Rebuild exactly what the student model sees at inference: system prompt, context block, tool turns, answer."""
    from finbot.prompts import SYSTEM_PROMPT, context_block
    from finbot.graph import text_of
    msgs, qi = state["messages"], state["question_idx"]
    ctx = context_block(text_of(msgs[qi]), state["flags"], state.get("redactions", []), state.get("sources", []),
                        state.get("rag_status", ""))
    out = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": ctx}]
    for m in msgs[qi + 1:]:
        if isinstance(m, AIMessage):
            rec = {"role": "assistant", "content": text_of(m)}
            if m.tool_calls:
                rec["tool_calls"] = [{"type": "function", "function": {"name": tc["name"], "arguments": tc["args"]}}
                                     for tc in m.tool_calls]
            out.append(rec)
        elif isinstance(m, ToolMessage):
            out.append({"role": "tool", "content": m.content})
        elif isinstance(m, HumanMessage):  # a reviewer note means a rewrite happened; such transcripts are not kept
            return {}
    return {"messages": out}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-kb", type=int, default=1000, help="questions generated from knowledge-base chunks")
    ap.add_argument("--n-calc", type=int, default=500, help="programmatic calculation questions")
    ap.add_argument("--out", default=str(ROOT / "train/data/sft.jsonl"))
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    from finbot.graph import FinBot
    from finbot.llm import get_chat_model
    from finbot.tools.registry import CALC_TOOLS
    rng = random.Random(a.seed)
    teacher = get_chat_model()
    bot = FinBot(model=teacher)
    bench = benchmark_questions()
    chunks = [json.loads(line) for line in (ROOT / "kb/index/chunks.jsonl").read_text().splitlines()]
    guide = [c for c in chunks if c["kind"] == "guidelines"]
    books = [c for c in chunks if c["kind"] == "books"]

    jobs = []
    while len(jobs) < a.n_kb:  # guidelines are oversampled: they are what the model knows least
        c = rng.choice(guide if rng.random() < 0.6 else books)
        persona, framing = rng.choice(PERSONAS), rng.choice(FRAMINGS)
        reply = teacher.invoke(QGEN_PROMPT.format(title=c["title"], n=2, persona=persona, framing=framing, text=c["text"]))
        try:
            qs = json.loads(re.search(r"\[.*\]", reply.content, re.S).group(0))
        except (AttributeError, ValueError):
            continue
        jobs += [("kb", q, None) for q in qs if isinstance(q, str)]
    jobs = jobs[: a.n_kb]
    for _ in range(a.n_calc):
        question, tool_name, args = calc_templates(rng)  # one draw, so question and expected call always match
        jobs.append(("calc", question, (tool_name, args)))

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    kept = dropped_dup = dropped_review = dropped_tool = 0
    tool_schemas = [t.tool_call_schema.model_json_schema() | {"name": t.name, "description": t.description} for t in CALC_TOOLS]
    with out.open("w", encoding="utf-8") as f:
        for i, job in enumerate(jobs, 1):
            question = job[1]
            if similar_to_benchmark(question, bench):
                dropped_dup += 1
                continue
            cfg = {"configurable": {"thread_id": f"gen-{i}", "use_rag": True}, "recursion_limit": 40}
            state = bot.graph.invoke({"user_input": question}, cfg)
            if not state.get("review", {}).get("passed") or state.get("revisions", 0):
                dropped_review += 1
                continue
            if job[0] == "calc":
                want_tool, want_args = job[2]
                calls = [tc for m in state["messages"][state["question_idx"] + 1:] if isinstance(m, AIMessage)
                         for tc in (m.tool_calls or [])]
                if not calls or calls[0]["name"] != want_tool or any(
                        abs(float(calls[0]["args"].get(k, -1)) - float(v)) > 1e-6
                        for k, v in want_args.items() if isinstance(v, (int, float))):
                    dropped_tool += 1
                    continue
            rec = to_record(state)
            if rec:
                rec["tools"] = tool_schemas
                rec["source"] = job[0]
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                kept += 1
            if i % 25 == 0:
                print(f"{i}/{len(jobs)} kept={kept} dup={dropped_dup} review={dropped_review} tool={dropped_tool}", flush=True)
    print(f"done: kept={kept} dropped: benchmark-like={dropped_dup}, failed review={dropped_review}, wrong tool={dropped_tool} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
