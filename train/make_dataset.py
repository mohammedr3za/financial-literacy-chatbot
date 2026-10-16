"""M6 step 1: build fine-tuning data by running FinBot's own graph with a stronger open "teacher" model.

Run on the rented GPU box, with vLLM serving the teacher (see train/README.md):
  FINBOT_BACKEND=openai FINBOT_BASE_URL=http://localhost:8000/v1 FINBOT_MODEL=Qwen/Qwen3.6-35B-A3B \
  uv run python train/make_dataset.py --n-kb 1000 --n-calc 500
Smoke test on the Mac with the local model as a stand-in teacher:
  uv run python train/make_dataset.py --n-kb 3 --n-calc 3 --out train/data/smoke.jsonl
From a collected question file (JSONL with id, question, type, gold_answer; see train/data/questions_v2):
  ... train/make_dataset.py --questions train/data/questions_v2/questions.jsonl --out train/data/sft_v2.jsonl
  Numeric questions with a gold answer are kept only if the answer contains it (within 1%). Every attempt, kept or
  not, goes to <out>.attempts.jsonl for evaluation. Re-running skips ids already attempted.

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
    """Programmatic calculation questions with the tool call we expect (varied, persona-flavoured wording)."""
    lakh = rng.choice([5, 8, 10, 15, 20, 25, 30, 40, 50, 75])
    rate = rng.choice([7.5, 8, 8.25, 8.5, 8.75, 9, 9.5, 10.5, 11, 12, 14])
    yrs = rng.choice([3, 5, 7, 10, 12, 15, 20, 25])
    sip = rng.choice([1000, 2000, 2500, 5000, 7500, 10000, 15000, 25000])
    ret = rng.choice([8, 10, 11, 12, 14, 15])
    infl = rng.choice([4, 5, 5.5, 6, 7])
    price = rng.choice([1200, 4999, 18500, 45000, 62000, 125000])
    gst = rng.choice([5, 12, 18, 28])
    m, freq = rng.choice([(12, "monthly"), (4, "quarterly"), (2, "half-yearly")])
    goal_q = rng.choice([f"How much should I invest every month to reach ₹{lakh} lakh in {yrs} years at {ret}% a year?",
                         f"I want ₹{lakh} lakh in {yrs} years. If I earn {ret}% a year, what monthly SIP do I need?",
                         f"What monthly saving gets me to ₹{lakh} lakh in {yrs} years, assuming {ret}% returns?"])
    return rng.choice([
        (rng.choice([f"What EMI will I pay on a ₹{lakh} lakh loan at {rate}% for {yrs} years?",
                     f"My loan is ₹{lakh} lakh for {yrs} years at {rate}%. What's the monthly EMI?"]),
         "loan_emi", {"principal": lakh * 1e5, "annual_rate_pct": rate, "months": yrs * 12}),
        (rng.choice([f"If I invest ₹{sip:,} every month at {ret}% a year for {yrs} years, how much will I have?",
                     f"I put ₹{sip:,} a month in a SIP. At {ret}% for {yrs} years, what does it grow to?"]),
         "sip_future_value", {"monthly": sip, "annual_return_pct": ret, "years": yrs}),
        (goal_q, "goal_saving", {"target": lakh * 1e5, "years": yrs, "annual_return_pct": ret}),
        (f"I need ₹{lakh} lakh in {yrs} years. How much should I invest today at {rate}% a year?", "tvm",
         {"solve_for": "pv", "annual_rate_pct": rate, "years": yrs, "fv": lakh * 1e5}),
        (f"My investment grew from ₹{sip:,} to ₹{sip * rng.choice([2, 3, 4]):,} in {yrs} years. What is the CAGR?", "tvm",
         {"solve_for": "annual_rate_pct", "years": yrs}),
        (rng.choice([f"How many years will my money take to double at {rate}% a year?", f"At {rate}% a year, when will my money double?"]),
         "tvm", {"solve_for": "years", "annual_rate_pct": rate}),
        (f"My {rng.choice(['FD', 'savings bond', 'debt fund'])} gives {rate}% and inflation is {infl}%. What is my real return?",
         "real_return", {"nominal_pct": rate, "inflation_pct": infl}),
        (f"What is the effective annual rate of {rate}% compounded {freq}?", "effective_annual_rate",
         {"nominal_pct": rate, "periods_per_year": m}),
        (f"A {rng.choice(['phone', 'laptop', 'scooter', 'fridge'])} costs ₹{price:,} before tax. With {gst}% GST, what is the final price?",
         "calculate", {}),
    ])


def similar_to_benchmark(q: str, bench: list[str], threshold: float = 0.85) -> bool:
    ql = q.lower()
    return any(difflib.SequenceMatcher(None, ql, b).ratio() >= threshold for b in bench)


def benchmark_questions() -> list[str]:
    qs = yaml.safe_load((ROOT / "bench/questions.yaml").read_text())
    calc = yaml.safe_load((ROOT / "bench/calc_problems.yaml").read_text())
    held = yaml.safe_load((ROOT / "bench/heldout_problems.yaml").read_text())
    out = [t for item in qs["questions"] for p in item["variants"].values() for t in p.values()]
    return [x.lower() for x in out + [p["question"] for p in calc["problems"] + held["problems"]]]


def to_record(state: dict, strip_revision: bool = False) -> dict:
    """Rebuild exactly what the student model sees at inference: system prompt, context block, tool turns, answer.
    strip_revision: when the first draft was sent back by the reviewer, keep only the corrected turn (drop the bad
    draft and the reviewer note), so the student learns the right behaviour directly."""
    from finbot.prompts import SYSTEM_PROMPT, context_block
    from finbot.graph import text_of
    msgs, qi = state["messages"], state["question_idx"]
    if strip_revision:
        notes = [i for i in range(qi + 1, len(msgs)) if isinstance(msgs[i], HumanMessage)]
        if len(notes) != 1:
            return {}
        msgs = msgs[: qi + 1] + msgs[notes[0] + 1:]
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
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from langchain_core.utils.function_calling import convert_to_openai_tool
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-kb", type=int, default=600, help="questions generated from knowledge-base chunks")
    ap.add_argument("--n-calc", type=int, default=300, help="programmatic calculation questions")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=str(ROOT / "train/data/sft.jsonl"))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--questions", help="JSONL of collected questions; replaces --n-kb/--n-calc generation")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    from finbot.graph import FinBot
    from finbot.llm import get_runnables
    from finbot.tools.registry import CALC_TOOLS
    rng = random.Random(a.seed)
    teacher, _ = get_runnables(CALC_TOOLS)
    bench = benchmark_questions()
    if a.questions:
        return run_questions(a, bench)
    chunks = [json.loads(line) for line in (ROOT / "kb/index/chunks.jsonl").read_text().splitlines()]
    guide = [c for c in chunks if c["kind"] == "guidelines"]
    books = [c for c in chunks if c["kind"] == "books"]
    picks = [(rng.choice(guide if rng.random() < 0.6 else books), rng.choice(PERSONAS), rng.choice(FRAMINGS)) for _ in range(a.n_kb // 2 + 20)]

    def make_questions(pick):
        c, persona, framing = pick
        try:
            reply = teacher.invoke(QGEN_PROMPT.format(title=c["title"], n=2, persona=persona, framing=framing, text=c["text"]))
            qs = json.loads(re.search(r"\[.*\]", reply.content, re.S).group(0))
            return [q for q in qs if isinstance(q, str) and 8 < len(q) < 400]
        except Exception:
            return []

    with ThreadPoolExecutor(a.workers) as pool:
        kb_qs = [q for qs in pool.map(make_questions, picks) for q in qs][: a.n_kb]
    jobs = [("kb", q, None) for q in kb_qs]
    for _ in range(a.n_calc):
        question, tool_name, args = calc_templates(rng)  # one draw, so question and expected call always match
        jobs.append(("calc", question, (tool_name, args)))
    print(f"{len(jobs)} questions ({len(kb_qs)} from documents, {a.n_calc} calculations)", flush=True)

    local = threading.local()
    lock = threading.Lock()
    tool_schemas = [convert_to_openai_tool(t) for t in CALC_TOOLS]
    counts = {"kept": 0, "dup": 0, "review": 0, "tool": 0, "error": 0, "done": 0}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    f = out.open("w", encoding="utf-8")

    def run(item):
        i, job = item
        question = job[1]
        if similar_to_benchmark(question, bench):
            return "dup", None
        if not hasattr(local, "bot"):
            local.bot = FinBot(model=None)
        try:
            state = local.bot.graph.invoke({"user_input": question},
                                           {"configurable": {"thread_id": f"gen-{i}", "use_rag": True}, "recursion_limit": 40})
        except Exception:
            return "error", None
        if not state.get("review", {}).get("passed") or state.get("revisions", 0):
            return "review", None
        if job[0] == "calc":
            want_tool, want_args = job[2]
            calls = [tc for m in state["messages"][state["question_idx"] + 1:] if isinstance(m, AIMessage) for tc in (m.tool_calls or [])]
            if not calls or calls[0]["name"] != want_tool or any(
                    abs(float(calls[0]["args"].get(k, -1)) - float(v)) > 1e-6 for k, v in want_args.items() if isinstance(v, (int, float))):
                return "tool", None
        rec = to_record(state)
        if not rec:
            return "review", None
        rec["tools"] = tool_schemas
        rec["source"] = job[0]
        return "kept", rec

    with ThreadPoolExecutor(a.workers) as pool:
        for status, rec in pool.map(run, enumerate(jobs)):
            with lock:
                counts[status] += 1
                counts["done"] += 1
                if rec:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()
                if counts["done"] % 25 == 0:
                    print(f"{counts['done']}/{len(jobs)} {counts}", flush=True)
    f.close()
    print(f"done: {counts} -> {out}", flush=True)
    return 0


def gold_numbers(gold) -> list[float]:
    """Numbers in a gold answer string like '26.61 days' or '$1,228,200; 5.78%'."""
    vals = []
    for m in re.findall(r"-?\(?\$?-?[\d,]*\.?\d+\)?", str(gold or "")):
        t = m.replace("$", "").replace(",", "")
        neg = t.startswith("(") and t.endswith(")")
        try:
            v = float(t.strip("()"))
        except ValueError:
            continue
        vals.append(-v if neg else v)
    return vals


def run_questions(a, bench) -> int:
    """Teacher transcripts for collected questions; every attempt is logged for evaluation."""
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from langchain_core.utils.function_calling import convert_to_openai_tool
    from bench.scoring import calc_correct
    from finbot.graph import FinBot
    from finbot.tools.registry import CALC_TOOLS
    items = [json.loads(line) for line in Path(a.questions).read_text(encoding="utf-8").splitlines() if line.strip()]
    out, attempts = Path(a.out), Path(a.out + ".attempts.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(line)["id"] for line in attempts.read_text().splitlines()} if attempts.exists() else set()
    todo = [q for q in items if q.get("id") not in done][: a.limit]
    print(f"{len(todo)} to run ({len(done)} already attempted) -> {out}", flush=True)
    schemas = [convert_to_openai_tool(t) for t in CALC_TOOLS]
    local, lock = threading.local(), threading.Lock()
    counts = {"kept": 0, "dup": 0, "review": 0, "wrong": 0, "error": 0}

    def run(q):
        if similar_to_benchmark(q["question"], bench):
            return q, "dup", None, {}
        if not hasattr(local, "bot"):
            local.bot = FinBot()
        try:
            t = local.bot.ask(q["question"], thread_id=f"gen-{q['id']}")
        except Exception as exc:
            return q, "error", None, {"error": repr(exc)[:300]}
        info = {"answer": t["answer"], "tools": t["tools"], "review": t["review"], "revisions": t["revisions"],
                "latency_s": t["latency_s"], "flags": t["flags"]}
        golds = gold_numbers(q.get("gold_answer")) if q.get("type") == "numeric" else []
        if golds:
            info["gold_ok"] = all(calc_correct(t["answer"], q["question"], g, 1.0)[0] for g in golds[:6])
        if not t["review"].get("passed") or t["revisions"] > 1:
            return q, "review", None, info
        if golds and not info["gold_ok"]:
            return q, "wrong", None, info
        if t["revisions"] and re.search(r"(?i)reviewer|previous (answer|draft|response)|apologi|as noted above", t["answer"]):
            return q, "review", None, info
        state = local.bot.graph.get_state({"configurable": {"thread_id": f"gen-{q['id']}"}}).values
        rec = to_record(state, strip_revision=bool(t["revisions"]))
        if not rec:
            return q, "review", None, info
        rec.update(tools=schemas, source=q.get("source", "collected"), qid=q["id"], qtype=q.get("type"), revised=bool(t["revisions"]))
        return q, "kept", rec, info

    with out.open("a", encoding="utf-8") as f, attempts.open("a", encoding="utf-8") as fa, ThreadPoolExecutor(a.workers) as pool:
        for i, (q, status, rec, info) in enumerate(pool.map(run, todo), 1):
            with lock:
                counts[status] += 1
                if rec:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
                fa.write(json.dumps({"id": q["id"], "type": q.get("type"), "topic": q.get("topic"), "status": status,
                                     "question": q["question"], "gold_answer": q.get("gold_answer"), **info},
                                    ensure_ascii=False) + "\n"); fa.flush()
                if i % 20 == 0:
                    print(f"{i}/{len(todo)} {counts}", flush=True)
    print(f"done: {counts} -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
