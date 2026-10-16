"""Merge, clean, decontaminate and balance teacher transcripts into the final SFT file (v2, 2026-10-17).

uv run python train/prepare_sft.py train/data/sft_v2.jsonl [more.jsonl ...] --out train/data/sft_v2_final.jsonl \\
    --external /Users/reza/MS3610/mcgraw/dataset/problems.jsonl /Users/reza/MS3610/mcgraw/dataset/test.jsonl results/qbank/questions.jsonl

Cleaning (what the 8B should NOT learn from the teacher):
- opening "Based on the provided text/context/sources," (the user never provided text) -> removed;
- answers that name internal tools (npv_irr, `loan_emi`, "the tool") -> dropped;
- single-$ LaTeX ($\\frac..$, $g$), which the web page shows as currency -> dropped;
- answers over --max-words (the prompt asks for ~120 words) -> dropped.
Decontamination: the user question is compared with every test question (benchmark, held-out, McGraw, team bank);
difflib ratio >= --sim drops it. Balance: concept answers without any tool call are capped at --max-concept share.
"""
import argparse
import difflib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
sys.path.insert(0, str(ROOT / "src"))
from filter_dataset import test_questions  # noqa: E402

OPENER = re.compile(r"^\s*(based on|according to|as per|from) (the )?(provided|given|above|retrieved|available) "
                    r"(text|context|sources?|information|excerpts?|documents?|data)\s*(\[S\d\](,\s*\[S\d\])*)?\s*[,:]\s*", re.I)
SOURCES_PHRASE = re.compile(r"\bthe (provided|given|retrieved) (sources|context|text|excerpts)\b", re.I)
BAD_DOLLAR = re.compile(r"(?<![$\\])\$\\[a-zA-Z]|(?<![$\w])\$[a-zA-Z]\$|\$-\$")


def tool_names() -> set[str]:
    from finbot.tools.registry import TOOLS_BY_NAME
    return set(TOOLS_BY_NAME)


def current_tools():
    """The tool definitions the served model sees now; every example is re-tagged with them."""
    from langchain_core.utils.function_calling import convert_to_openai_tool
    from finbot.tools.registry import CALC_TOOLS, TOOLS_BY_NAME
    return [convert_to_openai_tool(t) for t in CALC_TOOLS], TOOLS_BY_NAME


def calls_valid(rec: dict, by_name: dict) -> bool:
    """Every tool call must name a current tool and pass its argument schema."""
    for m in rec["messages"]:
        for tc in m.get("tool_calls") or []:
            fn = tc["function"]
            tool = by_name.get(fn["name"])
            if tool is None:
                return False
            try:
                schema = tool.args_schema
                args = fn["arguments"] if isinstance(fn["arguments"], dict) else json.loads(fn["arguments"])
                if schema is not None and hasattr(schema, "model_validate"):
                    schema.model_validate(args)
            except Exception:
                return False
    return True


def question_of(rec: dict, by_id: dict) -> str:
    if rec.get("qid") in by_id:
        return by_id[rec["qid"]]
    user = next((m["content"] for m in rec["messages"] if m["role"] == "user"), "")
    m = re.search(r"(?im)^(question|user question)\s*:\s*(.+)$", user)
    return (m.group(2) if m else user[-600:]).strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", default=str(ROOT / "train/data/sft_v2_final.jsonl"))
    ap.add_argument("--external", nargs="*", default=[])
    ap.add_argument("--questions", nargs="*", default=[str(ROOT / "train/data/questions_v2/questions.jsonl")])
    ap.add_argument("--sim", type=float, default=0.75)
    ap.add_argument("--max-words", type=int, default=350)
    ap.add_argument("--max-concept", type=float, default=0.45)
    a = ap.parse_args()
    by_id = {}
    for path in a.questions:
        if Path(path).exists():
            for line in open(path, encoding="utf-8"):
                q = json.loads(line)
                by_id[q["id"]] = q["question"]
    tests = test_questions([])
    for path in a.external:                      # McGraw files use "question", the team bank uses "q"
        for line in open(path, encoding="utf-8"):
            if line.strip():
                q = json.loads(line)
                tests.append((q.get("question") or q.get("q") or "").lower())
    names = tool_names()
    schemas, by_name = current_tools()
    # A tool is "named" only when written like code: backticked, snake_case, or name( — plain words like "calculate" are fine
    snake = [n for n in names if "_" in n]
    named = re.compile(r"`(" + "|".join(map(re.escape, names)) + r")`|\b(" + "|".join(map(re.escape, snake)) + r")\b|"
                       r"\b(" + "|".join(map(re.escape, names)) + r")\(|\bthe (" + "|".join(map(re.escape, names)) + r") tool\b")
    drops, seen, keep = Counter(), set(), []
    for path in a.inputs:
        for line in open(path, encoding="utf-8"):
            rec = json.loads(line)
            q = question_of(rec, by_id)
            key = re.sub(r"\W+", " ", q.lower()).strip()
            if key in seen:
                drops["duplicate"] += 1; continue
            seen.add(key)
            ql = q.lower()
            if any(difflib.SequenceMatcher(None, ql, t).ratio() >= a.sim for t in tests if abs(len(t) - len(ql)) < 0.5 * len(ql) + 40):
                drops["similar_to_test"] += 1; continue
            final = rec["messages"][-1]
            if final["role"] != "assistant" or not (final.get("content") or "").strip():
                drops["no_final_answer"] += 1; continue
            bad = False
            for m in rec["messages"]:
                if m["role"] != "assistant" or not m.get("content"):
                    continue
                text = OPENER.sub("", m["content"])
                text = text[:1].upper() + text[1:] if text else text
                text = SOURCES_PHRASE.sub("the official sources", text)
                if named.search(text):
                    drops["names_a_tool"] += 1; bad = True; break
                if BAD_DOLLAR.search(text):
                    drops["single_dollar_math"] += 1; bad = True; break
                if len(text.split()) > a.max_words:
                    drops["too_long"] += 1; bad = True; break
                m["content"] = text
            if bad:
                continue
            if not calls_valid(rec, by_name):
                drops["tool_call_no_longer_valid"] += 1; continue
            rec["tools"] = schemas
            rec["has_tool_call"] = any(m.get("tool_calls") for m in rec["messages"])
            keep.append(rec)
    rng = random.Random(3407)
    tools_ = [r for r in keep if r["has_tool_call"]]
    concept = [r for r in keep if not r["has_tool_call"]]
    cap = int(a.max_concept / (1 - a.max_concept) * len(tools_)) if tools_ and a.max_concept < 1 else len(concept)
    if len(concept) > cap:
        rng.shuffle(concept)
        drops["concept_over_cap"] += len(concept) - cap
        concept = concept[:cap]
    final = tools_ + concept
    rng.shuffle(final)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in final:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"kept {len(final)} ({len(tools_)} with tool calls, {len(concept)} concept) -> {a.out}")
    print("dropped:", dict(drops))
    print("by source:", Counter(r.get("source", "?") for r in final).most_common(12))
    print("tools used:", Counter(tc["function"]["name"] for r in final for m in r["messages"] for tc in m.get("tool_calls") or []).most_common())
    return 0


if __name__ == "__main__":
    sys.exit(main())
