"""LangGraph: START -> precheck -> retrieve -> agent <-> tools -> review -> END (SPEC section 3)."""
import json
import difflib
import re
import time
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from .config import Settings, get_settings
from .precheck import precheck as run_precheck
from .prompts import SYSTEM_PROMPT, context_block
from .review import (FOOTER, _strip_math_markup, asks_for_inputs, cut_hypothetical, prune_figures, review_answer,
                     reviewer_note, script_damaged, unsupported_figures, unverified_numbers)
from .tables import to_markdown
from .tools.financial_solver import KIND as SOLVER_KIND
from .tools.financial_solver import NO_INTEREST
from .tools.financial_solver import resolve as solver_name
from .routing import missing_tools
from .tools.fmt import MONEY_STYLE
from .tools.registry import CALC_TOOLS, TOOLS_BY_NAME

_THINK = re.compile(r"<think>.*?</think>", re.S)
GROUNDED_TOOLS = {"financial_ratios", "balance_sheet_check", "cash_flow_statement", "solve_financials", "equivalent_annual_cost",
                  "income_tax", "project_cash_flows"}
_WORD_NUMBERS = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen "
                                            "fourteen fifteen sixteen seventeen eighteen nineteen twenty".split())}
_WORD_NUMBERS.update({"thirty": 30, "forty": 40, "fifty": 50})
_WORD_NUMBER = re.compile(r"(?i)\b(" + "|".join(_WORD_NUMBERS) + r")\b")
_NUMBER = re.compile(r"(?<![\w.])\d{1,3}(?:,\d{2,3})+(?:\.\d+)?|(?<![\w.,])\d+(?:\.\d+)?|(?<![\w.])\.\d+")   # ".75" too
_SCALED = re.compile(r"(?i)(?<![\w.])(\d+(?:\.\d+)?|\.\d+)\s*(thousand|million|mn|billion|bn)\b")
_SCALE = {"thousand": 1e3, "million": 1e6, "mn": 1e6, "billion": 1e9, "bn": 1e9}
_PERCENT = re.compile(r"(?i)(?<![\w.,])(\d+(?:\.\d+)?|\.\d+)\s*(%|per ?cent)")
_NUMERIC_STRING = re.compile(r"\s*[-+]?(\d[\d,]*(\.\d+)?|\.\d+)\s*")


def numbers_in(text: str) -> set[float]:
    """Numbers written in the text, plus "4.5 million" style amounts in full."""
    found = {round(float(n.replace(",", "")), 4) for n in _NUMBER.findall(text or "")}
    return found | {round(float(n) * _SCALE[u.lower()], 4) for n, u in _SCALED.findall(text or "")}


def _arg_numbers(args, path=""):
    if isinstance(args, bool):
        return
    if isinstance(args, str) and _NUMERIC_STRING.fullmatch(args):   # "250000" sent as text is checked like a number
        yield path, float(args.replace(",", ""))
    elif isinstance(args, (int, float)):
        yield path, float(args)
    elif isinstance(args, dict):
        for k, v in args.items():
            yield from _arg_numbers(v, f"{path}.{k}" if path else str(k))
    elif isinstance(args, (list, tuple)):
        for i, v in enumerate(args):
            yield from _arg_numbers(v, f"{path}[{i}]")


def _solver_unit_problem(name: str, a: float, allowed: set, percents: set, percent_only: set) -> str | None:
    """solve_financials units: *_pct items take the percent number, ratios a plain number. None if the value is grounded."""
    kind = SOLVER_KIND.get(solver_name(name))
    if kind == "pct":
        fractions = {x for x in allowed if x < 1 and x not in percents}
        if (a in allowed and not (a < 1 and a not in percents)) or round(a / 100, 4) in fractions:
            return None
        return " (percent items take the percent number: 23 percent -> 23, .23 -> 23)"
    if kind in ("ratio", "days"):
        if (a in allowed and a not in percent_only) or round(a * 100, 4) in percents:
            return None
        return " (ratios are plain numbers: .75 -> 0.75, a 40 percent ratio -> 0.4)"
    return None if a in allowed else ""


def ungrounded_values(args: dict, user_text: str, tool: str | None = None, tool_text: str = "") -> list[str]:
    """Numbers in a grounded tool call that do not appear in the user's message (zeros are ignored).
    equivalent_annual_cost may also use a result of an earlier tool call in this turn (tool_text)."""
    allowed = numbers_in(user_text)
    if tool == "equivalent_annual_cost":
        allowed = allowed | numbers_in(tool_text)
        return [f"{p}={v:g}" for p, v in _arg_numbers(args) if v != 0 and round(abs(v), 4) not in allowed]
    if tool == "project_cash_flows":   # number words ("three-year") count; so does a unit change by 100 (".40 cents" = 0.004)
        allowed = allowed | numbers_in(tool_text) | {float(_WORD_NUMBERS[w.lower()]) for w in _WORD_NUMBER.findall(user_text or "")}
        allowed = allowed | {round(a / 100, 6) for a in allowed} | {round(a * 100, 4) for a in allowed}
        return [f"{p}={v:g}" for p, v in _arg_numbers(args) if v != 0 and round(abs(v), 6) not in allowed
                and round(abs(v), 4) not in allowed]
    diffs = {round(abs(a - b), 4) for a in allowed for b in allowed}
    percents = {round(float(n), 4) for n, _ in _PERCENT.findall(user_text or "")}
    percent_only = percents - numbers_in(_PERCENT.sub(" ", user_text or ""))
    bad = []
    for p, v in _arg_numbers(args):
        if tool == "solve_financials" and v != 0:
            hint = _solver_unit_problem(p, round(abs(v), 4), allowed, percents, percent_only)
            if hint is not None:
                bad.append(f"{p}={v:g}{hint}")
            continue
        if v == 0 or round(abs(v), 4) in allowed:
            continue
        # a working-capital change may be the model's own subtraction, but only if it is exactly closing − opening of given figures
        if p.startswith(("current_asset_changes.", "current_liability_changes.")) and "[" not in p and round(abs(v), 4) in diffs:
            continue
        bad.append(f"{p}={v:g}")
    return bad


_RATE_ARGS = {"loan_emi": ("annual_rate_pct",), "tvm": ("annual_rate_pct",), "sip_future_value": ("annual_return_pct",),
              "goal_saving": ("annual_return_pct",), "equivalent_annual_cost": ("rate_pct",)}
_ZERO_RATE_STATED = re.compile(r"(?i)(?<![\d.])0(\.0+)?\s*(%|per ?cent)|\bzero\b|interest[- ]free|\bno interest\b|"
                               r"without (any )?interest|\b0 interest\b")


_NO_RUNNING_COST = re.compile(r"(?i)\bno (annual |yearly |operating |running |maintenance )*costs?\b|\bzero\b|"
                              r"\bcosts? nothing to (run|operate)\b")


_BEGIN_CUE = re.compile(r"(?i)\b(beginning|start)\s+of\s+(each|every|the)\b|\bannuity due\b|\bin advance\b|"
                        r"\bat the (beginning|start)\b")
_TIMED_TOOLS = ("sip_future_value", "goal_saving")


def deposit_timing(tool: str, args: dict, flags: dict, user_text: str) -> dict:
    """Deposit timing convention (SPEC 3.1): a "beginning"/"start of" cue means the start of each period; otherwise a
    US-dollar (textbook) question means the end of each period (ordinary annuity), whatever the model passed; otherwise
    an omitted timing means the start of each month (an Indian SIP)."""
    if tool not in _TIMED_TOOLS or not isinstance(args, dict):
        return args
    if _BEGIN_CUE.search(user_text or ""):
        timing = "start"
    elif flags.get("currency") == "USD":
        timing = "end"
    else:
        timing = args.get("timing") or "start"
    return {**args, "timing": timing}


_AMT = r"(\d[\d,]*(?:\.\d+)?)"
_SOUP = [  # LaTeX-dollar soup around an amount -> a plain amount (SPEC 3.1 currency clean-up)
    (re.compile(r"\$\s*([-−])\s*\\?\$\s*" + _AMT + r"\s*\$(?!\$)"), r"\1$\2"),          # $-$1,650,000$  $-\$1,650,000$
    (re.compile(r"\$\$\s*" + _AMT + r"\s*\$(?!\$)"), r"$\1"),                              # $$412,500.00$
    (re.compile(r"(?<![$\\])\$\s*\\\$\s*" + _AMT + r"\s*\$(?!\$)"), r"$\1"),                # $\$412,500.00$
]
_SOUP_USD = (re.compile(r"(?<![$\\\w])\$" + _AMT + r"\$(?![$\d])"), r"$\1")               # $1,452,666.67$


def clean_currency(text: str, currency: str = "INR") -> str:
    for pattern, repl in _SOUP:
        text = pattern.sub(repl, text)
    if currency == "USD":
        text = _SOUP_USD[0].sub(_SOUP_USD[1], text)
    return text


_CAPITAL_GAIN = re.compile(r"(?i)\b(capital gains?|stcg|ltcg)\b|\b(profit|gains?|sold|sell|selling|sale)\b.{0,40}\b(shares?|"
                           r"stocks?|equity|equities|mutual funds?)\b|\b(shares?|stocks?|equity|mutual funds?)\b.{0,40}\b(profit|"
                           r"gains?|sold|sale)\b")
_OTHER_INCOME_CUE = re.compile(r"(?i)\b(interest|rent(al)?|fd|fixed deposit|savings account|other income|freelanc\w*|"
                               r"business|dividends?|pension)\b")


_CENTS = re.compile(r"(?i)(?<![\w.])(\d*\.?\d+)\s*cents?\b")


def cents_as_dollars(tool: str, args: dict, user_text: str) -> list[float]:
    """project_cash_flows: a cost the question writes in cents (".40 cents") passed as dollars (0.4 instead of 0.004)."""
    if tool != "project_cash_flows" or not isinstance(args, dict):
        return []
    cents = {round(float(x), 6) for x in _CENTS.findall(user_text or "")}
    v = args.get("variable_cost_per_unit")
    vals = v if isinstance(v, list) else [v]
    return [x for x in vals if isinstance(x, (int, float)) and round(float(x), 6) in cents]


def stock_profit_as_other_income(args: dict, user_text: str) -> bool:
    """income_tax with other_income when the question talks about profit from shares or funds and names no other income."""
    return bool(isinstance(args, dict) and args.get("other_income") and _CAPITAL_GAIN.search(user_text or "")
                and not _OTHER_INCOME_CUE.search(user_text or ""))


def invented_zero_rates(tool: str, args: dict, user_text: str) -> list[str]:
    """A small model fills an unknown rate with 0. A 0% rate is only accepted if the question says so. For
    equivalent_annual_cost the same holds for a zero running cost when no electricity figures are given."""
    if tool == "tvm" and args.get("solve_for") == "annual_rate_pct":   # the rate is the unknown and is ignored
        return []

    def zero(k):
        return isinstance(args.get(k), (int, float)) and not isinstance(args.get(k), bool) and args[k] == 0
    zeros = [k for k in _RATE_ARGS.get(tool, ()) if zero(k)]
    zeros = [] if not zeros or _ZERO_RATE_STATED.search(user_text or "") else zeros
    if tool == "equivalent_annual_cost" and zero("annual_cost") and not args.get("power_watts") \
            and not _NO_RUNNING_COST.search(user_text or ""):
        zeros.append("annual_cost")
    return zeros


def _argument_error(name: str, exc: Exception) -> str:
    """Pydantic errors as short "field: problem" text the model can act on."""
    errors = getattr(exc, "errors", None)
    if callable(errors):
        try:
            parts = [(".".join(str(x) for x in e.get("loc", ())) + ": " if e.get("loc") else "") + e.get("msg", "") for e in errors()]
            return f"Invalid arguments for {name}: {'; '.join(parts)}"
        except Exception:  # noqa: BLE001 - fall back to the plain message
            pass
    return f"Invalid arguments for {name}: {exc}"


_SOLVER_ADVICE = (" For solve_financials, pass each figure exactly as the question states it under its name; percent items (*_pct) "
                  "take the percent number (23 percent -> 23) and ratios are plain numbers (.75 -> 0.75). Never compute an input "
                  "yourself: put the item you need in targets.")
_BALANCE_ADVICE = (" For a balance sheet, pass every listed item to balance_sheet_check (it adds up the sections); if the "
                   "question asks for total equity (or total liabilities), pass that section as {}: the tool derives it as "
                   "the balancing figure.")
_EAC_ADVICE = ("For electricity, pass power_watts, hours_per_year, price_per_kwh and lifetime_hours exactly as the question gives "
               "them and leave out annual_cost and life_years; the tool works them out. Compute any other figure the question does "
               "not state with calculate first and pass its exact result. Costs are positive numbers.")


PARTIAL_NOTE = ("I could not verify the rest of this answer with a calculator, so that part is left out. Ask about it "
                "separately and I will work it out.")
NOT_RELIABLE = ("I could not compute this reliably, so I am not giving a figure. Please check that the question gives every "
                "input (amounts, rate, term) and ask again; I will work it out with the calculator.")
_NOT_RELIABLE_LOCAL = [   # the same message, first in the user's script (SPEC 5: reply in the user's language)
    (re.compile(r"[\u0900-\u097F]"), "मैं इसकी गणना भरोसेमंद ढंग से नहीं कर सका, इसलिए कोई आंकड़ा नहीं दे रहा हूँ। कृपया जाँचें कि प्रश्न "
                                    "में सभी जानकारी (राशि, ब्याज दर, अवधि) दी गई है और फिर से पूछें।"),
    (re.compile(r"[\u0B80-\u0BFF]"), "இதை நம்பகமாகக் கணக்கிட முடியவில்லை, எனவே எந்த எண்ணையும் தரவில்லை. கேள்வியில் தொகை, வட்டி "
                                    "விகிதம், காலம் ஆகிய அனைத்தும் உள்ளதா எனச் சரிபார்த்து மீண்டும் கேளுங்கள்."),
]


def not_reliable(question: str) -> str:
    local = next((text for script, text in _NOT_RELIABLE_LOCAL if script.search(question or "")), None)
    return f"{local}\n\n{NOT_RELIABLE}" if local else NOT_RELIABLE


def unverified_figures(answer: str, question: str, evidence: str, tools_used: list[str], flags: dict) -> list[str]:
    """Figures in a calculation answer that are not in the question (no successful tool call) or not in the evidence
    (question, sources, tool results)."""
    if not (flags.get("calculation_intent") or flags.get("calc_types")):
        return []
    text = _strip_math_markup(answer)
    if not tools_used:   # with no calculator, only figures restated exactly from the question may appear ($950 is not 9.50%)
        return unsupported_figures(text, question, plain_numbers=True, scale_variants=False)
    return unsupported_figures(text, evidence, rel_tol=0.005, plain_numbers=True)


def fabricated_figures(answer: str, question: str, evidence: str, tools_used: list[str], flags: dict) -> bool:
    return bool(unverified_figures(answer, question, evidence, tools_used, flags))


MIN_PRUNED_CHARS = 200
ARG_ALIASES = {"target_amount": "target", "goal": "target", "goal_amount": "target", "loan_amount": "principal",
               "monthly_investment": "monthly", "monthly_sip": "monthly", "number_of_years": "years", "tenure_years": "years",
               "life": ("years", "life_years"), "project_life": ("years", "life_years"), "useful_life": ("life_years", "years"),
               "profit": "gain", "capital_gain": "gain", "gain_amount": "gain"}
def _alias(name: str, fields: set) -> str | None:
    alias = ARG_ALIASES.get(name.lower())
    return next((a for a in ((alias,) if isinstance(alias, str) else alias or ()) if a in fields), None)


_RATE_LIKE = re.compile(r"(?i)^(annual_)?(interest_)?(rate|return)(_pct|_percent)?$|^(interest|discount)_rate(_pct)?$")


def normalize_args(tool: str, args: dict) -> tuple[dict, str | None]:
    """Tools built from a plain function drop unknown argument names silently (v3 lost the rate in 44 tvm calls).
    Map common aliases and a lone rate-like name onto the tool's one rate field; reject any other unknown name with a
    suggestion. Tools with a strict schema reject unknown names themselves: there only aliases are renamed (SPEC changelog
    2026-10-17, 2026-10-19)."""
    t = TOOLS_BY_NAME.get(tool)
    if t is None or not isinstance(args, dict):
        return args, None
    fields = set(t.args)
    if t.args_schema.model_config.get("extra") == "forbid":
        renamed = {}
        for k, v in args.items():
            target = k if k in fields else _alias(k, fields)
            renamed[target if target and target not in args and target not in renamed else k] = v
        return renamed, None
    rate_fields = [f for f in fields if re.search(r"(rate|return)_pct$", f)]
    out, unknown = {}, []
    for k, v in args.items():
        target = k if k in fields else _alias(k, fields)
        if target is None and _RATE_LIKE.search(k) and len(rate_fields) == 1:
            target = rate_fields[0]
        if target in fields and target not in out and (target == k or target not in args):
            out[target] = v
        else:
            unknown.append(k)
    if unknown:
        tips = [f"{k} (did you mean {m[0]}?)" if (m := difflib.get_close_matches(k, sorted(fields), n=1, cutoff=0.5)) else k
                for k in unknown]
        return out, f"Unknown argument(s) for {tool}: {', '.join(tips)}. Valid arguments: {', '.join(sorted(fields))}."
    return out, None


_MENTIONS_INTEREST = re.compile(r"(?i)\binterest\b(?!\s+rates?\b)")   # an interest line, not an interest rate
FORCE_TOOL_CODES = {"no_tool_for_numbers", "wrong_tool"}   # reviewer codes whose revision must call a tool (vLLM)


def calculation_question(flags: dict, question: str) -> bool:
    """A recognised calculation (SPEC 3.1): a calc_types route, or calculation intent with at least three figures in the
    question (a lump-sum or goal question the router does not name). Safety questions ("this coin will 10x") have fewer."""
    if flags.get("calc_types"):
        return True
    plain = re.sub(r"\(=\s*[\d.,]+\)", "", question or "")      # not the "(= 2500000)" notes precheck adds
    return bool(flags.get("calculation_intent")) and len(re.findall(r"\d+(?:[.,]\d+)*", plain)) >= 3


def without_unverified(answer: str, question: str, evidence: str, tools_used: list[str], flags: dict) -> str:
    """The final answer after a failed revision, with no unverified figure (SPEC 3.1). A recognised calculation gets the
    number-free fallback when no tool (or the wrong one) gave its key figures; with the right tool, only the sentences
    with unverified figures go, if a figure is left. A question that only sounds like a calculation (a scam
    check, "can this coin 10x"), or an answer that asks for missing inputs, loses only the sentences with unverified
    figures, if enough verified text is left."""
    bad = unverified_figures(answer, question, evidence, tools_used, flags)
    if not bad:
        return answer
    pruned = prune_figures(_strip_math_markup(answer), bad)
    clean = not unverified_figures(pruned, question, evidence, tools_used, flags)
    if not calculation_question(flags, question) or asks_for_inputs(answer):   # a request for inputs shows no result
        if len(pruned) >= MIN_PRUNED_CHARS and clean:
            return pruned
    elif tools_used and not missing_tools(flags.get("calc_types", []), tools_used):
        if clean and re.search(r"\d", pruned):    # the right tool gave the key figures: drop only a stray estimate
            return pruned
    elif tools_used and len(missing_tools(flags.get("calc_types", []), tools_used)) < len(flags.get("calc_types", [])):
        if clean and re.search(r"\d", pruned):    # some parts verified (a bond's YTM, not its effective yield): keep them
            return pruned.rstrip() + "\n\n" + PARTIAL_NOTE
    return not_reliable(question)


# ---------------------------------------------------------------- final clean-ups before an answer ships
_TOOL_ALT = "|".join(re.escape(n) for n in sorted(TOOLS_BY_NAME, key=len, reverse=True))
_SNAKE_ALT = "|".join(re.escape(n) for n in sorted(TOOLS_BY_NAME, key=len, reverse=True) if "_" in n)
_CALL_SHOWN = re.compile(rf"`?\b({_SNAKE_ALT})`?\s*\(|`?\b({_TOOL_ALT})`?\(")   # "perpetuity (…)" in prose is not a call
_NARRATED = re.compile(r"\b(function|tool)s?\s+(was|were|is|has been|have been)\s+(called|invoked|used|run|executed)\b"
                       r"|\b(called|invoked|calling|invoking|ran)\s+the\s+[\w ]{0,25}?\b(function|tool)\b|\btool calls?\b", re.I)
_TOOL_NAMED = re.compile(rf"(?:\b(?:the|a|an)\s+)?(?:`({_TOOL_ALT})`|\b({_SNAKE_ALT})\b)(?:\s+(?:tool|function|calculator))?"
                         rf"|(?:\b(?:the|a|an)\s+)?\b({_TOOL_ALT})\s+(?:tool|function)\b", re.I)


def strip_tool_narration(text: str) -> str:
    """Deterministic last step (SPEC 3.1): sentences that show a tool call or narrate one are removed, and any other
    mention of a registry tool name becomes "the calculator"."""
    out_lines = []
    for line in (text or "").split("\n"):
        kept = [sent for sent in re.split(r"(?<=[.!?])\s+", line) if not (_CALL_SHOWN.search(sent) or _NARRATED.search(sent))]
        line = " ".join(kept) if kept or not line.strip() else ""
        line = _TOOL_NAMED.sub(lambda m: "The calculator" if m.group(0)[:1].isupper() else "the calculator", line)
        out_lines.append(line)
    cleaned = re.sub(r"\n{3,}", "\n\n", "\n".join(out_lines)).strip()
    return cleaned or (text or "")


SHORT_TABLE_ROWS = 12
_OVERFLOW = re.compile(r"(?i)maximum context length|context[_ ]length|prompt is too long|too many tokens|reduce the length")
TOO_LONG = ("This conversation is too long for me to answer reliably. Please start a new conversation, or ask one part of "
            "the question at a time.")


def compact_tool_result(content, drop_tables: bool = False) -> str:
    """What the model sees of a tool result (SPEC 3.1 context budget): values, steps and short tables; a long table (or,
    with drop_tables, every table) becomes a note. The full result stays in the state, the trace and the UI."""
    try:
        out = json.loads(content)
    except (TypeError, ValueError):
        return content
    if not isinstance(out, dict) or "error" in out:
        return content
    out = {k: v for k, v in out.items() if k != "table_shown_to_user"}
    table = out.get("table")
    if isinstance(table, dict):
        rows = table.get("rows") or []
        if drop_tables or len(rows) > SHORT_TABLE_ROWS:
            out["table"] = f"table shown to user ({len(rows)} rows)"
        else:
            out["table"] = {"columns": table.get("columns"), "rows": rows, "note": "shown to user below the answer; do not copy it"}
    if drop_tables:
        out.pop("inputs", None)
    return json.dumps(out, ensure_ascii=False, separators=(",", ":"))


def _for_model(msgs: list, drop_tables: bool = False) -> list:
    return [ToolMessage(content=compact_tool_result(m.content, drop_tables), tool_call_id=m.tool_call_id, name=m.name, id=m.id)
            if isinstance(m, ToolMessage) else m for m in msgs]


# ---------------------------------------------------------------- multi-turn history budget (SPEC 3.1 agent)
HISTORY_TURNS = 2
MAX_PROMPT_TOKENS = 10_000          # history plus the current turn, when the context allows it
_PUNCT = re.compile(r"[^\w\s]")


def estimate_tokens(x) -> int:
    """Rough Qwen token count without a tokenizer: digits and punctuation are dense, words about 4.2 characters a token
    (within about 15% of the real tokenizer on prompts, schemas and tool results)."""
    if isinstance(x, (list, tuple)):
        return sum(estimate_tokens(m) for m in x)
    if not isinstance(x, str):
        text = text_of(x) if hasattr(x, "content") else str(x)
        calls = getattr(x, "tool_calls", None)
        return estimate_tokens(text + (json.dumps(calls, ensure_ascii=False) if calls else "")) + 4
    digits = sum(ch.isdigit() for ch in x)
    punct = len(_PUNCT.findall(x))
    return int(digits + punct * 0.5 + (len(x) - digits - punct) / 4.2) + 1


_TOOL_TOKENS: dict = {}


def prompt_budget(settings: Settings) -> int:
    """Tokens left for history plus the current turn: the context minus the output cap, the system prompt, the tool
    schemas and a margin, capped at MAX_PROMPT_TOKENS."""
    if "n" not in _TOOL_TOKENS:
        from .tools.schemas import compact_tool_schemas
        _TOOL_TOKENS["n"] = estimate_tokens("\n".join(json.dumps(t, ensure_ascii=False) for t in compact_tool_schemas(CALC_TOOLS)))
    room = settings.num_ctx - settings.output_cap() - estimate_tokens(SYSTEM_PROMPT) - _TOOL_TOKENS["n"] - 384
    return max(2000, min(MAX_PROMPT_TOKENS, int(room * 0.92)))


def earlier_turns(msgs: list) -> list[dict]:
    """Earlier turns as {question, calls, results, answer}; reviewer notes, tool payloads and failed calls are left out."""
    failed = {m.tool_call_id for m in msgs if isinstance(m, ToolMessage) and _is_error(m.content)}
    results = {m.tool_call_id: str(m.content) for m in msgs if isinstance(m, ToolMessage) and not _is_error(m.content)}
    turns, cur = [], None
    for m in msgs:
        if isinstance(m, HumanMessage) and m.name != "reviewer":
            cur = {"question": text_of(m), "calls": [], "results": [], "answer": ""}
            turns.append(cur)
        elif cur is not None and isinstance(m, AIMessage):
            ok = [tc for tc in (m.tool_calls or []) if tc["id"] not in failed]
            cur["calls"] += ok
            cur["results"] += [results[tc["id"]] for tc in ok if tc["id"] in results]
            if not m.tool_calls:
                cur["answer"] = text_of(m).replace(FOOTER.strip(), "").strip()
    return turns


def inputs_line(calls: list) -> str:
    """One line with the tool inputs of a turn, e.g. loan_emi(principal=2000000, annual_rate_pct=8.5, months=240)."""
    parts = []
    for tc in calls:
        args = ", ".join(f"{k}={json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict, str)) else v}"
                         for k, v in tc["args"].items() if v is not None)
        parts.append(f"{tc['name']}({args})")
    line = "; ".join(parts)
    return line if len(line) <= 600 else line[:597] + "..."


def history_messages(turns: list[dict], answer_chars: int | None = None) -> list:
    out = []
    for t in turns:
        q = t["question"]
        if t["calls"]:
            q += f"\n[Calculator inputs used for this question: {inputs_line(t['calls'])}]"
        a = t["answer"] or "(no answer)"
        if answer_chars is not None and len(a) > answer_chars:
            a = a[:answer_chars].rsplit(" ", 1)[0] + " …"
        out += [HumanMessage(content=q), AIMessage(content=a)]
    return out


def is_context_overflow(exc: Exception) -> bool:
    return bool(_OVERFLOW.search(str(exc)))


def _is_error(content) -> bool:
    return str(content).lstrip().startswith('{"error"')


class State(TypedDict, total=False):
    user_input: str
    messages: Annotated[list[AnyMessage], add_messages]
    question_idx: int
    flags: dict
    redactions: list[str]
    amounts: list[dict]
    sources: list[dict]
    sources_relevant: bool
    best_dense: float
    rag_status: str
    tool_iterations: int
    revisions: int
    review: dict
    tables: list[dict]
    served_by: str


def text_of(msg: AnyMessage) -> str:
    content = msg.content
    if isinstance(content, list):
        content = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return _THINK.sub("", content or "").strip()


def _load_kb(settings: Settings):
    index = settings.kb_dir / "index"
    if not (index / "chunks.jsonl").exists():
        return None
    from .kb.search import KnowledgeBase
    from .llm import get_embedder
    return KnowledgeBase.load(index, get_embedder(settings), settings.min_dense_score)


def build_graph(model=None, kb: Any = "auto", settings: Settings | None = None, checkpointer=None, forced_model=None):
    """forced_model: a tools model that must call a tool, used for the revision of a recognised calculation answered
    without one (SPEC 3.1). Built from the settings (vLLM only) when model is None; tests pass their own."""
    s = settings or get_settings()
    if model is None:
        from .llm import get_forced_tool_runnable, get_runnables
        model, model_with_tools = get_runnables(CALC_TOOLS, s)
        forced_model = get_forced_tool_runnable(CALC_TOOLS, s)
    else:
        model_with_tools = model.bind_tools(CALC_TOOLS)
    if kb == "auto":
        kb = _load_kb(s)
    budget = prompt_budget(s)

    def precheck(state: State) -> dict:
        prev = [m for m in state.get("messages", []) if isinstance(m, HumanMessage) and m.name != "reviewer"]
        # a follow-up of a follow-up ("And for 15 years?") is routed with the last two questions, not only the last one
        r = run_precheck(state.get("user_input", ""), " ".join(text_of(m) for m in prev[-2:]) if prev else None)
        return {"messages": [HumanMessage(content=r.text)], "question_idx": len(state.get("messages", [])),
                "flags": r.flags, "redactions": r.redactions, "amounts": r.amounts, "user_input": "",
                "sources": [], "sources_relevant": False, "best_dense": 0.0, "rag_status": "",
                "tool_iterations": 0, "revisions": 0, "review": {}, "tables": []}

    def retrieve(state: State, config) -> dict:
        if not config.get("configurable", {}).get("use_rag", True):
            return {"rag_status": "disabled"}
        if kb is None:
            return {"rag_status": "no_index"}
        msgs, qi = state["messages"], state["question_idx"]
        query = text_of(msgs[qi])
        if len(query.split()) < 6:  # short follow-up: add the previous user question for context
            prev = [m for m in msgs[:qi] if isinstance(m, HumanMessage) and m.name != "reviewer"]
            if prev:
                query = text_of(prev[-1]) + " " + query
        res = kb.search(query, k=s.top_k)
        return {"sources": res["hits"] if res["relevant"] else [], "sources_relevant": res["relevant"],
                "best_dense": res["best_dense"], "rag_status": "ok" if res["relevant"] else "no_match"}

    def agent(state: State) -> dict:
        msgs, qi = state["messages"], state["question_idx"]

        turns = earlier_turns(msgs[:qi])

        def ctx_for(sources) -> HumanMessage:
            return HumanMessage(context_block(text_of(msgs[qi]), state["flags"], state.get("redactions", []), sources,
                                              state.get("rag_status", "")))

        def model_input(trim: bool) -> list:
            """System prompt + at most the last 2 earlier turns as plain Q/A text + the current turn, within the budget.
            trim (after a context-length error): only the last turn, shortened; sources shortened; every table removed."""
            full = state.get("sources", [])
            short = [{**x, "text": x["text"][:400]} for x in full]
            levels = ([(1, 300, True, short)] if trim else
                      [(HISTORY_TURNS, None, False, full), (1, None, False, full), (1, 1500, False, full),
                       (1, 600, True, full), (1, 300, True, short), (0, None, True, short)])
            for k, chars, drop, sources in levels:
                candidate = [*history_messages(turns[-k:] if k else [], chars), ctx_for(sources),
                             *_for_model(msgs[qi + 1:], drop_tables=drop)]
                if estimate_tokens(candidate) <= budget:
                    break
            return [SystemMessage(SYSTEM_PROMPT), *candidate]

        capped = state.get("tool_iterations", 0) >= s.max_tool_iterations
        last = msgs[-1]
        codes = set(state.get("review", {}).get("codes", []))
        draft = next((m for m in reversed(msgs) if isinstance(m, AIMessage)), None)
        force = (forced_model is not None and not capped and calculation_question(state["flags"], text_of(msgs[qi]))
                 and isinstance(last, HumanMessage) and last.name == "reviewer" and FORCE_TOOL_CODES & codes
                 and "hypothetical_example" not in codes and not asks_for_inputs(text_of(draft) if draft else ""))
        runner = forced_model if force else model if capped else model_with_tools
        try:
            response = runner.invoke(model_input(trim=False))
        except Exception as exc:  # noqa: BLE001 - only a context-length error is retried
            if not is_context_overflow(exc):
                raise
            try:
                response = runner.invoke(model_input(trim=True))
            except Exception as exc2:  # noqa: BLE001
                if not is_context_overflow(exc2):
                    raise
                response = AIMessage(content=TOO_LONG)
        if capped and getattr(response, "tool_calls", None):  # defensive: never loop past the cap
            response = AIMessage(content=text_of(response) or "I could not finish the calculation.", id=response.id)
        if not getattr(response, "tool_calls", None) and isinstance(response.content, str):
            cleaned = clean_currency(response.content, state["flags"].get("currency", "INR"))
            if cleaned != response.content:
                response = response.model_copy(update={"content": cleaned})
        meta = getattr(response, "response_metadata", None) or {}
        return {"messages": [response], "served_by": meta.get("model_name") or meta.get("model") or "",
                "tool_iterations": state.get("tool_iterations", 0) + (1 if getattr(response, "tool_calls", None) else 0)}

    def tools(state: State) -> dict:
        """Run tool calls. Statement tools must only use figures from the user's message (SPEC 3.1)."""
        qi = state["question_idx"]
        user_text = text_of(state["messages"][qi])
        prev = earlier_turns(state["messages"][:qi])[-1:]   # a follow-up may reuse the previous turn's figures
        if prev:
            user_text += f"\n{prev[0]['question']}\n{inputs_line(prev[0]['calls'])}"
        # results of this turn's earlier successful tool calls (an EAC input may be a calculate result)
        results = [str(m.content) for m in state["messages"][qi + 1:] if isinstance(m, ToolMessage) and not _is_error(m.content)]
        out = []
        style = MONEY_STYLE.set(state.get("flags", {}).get("currency", "INR"))   # $ questions get $ formatting
        for tc in state["messages"][-1].tool_calls:
            fixed, arg_error = normalize_args(tc["name"], tc["args"])
            tc = {**tc, "args": deposit_timing(tc["name"], fixed, state.get("flags", {}), user_text)}
            bad = (ungrounded_values(tc["args"], user_text, tc["name"], "\n".join(results))
                   if tc["name"] in GROUNDED_TOOLS else [])
            zero_rates = invented_zero_rates(tc["name"], tc["args"], user_text)
            capital_gain = tc["name"] == "income_tax" and stock_profit_as_other_income(tc["args"], user_text)
            in_cents = cents_as_dollars(tc["name"], tc["args"], user_text)
            if arg_error:
                content = json.dumps({"error": arg_error}, ensure_ascii=False)
            elif in_cents:
                content = json.dumps({"error": f"The question gives this cost in cents ({in_cents[0]:g} cents), but "
                                      f"variable_cost_per_unit is in dollars. Pass variable_cost_cents_per_unit={in_cents[0]:g} "
                                      "instead and leave variable_cost_per_unit out; the tool converts cents to dollars."},
                                     ensure_ascii=False)
            elif capital_gain:
                content = json.dumps({"error": "other_income must not hold profit from selling shares or mutual funds: that is "
                                      "a capital gain, taxed at special rates outside the slabs and the 87A rebate. Call "
                                      "capital_gains for it (purchase and sale values, holding period) and income_tax with "
                                      "the salary figures only, then report both."}, ensure_ascii=False)
            elif bad and tc["name"] == "project_cash_flows":
                content = json.dumps({"error": f"These values are not in the question or in an earlier tool result: {', '.join(bad)}. "
                                      "Pass the figures exactly as the question gives them (sales, costs, investment, salvage, "
                                      "NWC, rates, MACRS rates); the tool works out depreciation, taxes, OCF, NWC recovery and "
                                      "after-tax salvage itself. Leave out sunk costs."}, ensure_ascii=False)
            elif bad and tc["name"] == "equivalent_annual_cost":
                content = json.dumps({"error": f"These values are not in the question or in an earlier tool result: {', '.join(bad)}. "
                                      + _EAC_ADVICE}, ensure_ascii=False)
            elif bad:
                content = json.dumps({"error": f"These values are not in the user's data: {', '.join(bad)}. Pass only figures "
                                      "that appear in the data, exactly as given. Leave out items that are not given, and give "
                                      "opening and closing balances instead of computing changes yourself."
                                      + (_SOLVER_ADVICE if tc["name"] == "solve_financials" else "") + _BALANCE_ADVICE},
                                     ensure_ascii=False)
            elif zero_rates == ["annual_cost"]:
                content = json.dumps({"error": "annual_cost=0 is not in the question. Give the yearly running cost, or for "
                                      "electricity pass power_watts, hours_per_year, price_per_kwh and lifetime_hours exactly as "
                                      "the question gives them; the tool works out the yearly cost and the life."})
            elif zero_rates:
                content = json.dumps({"error": f"{', '.join(f'{k}=0' for k in zero_rates)} is not in the question: a 0% rate is only "
                                      "valid if the question says so. If the rate is what you need to find, leave it out: loan_emi "
                                      "solves for whichever one of principal, annual_rate_pct, months and emi is missing."})
            elif tc["name"] not in TOOLS_BY_NAME:
                content = json.dumps({"error": f"Unknown tool {tc['name']}."})
            else:
                try:
                    content = TOOLS_BY_NAME[tc["name"]].invoke(tc["args"])
                except Exception as exc:  # invalid arguments: tell the model so it can retry
                    content = json.dumps({"error": _argument_error(tc["name"], exc)}, ensure_ascii=False)
                if tc["name"] == "solve_financials" and NO_INTEREST in str(content) and _MENTIONS_INTEREST.search(user_text):
                    content = json.dumps({"error": "The question mentions interest, so it cannot be taken as 0: pass the "
                                          "interest expense as interest=<amount> and call solve_financials again."})
            out.append(ToolMessage(content=content, tool_call_id=tc["id"], name=tc["name"]))
            if not _is_error(content):
                results.append(str(content))
        MONEY_STYLE.reset(style)
        return {"messages": out}

    def route_after_agent(state: State) -> str:
        return "tools" if getattr(state["messages"][-1], "tool_calls", None) else "review"

    def review(state: State) -> dict:
        msgs, qi = state["messages"], state["question_idx"]
        turn = msgs[qi + 1:]
        failed = {m.tool_call_id for m in turn if isinstance(m, ToolMessage) and str(m.content).lstrip().startswith('{"error"')}
        tools_used = [tc["name"] for m in turn if isinstance(m, AIMessage) for tc in (m.tool_calls or []) if tc["id"] not in failed]
        last_ai = next(m for m in reversed(msgs) if isinstance(m, AIMessage))
        answer = text_of(last_ai)
        restored = False
        notes = [i for i, m in enumerate(msgs) if i > qi and isinstance(m, HumanMessage) and m.name == "reviewer"]
        if notes:   # a Hindi/Tamil rewrite that lost its vowel signs: ship the clean first draft instead (SPEC 3.1)
            draft = next((m for m in reversed(msgs[:notes[-1]]) if isinstance(m, AIMessage) and not m.tool_calls), None)
            if draft is not None and script_damaged(text_of(draft), answer):
                answer, restored = text_of(draft), True
        srcs = state.get("sources", [])
        prev = earlier_turns(msgs[:qi])[-1:]     # a follow-up may quote the previous turn's question and tool results
        evidence = "\n".join([text_of(msgs[qi])] + [x["text"] for x in srcs]
                             + [m.content for m in turn if isinstance(m, ToolMessage)]
                             + [p["question"] + "\n" + "\n".join(p["results"]) for p in prev])
        result = review_answer(answer, state["flags"], tools_used, len(srcs), tuple(x.get("url", "") for x in srcs), evidence)
        prior = state.get("review", {}).get("first_draft_codes", [])
        if result.passed:
            out = {"review": {"passed": True, "codes": [], "first_draft_codes": prior, "footer": False}}
            cleaned = strip_tool_narration(answer)
            if cleaned != answer or restored:
                out["messages"] = [AIMessage(content=cleaned, id=last_ai.id)]
            return out
        if state.get("revisions", 0) < s.max_revisions:
            return {"messages": [HumanMessage(content=reviewer_note(result), name="reviewer")],
                    "revisions": state.get("revisions", 0) + 1,
                    "review": {"passed": False, "codes": result.codes, "first_draft_codes": result.codes}}
        if "hypothetical_example" in result.codes:
            answer = cut_hypothetical(answer)
        if "unverified_helpline" in result.codes:
            answer = prune_figures(answer, unverified_numbers(answer, evidence))
        checked = without_unverified(answer, text_of(msgs[qi]), evidence, tools_used, state["flags"])
        if checked != answer:
            answer = checked
        elif not answer.strip():   # the model gave up after the reviewer note: say so instead of showing only the footer
            answer = ("I could not complete this answer reliably. Please rephrase the question, or give the figures "
                      "(for example the cash flows year by year) and I will calculate it.")
        answer = strip_tool_narration(answer)
        return {"messages": [AIMessage(content=answer + FOOTER, id=last_ai.id)],
                "review": {"passed": False, "codes": result.codes, "first_draft_codes": prior, "footer": True}}

    def route_after_review(state: State) -> str:
        last = state["messages"][-1]
        return "agent" if isinstance(last, HumanMessage) and last.name == "reviewer" else "present"

    def present(state: State) -> dict:
        """Collect the exact tables the tools produced this turn (SPEC 4.2). Kept out of the model's memory."""
        tables = []
        for m in state["messages"][state["question_idx"] + 1:]:
            if isinstance(m, ToolMessage):
                try:
                    out = json.loads(m.content)
                except (TypeError, ValueError):
                    continue
                if isinstance(out, dict) and out.get("table"):
                    tables.append(out["table"])
        return {"tables": tables}

    g = StateGraph(State)
    g.add_node("precheck", precheck)
    g.add_node("retrieve", retrieve)
    g.add_node("agent", agent)
    g.add_node("tools", tools)
    g.add_node("review", review)
    g.add_node("present", present)
    g.add_edge(START, "precheck")
    g.add_edge("precheck", "retrieve")
    g.add_edge("retrieve", "agent")
    g.add_conditional_edges("agent", route_after_agent, {"tools": "tools", "review": "review"})
    g.add_edge("tools", "agent")
    g.add_conditional_edges("review", route_after_review, {"agent": "agent", "present": "present"})
    g.add_edge("present", END)
    return g.compile(checkpointer=checkpointer if checkpointer is not None else InMemorySaver())


_MD_TABLE = re.compile(r"(?:^[ \t]*\|.*\|[ \t]*\n?){2,}", re.M)


def strip_model_tables(text: str) -> str:
    """When exact tool tables exist, tables the model typed itself are removed so only exact figures are shown."""
    return re.sub(r"\n{3,}", "\n\n", _MD_TABLE.sub("", text)).strip()


_TABLE_ABOVE = re.compile(r"(?i)\b(the\s+)?(above\s+table|table\s+above)\b")


def fix_table_refs(text: str) -> str:
    """Tool tables render below the answer, so "the table above" / "the above table" become "the table below"."""
    def repl(m):
        out = ("the table below" if m.group(1) or m.group(2).lower().startswith("above") else "table below")
        return out[0].upper() + out[1:] if m.group(0)[0].isupper() else out
    return _TABLE_ABOVE.sub(repl, text)


def turn_trace(state: dict) -> dict:
    """Everything the UI's "How I got this" panel and the benchmark log need for the latest turn."""
    msgs, qi = state["messages"], state["question_idx"]
    turn = msgs[qi + 1:]
    results = {m.tool_call_id: m.content for m in turn if isinstance(m, ToolMessage)}
    tools = []
    for m in turn:
        if isinstance(m, AIMessage):
            for tc in m.tool_calls or []:
                raw = results.get(tc["id"], "")
                try:
                    parsed = json.loads(raw)
                except (TypeError, ValueError):
                    parsed = {"raw": raw}
                tools.append({"name": tc["name"], "args": tc["args"], "output": parsed})
    final = next((m for m in reversed(msgs) if isinstance(m, AIMessage)), None)
    tables = state.get("tables", [])
    answer_text = text_of(final) if final else ""
    if tables:
        answer_text = fix_table_refs(strip_model_tables(answer_text))
    return {
        "question_sent_to_model": text_of(msgs[qi]),
        "flags": state.get("flags", {}), "redactions": state.get("redactions", []), "amounts": state.get("amounts", []),
        "tools": tools,
        "sources": [{"n": i + 1, "title": h["title"], "page": h.get("page"), "url": h.get("url", ""), "file": h["file"],
                     "dense": h.get("dense"), "excerpt": h["text"][:300]} for i, h in enumerate(state.get("sources", []))],
        "rag_status": state.get("rag_status", ""), "best_dense": state.get("best_dense", 0.0),
        "review": state.get("review", {}), "revisions": state.get("revisions", 0),
        "served_by": state.get("served_by", ""),
        "answer_text": answer_text,
        "tables": tables,
        "answer": "\n\n".join([answer_text] + [to_markdown(t) for t in tables]),
    }


class FinBot:
    """Thin wrapper used by the CLI, the Streamlit app and the benchmark runner."""

    def __init__(self, model=None, kb: Any = "auto", settings: Settings | None = None, forced_model=None):
        self.settings = settings or get_settings()
        self.prompt_budget = prompt_budget(self.settings)
        self.graph = build_graph(model=model, kb=kb, settings=self.settings, forced_model=forced_model)

    def ask_iter(self, text: str, thread_id: str = "default", use_rag: bool = True):
        """Yields {"event": "step", "node": ..., "update": ...} per graph step, then {"event": "done", "trace": ...}."""
        cfg = {"configurable": {"thread_id": thread_id, "use_rag": use_rag}, "recursion_limit": 40}
        t0 = time.time()
        for chunk in self.graph.stream({"user_input": text}, cfg, stream_mode="updates"):
            for node, update in chunk.items():
                yield {"event": "step", "node": node, "update": update or {}}
        trace = turn_trace(self.graph.get_state(cfg).values)
        trace["latency_s"] = round(time.time() - t0, 2)
        yield {"event": "done", "trace": trace}

    def ask(self, text: str, thread_id: str = "default", use_rag: bool = True) -> dict:
        for event in self.ask_iter(text, thread_id, use_rag):
            if event["event"] == "done":
                return event["trace"]
        raise RuntimeError("graph finished without a result")
