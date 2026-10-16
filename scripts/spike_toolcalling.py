"""M0 spike: does the local model call calculator tools correctly, and how fast?

Usage: uv run python scripts/spike_toolcalling.py [model]
Tools here are stubs with the real signatures; the spike measures tool *selection*
and *arguments*, not computation.
"""
import statistics
import sys
import time

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

MODEL = sys.argv[1] if len(sys.argv) > 1 else "qwen3.5:9b"


@tool
def loan_emi(principal: float, annual_rate_pct: float, months: int) -> str:
    """Monthly EMI for a loan. principal in rupees, annual_rate_pct like 8.5, months like 240."""
    return '{"emi": 17356.46}'


@tool
def sip_future_value(monthly: float, annual_return_pct: float, years: float, step_up_pct: float = 0) -> str:
    """Future value of a monthly SIP. monthly in rupees, annual_return_pct like 12, years like 10."""
    return '{"corpus": 2323390.76}'


@tool
def calculate(expression: str) -> str:
    """Evaluate an arithmetic expression such as '245000*0.15*1.18'. Plain numbers, no commas."""
    return '{"result": 43365.0}'


@tool
def npv_irr(cashflows: list[float], discount_rate_pct: float | None = None) -> str:
    """NPV and IRR of cash flows. cashflows[0] is usually the negative investment."""
    return '{"irr_pct": 8.8963}'


TOOLS = [loan_emi, sip_future_value, calculate, npv_irr]
SYSTEM = ("You are a finance assistant. Use a tool for every calculation; never do arithmetic yourself. "
          "If no calculation is needed, answer directly without tools.")

CASES = [
    ("What EMI will I pay on a ₹20 lakh loan at 8.5% for 20 years?", "loan_emi",
     lambda a: abs(a.get("principal", 0) - 2_000_000) < 1 and abs(a.get("annual_rate_pct", 0) - 8.5) < 1e-6 and int(a.get("months", 0)) == 240),
    ("If I invest ₹10,000 every month in a SIP at 12% for 10 years, how much will I have?", "sip_future_value",
     lambda a: abs(a.get("monthly", 0) - 10000) < 1 and abs(a.get("annual_return_pct", 0) - 12) < 1e-6 and abs(a.get("years", 0) - 10) < 1e-6),
    ("What is 15% of 2,45,000, plus 18% GST on that amount?", "calculate",
     lambda a: "245000" in a.get("expression", "").replace(",", "")),
    ("A project costs ₹1,000 and returns ₹300, ₹400 and ₹500 over three years. What is its IRR?", "npv_irr",
     lambda a: [round(x) for x in a.get("cashflows", [])] == [-1000, 300, 400, 500]),
    ("Explain what a mutual fund expense ratio is, in two sentences.", None, lambda a: True),
]


def run(reasoning: bool):
    llm = ChatOllama(model=MODEL, reasoning=reasoning, num_ctx=16384, temperature=0.6, keep_alive="30m")
    bound = llm.bind_tools(TOOLS)
    ok, times = 0, []
    for q, want, check in CASES:
        t0 = time.time()
        msg = bound.invoke([SystemMessage(SYSTEM), HumanMessage(q)])
        calls = msg.tool_calls
        good = (not calls) if want is None else (len(calls) >= 1 and calls[0]["name"] == want and check(calls[0]["args"]))
        # Full round trip: send the stub tool result back and get the final answer.
        if calls:
            msgs = [SystemMessage(SYSTEM), HumanMessage(q), msg] + [
                ToolMessage(content=next(t for t in TOOLS if t.name == c["name"]).invoke(c["args"]), tool_call_id=c["id"]) for c in calls]
            final = bound.invoke(msgs)
        else:
            final = msg
        dt = time.time() - t0
        times.append(dt)
        ok += good
        meta = final.response_metadata or {}
        tps = (meta.get("eval_count", 0) / (meta.get("eval_duration", 1) / 1e9)) if meta.get("eval_duration") else float("nan")
        print(f"  [{'PASS' if good else 'FAIL'}] {dt:5.1f}s  {tps:5.1f} tok/s  calls={[(c['name'], c['args']) for c in calls]}")
        print(f"         answer: {final.content[:140]!r}")
    print(f"  => {ok}/5 correct, median {statistics.median(times):.1f}s, max {max(times):.1f}s")


if __name__ == "__main__":
    print(f"model={MODEL}")
    for r in (False, True):
        print(f"\nthinking={'on' if r else 'off'}")
        run(r)
