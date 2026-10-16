"""Loan solver (any one of EMI, rate, months, principal), loan routing, invented-zero-rate guard, tool-name narration
check, and "table above" -> "table below". Expected values come from independent closed forms written here."""
import math

import pytest
from pydantic import ValidationError

from finbot.tools.calculators import loan_emi

SAILBOAT = ("You want to borrow $91,000 from your local bank to buy a new sailboat. You can afford to make monthly payments "
            "of $1,750, but no more. Assuming monthly compounding, what is the highest rate you can afford on a 60-month APR loan?")


def pv_annuity(pmt, r, n):
    return pmt * n if r == 0 else pmt * (1 - (1 + r) ** -n) / r


def independent_monthly_rate(principal, pmt, n):
    lo, hi = 1e-9, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if pv_annuity(pmt, mid, n) > principal else (lo, mid)
    return (lo + hi) / 2


# ---------- loan_emi solves whichever one value is missing
def test_loan_solves_for_rate_sailboat():
    r = loan_emi(principal=91_000, months=60, emi=1_750)["result"]
    i = independent_monthly_rate(91_000, 1_750, 60)
    assert round(r["annual_rate_pct"], 2) == round(i * 1200, 2) == 5.78
    assert r["annual_rate_pct"] == pytest.approx(i * 1200, abs=1e-4)
    assert r["effective_annual_rate_pct"] == pytest.approx(((1 + i) ** 12 - 1) * 100, abs=1e-4)
    assert r["solved_for"] == "annual_rate_pct" and r["total_payment"] == pytest.approx(105_000, abs=0.01)


def test_loan_solves_for_principal():
    r = loan_emi(annual_rate_pct=9, months=240, emi=25_000)["result"]
    assert r["principal"] == pytest.approx(pv_annuity(25_000, 0.0075, 240), abs=0.01)


def test_loan_solves_for_months_with_smaller_final_payment():
    out = loan_emi(principal=500_000, annual_rate_pct=10, emi=12_000)
    r, i = out["result"], 10 / 1200
    exact = -math.log(1 - 500_000 * i / 12_000) / math.log(1 + i)
    assert r["months_exact"] == pytest.approx(exact, abs=1e-3)
    assert r["months"] == math.ceil(exact) and 0 < r["final_payment"] < 12_000
    assert r["total_payment"] == pytest.approx(12_000 * (r["months"] - 1) + r["final_payment"], abs=0.05)
    assert out["table"]["rows"][-2][5] == "0"                                    # fully repaid in the last year


def test_loan_emi_still_works_and_zero_rate_is_allowed_when_given():
    assert loan_emi(principal=2_000_000, annual_rate_pct=8.5, months=240)["result"]["emi"] == pytest.approx(17356.46, abs=0.01)
    assert loan_emi(principal=120_000, annual_rate_pct=0, months=12)["result"]["emi"] == pytest.approx(10_000)


@pytest.mark.parametrize("kwargs,words", [
    ({"principal": 91_000, "months": 60}, "exactly three"),
    ({"principal": 91_000, "annual_rate_pct": 0, "months": 60, "emi": 1_750}, "Leave out"),
    ({"principal": 500_000, "annual_rate_pct": 10, "emi": 4_000}, "never be repaid"),
    ({"principal": 91_000, "months": 60, "emi": 1_000}, "do not even repay"),
])
def test_loan_bad_combinations_are_explained(kwargs, words):
    with pytest.raises(ValueError, match=words):
        loan_emi(**kwargs)


def test_loan_tool_schema_is_strict_and_solves_rate():
    import json
    from finbot.tools.registry import TOOLS_BY_NAME
    out = json.loads(TOOLS_BY_NAME["loan_emi"].invoke({"principal": 91000, "months": 60, "emi": 1750}))
    assert round(out["result"]["annual_rate_pct"], 2) == 5.78
    with pytest.raises(ValidationError):
        TOOLS_BY_NAME["loan_emi"].invoke({"principal": 91000, "months": 60, "payment": 1750})


# ---------- routing
@pytest.mark.parametrize("q", [
    SAILBOAT,
    "How much can I borrow if I can pay ₹25,000 a month for 20 years at 9%?",
    "How many months will it take to repay a ₹5 lakh loan at 10% if I pay ₹12,000 a month?",
    "If I take a home loan of ₹30 lakh, what will my monthly payment be at 8.75% for 25 years?",
    "What is the APR on a car loan of $10,000 with monthly payments of $300 for 36 months?",
])
def test_loan_questions_route_to_loan_emi(q):
    from finbot.precheck import precheck
    from finbot.routing import REQUIRED_TOOLS
    f = precheck(q).flags
    kinds = f["calc_types"]
    assert f["calculation_intent"] and kinds and all(REQUIRED_TOOLS[k] == {"loan_emi"} for k in kinds), kinds


def test_loan_hint_says_leave_out_the_unknown():
    from finbot.routing import HINTS
    for k in ("loan", "emi"):
        assert "never pass 0" in HINTS[k] and "emi" in HINTS[k] and "months" in HINTS[k]


def test_savings_questions_are_not_loans():
    from finbot.precheck import precheck
    assert "loan" not in precheck("If I invest ₹10,000 every month in a SIP for 10 years at 12%, what will the corpus be?").flags["calc_types"]
    assert "loan" not in precheck("How many years will money take to double at 9% a year compounded annually?").flags["calc_types"]


# ---------- invented zero rates are bounced
def test_zero_rate_is_rejected_unless_the_question_says_so():
    from finbot.graph import invented_zero_rates
    assert invented_zero_rates("loan_emi", {"principal": 91000, "annual_rate_pct": 0, "months": 60}, SAILBOAT) == ["annual_rate_pct"]
    assert invented_zero_rates("goal_saving", {"target": 1e7, "years": 10, "annual_return_pct": 0}, "Reach ₹1 crore in 10 years") \
        == ["annual_return_pct"]
    for ok in ("an interest-free loan of ₹1,20,000 for 12 months", "a 0% loan of ₹50,000", "at zero interest", "with no interest"):
        assert invented_zero_rates("loan_emi", {"principal": 120000, "annual_rate_pct": 0, "months": 12}, ok) == []
    assert invented_zero_rates("tvm", {"solve_for": "annual_rate_pct", "annual_rate_pct": 0, "pv": 1, "fv": 2, "years": 5}, "x") == []
    assert invented_zero_rates("calculate", {"expression": "0"}, "x") == []


def test_graph_bounces_invented_zero_rate_then_solves_rate():
    from helpers import ScriptedModel, make_kb, tool_call
    from finbot.graph import FinBot
    model = ScriptedModel(responses=[
        tool_call("loan_emi", {"principal": 91000, "annual_rate_pct": 0, "months": 60}, "c1"),
        tool_call("loan_emi", {"principal": 91000, "months": 60, "emi": 1750}, "c2"),
        "The highest rate you can afford is an APR of about 5.78% (monthly rate × 12)."])
    t = FinBot(model=model, kb=make_kb()).ask(SAILBOAT, thread_id="sail")
    assert "error" in t["tools"][0]["output"] and "annual_rate_pct=0" in t["tools"][0]["output"]["error"]
    assert round(t["tools"][1]["output"]["result"]["annual_rate_pct"], 2) == 5.78
    assert t["review"]["passed"], t["review"]


# ---------- answers must not narrate tool calls
@pytest.mark.parametrize("text", [
    "The loan_emi function was called with the inputs principal 91000 and months 60.",
    "Using the calculate tool, the result is 5.",
    "I called tvm(solve_for='fv') and got ₹1,40,255.",
    "The tool was called with these values.",
])
def test_tool_narration_is_flagged(text):
    from finbot.review import review_answer
    r = review_answer(text, {"calculation_intent": True}, ["loan_emi"], 0)
    assert "tool_narration" in r.codes and "do not describe tool calls" in " ".join(r.issues)


@pytest.mark.parametrize("text", [
    "Let me calculate it: the EMI is ₹17,356.46.",
    "The TVM formula discounts each payment; your EMI is ₹17,356.46.",
])
def test_plain_words_are_not_narration(text):
    from finbot.review import review_answer
    assert "tool_narration" not in review_answer(text, {"calculation_intent": True}, ["loan_emi"], 0).codes


# ---------- tables render below the answer
def test_table_above_becomes_table_below_when_a_table_is_attached():
    from helpers import ScriptedModel, make_kb, tool_call
    from finbot.graph import FinBot, fix_table_refs
    assert fix_table_refs("The table above shows it. See the above table.") == "The table below shows it. See the table below."
    model = ScriptedModel(responses=[
        tool_call("loan_emi", {"principal": 2000000, "annual_rate_pct": 8.5, "months": 240}, "c1"),
        "Your EMI is ₹17,356.46. The table above shows the year-wise split."])
    t = FinBot(model=model, kb=make_kb()).ask("What EMI will I pay on a ₹20 lakh loan at 8.5% for 20 years?", thread_id="tb")
    assert "The table below shows" in t["answer_text"] and "table above" not in t["answer"]


# ---------- a table of project cash flows is not a financial statement
def test_project_cash_flow_table_routes_to_npv_only():
    from finbot.precheck import precheck
    q = ("Choose between two projects:\n| Year | Cash flow (A) | Cash flow (B) |\n|---:|---:|---:|\n| 0 | -$50,000 | -$20,000 |\n"
         "| 1 | 22,000 | 9,000 |\n| 2 | 22,000 | 9,000 |\n| 3 | 22,000 | 9,000 |\nAt a required return of 12 percent, what is the "
         "profitability index of each project?")
    f = precheck(q).flags
    assert f["calc_types"] == ["npv_irr"] and not f["statement_data"]
    assert precheck("What is the equivalent annual cost of a $3 bulb that costs $0.85 a year to run for 24 years at 10 percent?"
                    ).flags["calc_types"] == ["eac"]          # its own route since 2026-10-16 (SPEC 4.0)
    pasted = "Balance sheet\n| Item | Amount |\n| Cash | 1,20,000 |\n| Inventory | 2,50,000 |\n| Current liabilities | 1,80,000 |"
    assert "statement" in precheck(pasted + "\nCompute the current ratio").flags["calc_types"]


def test_empty_answer_after_revision_gets_a_fallback_not_just_the_footer():
    from helpers import ScriptedModel, make_kb
    from finbot.graph import FinBot
    model = ScriptedModel(responses=["", ""])
    t = FinBot(model=model, kb=make_kb()).ask("What is the equivalent annual cost of a $3 bulb at 10 percent?", thread_id="empty")
    assert t["review"]["footer"] and "could not complete" in t["answer_text"]
