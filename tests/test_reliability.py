"""Calculation reliability layer (SPEC 4.0). Expected values use plain closed-form formulas written here."""
import pytest

from finbot.precheck import precheck
from finbot.review import review_answer
from finbot.tools.calculators import effective_annual_rate, goal_saving, real_return, sip_future_value


def annuity(i, n):
    return ((1 + i) ** n - 1) / i


# ---------- goal_saving
def test_goal_saving_end_of_month_matches_formula():
    r = goal_saving(target=10_000_000, years=15, annual_return_pct=12, timing="end")["result"]
    assert r["monthly_required"] == pytest.approx(10_000_000 / annuity(0.01, 180), abs=0.01)   # ≈ 20,016.81
    assert r["check_target_reached"] == pytest.approx(10_000_000, abs=1)


def test_goal_saving_sip_timing_default_is_start_of_month():
    r = goal_saving(target=10_000_000, years=15, annual_return_pct=12)["result"]
    assert r["monthly_required"] == pytest.approx(10_000_000 / (annuity(0.01, 180) * 1.01), abs=0.01)  # ≈ 19,818.62
    sip = sip_future_value(monthly=r["monthly_required"], annual_return_pct=12, years=15)["result"]["corpus"]
    assert sip == pytest.approx(10_000_000, abs=1)   # consistent with the SIP tool


def test_goal_saving_with_existing_savings_and_step_up():
    base = goal_saving(target=10_000_000, years=15, annual_return_pct=12, timing="end")["result"]["monthly_required"]
    r = goal_saving(target=10_000_000, years=15, annual_return_pct=12, current_savings=500_000, timing="end")["result"]
    remaining = 10_000_000 - 500_000 * 1.01 ** 180
    assert r["monthly_required"] == pytest.approx(remaining / annuity(0.01, 180), abs=0.01) and r["monthly_required"] < base
    s = goal_saving(target=10_000_000, years=15, annual_return_pct=12, step_up_pct=10)["result"]
    assert s["monthly_required"] < base and s["check_target_reached"] == pytest.approx(10_000_000, abs=1)


def test_goal_saving_already_reached_and_bad_input():
    assert goal_saving(target=100_000, years=10, annual_return_pct=12, current_savings=100_000)["result"]["monthly_required"] == 0
    with pytest.raises(ValueError):
        goal_saving(target=-5, years=10, annual_return_pct=12)


def test_goal_saving_has_year_table_ending_at_target():
    t = goal_saving(target=10_000_000, years=15, annual_return_pct=12)["table"]
    assert len(t["rows"]) == 15
    assert float(t["rows"][-1][4].replace(",", "")) == pytest.approx(10_000_000, abs=1)


# ---------- real return, effective rate
def test_real_return():
    r = real_return(nominal_pct=8, inflation_pct=6)["result"]
    assert r["real_return_pct"] == pytest.approx((1.08 / 1.06 - 1) * 100, abs=1e-4) and r["approximation_pct"] == 2.0


def test_effective_annual_rate():
    assert effective_annual_rate(nominal_pct=12, periods_per_year=12)["result"]["effective_pct"] == pytest.approx(12.6825, abs=1e-4)
    assert effective_annual_rate(nominal_pct=8, periods_per_year=4)["result"]["effective_pct"] == pytest.approx(8.2432, abs=1e-4)


# ---------- router
@pytest.mark.parametrize("q,kind", [
    ("How much should I save each month to reach ₹1 crore in 15 years at 12%?", "goal_saving"),
    ("How much must I save at the end of each month to reach ₹1 crore in 15 years at 12% a year compounded monthly?", "goal_saving"),
    ("What monthly SIP do I need for a corpus of ₹50 lakh in 10 years at 11%?", "goal_saving"),
    ("How many years will money take to double at 9% a year compounded annually?", "years"),
    ("If my FD pays 8% and inflation is 6%, what is my real return?", "real_return"),
    ("What is the effective annual rate of 12% a year compounded monthly?", "effective_rate"),
    ("What is the EMI on a ₹5 lakh car loan at 9.25% for 5 years?", "emi"),
    ("If I invest ₹10,000 every month in a SIP for 10 years at 12%, what will the corpus be?", "sip_fv"),
    ("What is 15% of ₹2,45,000, plus 18% GST on that amount?", "percent"),
    ("What is the IRR of investing ₹50,000 and receiving ₹15,000, ₹20,000 and ₹25,000?", "npv_irr"),
])
def test_router_detects_type(q, kind):
    assert kind in precheck(q).flags["calc_types"]


def test_goal_question_is_not_also_routed_to_sip_fv():
    types = precheck("How much should I invest every month in a SIP to reach ₹1 crore in 15 years at 12%?").flags["calc_types"]
    assert "goal_saving" in types and "sip_fv" not in types


def test_concept_question_has_no_calc_type():
    assert precheck("Explain duration vs convexity").flags["calc_types"] == []


# ---------- enforcement and unsourced figures
def _flags(q):
    return precheck(q).flags


def test_wrong_tool_for_goal_is_rejected():
    q = "How much should I save each month to reach ₹1 crore in 15 years at 12%?"
    r = review_answer("Save ₹10,00,000 a month.", _flags(q), ["sip_future_value"], 0, (), q)
    assert "wrong_tool" in r.codes and "goal_saving" in " ".join(r.issues)


def test_rule_of_72_without_tool_is_rejected():
    q = "How many years will money take to double at 9% a year compounded annually?"
    r = review_answer("By the Rule of 72, it takes 72/9 = 8 years.", _flags(q), [], 0, (), q)
    assert "wrong_tool" in r.codes or "no_tool_for_numbers" in r.codes


def test_calc_answer_figures_must_come_from_tools_or_question():
    q = "What will ₹1,00,000 grow to in 5 years at 7% compounded annually?"
    evidence = q + ' {"result": {"fv": 140255.17}}'
    ok = review_answer("It grows to ₹1,40,255.17 (about ₹1.40 lakh) at 7%.", _flags(q), ["tvm"], 0, (), evidence)
    assert ok.passed, ok.codes
    bad = review_answer("It grows to ₹1,45,000.", _flags(q), ["tvm"], 0, (), evidence)
    assert "unsupported_figures" in bad.codes


def test_followup_inherits_previous_question_type():
    prev = "How much should I save each month to reach ₹1 crore in 15 years at 12%?"
    f = precheck("What if I already have ₹5 lakh saved?", previous_question=prev).flags
    assert f["followup"] and "goal_saving" in f["calc_types"]
    assert precheck("What if I already have ₹5 lakh saved?").flags["calc_types"] == []   # alone, nothing to route


def test_long_new_question_is_not_a_followup():
    prev = "How much should I save each month to reach ₹1 crore in 15 years at 12%?"
    f = precheck("Explain how the RBI ombudsman scheme handles complaints about bank charges and how long it takes", previous_question=prev).flags
    assert not f["followup"]


def test_followup_without_tool_is_rejected_in_graph():
    from helpers import ScriptedModel, make_kb, tool_call
    from finbot.graph import FinBot
    model = ScriptedModel(responses=[
        tool_call("goal_saving", {"target": 10000000, "years": 15, "annual_return_pct": 12}, "c1"), "Save ₹19,818.62 a month.",
        "Step 1: Use the goal_saving function with current savings of ₹5 lakh.",          # describes, does not call
        tool_call("goal_saving", {"target": 10000000, "years": 15, "annual_return_pct": 12, "current_savings": 500000}, "c2"),
        "With ₹5 lakh already saved, you need about ₹13,999 a month."])
    fb = FinBot(model=model, kb=make_kb())
    fb.ask("How much should I save each month to reach ₹1 crore in 15 years at 12%?", thread_id="fu")
    t = fb.ask("What if I already have ₹5 lakh saved?", thread_id="fu")
    assert t["revisions"] == 1 and t["tools"][-1]["args"]["current_savings"] == 500000


def test_sip_question_is_not_also_lump_sum():
    t = precheck("If I put ₹10,000 a month in a SIP for 15 years at 12% a year, how much will I have?").flags["calc_types"]
    assert t == ["sip_fv"]


@pytest.mark.parametrize("q", [
    "explain to me what is npv and how it works, and explain why we do discounting eli5",
    "What is IRR? eli5",
    "Explain bond duration in 2 sentences",
])
def test_concept_questions_without_quantities_are_not_routed(q):
    f = precheck(q).flags
    assert f["calc_types"] == [] and not f["calculation_intent"]


@pytest.mark.parametrize("q,kind", [
    ("What is the NPV at 10% of a project costing ₹50,000 with inflows of ₹20,000 for 3 years?", "npv_irr"),
    ("How many years to double at 9%?", "years"),
    ("EMI on 50k for 2 years at 14%", "emi"),
])
def test_real_quantities_still_route(q, kind):
    assert kind in precheck(q).flags["calc_types"]


def test_usd_questions_format_in_dollars():
    from finbot.tools.calculators import loan_emi
    from finbot.tools.fmt import MONEY_STYLE, format_inr, group
    assert precheck("Tralee has current liabilities of $445,000 and a quick ratio of .84").flags["currency"] == "USD"
    assert precheck("EMI on ₹20 lakh at 8.5%").flags["currency"] == "INR"
    tok = MONEY_STYLE.set("USD")
    try:
        assert format_inr(1228200) == "$1,228,200.00" and group(1228200) == "1,228,200" and group(-5000) == "(5,000)"
        assert loan_emi(principal=100000, annual_rate_pct=10, months=12)["table"]["title"].endswith("($)")
    finally:
        MONEY_STYLE.reset(tok)
    assert format_inr(1228200) == "₹12,28,200.00"   # default untouched


def test_dollar_and_plain_figures_are_checked_in_calculation_answers():
    q = "Nakatomi has sales of $675,000, costs of $337,000, depreciation of $81,000, interest of $50,500 and a 23 percent tax rate. What is net income?"
    f = precheck(q).flags
    bad = review_answer("EBIT is 257,000, so net income is $159,005.", f, ["calculate"], 0, (), q)
    assert "unsupported_figures" in bad.codes and "257,000" in bad.issues[-1]
    ok = review_answer("EBIT is 257,000, so net income is $159,005.", f, ["calculate"], 0, (),
                       q + ' {"steps": [{"value": 257000.0}], "value": 159005.0}')
    assert ok.passed, ok.codes


def test_failed_tool_call_does_not_count_as_using_a_tool():
    from helpers import ScriptedModel, make_kb, tool_call
    from finbot.graph import FinBot
    q = "Nakatomi has sales of $675,000, costs of $337,000, depreciation of $81,000, interest of $50,500 and a 23 percent tax rate. What is net income?"
    model = ScriptedModel(responses=[
        tool_call("financial_ratios", {"revenue": 675000, "tax_rate": 23}, "c1"),          # invalid argument -> error
        "Net income is $159,005.",                                                           # mental arithmetic -> rejected
        tool_call("calculate", {"expression": "(675000-337000-81000-50500)*(1-0.23)"}, "c2"),
        "Net income is $159,005."])
    t = FinBot(model=model, kb=make_kb()).ask(q, thread_id="failed-tool")
    assert t["revisions"] == 1 and t["review"]["passed"] and "no_tool_for_numbers" in t["review"]["first_draft_codes"]


@pytest.mark.parametrize("q,regulatory", [
    ("Nakatomi has sales of $675,000 and a tax rate of 23 percent. What is net income?", False),
    ("A firm earns ₹10 lakh and pays a 25% tax rate. What is its net income?", False),
    ("How are crypto gains taxed in India?", True),
    ("What is the new tax regime slab for ₹12 lakh income?", True),
])
def test_given_tax_rate_is_an_input_not_a_rule(q, regulatory):
    assert precheck(q).flags["regulatory_topic"] is regulatory
