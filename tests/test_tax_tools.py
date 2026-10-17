"""capital_gains, tax_rules and freelance income (SPEC 4.3, 4.4; test bank 2026-10-17). No model needed."""
import json

import pytest
from helpers import ScriptedModel, make_kb, tool_call

from finbot.graph import FinBot
from finbot.precheck import precheck
from finbot.tools.capital_gains import capital_gains
from finbot.tools.income_tax import income_tax
from finbot.tools.registry import TOOLS_BY_NAME
from finbot.tools.tax_rules import tax_rules


# ---------- capital_gains
def test_short_term_gain_on_listed_shares_is_taxed_at_20_percent():
    r = capital_gains(purchase_value=200_000, sale_value=260_000, holding_months=10)["result"]
    assert (r["gain"], r["term"], r["rate_pct"], r["tax_before_cess"], r["total_tax"]) == (60_000, "short-term", 20, 12_000, 12_480)


def test_long_term_gain_above_the_exemption_is_taxed_at_12_5_percent():
    r = capital_gains(purchase_value=300_000, sale_value=500_000, holding_months=18)["result"]
    assert (r["term"], r["exemption_applied"], r["taxable_gain"], r["tax_before_cess"], r["total_tax"]) == \
        ("long-term", 125_000, 75_000, 9_375, 9_750)


def test_long_term_gain_within_the_exemption_has_no_tax_and_used_exemption_counts():
    assert capital_gains(purchase_value=100_000, sale_value=200_000, holding_months=24)["result"]["total_tax"] == 0
    r = capital_gains(purchase_value=100_000, sale_value=200_000, holding_months=24, ltcg_exemption_used=100_000)["result"]
    assert (r["exemption_applied"], r["taxable_gain"], r["tax_before_cess"]) == (25_000, 75_000, 9_375)


def test_twelve_months_exactly_is_still_short_term_and_expenses_reduce_the_gain():
    r = capital_gains(purchase_value=100_000, sale_value=150_000, holding_months=12, expenses=1_000)["result"]
    assert (r["term"], r["gain"], r["tax_before_cess"]) == ("short-term", 49_000, 9_800)


def test_capital_gains_rejects_debt_funds_and_a_loss_has_no_tax():
    with pytest.raises(ValueError, match="slab"):
        capital_gains(purchase_value=100_000, sale_value=120_000, holding_months=30, asset="debt_fund")
    r = capital_gains(purchase_value=100_000, sale_value=80_000, holding_months=6)["result"]
    assert r["gain"] == -20_000 and r["total_tax"] == 0 and "loss" in " ".join(r["notes"]).lower()


def test_capital_gains_tool_is_registered_and_mentions_the_rebate_rule():
    out = json.loads(TOOLS_BY_NAME["capital_gains"].invoke({"purchase_value": 200000, "sale_value": 260000, "holding_months": 10}))
    assert out["result"]["tax_before_cess"] == 12_000 and any("87A" in n for n in out["result"]["notes"])


# ---------- tax_rules
def test_tax_rules_give_the_official_figures_used_by_the_calculators():
    r = tax_rules(financial_year="2025-26")["result"]
    assert r["new_regime"]["standard_deduction"] == 75_000 and r["new_regime"]["rebate_87a_max"] == 60_000
    assert r["new_regime"]["zero_tax_taxable_income"] == 1_200_000 and r["new_regime"]["zero_tax_salary"] == 1_275_000
    assert r["deductions"]["80ccd1b"] == 50_000 and r["deductions"]["80c"] == 150_000
    assert r["capital_gains"]["ltcg_rate_pct"] == 12.5 and r["capital_gains"]["ltcg_exemption"] == 125_000
    assert tax_rules(topic="capital_gains")["result"].keys() == {"capital_gains", "financial_year", "law", "sources"}


# ---------- freelance income: other_income, no standard deduction
def test_freelance_income_gets_no_standard_deduction_but_gets_marginal_relief():
    r = income_tax(gross_salary=0, other_income=1_250_000, financial_year="2025-26", regime="new")["result"]["by_regime"]["new"]
    assert r["total_deductions"] == 0 and r["total_tax"] == 52_000


def test_income_tax_still_needs_some_income():
    with pytest.raises(ValueError):
        income_tax(gross_salary=0, other_income=0)


# ---------- routing
@pytest.mark.parametrize("q", [
    "Bought listed shares for ₹2 lakh ten months ago and selling now for ₹2.6 lakh. What is the capital gains tax?",
    "I sold equity mutual fund units after 2 years with a profit of ₹3 lakh. How much LTCG tax will I pay?"])
def test_capital_gains_questions_are_routed_to_capital_gains(q):
    f = precheck(q).flags
    assert "capital_gains" in f["calc_types"] and "income_tax" not in f["calc_types"], f["calc_types"]


def test_salary_and_share_profit_question_keeps_both_routes():
    f = precheck("I earn 45k a month and made ₹2 lakh profit from stocks this year. What's my income tax and the tax on the profit?").flags
    assert {"capital_gains", "income_tax"} <= set(f["calc_types"])


@pytest.mark.parametrize("q", [
    "For a salaried taxpayer choosing the new regime in FY 2025-26, how much standard deduction is allowed?",
    "Over and above the ₹1.5 lakh limit of section 80C, how much extra deduction do I get for NPS under the old regime?"])
def test_tax_rule_questions_are_routed_to_tax_rules(q):
    assert "tax_rules" in precheck(q).flags["calc_types"]


def test_rule_answer_quoting_tax_rules_is_kept():
    q = "For a salaried taxpayer choosing the new regime in FY 2025-26, how much standard deduction is allowed?"
    model = ScriptedModel(responses=[tool_call("tax_rules", {"financial_year": "2025-26", "topic": "new_regime"}, "c1"),
                                     "The standard deduction under the new regime is ₹75,000 for FY 2025-26."])
    t = FinBot(model=model, kb=make_kb()).ask(q, thread_id="rules")
    assert "75,000" in t["answer_text"] and "could not compute" not in t["answer_text"]


# ---------- follow-ups from the fixed-code test-bank run (2026-10-17)
def test_new_regime_rules_say_80ccd1b_is_not_allowed():
    allowed = tax_rules(topic="new_regime")["result"]["new_regime"]["deductions_allowed"]
    assert "80CCD(1B)" in allowed and "no " in allowed


def test_income_tax_without_income_points_rule_questions_to_tax_rules():
    out = json.loads(TOOLS_BY_NAME["income_tax"].invoke({"regime": "new"}))
    assert "tax_rules" in out["error"]


def test_duration_and_two_stage_dividend_questions_may_be_answered_with_calculate():
    from finbot.routing import missing_tools
    assert missing_tools(["bond"], ["calculate"]) == [] and missing_tools(["perpetuity"], ["calculate"]) == []
    assert missing_tools(["bond"], ["loan_emi"]) == ["bond"]
