"""Fixes from the live 27B smoke run (2026-10-17): cents in project costs, the balance-sheet balancing hint, follow-up
chains keep their calculation route, and "a guaranteed 9% saving" from paying off a loan is not a return guarantee."""
import re

from helpers import ScriptedModel, make_kb, tool_call

from finbot.graph import FinBot
from finbot.review import review_answer
from finbot.tools.calculators import project_cash_flows

BID = dict(years=4, tax_rate_pct=25, rate_pct=10, investment=1_200_000, depreciation="straight_line", units=2_000_000,
           fixed_costs=250_000, nwc_initial=150_000, nwc_annual_addition=20_000, salvage=100_000,
           opportunity_cost_now=300_000, opportunity_value_end=320_000, solve_bid_price=True)
BID_Q = ("Supply 2 million gaskets a year for four years. A plot of land worth $300,000 after tax today can be sold for "
         "$320,000 after tax in four years. Equipment costs $1.2 million, depreciated straight-line to zero over four years, "
         "and can be sold for $100,000 at the end. Net working capital is $150,000 now plus $20,000 each year after. "
         "Production costs are .40 cents per gasket and fixed costs are $250,000 a year. The tax rate is 25 percent and "
         "the required return 10 percent. What bid price should you submit?")


def a_factor(r, n):
    return (1 - (1 + r) ** -n) / r


def bid_by_hand():
    a4, a3, d4 = a_factor(0.10, 4), a_factor(0.10, 3), 1.1 ** -4
    return (1_650_000 + 118_500 * a4 + 20_000 * a3 - (150_000 + 60_000 + 75_000 + 320_000) * d4) / (1_500_000 * a4)


def test_cost_in_cents_has_its_own_argument():
    r = project_cash_flows(variable_cost_cents_per_unit=0.4, **BID)["result"]
    assert abs(r["bid_price"] - bid_by_hand()) < 1e-7


def test_cents_passed_as_dollars_are_bounced():
    model = ScriptedModel(responses=[
        tool_call("project_cash_flows", {**BID, "variable_cost_per_unit": 0.4}, "c1"),
        tool_call("project_cash_flows", {**BID, "variable_cost_cents_per_unit": 0.4}, "c2"),
        f"The bid price is ${bid_by_hand():.5f} per gasket."])
    t = FinBot(model=model, kb=make_kb()).ask(BID_Q, thread_id="cents")
    err = t["tools"][0]["output"]["error"]
    assert "cents" in err and "variable_cost_cents_per_unit" in err
    assert abs(t["tools"][1]["output"]["result"]["bid_price"] - bid_by_hand()) < 1e-7 and t["review"]["passed"], t["review"]


def test_derived_balance_sheet_total_gets_the_balancing_hint():
    q = "Cash $6,292, receivables $8,190, inventory $14,538, net fixed assets $51,218, payables $4,624, long-term debt $20,770. Prepare the balance sheet."
    model = ScriptedModel(responses=[
        tool_call("balance_sheet_check", {"assets": {"Cash": 6292, "Receivables": 8190, "Inventory": 14538, "Net fixed assets": 51218},
                                          "liabilities": {"Payables": 4624, "Long-term debt": 20770}, "equity": {"Total equity": 54854}}, "c1"),
        tool_call("balance_sheet_check", {"assets": {"Cash": 6292, "Receivables": 8190, "Inventory": 14538, "Net fixed assets": 51218},
                                          "liabilities": {"Payables": 4624, "Long-term debt": 20770}, "equity": {}}, "c2"),
        "Total assets are $80,238 and equity, the balancing figure, is $54,844."])
    t = FinBot(model=model, kb=make_kb()).ask(q, thread_id="bs")
    assert "{}" in t["tools"][0]["output"]["error"] and "balancing figure" in t["tools"][0]["output"]["error"]
    assert "error" not in t["tools"][1]["output"]


def test_follow_up_chain_keeps_the_loan_route():
    model = ScriptedModel(responses=[
        tool_call("loan_emi", {"principal": 2000000, "annual_rate_pct": 8.5, "months": 240}, "c1"), "Your EMI is ₹17,356.46.",
        tool_call("loan_emi", {"principal": 2000000, "annual_rate_pct": 9, "months": 240}, "c2"), "At 9% it is ₹17,994.52.",
        "For 15 years at 9% the EMI is ₹20,358.13.", "For 15 years at 9%, your EMI would be ₹20,358.13."])
    fb = FinBot(model=model, kb=make_kb())
    fb.ask("What EMI will I pay on a ₹20 lakh loan at 8.5% for 20 years?", thread_id="ch")
    fb.ask("What if the rate were 9%?", thread_id="ch")
    t = fb.ask("And for 15 years at 9%?", thread_id="ch")
    assert t["flags"]["calc_types"] and t["flags"]["calculation_intent"]
    assert "20,358.13" not in t["answer"] and "could not compute this reliably" in t["answer_text"]


def test_guaranteed_saving_from_paying_off_a_loan_is_not_a_return_guarantee():
    flags = {"calculation_intent": False}
    assert "guarantee_language" not in review_answer("Paying off the loan is a guaranteed 9% saving on interest.", flags, [], 0).codes
    assert "guarantee_language" in review_answer("This fund gives guaranteed 12% returns.", flags, [], 0).codes
    assert "guarantee_language" in review_answer("You get a guaranteed 12% every year.", flags, [], 0).codes
