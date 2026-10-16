"""Routing and conventions (SPEC 3.1, 4.0), currency clean-up, and the HRA city list (SPEC 4.3)."""
import pytest
from helpers import ScriptedModel, make_kb, tool_call

from finbot.graph import FinBot, clean_currency, deposit_timing
from finbot.precheck import precheck
from finbot.tools.calculators import sip_future_value
from finbot.tools.income_tax import income_tax

RETIRE = ("You plan to save for retirement over the next 30 years. You will invest $500 per month in a stock account and "
          "$200 per month in a bond account. The stock account returns 8 percent and the bond account 4 percent. How much "
          "will you have when you retire?")


def fv_ordinary(p, annual_pct, years):
    i = annual_pct / 1200
    return p * ((1 + i) ** (years * 12) - 1) / i


# ---------- bond account / bond fund is not bond pricing
def test_bond_account_is_not_bond_pricing():
    assert "bond" not in precheck(RETIRE).flags["calc_types"]
    assert "bond" not in precheck("I put ₹5,000 a month in a bond fund at 7% for 10 years. What will it grow to?").flags["calc_types"]
    assert "bond" in precheck("What is the price of a 5-year bond with a 7% coupon at a YTM of 8%?").flags["calc_types"]


# ---------- deposit timing: USD textbook = end of period unless "beginning"; INR SIP = start of month
def test_sip_timing_parameter():
    assert sip_future_value(monthly=500, annual_return_pct=8, years=30, timing="end")["result"]["corpus"] == \
        pytest.approx(fv_ordinary(500, 8, 30), abs=0.01)
    assert sip_future_value(monthly=500, annual_return_pct=8, years=30)["result"]["corpus"] == \
        pytest.approx(fv_ordinary(500, 8, 30) * (1 + 8 / 1200), abs=0.01)


@pytest.mark.parametrize("question,currency,passed,expected", [
    (RETIRE, "USD", {}, "end"),
    (RETIRE, "USD", {"timing": "start"}, "end"),                       # textbook convention wins without a "beginning" cue
    (RETIRE.replace("invest $500 per month", "invest $500 at the beginning of each month"), "USD", {}, "start"),
    ("I invest ₹10,000 every month in a SIP for 10 years at 12%. What will the corpus be?", "INR", {}, "start"),
    ("I save ₹10,000 at the end of each month for 10 years at 12%.", "INR", {"timing": "end"}, "end"),
])
def test_deposit_timing_convention(question, currency, passed, expected):
    args = {"monthly": 500, "annual_return_pct": 8, "years": 30, **passed}
    assert deposit_timing("sip_future_value", args, {"currency": currency}, question)["timing"] == expected
    assert deposit_timing("goal_saving", {"target": 1e6, "years": 10, "annual_return_pct": 8, **passed},
                          {"currency": currency}, question)["timing"] == expected
    assert deposit_timing("tvm", {"solve_for": "fv"}, {"currency": currency}, question) == {"solve_for": "fv"}


def test_graph_applies_end_of_month_to_a_dollar_question():
    model = ScriptedModel(responses=[
        tool_call("sip_future_value", {"monthly": 500, "annual_return_pct": 8, "years": 30}, "c1"),
        tool_call("sip_future_value", {"monthly": 200, "annual_return_pct": 4, "years": 30}, "c2"),
        f"The stock account grows to ${fv_ordinary(500, 8, 30):,.2f}."])
    t = FinBot(model=model, kb=make_kb()).ask(RETIRE, thread_id="usd")
    outs = [x["output"] for x in t["tools"]]
    assert all(o["inputs"]["timing"] == "end" for o in outs)
    assert outs[0]["result"]["corpus"] == pytest.approx(fv_ordinary(500, 8, 30), abs=0.01)


# ---------- currency clean-up of LaTeX-dollar soup
@pytest.mark.parametrize("raw,clean", [
    ("Year 0: $-$1,650,000$", "Year 0: -$1,650,000"),
    ("Year 1: $$412,500.00$ and year 2 too", "Year 1: $412,500.00 and year 2 too"),
    ("NPV is $-\\$1,650,000$.", "NPV is -$1,650,000."),
    ("OCF is $\\$412,500.00$ a year", "OCF is $412,500.00 a year"),
    ("Year 3: $540,000.00$.", "Year 3: $540,000.00."),
    ("A loss of $−$411,598.27$", "A loss of −$411,598.27"),
])
def test_currency_soup_is_cleaned(raw, clean):
    assert clean_currency(raw, "USD") == clean


@pytest.mark.parametrize("text", [
    "$$ PV = \\frac{1{,}100}{1.10} = 1{,}000 $$",
    "It costs $5 and $6 a year.",
    "The price is $1,018$$ hmm",
    "Inline maths \\( x = 5 \\) stays.",
])
def test_real_maths_and_plain_amounts_are_untouched(text):
    assert clean_currency(text, "USD") == text


def test_graph_cleans_the_final_answer():
    model = ScriptedModel(responses=["Your NPV is $-$1,650,000$ in year 0."])
    t = FinBot(model=model, kb=make_kb()).ask("Explain what a negative cash flow in year 0 means", thread_id="soup")
    assert "-$1,650,000" in t["answer_text"] and "$-$" not in t["answer"]


# ---------- income_tax names the 50% HRA cities for the year
def test_income_tax_lists_hra_50pct_cities():
    new = income_tax(gross_salary=3_000_000)["result"]["hra_50pct_cities"]
    old = income_tax(gross_salary=3_000_000, financial_year="2025-26")["result"]["hra_50pct_cities"]
    assert len(new) == 8 and {"Bengaluru", "Hyderabad", "Pune", "Ahmedabad", "Mumbai", "Delhi", "Kolkata", "Chennai"} == set(new)
    assert set(old) == {"Mumbai", "Delhi", "Kolkata", "Chennai"}
