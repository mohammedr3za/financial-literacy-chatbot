"""v4 step A (SPEC changelog 2026-10-17): tool-argument aliases, growing annuity, new routes. No model needed."""
import json

import pytest
from helpers import ScriptedModel, make_kb, tool_call

from finbot.graph import FinBot, normalize_args
from finbot.precheck import precheck
from finbot.routing import REQUIRED_TOOLS
from finbot.tools.calculators import perpetuity


# ---------- tool arguments: aliases mapped, other unknown names rejected (v3: 44 tvm calls lost the rate)
def test_rate_aliases_map_to_the_tools_rate_field():
    args, err = normalize_args("tvm", {"solve_for": "fv", "rate": 8, "years": 5, "pv": 1000})
    assert err is None and args == {"solve_for": "fv", "annual_rate_pct": 8, "years": 5, "pv": 1000}
    args, err = normalize_args("sip_future_value", {"monthly": 5000, "interest_rate": 12, "years": 10})
    assert err is None and args["annual_return_pct"] == 12
    args, err = normalize_args("goal_saving", {"target_amount": 1_000_000, "years": 10, "annual_return_pct": 12})
    assert err is None and args["target"] == 1_000_000


def test_other_unknown_arguments_are_rejected_with_a_suggestion():
    _, err = normalize_args("tvm", {"solve_for": "fv", "annual_rate_pct": 8, "yrs": 5, "pv": 1000})
    assert err and "yrs" in err and "years" in err


def test_strict_tools_and_correct_calls_are_untouched():
    call = {"principal": 2000000, "annual_rate_pct": 8.5, "months": 240}
    assert normalize_args("loan_emi", call) == (call, None)
    ok = {"solve_for": "fv", "annual_rate_pct": 8, "years": 5, "pv": 1000}
    assert normalize_args("tvm", ok) == (ok, None)


def test_graph_runs_a_tool_called_with_an_aliased_rate():
    q = "If I invest ₹1,000 at 8% a year for 5 years, what will it grow to?"
    model = ScriptedModel(responses=[tool_call("tvm", {"solve_for": "fv", "rate": 8, "years": 5, "pv": 1000}, "c1"),
                                     "It will grow to ₹1,469.33."])
    t = FinBot(model=model, kb=make_kb()).ask(q, thread_id="alias")
    assert t["tools"][0]["output"]["result"]["fv"] == pytest.approx(1469.33, abs=0.01)


# ---------- growing annuity (team report: lottery growing 3.2% for 25 years at 7.2%)
def test_growing_annuity_present_value():
    r, g, n, c = 0.072, 0.032, 25, 1_000_000
    expected = c / (r - g) * (1 - ((1 + g) / (1 + r)) ** n)
    assert perpetuity(cash_flow=c, rate_pct=7.2, growth_pct=3.2, periods=n)["result"]["pv"] == pytest.approx(expected, abs=0.01)
    assert perpetuity(cash_flow=100, rate_pct=5, growth_pct=5, periods=10)["result"]["pv"] == pytest.approx(10 * 100 / 1.05, abs=0.01)
    assert perpetuity(cash_flow=100, rate_pct=4, growth_pct=6, periods=10)["result"]["pv"] > 0      # r < g is fine when finite


# ---------- routes
@pytest.mark.parametrize("q,kind,tool", [
    ("What is the simple interest on ₹50,000 at 8% per annum for 3 years?", "simple_interest", "calculate"),
    ("A company needs $200,000 in 8 years. How much must it put into a sinking fund each year at 6%?", "sinking_fund", "tvm"),
    ("If I deposit ₹1.5 lakh in PPF at the start of every year for 15 years at 7.1%, what will I have?", "ppf", "tvm"),
    ("You win a lottery paying $1,000,000 in one year, growing 3.2% a year for 25 years. Discount rate is 7.2%. What is the present value?",
     "growing_annuity", "perpetuity"),
])
def test_new_routes(q, kind, tool):
    f = precheck(q).flags
    assert kind in f["calc_types"], f["calc_types"]
    assert tool in REQUIRED_TOOLS[kind]


def test_sip_future_value_needs_the_sip_tool():
    assert REQUIRED_TOOLS["sip_fv"] == {"sip_future_value"}
