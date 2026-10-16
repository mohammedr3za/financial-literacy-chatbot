"""equivalent_annual_cost (SPEC 4): textbook EAC with costs negative, its route, input grounding and the graph flow.
Expected values use the closed-form annuity factor written here, not the tool's code."""
import json

import pytest

from finbot.tools.calculators import equivalent_annual_cost

# The tool-value tests use the textbook bulb figures the user gave. The question-level tests below use a made-up scenario
# with different numbers: a $0.50 40-watt bulb (40 W × 750 h × $0.12/kWh = $3.60 a year; 1,500 h ÷ 750 h = 2 years) and a
# $4.00 9-watt LED ($0.81 a year; 15,000 h ÷ 750 h = 20 years), at 8%.
BULBS_Q = ("Compare two lamps. A 40-watt bulb costs $0.50 and lasts 1,500 hours; a 9-watt LED costs $4.00 and lasts "
           "15,000 hours. Power costs $0.12 per kilowatt-hour, the lamp is on 750 hours a year and the required "
           "return is 8 percent. What is the equivalent annual cost of each?")


def pvifa(r, n):
    return (1 - (1 + r) ** -n) / r


def test_incandescent_eac_matches_textbook():
    r = equivalent_annual_cost(purchase_cost=0.37, annual_cost=3.39, life_years=2, rate_pct=10)["result"]
    assert r["annuity_factor"] == pytest.approx(pvifa(0.1, 2), abs=1e-6)
    assert r["pv_of_costs"] == pytest.approx(-(0.37 + 3.39 * pvifa(0.1, 2)), abs=1e-4)      # −6.25347
    assert r["eac"] == pytest.approx(-3.6032, abs=1e-4)


def test_led_eac_matches_textbook():
    r = equivalent_annual_cost(purchase_cost=3, annual_cost=0.8475, life_years=24, rate_pct=10)["result"]
    assert r["pv_of_costs"] == pytest.approx(-10.61457, abs=1e-4)
    assert r["eac"] == pytest.approx(-1.1814, abs=1e-4)


def test_eac_agrees_with_npv_irr_equivalent_annual_amount():
    from finbot.tools.calculators import npv_irr
    a = npv_irr([-3.0] + [-0.8475] * 24, 10)["result"]["equivalent_annual_amount"]
    assert equivalent_annual_cost(3, 0.8475, 24, 10)["result"]["eac"] == pytest.approx(a, abs=1e-4)


def test_salvage_reduces_the_cost():
    # machine: 50,000 now, 8,000 a year, 5 years, salvage 5,000, 12%
    r = equivalent_annual_cost(purchase_cost=50_000, annual_cost=8_000, life_years=5, rate_pct=12, salvage=5_000)["result"]
    pv = -50_000 - 8_000 * pvifa(0.12, 5) + 5_000 / 1.12 ** 5
    assert r["pv_of_costs"] == pytest.approx(pv, abs=0.01)
    assert r["eac"] == pytest.approx(pv / pvifa(0.12, 5), abs=0.01)


def test_costs_given_as_negative_numbers_are_read_as_costs():
    r = equivalent_annual_cost(purchase_cost=-0.37, annual_cost=-3.39, life_years=2, rate_pct=10)["result"]
    assert r["eac"] == pytest.approx(-3.6032, abs=1e-4)


def test_electricity_usage_inputs_give_the_running_cost_and_life():
    inc = equivalent_annual_cost(purchase_cost=0.37, rate_pct=10, power_watts=60, hours_per_year=500, price_per_kwh=0.113,
                                 lifetime_hours=1000)["result"]
    assert inc["annual_cost"] == pytest.approx(3.39) and inc["life_years"] == 2
    assert inc["eac"] == pytest.approx(-3.6032, abs=1e-4)
    led = equivalent_annual_cost(purchase_cost=3, rate_pct=10, power_watts=15, hours_per_year=500, price_per_kwh=0.113,
                                 lifetime_hours=12000)
    assert led["result"]["annual_cost"] == pytest.approx(0.8475) and led["result"]["life_years"] == 24
    assert led["result"]["eac"] == pytest.approx(-1.1814, abs=1e-4)
    assert any("15 ÷ 1000 × 500 × 0.113" in w for w in led["working"])
    assert any("12,000 ÷ 500" in w for w in led["working"])


@pytest.mark.parametrize("kwargs,words", [
    ({"purchase_cost": 3, "rate_pct": 10, "life_years": 24}, "annual_cost"),                       # no running cost at all
    ({"purchase_cost": 3, "rate_pct": 10, "power_watts": 15, "life_years": 24}, "price_per_kwh"),  # usage incomplete
    ({"purchase_cost": 3, "rate_pct": 10, "annual_cost": 1}, "life_years"),                        # no life
    ({"purchase_cost": 3, "rate_pct": 10, "annual_cost": 1, "lifetime_hours": 1300, "hours_per_year": 500}, "whole number"),
    ({"purchase_cost": 3, "rate_pct": 10, "annual_cost": 5, "power_watts": 15, "hours_per_year": 500, "price_per_kwh": 0.113,
      "life_years": 24}, "does not match"),
])
def test_usage_inputs_are_checked(kwargs, words):
    with pytest.raises(ValueError, match=words):
        equivalent_annual_cost(**kwargs)


def test_zero_rate_and_table():
    out = equivalent_annual_cost(purchase_cost=100, annual_cost=10, life_years=4, rate_pct=0)
    assert out["result"]["annuity_factor"] == 4 and out["result"]["eac"] == pytest.approx(-35)
    t = out["table"]
    assert t["columns"][:2] == ["Year", "Cash flow"] and len(t["rows"]) == 6          # years 0-4 plus a total row
    assert t["rows"][0][1] == "(100.00)"


@pytest.mark.parametrize("kwargs,words", [
    ({"purchase_cost": 3, "annual_cost": 1, "life_years": 0, "rate_pct": 10}, "life_years"),
    ({"purchase_cost": 3, "annual_cost": 1, "life_years": 2.5, "rate_pct": 10}, "whole number"),
    ({"purchase_cost": 3, "annual_cost": 1, "life_years": 2, "rate_pct": -100}, "rate_pct"),
])
def test_bad_inputs_are_explained(kwargs, words):
    with pytest.raises(ValueError, match=words):
        equivalent_annual_cost(**kwargs)


def test_registered_tool_and_dollar_formatting():
    from finbot.tools.fmt import MONEY_STYLE
    from finbot.tools.registry import CALC_TOOLS, TOOLS_BY_NAME
    assert TOOLS_BY_NAME["equivalent_annual_cost"] in CALC_TOOLS
    tok = MONEY_STYLE.set("USD")
    try:
        out = json.loads(TOOLS_BY_NAME["equivalent_annual_cost"].invoke(
            {"purchase_cost": 0.37, "annual_cost": 3.39, "life_years": 2, "rate_pct": 10}))
    finally:
        MONEY_STYLE.reset(tok)
    assert out["formatted"]["eac"] == "-$3.60" and "table_shown_to_user" in out


# ---------- routing
@pytest.mark.parametrize("q", [
    BULBS_Q,
    "What is the EAC of a ₹2 lakh machine that costs ₹30,000 a year to run for 6 years at 11%?",
    "Find the equivalent annual annuity of a $3 bulb that costs $0.85 a year to run for 24 years at 10 percent.",
])
def test_eac_questions_route_to_the_eac_tool(q):
    from finbot.precheck import precheck
    from finbot.routing import HINTS, REQUIRED_TOOLS
    f = precheck(q).flags
    assert f["calc_types"] == ["eac"] and not f["statement_data"] and f["calculation_intent"]
    assert "equivalent_annual_cost" in REQUIRED_TOOLS["eac"] and HINTS["eac"].startswith("Call equivalent_annual_cost")
    assert "power_watts" in HINTS["eac"] and "lifetime_hours" in HINTS["eac"]


def test_npv_hint_no_longer_asks_for_long_cost_lists():
    from finbot.routing import HINTS
    assert "EVERY" not in HINTS["npv_irr"] and "equivalent_annual_cost" in HINTS["npv_irr"]


# ---------- grounding: inputs come from the question or an earlier tool result in the turn
def test_eac_inputs_must_come_from_the_question_or_a_tool_result():
    from finbot.graph import ungrounded_values
    args = {"purchase_cost": 0.50, "annual_cost": 3.6, "life_years": 2, "rate_pct": 8}
    assert ungrounded_values(args, BULBS_Q, "equivalent_annual_cost") == ["annual_cost=3.6", "life_years=2"]
    calc_out = json.dumps({"result": {"value": 3.5999999999999996, "rounded": 3.6}}) + json.dumps({"result": {"value": 2.0}})
    assert ungrounded_values(args, BULBS_Q, "equivalent_annual_cost", tool_text=calc_out) == []
    # a life in years that is neither stated nor a tool result is rejected, even when it is some quotient of two
    # question figures (a live run passed 12,000 ÷ 1,000 = 12 instead of 12,000 ÷ 500 = 24)
    led = {"purchase_cost": 4.00, "annual_cost": 0.81, "life_years": 20, "rate_pct": 8}
    assert ungrounded_values(led, BULBS_Q, "equivalent_annual_cost", tool_text='{"value": 0.8099999999999998}') == ["life_years=20"]
    assert ungrounded_values(led, BULBS_Q, "equivalent_annual_cost", tool_text='{"value": 0.81} {"value": 20.0}') == []
    # usage figures are copied from the question, so they are grounded directly
    usage = {"purchase_cost": 4.00, "rate_pct": 8, "power_watts": 9, "hours_per_year": 750, "price_per_kwh": 0.12,
             "lifetime_hours": 15000}
    assert ungrounded_values(usage, BULBS_Q, "equivalent_annual_cost") == []


def test_zero_rate_and_invented_zero_running_cost_are_bounced_for_eac():
    from finbot.graph import invented_zero_rates
    assert invented_zero_rates("equivalent_annual_cost", {"rate_pct": 0}, BULBS_Q) == ["rate_pct"]
    # a live run passed annual_cost=0 for both bulbs; zero is only accepted when the usage figures are given instead
    assert invented_zero_rates("equivalent_annual_cost", {"rate_pct": 8, "annual_cost": 0, "life_years": 2}, BULBS_Q) == ["annual_cost"]
    assert invented_zero_rates("equivalent_annual_cost", {"rate_pct": 8, "annual_cost": 0, "power_watts": 9,
                                                          "hours_per_year": 750, "price_per_kwh": 0.12}, BULBS_Q) == []
    assert invented_zero_rates("equivalent_annual_cost", {"rate_pct": 8, "annual_cost": 0},
                               "A press costs $9,000, has no running costs and lasts 3 years at 8%.") == []


def test_graph_eac_with_usage_figures_passes_review_in_one_round():
    from helpers import ScriptedModel, make_kb
    from finbot.graph import FinBot
    from langchain_core.messages import AIMessage
    usage = {"rate_pct": 8, "hours_per_year": 750, "price_per_kwh": 0.12}
    model = ScriptedModel(responses=[
        AIMessage(content="", tool_calls=[
            {"name": "equivalent_annual_cost", "args": {"purchase_cost": 0.5, "power_watts": 40, "lifetime_hours": 1500, **usage},
             "id": "c1", "type": "tool_call"},
            {"name": "equivalent_annual_cost", "args": {"purchase_cost": 4, "power_watts": 9, "lifetime_hours": 15000, **usage},
             "id": "c2", "type": "tool_call"}]),
        "The 40-watt bulb costs $3.60 a year over a 2-year life, so its EAC is -$3.88. The LED costs $0.81 a year over "
        "20 years, so its EAC is -$1.22."])
    t = FinBot(model=model, kb=make_kb()).ask(BULBS_Q, thread_id="eac-usage")
    assert [x["output"]["result"]["eac"] for x in t["tools"]] == pytest.approx(
        [-(0.5 + 3.6 * pvifa(0.08, 2)) / pvifa(0.08, 2), -(4 + 0.81 * pvifa(0.08, 20)) / pvifa(0.08, 20)], abs=1e-4)
    assert t["review"]["passed"], t["review"]
    assert t["revisions"] == 0 and len(t["tables"]) == 2


def test_graph_eac_flow_calculate_then_eac_passes_review():
    from helpers import ScriptedModel, make_kb, tool_call
    from finbot.graph import FinBot
    from langchain_core.messages import AIMessage

    def calls(*items):
        return AIMessage(content="", tool_calls=[{"name": n, "args": a, "id": i, "type": "tool_call"} for n, a, i in items])

    model = ScriptedModel(responses=[
        tool_call("equivalent_annual_cost", {"purchase_cost": 0.5, "annual_cost": 3.6, "life_years": 2, "rate_pct": 8}, "c0"),
        calls(("calculate", {"expression": "40/1000*750*0.12"}, "c1"), ("calculate", {"expression": "9/1000*750*0.12"}, "c2"),
              ("calculate", {"expression": "1500/750"}, "c5"), ("calculate", {"expression": "15000/750"}, "c6")),
        calls(("equivalent_annual_cost", {"purchase_cost": 0.5, "annual_cost": 3.6, "life_years": 2, "rate_pct": 8}, "c3"),
              ("equivalent_annual_cost", {"purchase_cost": 4, "annual_cost": 0.81, "life_years": 20, "rate_pct": 8}, "c4")),
        "The 40-watt bulb costs $3.60 a year to run over a 2-year life, so its EAC is -$3.88. The LED costs $0.81 a "
        "year over 20 years, so its EAC is -$1.22. The LED is cheaper per year."])
    t = FinBot(model=model, kb=make_kb()).ask(BULBS_Q, thread_id="eac")
    assert "error" in t["tools"][0]["output"] and "annual_cost=3.6" in t["tools"][0]["output"]["error"]
    assert "calculate" in t["tools"][0]["output"]["error"]
    eacs = [x["output"]["result"]["eac"] for x in t["tools"] if x["name"] == "equivalent_annual_cost" and "result" in x["output"]]
    expected = [-(0.5 + 3.6 * pvifa(0.08, 2)) / pvifa(0.08, 2), -(4 + 0.81 * pvifa(0.08, 20)) / pvifa(0.08, 20)]
    assert eacs == pytest.approx(expected, abs=1e-4)
    assert t["review"]["passed"], t["review"]
    assert len(t["tables"]) == 2
