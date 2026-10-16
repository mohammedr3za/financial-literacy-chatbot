"""equivalent_annual_cost (SPEC 4): textbook EAC with costs negative, its route, input grounding and the graph flow.
Expected values use the closed-form annuity factor written here, not the tool's code."""
import json

import pytest

from finbot.tools.calculators import equivalent_annual_cost

# Our own scenario (not a course-set problem): two circulation pumps at 7%, run 2,000 hours a year at $0.15/kWh.
# Standard pump: $180, 750 W (0.75 kW × 2,000 h × $0.15 = $225 a year), 6,000 h ÷ 2,000 h = 3 years.
# Efficient pump: $420, 550 W (0.55 kW × 2,000 h × $0.15 = $165 a year), 16,000 h ÷ 2,000 h = 8 years.
# PVIFA(7%, 3) = 2.624316, so EAC = −(180 + 225 × 2.624316) ÷ 2.624316 = −770.4711 ÷ 2.624316 = −293.5893;
# PVIFA(7%, 8) = 5.971299, so EAC = −(420 + 165 × 5.971299) ÷ 5.971299 = −1,405.2643 ÷ 5.971299 = −235.3365.
PUMPS_Q = ("Compare two circulation pumps. A standard pump costs $180, draws 750 watts and lasts 6,000 hours; an efficient "
           "pump costs $420, draws 550 watts and lasts 16,000 hours. Power costs $0.15 per kilowatt-hour, the pump runs "
           "2,000 hours a year and the required return is 7 percent. What is each pump's equivalent annual cost?")


def pvifa(r, n):
    return (1 - (1 + r) ** -n) / r


def test_short_life_asset_eac_matches_closed_form():
    r = equivalent_annual_cost(purchase_cost=180, annual_cost=225, life_years=3, rate_pct=7)["result"]
    assert r["annuity_factor"] == pytest.approx(pvifa(0.07, 3), abs=1e-6)
    assert r["pv_of_costs"] == pytest.approx(-(180 + 225 * pvifa(0.07, 3)), abs=1e-4)      # −770.4711
    assert r["eac"] == pytest.approx(-293.5893, abs=1e-4)


def test_long_life_asset_eac_matches_closed_form():
    r = equivalent_annual_cost(purchase_cost=420, annual_cost=165, life_years=8, rate_pct=7)["result"]
    assert r["pv_of_costs"] == pytest.approx(-1405.2643, abs=1e-4)
    assert r["eac"] == pytest.approx(-235.3365, abs=1e-4)


def test_eac_agrees_with_npv_irr_equivalent_annual_amount():
    from finbot.tools.calculators import npv_irr
    a = npv_irr([-420.0] + [-165.0] * 8, 7)["result"]["equivalent_annual_amount"]
    assert equivalent_annual_cost(420, 165, 8, 7)["result"]["eac"] == pytest.approx(a, abs=1e-4)


def test_salvage_reduces_the_cost():
    # machine: 50,000 now, 8,000 a year, 5 years, salvage 5,000, 12%
    r = equivalent_annual_cost(purchase_cost=50_000, annual_cost=8_000, life_years=5, rate_pct=12, salvage=5_000)["result"]
    pv = -50_000 - 8_000 * pvifa(0.12, 5) + 5_000 / 1.12 ** 5
    assert r["pv_of_costs"] == pytest.approx(pv, abs=0.01)
    assert r["eac"] == pytest.approx(pv / pvifa(0.12, 5), abs=0.01)


def test_costs_given_as_negative_numbers_are_read_as_costs():
    r = equivalent_annual_cost(purchase_cost=-180, annual_cost=-225, life_years=3, rate_pct=7)["result"]
    assert r["eac"] == pytest.approx(-293.5893, abs=1e-4)


def test_electricity_usage_inputs_give_the_running_cost_and_life():
    std = equivalent_annual_cost(purchase_cost=180, rate_pct=7, power_watts=750, hours_per_year=2000, price_per_kwh=0.15,
                                 lifetime_hours=6000)["result"]
    assert std["annual_cost"] == pytest.approx(225) and std["life_years"] == 3
    assert std["eac"] == pytest.approx(-293.5893, abs=1e-4)
    eff = equivalent_annual_cost(purchase_cost=420, rate_pct=7, power_watts=550, hours_per_year=2000, price_per_kwh=0.15,
                                 lifetime_hours=16000)
    assert eff["result"]["annual_cost"] == pytest.approx(165) and eff["result"]["life_years"] == 8
    assert eff["result"]["eac"] == pytest.approx(-235.3365, abs=1e-4)
    assert any("550 ÷ 1000 × 2,000 × 0.15" in w for w in eff["working"])
    assert any("16,000 ÷ 2,000" in w for w in eff["working"])


@pytest.mark.parametrize("kwargs,words", [
    ({"purchase_cost": 420, "rate_pct": 7, "life_years": 8}, "annual_cost"),                         # no running cost at all
    ({"purchase_cost": 420, "rate_pct": 7, "power_watts": 550, "life_years": 8}, "price_per_kwh"),   # usage incomplete
    ({"purchase_cost": 420, "rate_pct": 7, "annual_cost": 1}, "life_years"),                         # no life
    ({"purchase_cost": 420, "rate_pct": 7, "annual_cost": 1, "lifetime_hours": 4500, "hours_per_year": 2000}, "whole number"),
    ({"purchase_cost": 420, "rate_pct": 7, "annual_cost": 200, "power_watts": 550, "hours_per_year": 2000, "price_per_kwh": 0.15,
      "life_years": 8}, "does not match"),                                                          # 200 vs 165
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
    ({"purchase_cost": 50, "annual_cost": 1, "life_years": 0, "rate_pct": 7}, "life_years"),
    ({"purchase_cost": 50, "annual_cost": 1, "life_years": 2.5, "rate_pct": 7}, "whole number"),
    ({"purchase_cost": 50, "annual_cost": 1, "life_years": 2, "rate_pct": -100}, "rate_pct"),
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
            {"purchase_cost": 180, "annual_cost": 225, "life_years": 3, "rate_pct": 7}))
    finally:
        MONEY_STYLE.reset(tok)
    assert out["formatted"]["eac"] == "-$293.59" and "table_shown_to_user" in out


# ---------- routing
@pytest.mark.parametrize("q", [
    PUMPS_Q,
    "What is the EAC of a ₹2 lakh machine that costs ₹30,000 a year to run for 6 years at 11%?",
    "Find the equivalent annual annuity of a $420 pump that costs $165 a year to run for 8 years at 7 percent.",
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
    args = {"purchase_cost": 180, "annual_cost": 225, "life_years": 3, "rate_pct": 7}
    assert ungrounded_values(args, PUMPS_Q, "equivalent_annual_cost") == ["annual_cost=225", "life_years=3"]
    calc_out = json.dumps({"result": {"value": 224.99999999999997, "rounded": 225}}) + json.dumps({"result": {"value": 3.0}})
    assert ungrounded_values(args, PUMPS_Q, "equivalent_annual_cost", tool_text=calc_out) == []
    # a life in years that is neither stated nor a tool result is rejected, even when it is some quotient of two
    # question figures (a live run divided by the wrong figure instead of by the hours run a year)
    eff = {"purchase_cost": 420, "annual_cost": 165, "life_years": 8, "rate_pct": 7}
    assert ungrounded_values(eff, PUMPS_Q, "equivalent_annual_cost", tool_text='{"value": 165.00000000000003}') == ["life_years=8"]
    assert ungrounded_values(eff, PUMPS_Q, "equivalent_annual_cost", tool_text='{"value": 165.0} {"value": 8.0}') == []
    # usage figures are copied from the question, so they are grounded directly
    usage = {"purchase_cost": 420, "rate_pct": 7, "power_watts": 550, "hours_per_year": 2000, "price_per_kwh": 0.15,
             "lifetime_hours": 16000}
    assert ungrounded_values(usage, PUMPS_Q, "equivalent_annual_cost") == []


def test_zero_rate_and_invented_zero_running_cost_are_bounced_for_eac():
    from finbot.graph import invented_zero_rates
    assert invented_zero_rates("equivalent_annual_cost", {"rate_pct": 0}, PUMPS_Q) == ["rate_pct"]
    # a live run passed annual_cost=0 for both options; zero is only accepted when the usage figures are given instead
    assert invented_zero_rates("equivalent_annual_cost", {"rate_pct": 7, "annual_cost": 0, "life_years": 3}, PUMPS_Q) == ["annual_cost"]
    assert invented_zero_rates("equivalent_annual_cost", {"rate_pct": 7, "annual_cost": 0, "power_watts": 550,
                                                          "hours_per_year": 2000, "price_per_kwh": 0.15}, PUMPS_Q) == []
    assert invented_zero_rates("equivalent_annual_cost", {"rate_pct": 8, "annual_cost": 0},
                               "A press costs $9,000, has no running costs and lasts 3 years at 8%.") == []


def test_graph_eac_with_usage_figures_passes_review_in_one_round():
    from helpers import ScriptedModel, make_kb
    from finbot.graph import FinBot
    from langchain_core.messages import AIMessage
    usage = {"rate_pct": 7, "hours_per_year": 2000, "price_per_kwh": 0.15}
    model = ScriptedModel(responses=[
        AIMessage(content="", tool_calls=[
            {"name": "equivalent_annual_cost", "args": {"purchase_cost": 180, "power_watts": 750, "lifetime_hours": 6000, **usage},
             "id": "c1", "type": "tool_call"},
            {"name": "equivalent_annual_cost", "args": {"purchase_cost": 420, "power_watts": 550, "lifetime_hours": 16000, **usage},
             "id": "c2", "type": "tool_call"}]),
        "The standard pump costs $225.00 a year over a 3-year life, so its EAC is -$293.59. The efficient pump costs "
        "$165.00 a year over 8 years, so its EAC is -$235.34."])
    t = FinBot(model=model, kb=make_kb()).ask(PUMPS_Q, thread_id="eac-usage")
    assert [x["output"]["result"]["eac"] for x in t["tools"]] == pytest.approx(
        [-(180 + 225 * pvifa(0.07, 3)) / pvifa(0.07, 3), -(420 + 165 * pvifa(0.07, 8)) / pvifa(0.07, 8)], abs=1e-4)
    assert t["review"]["passed"], t["review"]
    assert t["revisions"] == 0 and len(t["tables"]) == 2


def test_graph_eac_flow_calculate_then_eac_passes_review():
    from helpers import ScriptedModel, make_kb, tool_call
    from finbot.graph import FinBot
    from langchain_core.messages import AIMessage

    def calls(*items):
        return AIMessage(content="", tool_calls=[{"name": n, "args": a, "id": i, "type": "tool_call"} for n, a, i in items])

    model = ScriptedModel(responses=[
        tool_call("equivalent_annual_cost", {"purchase_cost": 180, "annual_cost": 225, "life_years": 3, "rate_pct": 7}, "c0"),
        calls(("calculate", {"expression": "750/1000*2000*0.15"}, "c1"), ("calculate", {"expression": "550/1000*2000*0.15"}, "c2"),
              ("calculate", {"expression": "6000/2000"}, "c5"), ("calculate", {"expression": "16000/2000"}, "c6")),
        calls(("equivalent_annual_cost", {"purchase_cost": 180, "annual_cost": 225, "life_years": 3, "rate_pct": 7}, "c3"),
              ("equivalent_annual_cost", {"purchase_cost": 420, "annual_cost": 165, "life_years": 8, "rate_pct": 7}, "c4")),
        "The standard pump costs $225.00 a year to run over a 3-year life, so its EAC is -$293.59. The efficient pump "
        "costs $165.00 a year over 8 years, so its EAC is -$235.34. The efficient pump is cheaper per year."])
    t = FinBot(model=model, kb=make_kb()).ask(PUMPS_Q, thread_id="eac")
    assert "error" in t["tools"][0]["output"] and "annual_cost=225" in t["tools"][0]["output"]["error"]
    assert "calculate" in t["tools"][0]["output"]["error"]
    eacs = [x["output"]["result"]["eac"] for x in t["tools"] if x["name"] == "equivalent_annual_cost" and "result" in x["output"]]
    expected = [-(180 + 225 * pvifa(0.07, 3)) / pvifa(0.07, 3), -(420 + 165 * pvifa(0.07, 8)) / pvifa(0.07, 8)]
    assert eacs == pytest.approx(expected, abs=1e-4)
    assert t["review"]["passed"], t["review"]
    assert len(t["tables"]) == 2
