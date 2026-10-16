"""solve_financials (SPEC 4.1): textbook identities, derivation steps, consistency warnings, strict names, grounding, graph.
Expected values are worked by hand from the standard Ross/Westerfield/Jordan formulas, not taken from the code."""
import json

import pytest
from pydantic import ValidationError

from finbot.tools.financial_solver import solve_financials
from finbot.tools.fmt import MONEY_STYLE

TRALEE = ("Tralee Corporation has current liabilities of $445,000, a quick ratio of .84, inventory turnover of 6, "
          "and a current ratio of 1.3. What is the cost of goods sold?")
TRALEE_ARGS = {"current_liabilities": 445000, "quick_ratio": 0.84, "inventory_turnover": 6, "current_ratio": 1.3,
               "targets": ["cogs"]}
NAKATOMI = ("Nakatomi, Inc., has sales of $675,000, costs of $337,000, depreciation expense of $81,000, interest expense "
            "of $50,500, and a tax rate of 23 percent. What is the net income for this firm?")
NAKATOMI_ARGS = {"sales": 675000, "costs": 337000, "depreciation": 81000, "interest": 50500, "tax_rate_pct": 23,
                 "targets": ["net_income"]}


def step(out, var):
    return next(s for s in out["steps"] if s["variable"] == var)


@pytest.fixture
def usd():
    tok = MONEY_STYLE.set("USD")      # the graph sets this for questions in dollars
    yield
    MONEY_STYLE.reset(tok)


def test_tralee_cogs_goes_through_quick_assets_and_inventory(usd):
    out = solve_financials(**TRALEE_ARGS)
    assert out["result"]["cogs"] == pytest.approx(1_228_200)
    assert step(out, "current_assets")["value"] == pytest.approx(578_500)      # 1.3 × 445,000
    assert step(out, "quick_assets")["value"] == pytest.approx(373_800)        # 0.84 × 445,000
    inv = step(out, "inventory")
    assert inv["value"] == pytest.approx(204_700)
    assert inv["formula"] == "inventory = current_assets − quick_assets"
    assert inv["substituted"] == "inventory = 578,500 − 373,800"
    assert [s["variable"] for s in out["steps"]] == ["current_assets", "quick_assets", "inventory", "cogs"]
    assert step(out, "current_assets")["formula"] == "current_assets = current_ratio × current_liabilities"
    assert step(out, "cogs")["substituted"] == "cogs = 6 × 204,700"
    assert "warnings" not in out and "missing" not in out


def test_nakatomi_income_statement(usd):
    out = solve_financials(**NAKATOMI_ARGS)
    vals = {s["variable"]: s["value"] for s in out["steps"]}
    assert vals == pytest.approx({"ebit": 257_000, "ebt": 206_500, "taxes": 47_495, "net_income": 159_005})
    assert step(out, "ebit")["formula"] == "ebit = sales − costs − depreciation"
    assert step(out, "taxes")["substituted"] == "taxes = 206,500 × 23 ÷ 100"
    assert out["result"] == {"net_income": 159005.0}


def test_dupont_roe_from_ratios_only():
    out = solve_financials(profit_margin_pct=6, total_asset_turnover=1.5, equity_multiplier=2, targets=["roe_pct"])
    assert out["result"]["roe_pct"] == pytest.approx(18.0)                       # 6% × 1.5 × 2
    assert out["formatted"]["roe_pct"] == "18.00%"


def test_equity_multiplier_from_debt_equity_then_roe_and_net_income():
    # SnowGlobal: D/E .80, ROA 8.7%, equity $735,000 -> EM 1.8, ROE 15.66%, NI 115,101
    out = solve_financials(debt_equity_ratio=0.8, roa_pct=8.7, total_equity=735000,
                           targets=["equity_multiplier", "roe_pct", "net_income"])
    assert out["result"] == pytest.approx({"equity_multiplier": 1.8, "roe_pct": 15.66, "net_income": 115_101})


def test_net_fixed_assets_chain_le_bleu():
    # LTD ratio .45, current ratio 1.25, CL 875, sales 5,780, PM 9.5%, ROE 18.5% -> NFA 5,177.81
    out = solve_financials(long_term_debt_ratio=0.45, current_ratio=1.25, current_liabilities=875, sales=5780,
                           profit_margin_pct=9.5, roe_pct=18.5, targets=["net_fixed_assets"])
    ni = 0.095 * 5780
    te = ni / 0.185
    ltd = 0.45 / 0.55 * te
    ta = 875 + ltd + te
    assert out["result"]["net_fixed_assets"] == pytest.approx(ta - 1.25 * 875, abs=0.01)
    assert out["result"]["net_fixed_assets"] == pytest.approx(5177.81, abs=0.01)


def test_sustainable_growth_rate():
    out = solve_financials(roe_pct=15, payout_ratio_pct=40, targets=["sustainable_growth_pct"])
    assert step(out, "retention_ratio_pct")["value"] == pytest.approx(60)
    assert out["result"]["sustainable_growth_pct"] == pytest.approx(0.09 / 0.91 * 100, abs=1e-4)   # 9.8901%


def test_inconsistent_inputs_give_a_warning():
    out = solve_financials(current_assets=500, current_liabilities=250, current_ratio=3, targets=["net_working_capital"])
    assert out["result"]["net_working_capital"] == pytest.approx(250)
    assert any("current_ratio" in w for w in out["warnings"])


def test_consistent_redundant_inputs_do_not_warn():
    out = solve_financials(current_assets=578500, current_liabilities=445000, current_ratio=1.3, targets=["net_working_capital"])
    assert "warnings" not in out


def test_negative_result_for_an_item_that_cannot_be_negative_is_flagged():
    out = solve_financials(current_liabilities=100, current_ratio=1.0, quick_ratio=1.2, targets=["inventory"])
    assert out["result"]["inventory"] == pytest.approx(-20)
    assert any("negative" in w and "inventory" in w for w in out["warnings"])


def test_underdetermined_target_reports_what_is_missing():
    out = solve_financials(sales=1000, targets=["net_income"])
    assert "net_income" not in out["result"]
    msg = out["missing"]["net_income"]
    assert "cannot be derived" in msg and "taxes" in msg and "ebt" in msg


def test_zero_inputs_are_treated_as_not_given():
    out = solve_financials(sales=675000, costs=337000, depreciation=81000, interest=50500, tax_rate_pct=23, dividends=0,
                           targets=["net_income"])
    assert out["result"]["net_income"] == pytest.approx(159_005)
    assert out["ignored_zero_inputs"] == ["dividends"]


def test_unknown_names_are_rejected_with_suggestions():
    from finbot.tools.registry import TOOLS_BY_NAME
    tool = TOOLS_BY_NAME["solve_financials"]
    with pytest.raises(ValidationError) as e:
        tool.invoke({"sales": 675000, "tax_rate": 23, "targets": ["net_income"]})
    assert "tax_rate_pct" in str(e.value) and "Valid names" in str(e.value)
    with pytest.raises(ValidationError) as e:
        tool.invoke({"sales": 675000, "targets": ["cost_of_goods_sold"]})
    assert "cogs" in str(e.value)
    with pytest.raises(ValueError):
        solve_financials(sales=1, targets=["nonsense"])


def test_tool_json_has_every_step_value_and_a_table():
    from finbot.tools.registry import TOOLS_BY_NAME
    out = json.loads(TOOLS_BY_NAME["solve_financials"].invoke(TRALEE_ARGS))
    assert out["result"]["cogs"] == pytest.approx(1_228_200) and "table_shown_to_user" in out
    text = json.dumps(out)
    for s in out["steps"]:
        assert str(s["value"]) in text
    t = out["table"]
    assert t["columns"] == ["Step", "Formula", "Value"]
    assert any(r[0] == "Inventory" and "Current assets − Quick assets" in r[1] for r in t["rows"])


def test_money_is_formatted_in_the_question_currency():
    tok = MONEY_STYLE.set("USD")
    try:
        out = solve_financials(**TRALEE_ARGS)
        assert out["formatted"]["cogs"] == "$1,228,200.00"
        assert step(out, "inventory")["formatted"] == "$204,700.00"
        assert out["table"]["title"].endswith("($)")
    finally:
        MONEY_STYLE.reset(tok)
    assert solve_financials(**TRALEE_ARGS)["formatted"]["cogs"] == "₹12,28,200.00"


# ---------- grounding: figures passed must be the question's figures, in the right units
def test_grounding_accepts_question_figures_as_written():
    from finbot.graph import ungrounded_values
    assert ungrounded_values({k: v for k, v in TRALEE_ARGS.items()}, TRALEE, "solve_financials") == []
    assert ungrounded_values(NAKATOMI_ARGS, NAKATOMI, "solve_financials") == []


def test_grounding_checks_percent_and_ratio_units():
    from finbot.graph import ungrounded_values
    bad = ungrounded_values({"sales": 675000, "tax_rate_pct": 0.23}, NAKATOMI, "solve_financials")
    assert len(bad) == 1 and bad[0].startswith("tax_rate_pct=0.23")
    assert ungrounded_values({"tax_rate_pct": 23}, "The tax rate is .23 and sales are $500.", "solve_financials") == []
    assert ungrounded_values({"tax_rate_pct": 0.23}, "The tax rate is .23 and sales are $500.", "solve_financials") != []
    assert ungrounded_values({"debt_ratio": 0.4}, "Its total debt ratio is 40 percent.", "solve_financials") == []
    assert ungrounded_values({"debt_ratio": 40}, "Its total debt ratio is 40 percent.", "solve_financials") != []
    assert ungrounded_values({"total_assets": 4_500_000}, "Total assets are $4.5 million.", "solve_financials") == []
    assert ungrounded_values({"inventory": 204700}, TRALEE, "solve_financials") == ["inventory=204700"]


def test_numeric_strings_are_grounded_too():
    from finbot.graph import ungrounded_values
    assert ungrounded_values({"sales": "999999"}, NAKATOMI, "solve_financials") == ["sales=999999"]


# ---------- routing and graph
@pytest.mark.parametrize("q", [
    TRALEE, NAKATOMI,
    "A firm has net income of $50,000 and 10,000 shares outstanding. What is its earnings per share?",
    "If a firm has an ROE of 15 percent and a payout ratio of 40 percent, what is its sustainable growth rate?",
    "Sales are $500,000, costs are $300,000 and the tax rate is 21 percent. What is net income?",
])
def test_statement_questions_route_to_solver(q):
    from finbot.precheck import precheck
    from finbot.routing import HINTS, REQUIRED_TOOLS
    f = precheck(q).flags
    assert "statement" in f["calc_types"] and f["calculation_intent"]
    assert "solve_financials" in REQUIRED_TOOLS["statement"] and HINTS["statement"].startswith("Call solve_financials FIRST")


def test_graph_solver_call_passes_review_with_table():
    from helpers import ScriptedModel, make_kb, tool_call
    from finbot.graph import FinBot
    model = ScriptedModel(responses=[
        tool_call("solve_financials", TRALEE_ARGS, "c1"),
        "Current assets are $578,500 and quick assets are $373,800, so inventory is $204,700. "
        "With inventory turnover of 6, the cost of goods sold is $1,228,200."])
    t = FinBot(model=model, kb=make_kb()).ask(TRALEE, thread_id="solver")
    assert t["tools"][0]["output"]["result"]["cogs"] == pytest.approx(1_228_200)
    assert t["review"]["passed"], t["review"]
    assert t["revisions"] == 0 and len(t["tables"]) == 1 and "$1,228,200" in t["answer"]
    assert "solve_financials FIRST" in str(model.calls[0][-1].content)


def test_graph_rejects_invented_solver_input_then_accepts_retry():
    from helpers import ScriptedModel, make_kb, tool_call
    from finbot.graph import FinBot
    model = ScriptedModel(responses=[
        tool_call("solve_financials", {**NAKATOMI_ARGS, "tax_rate_pct": 0.23}, "c1"),
        tool_call("solve_financials", NAKATOMI_ARGS, "c2"),
        "EBIT is $257,000, EBT is $206,500 and taxes are $47,495, so net income is $159,005."])
    t = FinBot(model=model, kb=make_kb()).ask(NAKATOMI, thread_id="solver2")
    assert "error" in t["tools"][0]["output"] and "23 percent" in t["tools"][0]["output"]["error"]
    assert t["tools"][1]["output"]["result"]["net_income"] == pytest.approx(159_005)
    assert t["review"]["passed"], t["review"]
