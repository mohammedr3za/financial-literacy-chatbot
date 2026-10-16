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
        tool.invoke({"sales": 675000, "taxrate": 23, "targets": ["net_income"]})     # tax_rate itself is an accepted alias
    assert "tax_rate_pct" in str(e.value) and "Valid names" in str(e.value)
    with pytest.raises(ValidationError) as e:
        tool.invoke({"sales": 675000, "targets": ["gross_cost"]})
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


# ---------- live defects (2026-10-16): worked back through net income, credit sales, DuPont, debt ratio
def test_ebt_from_net_income_and_tax_rate_then_depreciation():
    # sales 68,000, costs 31,500, addition to RE 6,100, dividends 2,100, interest 4,800, tax 24%:
    # NI = 8,200 -> EBT = 8,200 / 0.76 = 10,789.47 -> EBIT = 15,589.47 -> depreciation = 20,910.53
    out = solve_financials(sales=68000, costs=31500, addition_to_retained_earnings=6100, dividends=2100, interest=4800,
                           tax_rate_pct=24, targets=["depreciation"])
    vals = {s["variable"]: s["value"] for s in out["steps"]}
    assert vals["net_income"] == pytest.approx(8200)
    assert vals["ebt"] == pytest.approx(8200 / 0.76, abs=0.01)
    assert vals["ebit"] == pytest.approx(8200 / 0.76 + 4800, abs=0.01)
    assert out["result"]["depreciation"] == pytest.approx(68000 - 31500 - (8200 / 0.76 + 4800), abs=0.01)
    assert round(out["result"]["depreciation"]) == 20911
    assert step(out, "ebt")["formula"] == "ebt = net_income ÷ (1 − tax_rate_pct ÷ 100)"
    assert "warnings" not in out and "missing" not in out


def test_net_income_shortcut_agrees_with_the_two_step_route():
    out = solve_financials(**{k: v for k, v in NAKATOMI_ARGS.items() if k != "targets"}, targets=["net_income", "taxes"])
    assert out["result"] == pytest.approx({"net_income": 159_005, "taxes": 47_495})
    assert "warnings" not in out


def test_days_sales_in_receivables_with_credit_sales():
    # NI 190,000, profit margin 9.1%, 85% of sales on credit, receivables 129,370 -> 26.61 days
    out = solve_financials(net_income=190000, profit_margin_pct=9.1, credit_sales_pct=85, receivables=129370,
                           targets=["days_sales_in_receivables"])
    sales = 190000 / 0.091
    assert step(out, "credit_sales")["value"] == pytest.approx(0.85 * sales, abs=0.01)
    assert step(out, "receivables_turnover")["formula"] == "receivables_turnover = credit_sales ÷ receivables"
    assert out["result"]["days_sales_in_receivables"] == pytest.approx(365 / (0.85 * sales / 129370), abs=1e-4)
    assert round(out["result"]["days_sales_in_receivables"], 2) == 26.61
    assert "warnings" not in out


def test_credit_sales_amount_given_directly():
    out = solve_financials(credit_sales=1_000_000, receivables=125_000, sales=1_600_000, targets=["receivables_turnover"])
    assert out["result"]["receivables_turnover"] == pytest.approx(8.0)
    assert "warnings" not in out              # sales ÷ receivables (12.8) is not active when credit figures are given


def test_without_credit_figures_receivables_turnover_uses_total_sales():
    out = solve_financials(sales=1_600_000, receivables=125_000, targets=["receivables_turnover", "days_sales_in_receivables"])
    assert out["result"]["receivables_turnover"] == pytest.approx(12.8)
    assert step(out, "receivables_turnover")["formula"] == "receivables_turnover = sales ÷ receivables"
    assert all(s["variable"] != "credit_sales" for s in out["steps"])


def test_dupont_net_income_from_debt_equity_and_roe():
    # sales 6,269, total assets 2,885, D/E 1.80, ROE 12% -> NI = 0.12 × 2,885 ÷ 2.8 = 123.64
    out = solve_financials(sales=6269, total_assets=2885, debt_equity_ratio=1.8, roe_pct=12, targets=["net_income"])
    assert out["result"]["net_income"] == pytest.approx(0.12 * 2885 / 2.8, abs=1e-4)
    assert round(out["result"]["net_income"], 2) == 123.64
    assert "warnings" not in out


@pytest.mark.parametrize("debt_ratio,roa,roe", [(0.35, 9, 13.85), (0.25, 13, 17.33)])
def test_roe_from_debt_ratio_and_roa(debt_ratio, roa, roe):
    out = solve_financials(debt_ratio=debt_ratio, roa_pct=roa, targets=["roe_pct"])
    assert out["result"]["roe_pct"] == pytest.approx(roa / (1 - debt_ratio), abs=1e-6)
    assert round(out["result"]["roe_pct"], 2) == roe


CREDIT_Q = ("A retailer has net income of $250,000 on a profit margin of 8 percent. Receivables are $150,000 and 70 percent "
            "of its sales are on credit. What is its days' sales in receivables?")


def test_credit_question_routes_to_solver_only_and_is_grounded():
    from finbot.graph import ungrounded_values
    from finbot.precheck import precheck
    assert precheck(CREDIT_Q).flags["calc_types"] == ["statement"]       # no plain-percentage route on statement questions
    args = {"net_income": 250000, "profit_margin_pct": 8, "credit_sales_pct": 70, "receivables": 150000,
            "targets": ["days_sales_in_receivables"]}
    assert ungrounded_values(args, CREDIT_Q, "solve_financials") == []
    assert ungrounded_values({**args, "credit_sales_pct": 0.7}, CREDIT_Q, "solve_financials")[0].startswith("credit_sales_pct=0.7")


def test_graph_wrong_calculator_working_is_caught_and_solver_fixes_it():
    """The answer shows the right DuPont working but states the result of a different expression; review catches the
    mismatch, and the rewrite (after solve_financials) passes."""
    from helpers import ScriptedModel, make_kb, tool_call
    from finbot.graph import FinBot
    q = ("Orbit Ltd has sales of $5,000, total assets of $2,000 and a debt-equity ratio of 1.5. If its return on equity is "
         "15 percent, what is its net income?")
    model = ScriptedModel(responses=[
        tool_call("calculate", {"expression": "15*(2000*1.5)/(1+1.5)"}, "c1"),
        "Net income = ROE × equity = $$0.15 \\times \\frac{2000}{1 + 1.5}$$ so net income is $18,000.",
        tool_call("solve_financials", {"sales": 5000, "total_assets": 2000, "debt_equity_ratio": 1.5, "roe_pct": 15,
                                       "targets": ["net_income"]}, "c2"),
        "The equity multiplier is 1 + 1.5 = 2.5, so equity is $800 and net income is 15% of $800 = $120."])
    t = FinBot(model=model, kb=make_kb()).ask(q, thread_id="dupont")
    assert t["revisions"] == 1 and "arithmetic_mismatch" in t["review"]["first_draft_codes"]
    note = str(model.calls[2][-1].content)
    assert "evaluates to 120" in note and "18,000" in note
    assert t["tools"][-1]["output"]["result"]["net_income"] == pytest.approx(120)
    assert t["review"]["passed"], t["review"]


# ---------- common alternative names are mapped, not rejected (a live run sent interest_expense, then dropped interest)
def test_alias_names_are_mapped_to_canonical_items():
    from finbot.tools.registry import TOOLS_BY_NAME
    args = {"sales": 68000, "costs": 31500, "addition_to_retained_earnings": 6100, "dividends_paid": 2100,
            "interest_expense": 4800, "tax_rate_pct": 24, "targets": ["depreciation_expense"]}
    out = json.loads(TOOLS_BY_NAME["solve_financials"].invoke(args))
    assert out["result"]["depreciation"] == pytest.approx(20910.53, abs=0.01)
    assert solve_financials(revenue=600, net_profit=60, targets=["profit_margin_pct"])["result"]["profit_margin_pct"] == pytest.approx(10)


def test_alias_conflicting_with_its_canonical_name_is_an_error():
    with pytest.raises(ValueError, match="interest"):
        solve_financials(interest=100, interest_expense=200, sales=1000, targets=["ebt"])


def test_grounding_checks_units_under_alias_names_too():
    from finbot.graph import ungrounded_values
    bad = ungrounded_values({"sales": 675000, "tax_rate": 0.23}, NAKATOMI, "solve_financials")
    assert len(bad) == 1 and bad[0].startswith("tax_rate=0.23") and "23 percent" in bad[0]
    assert ungrounded_values({"interest_expense": 50500, "tax_rate": 23}, NAKATOMI, "solve_financials") == []


@pytest.mark.parametrize("alias,name", [
    ("interest_expense", "interest"), ("accounts_receivable", "receivables"), ("return_on_equity_pct", "roe_pct"),
    ("return_on_equity", "roe_pct"), ("return_on_assets_percent", "roa_pct"), ("debt_to_assets", "debt_ratio"),
    ("net_profit_margin", "profit_margin_pct"), ("credit_sales_percentage", "credit_sales_pct"), ("tax_rate", "tax_rate_pct"),
    ("sales", "sales"), ("nonsense_item", "nonsense_item"),
])
def test_alias_names_seen_in_live_runs_resolve(alias, name):
    from finbot.tools.financial_solver import resolve
    assert resolve(alias) == name


def test_live_alias_calls_now_solve():
    from finbot.tools.registry import TOOLS_BY_NAME
    tool = TOOLS_BY_NAME["solve_financials"]
    y3k = json.loads(tool.invoke({"debt_equity_ratio": 1.8, "return_on_equity_pct": 12, "sales": 6269, "total_assets": 2885,
                                  "targets": ["net_income"]}))
    assert round(y3k["result"]["net_income"], 2) == 123.64
    dsr = json.loads(tool.invoke({"net_income": 190000, "profit_margin_pct": 9.1, "accounts_receivable": 129370,
                                  "credit_sales_pct": 85, "targets": ["sales", "days_sales_in_receivables"], "tax_rate_pct": 0}))
    assert round(dsr["result"]["days_sales_in_receivables"], 2) == 26.61
