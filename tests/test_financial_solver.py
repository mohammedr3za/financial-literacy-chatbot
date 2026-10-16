"""solve_financials (SPEC 4.1): textbook identities, derivation steps, consistency warnings, strict names, grounding, graph.
Expected values are worked by hand from the standard Ross/Westerfield/Jordan formulas, not taken from the code."""
import json

import pytest
from pydantic import ValidationError

from finbot.tools.financial_solver import solve_financials
from finbot.tools.fmt import MONEY_STYLE

# Our own scenarios (not course-set problems). Worked by hand:
# LIQUIDITY: current assets = 1.6 × 320,000 = 512,000; quick assets = 0.75 × 320,000 = 240,000;
#            inventory = 512,000 − 240,000 = 272,000; COGS = 5 × 272,000 = 1,360,000.
# INCOME:    EBIT = 548,000 − 262,500 − 71,000 = 214,500; EBT = 214,500 − 38,500 = 176,000;
#            taxes = 176,000 × 0.30 = 52,800; net income = 176,000 − 52,800 = 123,200.
LIQUIDITY_Q = ("Corvid Trading has current liabilities of $320,000, a quick ratio of .75, inventory turnover of 5, "
               "and a current ratio of 1.6. What is its cost of goods sold?")
LIQUIDITY_ARGS = {"current_liabilities": 320000, "quick_ratio": 0.75, "inventory_turnover": 5, "current_ratio": 1.6,
                  "targets": ["cogs"]}
INCOME_Q = ("Halvorsen Ltd. has sales of $548,000, costs of $262,500, depreciation expense of $71,000, interest expense "
            "of $38,500, and a tax rate of 30 percent. What is its net income?")
INCOME_ARGS = {"sales": 548000, "costs": 262500, "depreciation": 71000, "interest": 38500, "tax_rate_pct": 30,
               "targets": ["net_income"]}


def step(out, var):
    return next(s for s in out["steps"] if s["variable"] == var)


@pytest.fixture
def usd():
    tok = MONEY_STYLE.set("USD")      # the graph sets this for questions in dollars
    yield
    MONEY_STYLE.reset(tok)


def test_cogs_from_liquidity_ratios_goes_through_quick_assets_and_inventory(usd):
    out = solve_financials(**LIQUIDITY_ARGS)
    assert out["result"]["cogs"] == pytest.approx(1_360_000)                    # 5 × 272,000
    assert step(out, "current_assets")["value"] == pytest.approx(512_000)      # 1.6 × 320,000
    assert step(out, "quick_assets")["value"] == pytest.approx(240_000)        # 0.75 × 320,000
    inv = step(out, "inventory")
    assert inv["value"] == pytest.approx(272_000)                              # 512,000 − 240,000
    assert inv["formula"] == "inventory = current_assets − quick_assets"
    assert inv["substituted"] == "inventory = 512,000 − 240,000"
    assert [s["variable"] for s in out["steps"]] == ["current_assets", "quick_assets", "inventory", "cogs"]
    assert step(out, "current_assets")["formula"] == "current_assets = current_ratio × current_liabilities"
    assert step(out, "cogs")["substituted"] == "cogs = 5 × 272,000"
    assert "warnings" not in out and "missing" not in out


def test_income_statement_net_income_steps(usd):
    out = solve_financials(**INCOME_ARGS)
    vals = {s["variable"]: s["value"] for s in out["steps"]}
    assert vals == pytest.approx({"ebit": 214_500, "ebt": 176_000, "taxes": 52_800, "net_income": 123_200})
    assert step(out, "ebit")["formula"] == "ebit = sales − costs − depreciation"
    assert step(out, "taxes")["substituted"] == "taxes = 176,000 × 30 ÷ 100"
    assert out["result"] == {"net_income": 123200.0}


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
    out = solve_financials(current_assets=512000, current_liabilities=320000, current_ratio=1.6, targets=["net_working_capital"])
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
    out = solve_financials(sales=548000, costs=262500, depreciation=71000, interest=38500, tax_rate_pct=30, dividends=0,
                           targets=["net_income"])
    assert out["result"]["net_income"] == pytest.approx(123_200)
    assert out["ignored_zero_inputs"] == ["dividends"]


def test_unknown_names_are_rejected_with_suggestions():
    from finbot.tools.registry import TOOLS_BY_NAME
    tool = TOOLS_BY_NAME["solve_financials"]
    with pytest.raises(ValidationError) as e:
        tool.invoke({"sales": 548000, "taxrate": 30, "targets": ["net_income"]})     # tax_rate itself is an accepted alias
    assert "tax_rate_pct" in str(e.value) and "Valid names" in str(e.value)
    with pytest.raises(ValidationError) as e:
        tool.invoke({"sales": 548000, "targets": ["gross_cost"]})
    assert "cogs" in str(e.value)
    with pytest.raises(ValueError):
        solve_financials(sales=1, targets=["nonsense"])


def test_tool_json_has_every_step_value_and_a_table():
    from finbot.tools.registry import TOOLS_BY_NAME
    out = json.loads(TOOLS_BY_NAME["solve_financials"].invoke(LIQUIDITY_ARGS))
    assert out["result"]["cogs"] == pytest.approx(1_360_000) and "table_shown_to_user" in out
    text = json.dumps(out)
    for s in out["steps"]:
        assert str(s["value"]) in text
    t = out["table"]
    assert t["columns"] == ["Step", "Formula", "Value"]
    assert any(r[0] == "Inventory" and "Current assets − Quick assets" in r[1] for r in t["rows"])


def test_money_is_formatted_in_the_question_currency():
    tok = MONEY_STYLE.set("USD")
    try:
        out = solve_financials(**LIQUIDITY_ARGS)
        assert out["formatted"]["cogs"] == "$1,360,000.00"
        assert step(out, "inventory")["formatted"] == "$272,000.00"
        assert out["table"]["title"].endswith("($)")
    finally:
        MONEY_STYLE.reset(tok)
    assert solve_financials(**LIQUIDITY_ARGS)["formatted"]["cogs"] == "₹13,60,000.00"


# ---------- grounding: figures passed must be the question's figures, in the right units
def test_grounding_accepts_question_figures_as_written():
    from finbot.graph import ungrounded_values
    assert ungrounded_values({k: v for k, v in LIQUIDITY_ARGS.items()}, LIQUIDITY_Q, "solve_financials") == []
    assert ungrounded_values(INCOME_ARGS, INCOME_Q, "solve_financials") == []


def test_grounding_checks_percent_and_ratio_units():
    from finbot.graph import ungrounded_values
    bad = ungrounded_values({"sales": 548000, "tax_rate_pct": 0.3}, INCOME_Q, "solve_financials")
    assert len(bad) == 1 and bad[0].startswith("tax_rate_pct=0.3")
    assert ungrounded_values({"tax_rate_pct": 23}, "The tax rate is .23 and sales are $500.", "solve_financials") == []
    assert ungrounded_values({"tax_rate_pct": 0.23}, "The tax rate is .23 and sales are $500.", "solve_financials") != []
    assert ungrounded_values({"debt_ratio": 0.4}, "Its total debt ratio is 40 percent.", "solve_financials") == []
    assert ungrounded_values({"debt_ratio": 40}, "Its total debt ratio is 40 percent.", "solve_financials") != []
    assert ungrounded_values({"total_assets": 4_500_000}, "Total assets are $4.5 million.", "solve_financials") == []
    assert ungrounded_values({"inventory": 272000}, LIQUIDITY_Q, "solve_financials") == ["inventory=272000"]


def test_numeric_strings_are_grounded_too():
    from finbot.graph import ungrounded_values
    assert ungrounded_values({"sales": "999999"}, INCOME_Q, "solve_financials") == ["sales=999999"]


# ---------- routing and graph
@pytest.mark.parametrize("q", [
    LIQUIDITY_Q, INCOME_Q,
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
        tool_call("solve_financials", LIQUIDITY_ARGS, "c1"),
        "Current assets are $512,000 and quick assets are $240,000, so inventory is $272,000. "
        "With inventory turnover of 5, the cost of goods sold is $1,360,000."])
    t = FinBot(model=model, kb=make_kb()).ask(LIQUIDITY_Q, thread_id="solver")
    assert t["tools"][0]["output"]["result"]["cogs"] == pytest.approx(1_360_000)
    assert t["review"]["passed"], t["review"]
    assert t["revisions"] == 0 and len(t["tables"]) == 1 and "$1,360,000" in t["answer"]
    assert "solve_financials FIRST" in str(model.calls[0][-1].content)


def test_graph_rejects_invented_solver_input_then_accepts_retry():
    from helpers import ScriptedModel, make_kb, tool_call
    from finbot.graph import FinBot
    model = ScriptedModel(responses=[
        tool_call("solve_financials", {**INCOME_ARGS, "tax_rate_pct": 0.3}, "c1"),
        tool_call("solve_financials", INCOME_ARGS, "c2"),
        "EBIT is $214,500, EBT is $176,000 and taxes are $52,800, so net income is $123,200."])
    t = FinBot(model=model, kb=make_kb()).ask(INCOME_Q, thread_id="solver2")
    err = t["tools"][0]["output"].get("error", "")
    assert "tax_rate_pct=0.3" in err and "percent items take the percent number" in err
    assert t["tools"][1]["output"]["result"]["net_income"] == pytest.approx(123_200)
    assert t["review"]["passed"], t["review"]


# ---------- live defects (2026-10-16): worked back through net income, credit sales, DuPont, debt ratio
def test_ebt_from_net_income_and_tax_rate_then_depreciation():
    # sales 92,000, costs 41,000, addition to RE 7,400, dividends 3,100, interest 5,600, tax 28%:
    # NI = 10,500 -> EBT = 10,500 / 0.72 = 14,583.33 -> EBIT = 20,183.33 -> depreciation = 51,000 − 20,183.33 = 30,816.67
    out = solve_financials(sales=92000, costs=41000, addition_to_retained_earnings=7400, dividends=3100, interest=5600,
                           tax_rate_pct=28, targets=["depreciation"])
    vals = {s["variable"]: s["value"] for s in out["steps"]}
    assert vals["net_income"] == pytest.approx(10500)
    assert vals["ebt"] == pytest.approx(10500 / 0.72, abs=0.01)
    assert vals["ebit"] == pytest.approx(10500 / 0.72 + 5600, abs=0.01)
    assert out["result"]["depreciation"] == pytest.approx(92000 - 41000 - (10500 / 0.72 + 5600), abs=0.01)
    assert round(out["result"]["depreciation"]) == 30817
    assert step(out, "ebt")["formula"] == "ebt = net_income ÷ (1 − tax_rate_pct ÷ 100)"
    assert "warnings" not in out and "missing" not in out


def test_net_income_shortcut_agrees_with_the_two_step_route():
    out = solve_financials(**{k: v for k, v in INCOME_ARGS.items() if k != "targets"}, targets=["net_income", "taxes"])
    assert out["result"] == pytest.approx({"net_income": 123_200, "taxes": 52_800})
    assert "warnings" not in out


def test_days_sales_in_receivables_with_credit_sales():
    # NI 236,000, profit margin 7.4%, 80% of sales on credit, receivables 214,600:
    # sales = 236,000 / 0.074 = 3,189,189.19; credit sales = 2,551,351.35; turnover = 11.8889 -> 365 / 11.8889 = 30.70 days
    out = solve_financials(net_income=236000, profit_margin_pct=7.4, credit_sales_pct=80, receivables=214600,
                           targets=["days_sales_in_receivables"])
    sales = 236000 / 0.074
    assert step(out, "credit_sales")["value"] == pytest.approx(0.80 * sales, abs=0.01)
    assert step(out, "receivables_turnover")["formula"] == "receivables_turnover = credit_sales ÷ receivables"
    assert out["result"]["days_sales_in_receivables"] == pytest.approx(365 / (0.80 * sales / 214600), abs=1e-4)
    assert round(out["result"]["days_sales_in_receivables"], 2) == 30.70
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
    # sales 8,450, total assets 3,920, D/E 1.35, ROE 14% -> equity = 3,920 ÷ 2.35 = 1,668.09; NI = 0.14 × 1,668.09 = 233.53
    out = solve_financials(sales=8450, total_assets=3920, debt_equity_ratio=1.35, roe_pct=14, targets=["net_income"])
    assert out["result"]["net_income"] == pytest.approx(0.14 * 3920 / 2.35, abs=1e-4)
    assert round(out["result"]["net_income"], 2) == 233.53
    assert "warnings" not in out


# ROE = ROA ÷ (1 − debt ratio): 7 ÷ 0.55 = 12.727 -> 12.73; 10 ÷ 0.70 = 14.286 -> 14.29
@pytest.mark.parametrize("debt_ratio,roa,roe", [(0.45, 7, 12.73), (0.30, 10, 14.29)])
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
    args = {"sales": 92000, "costs": 41000, "addition_to_retained_earnings": 7400, "dividends_paid": 3100,
            "interest_expense": 5600, "tax_rate_pct": 28, "targets": ["depreciation_expense"]}
    out = json.loads(TOOLS_BY_NAME["solve_financials"].invoke(args))
    assert out["result"]["depreciation"] == pytest.approx(30816.67, abs=0.01)      # as worked above
    assert solve_financials(revenue=600, net_profit=60, targets=["profit_margin_pct"])["result"]["profit_margin_pct"] == pytest.approx(10)


def test_alias_conflicting_with_its_canonical_name_is_an_error():
    with pytest.raises(ValueError, match="interest"):
        solve_financials(interest=100, interest_expense=200, sales=1000, targets=["ebt"])


def test_grounding_checks_units_under_alias_names_too():
    from finbot.graph import ungrounded_values
    bad = ungrounded_values({"sales": 548000, "tax_rate": 0.3}, INCOME_Q, "solve_financials")
    assert len(bad) == 1 and bad[0].startswith("tax_rate=0.3") and "percent items take the percent number" in bad[0]
    assert ungrounded_values({"interest_expense": 38500, "tax_rate": 30}, INCOME_Q, "solve_financials") == []


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
    dupont = json.loads(tool.invoke({"debt_equity_ratio": 1.35, "return_on_equity_pct": 14, "sales": 8450, "total_assets": 3920,
                                     "targets": ["net_income"]}))
    assert round(dupont["result"]["net_income"], 2) == 233.53                    # 0.14 × 3,920 ÷ 2.35
    dsr = json.loads(tool.invoke({"net_income": 236000, "profit_margin_pct": 7.4, "accounts_receivable": 214600,
                                  "credit_sales_pct": 80, "targets": ["sales", "days_sales_in_receivables"], "tax_rate_pct": 0}))
    assert round(dsr["result"]["days_sales_in_receivables"], 2) == 30.70         # 365 × 214,600 ÷ 2,551,351.35


# ---------- other_expenses: EBIT = sales − cogs − other expenses − depreciation; one call per year (SPEC 4.1, 2026-10-17)
@pytest.mark.parametrize("year,figures,ebit,ni", [
    (2025, dict(sales=25000, cogs=9000, other_expenses=2500, depreciation=1500, interest=1000), 12000, 8690),   # (12000-1000)×0.79
    (2026, dict(sales=23000, cogs=8800, other_expenses=2200, depreciation=1600, interest=1100), 10400, 7347),   # (10400-1100)×0.79
])
def test_income_statement_with_cogs_and_other_expenses(year, figures, ebit, ni):
    out = solve_financials(targets=["ebit", "net_income"], tax_rate_pct=21, **figures)
    assert out["result"]["ebit"] == ebit and round(out["result"]["net_income"], 2) == ni
    assert any(st["formula"] == "ebit = sales − cogs − other_expenses − depreciation" for st in out["steps"])


def test_other_expenses_aliases_and_costs_path_unchanged():
    out = solve_financials(targets=["ebit"], sales=1000, cost_of_goods_sold=400, sga=100, depreciation=50)
    assert out["result"]["ebit"] == 450
    assert solve_financials(targets=["ebit"], sales=1000, costs=500, depreciation=50)["result"]["ebit"] == 450
