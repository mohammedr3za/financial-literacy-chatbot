"""Financial-statement tools (SPEC 4.1). Reference company and cash flow problem worked by hand."""
import pytest

from finbot.tools.statements import balance_sheet_check, cash_flow_statement, financial_ratios

CO = dict(current_assets=200_000, current_liabilities=100_000, inventory=50_000, cash=30_000, total_assets=500_000,
          total_liabilities=200_000, total_equity=300_000, total_debt=150_000, revenue=600_000, cogs=360_000, ebit=90_000,
          interest_expense=15_000, net_income=50_000, receivables=60_000, payables=40_000, operating_cash_flow=70_000,
          capex=25_000, depreciation=20_000)


def test_ratios_reference_company():
    r = financial_ratios(**CO)["result"]
    expected = {"current_ratio": 2.0, "quick_ratio": 1.5, "cash_ratio": 0.3, "working_capital": 100_000, "debt_to_equity": 0.5,
                "debt_ratio": 0.4, "equity_multiplier": 1.6667, "interest_coverage": 6.0, "gross_margin_pct": 40.0,
                "operating_margin_pct": 15.0, "net_margin_pct": 8.3333, "roa_pct": 10.0, "roe_pct": 16.6667, "asset_turnover": 1.2,
                "inventory_turnover": 7.2, "days_inventory": 50.6944, "days_sales_outstanding": 36.5, "days_payables": 40.5556,
                "free_cash_flow": 45_000, "ebitda": 110_000}
    for k, v in expected.items():
        assert r[k] == pytest.approx(v, abs=1e-3), k
    assert r["dupont_roe_pct"] == pytest.approx(r["roe_pct"], abs=1e-6)


def test_ratios_partial_inputs_compute_what_they_can():
    out = financial_ratios(current_assets=150, current_liabilities=100)
    assert out["result"] == {"current_ratio": 1.5, "working_capital": 50.0}
    assert out["table"]["rows"][0][1] == "Current ratio"


def test_ratios_need_two_items():
    with pytest.raises(ValueError):
        financial_ratios(revenue=100)


def test_ratio_table_has_formulas():
    rows = financial_ratios(**CO)["table"]["rows"]
    row = next(r for r in rows if r[1] == "Return on equity")
    assert [r[0] for r in rows].count("Profitability") == 1  # group name shown once
    assert row[2] == "Net income ÷ Total equity" and row[3] == "16.67%"


def test_balance_sheet_balances_and_common_size():
    out = balance_sheet_check(assets={"Cash": 50_000, "Receivables": 30_000, "Inventory": 40_000, "Plant": 180_000},
                              liabilities={"Payables": 40_000, "Bank loan": 110_000},
                              equity={"Share capital": 100_000, "Reserves": 50_000})
    r = out["result"]
    assert r["total_assets"] == 300_000 and r["balanced"] and r["difference"] == 0
    plant = next(row for row in out["table"]["rows"] if row[1] == "Plant")
    assert plant[3] == "60.0%"


def test_balance_sheet_difference_reported():
    r = balance_sheet_check(assets={"Cash": 100}, liabilities={"Loan": 50}, equity={"Capital": 40})["result"]
    assert not r["balanced"] and r["difference"] == 10


def test_cash_flow_indirect_method_reconciles():
    out = cash_flow_statement(
        net_income=50_000, depreciation=10_000,
        current_asset_changes={"Receivables": 5_000, "Inventory": [23_000, 20_000]},
        current_liability_changes={"Payables": 4_000},
        investing={"Purchase of equipment": -30_000, "Sale of investments": 8_000},
        financing={"Loan raised": 20_000, "Dividends paid": -12_000},
        opening_cash=15_000, closing_cash=63_000)
    r = out["result"]
    assert (r["cfo"], r["cfi"], r["cff"], r["net_change"], r["computed_closing_cash"]) == (62_000, -22_000, 8_000, 48_000, 63_000)
    assert r["reconciles"] is True
    rows = out["table"]["rows"]
    assert any(row[1] == "Increase in Receivables" and row[2] == "(5,000)" for row in rows)
    assert any(row[1] == "Decrease in Inventory" and row[2] == "3,000" for row in rows)


def test_cash_flow_flags_mismatch():
    r = cash_flow_statement(net_income=100, opening_cash=0, closing_cash=150)["result"]
    assert r["reconciles"] is False and r["unexplained_difference"] == 50


def test_working_capital_entry_must_be_change_or_pair():
    with pytest.raises(ValueError):
        cash_flow_statement(net_income=1, current_asset_changes={"Stock": [1, 2, 3]})


def test_zero_items_are_treated_as_not_given():
    r = financial_ratios(ebit=260, depreciation=0, revenue=2400)["result"]
    assert "ebitda" not in r and r["operating_margin_pct"] == 10.8333


def test_missing_equity_is_derived_as_the_balancing_figure():
    # current assets 5,760 + net fixed assets 25,500; current liabilities 5,200 + long-term debt 12,500 -> equity 13,560
    out = balance_sheet_check(assets={"Current assets": 5_760, "Net fixed assets": 25_500},
                              liabilities={"Current liabilities": 5_200, "Long-term debt": 12_500},
                              equity={"Shareholders' equity": 0})
    r = out["result"]
    assert r["total_equity"] == 13_560 and r["derived"] == {"total_equity": 13_560}
    assert r["balanced"] and r["difference"] == 0
    rows = out["table"]["rows"]
    assert any(row[1] == "Shareholders' equity (derived)" and row[2] == "13,560" for row in rows)
    assert "Not balanced" not in out["table"]["note"] and "derived" in out["table"]["note"]


def test_missing_section_is_derived_when_empty_and_liabilities_too():
    eq = balance_sheet_check(assets={"Cash": 100, "Plant": 400}, liabilities={"Loan": 150}, equity={})
    assert eq["result"]["total_equity"] == 350 and eq["result"]["derived"] == {"total_equity": 350}
    assert any(row[1] == "Total equity (derived)" for row in eq["table"]["rows"])
    li = balance_sheet_check(assets={"Cash": 100, "Plant": 400}, liabilities={"Total liabilities": 0}, equity={"Capital": 300})
    assert li["result"]["total_liabilities"] == 200 and li["result"]["balanced"]
    assert any(row[1] == "Total liabilities (derived)" for row in li["table"]["rows"])


def test_given_sections_are_still_checked_not_derived():
    r = balance_sheet_check(assets={"Cash": 100}, liabilities={"Loan": 50}, equity={"Capital": 40})["result"]
    assert not r["balanced"] and r["difference"] == 10 and "derived" not in r
    both_missing = balance_sheet_check(assets={"Cash": 100}, liabilities={}, equity={})["result"]
    assert not both_missing["balanced"] and "derived" not in both_missing
