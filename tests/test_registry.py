import json

from finbot.tools.registry import CALC_TOOLS, TOOLS_BY_NAME


def test_all_tools_have_described_args():
    for t in CALC_TOOLS:
        props = t.args_schema.model_json_schema()["properties"]
        assert t.description, t.name
        for name, spec in props.items():
            assert spec.get("description"), f"{t.name}.{name} has no description"


def test_tool_returns_json_result():
    out = json.loads(TOOLS_BY_NAME["loan_emi"].invoke({"principal": 2000000, "annual_rate_pct": 8.5, "months": 240}))
    assert out["formatted"]["emi"] == "₹17,356.46"


def test_tool_error_is_returned_not_raised():
    out = json.loads(TOOLS_BY_NAME["calculate"].invoke({"expression": "__import__('os')"}))
    assert "error" in out
    out = json.loads(TOOLS_BY_NAME["loan_emi"].invoke({"principal": -1, "annual_rate_pct": 8.5, "months": 240}))
    assert "error" in out and "lakh" in out["error"]


def test_statement_tools_registered():
    out = json.loads(TOOLS_BY_NAME["cash_flow_statement"].invoke({
        "net_income": 100, "depreciation": 20, "current_asset_changes": {"Stock": [50, 40]}, "opening_cash": 10}))
    assert out["result"]["cfo"] == 130 and out["result"]["computed_closing_cash"] == 140
    assert "table_shown_to_user" in out
    out = json.loads(TOOLS_BY_NAME["financial_ratios"].invoke({"current_assets": 300, "current_liabilities": 150}))
    assert out["result"]["current_ratio"] == 2.0
    out = json.loads(TOOLS_BY_NAME["balance_sheet_check"].invoke({"assets": {"Cash": 10}, "liabilities": {}, "equity": {"Capital": 10}}))
    assert out["result"]["balanced"] is True


def test_unknown_argument_is_rejected_not_silently_dropped():
    import pytest
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        TOOLS_BY_NAME["cash_flow_statement"].invoke({"net_income": 100, "operating": {"Debtors": 5}})
