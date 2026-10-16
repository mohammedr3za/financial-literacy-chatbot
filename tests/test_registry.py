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
