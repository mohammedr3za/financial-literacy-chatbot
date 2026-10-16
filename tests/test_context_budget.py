"""Context budget on the 16,384-token vLLM path (SPEC 3.1 agent): compact tool schemas, compacted tool results for the
model (full results stay in the trace), an output cap per backend, and one retry with trimmed history on a
context-length error."""
import json

import pytest
from helpers import ScriptedModel, make_kb, tool_call
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_tool

from finbot.config import Settings
from finbot.graph import FinBot, compact_tool_result
from finbot.tools.registry import CALC_TOOLS
from finbot.tools.schemas import compact_tool_schemas

OVERFLOW = RuntimeError("Error code: 400 - {'object': 'error', 'message': \"This model's maximum context length is 16384 "
                        "tokens. However, you requested 16385 tokens (15685 in the messages, 700 in the completion).\"}")


# ---------- (a) compact schemas
def test_compact_schemas_drop_title_null_defaults_and_null_unions():
    compact = compact_tool_schemas(CALC_TOOLS)
    assert [t["function"]["name"] for t in compact] == [t.name for t in CALC_TOOLS]
    text = json.dumps(compact)
    assert '"title"' not in text and '"default": null' not in text and '{"type": "null"}' not in text
    for full, small in zip((convert_to_openai_tool(t) for t in CALC_TOOLS), compact):
        fp, sp = full["function"]["parameters"], small["function"]["parameters"]
        assert set(fp["properties"]) == set(sp["properties"]), full["function"]["name"]   # no argument is lost
        assert fp.get("required", []) == sp.get("required", [])
        for name, prop in sp["properties"].items():
            assert "type" in prop or "anyOf" in prop, (full["function"]["name"], name)  # types survive the anyOf collapse


def test_compact_schemas_are_much_smaller():
    full = sum(len(json.dumps(convert_to_openai_tool(t), ensure_ascii=False)) for t in CALC_TOOLS)
    small = sum(len(json.dumps(t, ensure_ascii=False)) for t in compact_tool_schemas(CALC_TOOLS))
    assert small < 0.65 * full, (small, full)
    sf_full = len(json.dumps(convert_to_openai_tool(next(t for t in CALC_TOOLS if t.name == "solve_financials"))))
    sf_small = len(json.dumps(next(t for t in compact_tool_schemas(CALC_TOOLS) if t["function"]["name"] == "solve_financials")))
    assert sf_small < 0.4 * sf_full, (sf_small, sf_full)


def test_openai_backend_binds_compact_schemas_and_its_own_output_cap():
    from finbot.llm import get_runnables
    s = Settings(backend="openai", base_url="http://127.0.0.1:9/v1", model="m", fallback_model="none")
    plain, bound = get_runnables(CALC_TOOLS, s)
    assert bound.kwargs["tools"] == compact_tool_schemas(CALC_TOOLS)
    assert plain.max_tokens == 1500


# ---------- (c) output cap per backend
def test_output_cap_defaults_per_backend(monkeypatch):
    for k in ("FINBOT_MAX_OUTPUT_TOKENS", "FINBOT_MAX_OUTPUT_TOKENS_OPENAI", "FINBOT_MAX_OUTPUT_TOKENS_OLLAMA"):
        monkeypatch.delenv(k, raising=False)
    s = Settings()
    assert s.output_cap("openai") == 1500 and s.output_cap("ollama") == 700
    monkeypatch.setenv("FINBOT_MAX_OUTPUT_TOKENS", "800")           # the older name sets the Ollama cap only
    monkeypatch.setenv("FINBOT_MAX_OUTPUT_TOKENS_OPENAI", "2000")
    s = Settings()
    assert s.output_cap("openai") == 2000 and s.output_cap("ollama") == 800
    monkeypatch.setenv("FINBOT_MAX_OUTPUT_TOKENS_OLLAMA", "600")
    assert Settings().output_cap("ollama") == 600


# ---------- (b) compact tool results
def test_long_table_is_replaced_by_a_note_for_the_model_but_kept_in_trace():
    model = ScriptedModel(responses=[tool_call("loan_emi", {"principal": 2000000, "annual_rate_pct": 8.5, "months": 240}, "c1"),
                                     "Your EMI is ₹17,356.46 for 240 months."])
    t = FinBot(model=model, kb=make_kb()).ask("What EMI will I pay on a ₹20 lakh loan at 8.5% for 20 years?", thread_id="lt")
    seen = [m for m in model.calls[1] if isinstance(m, ToolMessage)][-1].content
    assert "table shown to user (21 rows)" in seen and "Opening balance" not in seen and "17356.46" in seen
    assert len(t["tools"][0]["output"]["table"]["rows"]) == 21 and t["tables"]            # full result in trace and UI
    assert len(seen) < 0.5 * len(json.dumps(t["tools"][0]["output"], ensure_ascii=False))


def test_short_table_stays_visible_to_the_model():
    out = json.dumps({"result": {"npv": 1.5}, "table": {"title": "T", "columns": ["Year", "Cash flow"],
                                                         "rows": [["0", "(100)"], ["1", "60"], ["2", "60"]], "align": [],
                                                         "bold_rows": [], "note": None},
                      "table_shown_to_user": "This table is displayed to the user automatically below your answer."})
    small = json.loads(compact_tool_result(out))
    assert small["table"] == {"columns": ["Year", "Cash flow"], "rows": [["0", "(100)"], ["1", "60"], ["2", "60"]],
                              "note": "shown to user below the answer; do not copy it"}
    assert "table_shown_to_user" not in small
    assert json.loads(compact_tool_result(out, drop_tables=True))["table"] == "table shown to user (3 rows)"
    assert compact_tool_result("not json") == "not json"
    assert compact_tool_result('{"error": "bad"}') == '{"error": "bad"}'


# ---------- (d) context-length 400: one retry with trimmed history
def test_context_overflow_retries_once_with_trimmed_history():
    model = ScriptedModel(responses=["Duration measures rate sensitivity.", "Convexity is curvature. " * 100, OVERFLOW,
                                     tool_call("loan_emi", {"principal": 2000000, "annual_rate_pct": 8.5, "months": 240}, "c1"),
                                     "Your EMI is ₹17,356.46."])
    fb = FinBot(model=model, kb=make_kb())
    fb.ask("Explain bond duration in simple words please", thread_id="ovf")
    fb.ask("Explain convexity in simple words please", thread_id="ovf")
    t = fb.ask("What EMI will I pay on a ₹20 lakh loan at 8.5% for 20 years?", thread_id="ovf")
    retry = model.calls[3]
    assert not any("bond duration" in str(m.content) for m in retry)     # only the last earlier turn is kept ...
    assert any("Explain convexity" in str(m.content) for m in retry)
    assert all(len(str(m.content)) < 400 for m in retry if isinstance(m, AIMessage))   # ... with its answer shortened
    assert any("₹20 lakh" in str(m.content) for m in retry)              # the question is kept
    assert t["review"]["passed"] and "17,356.46" in t["answer"]


def test_second_overflow_gives_a_number_free_message_and_other_errors_propagate():
    model = ScriptedModel(responses=[OVERFLOW, OVERFLOW, OVERFLOW, OVERFLOW])
    t = FinBot(model=model, kb=make_kb()).ask("What EMI will I pay on a ₹20 lakh loan at 8.5% for 20 years?", thread_id="o2")
    assert "too long" in t["answer_text"]
    model = ScriptedModel(responses=[RuntimeError("Error code: 500 - server crashed")])
    with pytest.raises(RuntimeError, match="server crashed"):
        FinBot(model=model, kb=make_kb()).ask("What EMI will I pay on a ₹20 lakh loan at 8.5% for 20 years?", thread_id="o3")
