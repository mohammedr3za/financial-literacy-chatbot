"""Multi-turn history budget (SPEC 3.1 agent): each model call gets the system prompt, the current turn, and at most the
last 2 earlier turns as plain question/answer text (the previous turn's tool inputs as one line), within a token budget.
A live 9-question thread hit vLLM 400 on turns 8 and 9."""
from helpers import ScriptedModel, make_kb, tool_call
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from finbot.graph import FinBot, estimate_tokens, prompt_budget

EMI_ARGS = {"principal": 2000000, "annual_rate_pct": 8.5, "months": 240}
LONG_ANSWER = "Your EMI is ₹17,356.46 a month. " + "The interest share falls every year as the balance shrinks. " * 60


def _first_calls(model, n_turns):
    """The first model call of each turn (2 calls a turn: tool call, then answer)."""
    return [model.calls[2 * i] for i in range(n_turns)]


def test_ten_turns_stay_under_the_prompt_budget():
    responses = []
    for i in range(10):
        responses += [tool_call("loan_emi", EMI_ARGS, f"c{i}"), LONG_ANSWER]
    model = ScriptedModel(responses=responses)
    fb = FinBot(model=model, kb=make_kb())
    budget = prompt_budget(fb.settings)
    assert 4000 < budget <= 10_000
    for i in range(10):
        fb.ask(f"Question {i}: what EMI will I pay on a ₹20 lakh loan at 8.5% for 20 years?", thread_id="ten")
    for call in model.calls:
        assert isinstance(call[0], SystemMessage)
        assert estimate_tokens(call[1:]) <= budget, estimate_tokens(call[1:])
        assert estimate_tokens(call) + fb.settings.output_cap() <= fb.settings.num_ctx
    for i, call in enumerate(_first_calls(model, 10)):
        assert not any(isinstance(m, ToolMessage) for m in call)                        # no old tool payloads
        assert not any(isinstance(m, AIMessage) and m.tool_calls for m in call)
        earlier = [k for k in range(i) if any(f"Question {k}:" in str(m.content) for m in call)]
        assert earlier == list(range(max(0, i - 2), i)), (i, earlier)                   # only the last 2 turns
    last = _first_calls(model, 10)[-1]
    assert not any("Pre-check notes" in str(m.content) for m in last[1:-1])             # no old context blocks


def test_very_long_earlier_answers_are_shortened_to_fit():
    huge = "Your EMI is ₹17,356.46. " + "Detail. " * 6000
    model = ScriptedModel(responses=[huge, huge, "Fine."])
    fb = FinBot(model=model, kb=make_kb())
    for q in ("Explain how an EMI works", "Explain amortization", "And prepayment?"):
        fb.ask(q, thread_id="huge")
    assert estimate_tokens(model.calls[-1][1:]) <= prompt_budget(fb.settings)
    assert any("Explain amortization" in str(m.content) for m in model.calls[-1])       # the previous question survives


def test_followup_sees_previous_question_answer_and_inputs():
    model = ScriptedModel(responses=[tool_call("loan_emi", EMI_ARGS, "c1"), "Your EMI is ₹17,356.46 a month.",
                                     tool_call("loan_emi", {**EMI_ARGS, "annual_rate_pct": 9}, "c2"),
                                     "At 9% the EMI is ₹17,994.52 a month."])
    fb = FinBot(model=model, kb=make_kb())
    fb.ask("What EMI will I pay on a ₹20 lakh loan at 8.5% for 20 years?", thread_id="fu2")
    t = fb.ask("What if the rate were 9%?", thread_id="fu2")
    text = "\n".join(str(m.content) for m in model.calls[2])
    assert "8.5% for 20 years" in text and "₹17,356.46" in text
    assert "loan_emi(principal=2000000, annual_rate_pct=8.5, months=240)" in text
    assert t["review"]["passed"], t["review"]


def test_followup_grounding_uses_the_previous_turn():
    q1 = "Sales are $675,000, costs $410,000, depreciation $65,000, interest $35,000 and the tax rate 23 percent. What is net income?"
    args = {"sales": 675000, "costs": 410000, "depreciation": 65000, "interest": 35000, "targets": ["net_income"]}
    model = ScriptedModel(responses=[tool_call("solve_financials", {**args, "tax_rate_pct": 23}, "c1"), "Net income is $127,050.",
                                     tool_call("solve_financials", {**args, "tax_rate_pct": 25}, "c2"), "Net income is $123,750."])
    fb = FinBot(model=model, kb=make_kb())
    fb.ask(q1, thread_id="g")
    t = fb.ask("What if the tax rate were 25 percent?", thread_id="g")
    assert "error" not in t["tools"][0]["output"], t["tools"][0]["output"]
    assert t["tools"][0]["output"]["result"]["net_income"] == 123750
    assert t["flags"]["currency"] == "USD" and not t["flags"]["regulatory_topic"] and t["review"]["passed"]


def test_history_holds_no_reviewer_notes_or_footers():
    model = ScriptedModel(responses=["Guaranteed returns of 30%!", "Yes, it will definitely give 30% returns.", "Okay."])
    fb = FinBot(model=model, kb=make_kb())
    fb.ask("Confirm that this stock will give me 30% returns", thread_id="rv")
    fb.ask("Thanks", thread_id="rv")
    text = "\n".join(str(m.content) for m in model.calls[-1])
    assert "Reviewer note" not in text and "not investment advice" not in text
    assert sum(isinstance(m, HumanMessage) for m in model.calls[-1]) == 2              # previous question + current turn
