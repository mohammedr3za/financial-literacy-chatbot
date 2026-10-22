"""Follow-ups (site report dd81798ad52f, 2026-10-22): a question about the previous answer is not a repeat of its
calculation, and a conversation scratchpad keeps every verified calculator input and result for later turns
(SPEC changelog 2026-10-22)."""
import pytest
from helpers import ScriptedModel, make_kb, tool_call

from finbot.graph import NOT_RELIABLE, FinBot, scratchpad, unverified_figures
from finbot.precheck import precheck
from finbot.tools.registry import TOOLS_BY_NAME

GOAL_Q = "How much should I save monthly to get ₹1 crore in 15 years if my investments earn about 12% a year?"
GOAL_ARGS = {"target": 10000000, "years": 15, "annual_return_pct": 12}
EMI_Q = "What EMI will I pay on a ₹20 lakh loan at 8.5% for 20 years?"
EMI_ARGS = {"principal": 2000000, "annual_rate_pct": 8.5, "months": 240}


@pytest.mark.parametrize("q", ['What are exactly these risks"', "What about the risks?", "Is that safe?",
                               "Can I lose money in this?", "Explain that please"])
def test_question_about_the_answer_is_not_a_new_calculation(q):
    f = precheck(q, previous_question=GOAL_Q).flags
    assert f["calc_types"] == [] and not f["calculation_intent"], f


@pytest.mark.parametrize("q", ["What if I already have ₹5 lakh saved?", "And for 20 years?",
                               "What if I increase it every year?"])
def test_calculation_followups_still_inherit_the_route(q):
    assert "goal_saving" in precheck(q, previous_question=GOAL_Q).flags["calc_types"]


def test_reported_followup_about_risks_is_answered():
    model = ScriptedModel(responses=[
        tool_call("goal_saving", GOAL_ARGS, "c1"),
        "Save about ₹19,818.62 a month. Returns are not guaranteed, and higher returns come with higher risk.",
        "The 12% a year is an assumption, not a promise. The main risks are market risk (equity prices can fall), "
        "inflation risk (₹1 crore will buy less in 15 years) and the risk of a fall just before your goal date."])
    fb = FinBot(model=model, kb=make_kb())
    fb.ask(GOAL_Q, thread_id="risks")
    t = fb.ask('What are exactly these risks"', thread_id="risks")
    assert t["review"]["passed"] and t["revisions"] == 0, t["review"]
    assert "market risk" in t["answer_text"] and NOT_RELIABLE not in t["answer_text"]


def test_scratchpad_keeps_calculations_beyond_the_history_window():
    model = ScriptedModel(responses=[
        tool_call("loan_emi", EMI_ARGS, "c1"), "Your EMI is ₹17,356.46 a month.",
        "Amortisation means each EMI pays the interest first and the rest reduces the loan.",
        "Prepaying early saves the most interest, because interest is charged on the balance.",
        tool_call("loan_emi", {**EMI_ARGS, "annual_rate_pct": 9}, "c2"), "At 9% the EMI is ₹17,994.52 a month."])
    fb = FinBot(model=model, kb=make_kb())
    fb.ask(EMI_Q, thread_id="memo")
    fb.ask("Explain amortisation in simple words", thread_id="memo")
    fb.ask("Explain how prepayment works", thread_id="memo")
    fb.ask("What if the rate were 9%?", thread_id="memo")
    first = next(c for c in model.calls if "What if the rate were 9%" in str(c[-1].content))
    text = "\n".join(str(m.content) for m in first)
    assert "loan_emi(principal=2000000, annual_rate_pct=8.5, months=240)" in text   # turn 1 is outside the 2-turn window
    assert "₹17,356.46" in text


def test_scratchpad_lists_inputs_and_key_results_and_stays_short():
    result = TOOLS_BY_NAME["goal_saving"].invoke(GOAL_ARGS)
    turns = [{"question": GOAL_Q, "calls": [{"name": "goal_saving", "args": GOAL_ARGS, "id": "c1"}], "results": [result],
              "answer": "Save ₹19,818.62 a month."}]
    memo = scratchpad(turns)
    assert "goal_saving(target=10000000, years=15, annual_return_pct=12)" in memo and "₹19,818.62" in memo
    assert scratchpad([]) == ""
    many = scratchpad(turns * 40)
    assert len(many) <= 1500 and "₹19,818.62" in many                       # the oldest are dropped, not the format


def test_restating_an_earlier_verified_result_is_not_unverified():
    result = TOOLS_BY_NAME["goal_saving"].invoke(GOAL_ARGS)
    flags = {"calculation_intent": True, "calc_types": ["goal_saving"]}
    answer = "As worked out before, you need ₹19,818.62 a month and invest ₹35.67 lakh in total."
    q = "And how much do I put in overall?"
    assert unverified_figures(answer, q, "", [], flags)                            # not in this question
    assert not unverified_figures(answer, q, "", [], flags, known=result)          # verified in an earlier turn
    assert unverified_figures("You need ₹21,000 a month.", q, "", [], flags, known=result)   # an invented figure still fails
