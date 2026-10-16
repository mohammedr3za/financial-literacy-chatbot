"""No fabricated numbers (SPEC 3.1 review): after the one revision, a calculation answer with no successful tool call (and
figures that are not in the question) or with unsupported figures is replaced by a number-free fallback."""
import re

from helpers import ScriptedModel, make_kb, tool_call

from finbot.graph import FinBot
from finbot.precheck import precheck
from finbot.routing import HINTS
from finbot.tools.calculators import loan_emi

VAN = "I can pay at most $950 a month for 60 months on a $48,000 camper-van loan. What is the highest APR I can afford, with monthly compounding?"
VAN_ARGS = {"principal": 48000, "months": 60, "emi": 950}
VAN_APR = f"{loan_emi(**VAN_ARGS)['result']['annual_rate_pct']:.2f}%"


def ask(question, *responses, tid="t"):
    model = ScriptedModel(responses=list(responses))
    return FinBot(model=model, kb=make_kb()).ask(question, thread_id=tid), model


def test_no_tool_twice_gives_number_free_fallback():
    t, model = ask(VAN, "The highest rate you can afford is about 9.50% APR.",
                   "After checking, the highest APR you can afford is 9.50%.", tid="van")
    assert t["revisions"] == 1 and not t["review"]["passed"]
    assert "could not compute this reliably" in t["answer_text"]
    assert "9.50" not in t["answer"] and not re.search(r"\d", t["answer_text"])
    note = str(model.calls[1][-1].content)        # the reviewer note repeats the loan-rate hint
    assert "Reviewer note" in note and "loan_emi(principal=" in note and "annual_rate_pct out" in note


def test_unsupported_figures_after_revision_give_fallback():
    q = "What EMI will I pay on a ₹20 lakh loan at 8.5% for 20 years?"
    call = {"principal": 2000000, "annual_rate_pct": 8.5, "months": 240}
    t, _ = ask(q, tool_call("loan_emi", call, "c1"), "Your EMI is ₹18,000 a month.",
               "Your EMI is ₹18,500 a month.", tid="emi")
    assert t["review"]["codes"] == ["unsupported_figures"]
    assert "could not compute this reliably" in t["answer_text"] and "18,500" not in t["answer"]


def test_answer_that_only_restates_the_question_keeps_its_text():
    q = "What EMI will I pay on a ₹20 lakh loan?"
    text = "To work out the EMI on ₹20 lakh I need the interest rate and the loan tenure. Please share them."
    t, _ = ask(q, text, text, tid="ask")
    assert not t["review"]["passed"] and "need the interest rate" in t["answer_text"]
    assert "could not compute" not in t["answer_text"]


def test_correct_tool_answer_is_untouched():
    t, _ = ask(VAN, tool_call("loan_emi", VAN_ARGS, "c1"), f"The highest rate you can afford is an APR of {VAN_APR}.", tid="ok")
    assert t["review"]["passed"] and VAN_APR in t["answer_text"]


def test_loan_rate_question_gets_the_loan_rate_route_and_hint():
    f = precheck(VAN).flags
    assert "loan_rate" in f["calc_types"] and "loan" not in f["calc_types"]
    hint = HINTS["loan_rate"]
    assert "loan_emi(principal=" in hint and "emi=" in hint and "months=" in hint and "annual_rate_pct out" in hint
    assert "loan_rate" in precheck("What is the APR on a car loan of $10,000 with monthly payments of $300 for 36 months?").flags["calc_types"]
    assert "loan_rate" not in precheck("What EMI will I pay on a ₹20 lakh loan at 8.5% for 20 years?").flags["calc_types"]


def test_context_block_carries_the_loan_rate_hint():
    _, model = ask(VAN, tool_call("loan_emi", VAN_ARGS, "c1"), f"The APR is {VAN_APR}.", tid="ctx")
    assert "annual_rate_pct out" in str(model.calls[0][-1].content)
