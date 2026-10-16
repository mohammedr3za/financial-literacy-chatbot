"""Question-bank addendum (SPEC changelog 2026-10-17): Hindi and Tamil amounts and routing, equity capital gains in
income_tax, decision questions without amounts, a final tool-narration clean-up, and no made-up examples when inputs
are missing."""
import json
import re

import pytest
from helpers import ScriptedModel, make_kb, tool_call

from finbot.graph import FinBot, strip_tool_narration
from finbot.precheck import precheck
from finbot.review import review_answer
from finbot.tools.registry import TOOLS_BY_NAME

HINDI_EMI = "₹20 लाख के होम लोन पर 8.5% ब्याज दर से 20 साल के लिए मेरी EMI कितनी होगी?"


def ask(question, *responses, tid="t"):
    model = ScriptedModel(responses=list(responses))
    return FinBot(model=model, kb=make_kb()).ask(question, thread_id=tid), model


# ---------- 1. Indian-language amounts and routing
@pytest.mark.parametrize("text,value", [
    (HINDI_EMI, 2_000_000), ("मुझे 1.5 करोड़ जमा करने हैं", 15_000_000), ("50 लाखों का लक्ष्य", 5_000_000),
    ("₹30 லட்சம் வீட்டுக் கடன்", 3_000_000), ("2 கோடி இலக்கு", 20_000_000),
])
def test_indian_language_amounts_are_converted(text, value):
    r = precheck(text)
    assert any(a["value"] == value for a in r.amounts), r.amounts
    assert f"(= {value})" in r.text


@pytest.mark.parametrize("q,kind", [
    (HINDI_EMI, "emi"),
    ("₹30 லட்சம் வீட்டுக் கடனுக்கு 9% வட்டியில் 20 ஆண்டுகளுக்கு EMI எவ்வளவு?", "emi"),
    ("₹5 लाख के लोन पर 10% ब्याज पर हर महीने ₹12,000 देने पर कितने महीने लगेंगे?", "loan"),
    ("मैं हर महीने ₹10,000 की SIP 10 साल के लिए 12% पर करूँ तो कितना मिलेगा?", "sip_fv"),
])
def test_hindi_and_tamil_calculations_are_routed(q, kind):
    f = precheck(q).flags
    assert kind in f["calc_types"] and f["calculation_intent"], f


def test_hindi_emi_without_a_tool_gets_the_number_free_fallback():
    t, _ = ask(HINDI_EMI, "आपकी EMI ₹17,541 होगी।", "आपकी मासिक EMI लगभग ₹17,541 होगी।", tid="hi")
    assert "17,541" not in t["answer"] and "could not compute this reliably" in t["answer_text"]
    assert re.search(r"[ऀ-ॿ]", t["answer_text"])                 # the fallback also speaks Hindi
    assert not re.search(r"\d", t["answer_text"])


def test_hindi_emi_with_the_tool_passes():
    t, _ = ask(HINDI_EMI, tool_call("loan_emi", {"principal": 2000000, "annual_rate_pct": 8.5, "months": 240}, "c1"),
               "आपकी मासिक EMI ₹17,356.46 होगी।", tid="hi2")
    assert t["review"]["passed"], t["review"]


# ---------- 2. equity capital gains are rejected by income_tax, never folded into other_income
def test_income_tax_rejects_equity_capital_gains_with_a_clear_message():
    out = json.loads(TOOLS_BY_NAME["income_tax"].invoke({"gross_salary": 1000000, "capital_gains_equity_stcg": 200000}))
    assert "error" in out
    msg = out["error"].lower()
    assert "special rates" in msg and "87a" in msg and "other_income" in msg and "not compute" in msg
    out = json.loads(TOOLS_BY_NAME["income_tax"].invoke({"gross_salary": 1000000, "capital_gains_equity_ltcg": 150000}))
    assert "error" in out


def test_stock_profit_passed_as_other_income_is_bounced():
    q = "My salary is ₹10 lakh and I made ₹2 lakh profit selling shares this year. How much income tax do I pay?"
    t, _ = ask(q, tool_call("income_tax", {"gross_salary": 1000000, "other_income": 200000}, "c1"),
               tool_call("income_tax", {"gross_salary": 1000000}, "c2"),
               *["On the salary alone the tax is ₹0 in the new regime. The ₹2 lakh share profit is a capital gain, taxed "
                 "separately at special rates, so I have not included it."] * 2, tid="cg")
    first = t["tools"][0]["output"]
    assert "error" in first and "capital gain" in first["error"].lower()
    assert "error" not in t["tools"][1]["output"]


def test_tool_description_and_prompt_rule_mention_capital_gains():
    from finbot.prompts import SYSTEM_PROMPT
    assert "capital_gains_equity" in TOOLS_BY_NAME["income_tax"].description or \
        "capital_gains_equity_stcg" in json.dumps(TOOLS_BY_NAME["income_tax"].args)
    rule12 = next(line for line in SYSTEM_PROMPT.splitlines() if line.startswith("12."))
    assert "other_income" in rule12 and ("shares" in rule12 or "mutual" in rule12)


# ---------- 3. decision questions without amounts are concept answers
@pytest.mark.parametrize("q", [
    "Should I pay off a 9% loan or invest in a fund?",
    "Is it better to prepay my home loan EMI or invest in a 12% SIP?",
])
def test_decision_questions_without_amounts_are_not_loan_calculations(q):
    f = precheck(q).flags
    assert not {"loan", "loan_rate", "emi"} & set(f["calc_types"]), f["calc_types"]


def test_loan_questions_with_amounts_still_route():
    assert "emi" in precheck("What EMI will I pay on a ₹20 lakh loan at 8.5% for 20 years?").flags["calc_types"]
    assert "loan" in precheck("How much can I borrow if I can pay ₹25,000 a month for 20 years at 9%?").flags["calc_types"]


# ---------- 4. final clean-up of tool narration
@pytest.mark.parametrize("raw,expect_in,expect_out", [
    ("The `npv_irr` tool gives an NPV of $1,000 at 10%.", "The calculator gives an NPV of $1,000 at 10%.", "npv_irr"),
    ("To compute it, the `loan_emi` tool requires the loan amount. Please share it.", "the calculator requires the loan amount",
     "loan_emi"),
    ("Your EMI is ₹17,356.46. I called loan_emi(principal=2000000, months=240) to get it.", "Your EMI is ₹17,356.46.", "called"),
    ("Using goal_saving, you need ₹19,818.62 a month.", "you need ₹19,818.62 a month", "goal_saving"),
])
def test_strip_tool_narration(raw, expect_in, expect_out):
    out = strip_tool_narration(raw)
    assert expect_in in out and expect_out not in out


def test_plain_answers_are_unchanged_by_the_cleanup():
    text = "Your EMI is ₹17,356.46. Let me calculate the total: it is ₹41,65,551.54. The TVM formula discounts each payment."
    assert strip_tool_narration(text) == text


def test_narration_left_after_revision_is_removed_before_shipping():
    q = "What EMI will I pay on a ₹20 lakh loan at 8.5% for 20 years?"
    call = tool_call("loan_emi", {"principal": 2000000, "annual_rate_pct": 8.5, "months": 240}, "c1")
    t, _ = ask(q, call, "The `loan_emi` tool gives an EMI of ₹17,356.46.", "The `loan_emi` tool gives an EMI of ₹17,356.46.",
               tid="narr")
    assert "loan_emi" not in t["answer"] and "₹17,356.46" in t["answer_text"]


# ---------- 5. missing inputs: ask, do not compute a made-up example
NPV_Q = "What is the NPV of a project that costs 10,000?"
HYPO = ("To find the NPV I need the yearly cash flows and the discount rate. Please share them. For example, if the project "
        "returned 4,000 a year for 3 years at 10%, the NPV would be about −52.59 (a hypothetical illustration).")


def test_made_up_example_after_asking_for_inputs_is_flagged():
    flags = precheck(NPV_Q).flags
    r = review_answer(HYPO, flags, [], 0)
    assert "hypothetical_example" in r.codes and "do not compute" in " ".join(r.issues).lower()
    ok = review_answer("To find the NPV I need the yearly cash flows and the discount rate. Please share them.", flags, [], 0)
    assert "hypothetical_example" not in ok.codes


def test_made_up_example_is_cut_if_the_rewrite_keeps_it():
    t, _ = ask(NPV_Q, HYPO, HYPO, tid="hypo")
    assert "Please share them" in t["answer_text"] and "4,000" not in t["answer_text"] and "52.59" not in t["answer_text"]


def test_prompt_says_ask_for_missing_inputs():
    from finbot.prompts import SYSTEM_PROMPT
    assert "missing" in SYSTEM_PROMPT and "example" in SYSTEM_PROMPT.split("missing", 1)[1][:300]


# ---------- tool names that are also English words ("perpetuity (…)", "calculate (…)") are prose, not calls
def test_word_tool_names_followed_by_a_bracket_in_prose_are_not_narration():
    text = "A growing perpetuity (cash flows that never end) is worth $1,428,571.43 here."
    assert strip_tool_narration(text) == text
    assert "tool_narration" not in review_answer(text, {"calculation_intent": True}, ["perpetuity"], 0).codes
    assert "tool_narration" in review_answer("I ran perpetuity(cash_flow=1) to get it.", {"calculation_intent": True},
                                             ["perpetuity"], 0).codes


# ---------- a follow-up may quote the previous turn's tool result
def test_followup_answer_may_quote_the_previous_result():
    from finbot.tools.calculators import loan_emi
    emi9 = loan_emi(principal=2000000, annual_rate_pct=9, months=240)["result"]["emi"]
    t1 = tool_call("loan_emi", {"principal": 2000000, "annual_rate_pct": 8.5, "months": 240}, "c1")
    t2 = tool_call("loan_emi", {"principal": 2000000, "annual_rate_pct": 9, "months": 240}, "c2")
    model = ScriptedModel(responses=[t1, "Your EMI is ₹17,356.46.", t2, f"At 9% the EMI is ₹{emi9:,.2f}, up from ₹17,356.46."])
    fb = FinBot(model=model, kb=make_kb())
    fb.ask("What EMI will I pay on a ₹20 lakh loan at 8.5% for 20 years?", thread_id="q")
    t = fb.ask("What if the rate were 9%?", thread_id="q")
    assert t["review"]["passed"], t["review"]
