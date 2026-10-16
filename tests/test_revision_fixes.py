"""SPEC 3.1 (2026-10-17): a question that only sounds like a calculation loses only its sentences with unverified figures,
not the whole answer; on the vLLM path the revision of a recognised calculation must call a tool."""
from helpers import ScriptedModel, make_kb, tool_call

from finbot.config import Settings
from finbot.graph import FinBot
from finbot.llm import get_forced_tool_runnable
from finbot.review import prune_figures
from finbot.tools.calculators import loan_emi

VAN = "I can pay at most $950 a month for 60 months on a $48,000 camper-van loan. What is the highest APR I can afford, with monthly compounding?"
VAN_ARGS = {"principal": 48000, "months": 60, "emi": 950}
VAN_APR = f"{loan_emi(**VAN_ARGS)['result']['annual_rate_pct']:.2f}%"
COIN = "Can a meme coin really go up 20x, and how much of my savings should go into it?"
COIN_DRAFT = "A 20x rise is possible but rare. Keep it to 3.7% of your savings."
COIN_REVISED = (
    "A 20x rise is possible but very rare, and most small coins lose most of their value instead.\n\n"
    "**How much to put in**\n"
    "In past crashes many small coins fell by 87% or more.\n\n"
    "**Key risks**\n"
    "* Prices can swing wildly within days, and no regulator protects you if an exchange fails.\n"
    "* Coins produce no cash flows, so nothing underneath supports the price. Some fell 87% in a year.\n\n"
    "Build an emergency fund first and never invest money you will need soon. For a plan that fits you, consult a "
    "SEBI-registered investment adviser.")


def bot(*responses, forced=None):
    model = ScriptedModel(responses=list(responses))
    return FinBot(model=model, kb=make_kb(), forced_model=forced), model


# ---------- forced tool call on the revision of a recognised calculation
def test_revision_of_a_calculation_answered_by_hand_is_forced_to_call_a_tool():
    forced = ScriptedModel(responses=[tool_call("loan_emi", VAN_ARGS, "c1")])
    fb, _ = bot("The highest rate you can afford is about 9.50% APR.",
                f"The highest rate you can afford is an APR of {VAN_APR}.", forced=forced)
    t = fb.ask(VAN, thread_id="force")
    assert len(forced.calls) == 1 and "Reviewer note" in str(forced.calls[0][-1].content)
    assert t["revisions"] == 1 and t["review"]["passed"], t["review"]
    assert VAN_APR in t["answer_text"] and t["tools"][0]["name"] == "loan_emi"


def test_no_forced_call_for_a_question_without_a_calculation_route():
    forced = ScriptedModel(responses=[])          # any call would raise "ran out of responses"
    fb, _ = bot(COIN_DRAFT, tool_call("calculate", {"expression": "20**(1/5) - 1"}, "c1"), COIN_REVISED, forced=forced)
    fb.ask(COIN, thread_id="noforce")
    assert forced.calls == []


def test_first_draft_is_never_forced():
    forced = ScriptedModel(responses=[])
    fb, _ = bot(tool_call("loan_emi", VAN_ARGS, "c1"), f"The APR is {VAN_APR}.", forced=forced)
    assert fb.ask(VAN, thread_id="first")["review"]["passed"] and forced.calls == []


def test_forced_runnable_only_on_the_openai_backend():
    assert get_forced_tool_runnable([loan_emi], Settings(backend="ollama")) is None
    forced = get_forced_tool_runnable([loan_emi], Settings(backend="openai", fallback_model="none"))
    assert forced.kwargs["tool_choice"] == "required"
    with_fallback = get_forced_tool_runnable([loan_emi], Settings(backend="openai", fallback_model="qwen3:8b"))
    assert with_fallback.runnable.kwargs["tool_choice"] == "required"


# ---------- precise fallback: only the sentences with unverified figures go
def test_safety_answer_keeps_its_verified_text_when_a_figure_is_unverified():
    fb, _ = bot(COIN_DRAFT, tool_call("calculate", {"expression": "20**(1/5) - 1"}, "c1"), COIN_REVISED)
    t = fb.ask(COIN, thread_id="prune")
    text = t["answer_text"]
    assert "could not compute" not in text and "87%" not in text and "3.7%" not in text
    assert "emergency fund" in text and "**Key risks**" in text and "Prices can swing wildly" in text
    assert "Coins produce no cash flows, so nothing underneath supports the price." in text
    assert "**How much to put in**" not in text          # its only sentence went, so the heading goes too
    assert "SEBI-registered investment adviser" in t["answer"]


def test_safety_answer_with_too_little_verified_text_gets_the_fallback():
    fb, _ = bot(COIN_DRAFT, "Put 3.7% of your savings into it; many coins fell 87%.")
    t = fb.ask(COIN, thread_id="short")
    assert "could not compute this reliably" in t["answer_text"] and "3.7%" not in t["answer"]


def test_recognised_calculation_still_gets_the_number_free_fallback():
    long_wrong = ("The highest rate you can afford is about 9.50% APR. " + "Loan rates depend on your credit history and "
                  "the lender, so compare offers from several banks and read the fine print before you sign anything. " * 3)
    fb, _ = bot(long_wrong, long_wrong)
    t = fb.ask(VAN, thread_id="calc")
    assert "could not compute this reliably" in t["answer_text"] and "9.50" not in t["answer"]


def test_prune_figures_drops_sentences_rows_and_empty_headings():
    text = ("Intro without numbers stays here.\n\n**Returns**\nIt grew 87% last year.\n\n**Risks**\n"
            "* Volatile. It fell 87% once.\n* No cash flows.\n\n| Year | Change |\n|--|--|\n| 1 | 87% |\n| 2 | flat |")
    out = prune_figures(text, ["87%"])
    assert "87%" not in out and "**Returns**" not in out
    assert "**Risks**" in out and "* Volatile." in out and "* No cash flows." in out
    assert "| 2 | flat |" in out and "Intro without numbers stays here." in out
    assert prune_figures(text, []) == text


# ---------- compact tool schemas on Ollama (finbot-8b-v2 was trained with them)
def test_ollama_sends_full_schemas_unless_compact_tools_is_on():
    from finbot.llm import get_runnables
    from finbot.tools.registry import CALC_TOOLS
    from finbot.tools.schemas import compact_tool_schemas
    _, full = get_runnables(CALC_TOOLS, Settings(backend="ollama"))
    _, compact = get_runnables(CALC_TOOLS, Settings(backend="ollama", compact_tools_ollama=True))
    assert compact.kwargs["tools"] == compact_tool_schemas(CALC_TOOLS)
    assert len(str(full.kwargs["tools"])) > len(str(compact.kwargs["tools"])) * 1.3


def test_draft_that_asks_for_missing_inputs_is_not_forced_to_call_a_tool():
    forced = ScriptedModel(responses=[])          # forcing here made the model call npv_irr([-10000]) and lose the request
    ask_inputs = ("To work out the NPV I need the cash inflows for each year and the discount rate. The cost of "
                  "₹10,000 is given. Please share the cash flows and the rate.")
    fb, _ = bot(ask_inputs, ask_inputs, forced=forced)
    t = fb.ask("What is the NPV of a project that costs ₹10,000?", thread_id="inputs")
    assert forced.calls == [] and "Please share the cash flows" in t["answer_text"]


def test_request_for_inputs_on_a_calculation_route_keeps_its_text_without_unverified_figures():
    ask_inputs = ("To work out the NPV I need the cash inflows for each year and the discount rate. The cost of ₹10,000 is "
                  "given, but the yearly inflows and the project life are not. For example, at a 10% rate the NPV would be "
                  "₹2,437. Please share the cash flows, the rate and the number of years, and I will calculate it exactly.")
    fb, _ = bot(ask_inputs, ask_inputs)
    t = fb.ask("What is the NPV of a project that costs ₹10,000?", thread_id="inputs-prune")
    text = t["answer_text"]
    assert "could not compute" not in text and "2,437" not in t["answer"]
    assert "Please share the cash flows, the rate and the number of years" in text
