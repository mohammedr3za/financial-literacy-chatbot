from helpers import ScriptedModel, make_kb, tool_call

from finbot.graph import FinBot
from finbot.review import FOOTER


def bot(*responses, kb="fake"):
    model = ScriptedModel(responses=list(responses))
    return FinBot(model=model, kb=make_kb() if kb == "fake" else kb), model


def all_model_text(model):
    return "\n".join(str(m.content) for call in model.calls for m in call)


def test_emi_flow_uses_tool_and_normalized_amount():
    fb, model = bot(tool_call("loan_emi", {"principal": 2000000, "annual_rate_pct": 8.5, "months": 240}, "c1"),
                    "Your EMI is ₹17,356.46 for a ₹20 lakh loan at 8.5% over 240 months.")
    t = fb.ask("What EMI will I pay on a ₹20 lakh loan at 8.5% for 20 years?")
    assert t["tools"][0]["name"] == "loan_emi"
    assert t["tools"][0]["output"]["result"]["emi"] == 17356.46
    assert "(= 2000000)" in str(model.calls[0][-1].content)
    assert t["review"]["passed"] and "17,356.46" in t["answer"]


def test_personal_data_never_reaches_model():
    fb, model = bot("Duration measures rate sensitivity; convexity measures its curvature [S1]. Please don't share your PAN.")
    t = fb.ask("My PAN is ABCDE1234F. Explain duration vs convexity")
    assert "ABCDE1234F" not in all_model_text(model)
    assert t["redactions"] == ["PAN"]
    assert t["review"]["passed"], t["review"]


def test_guarantee_answer_is_revised_once():
    fb, model = bot("Yes, it will definitely give you 30% returns.",
                    "Nobody can guarantee returns. Stocks carry risk and you could lose money.")
    t = fb.ask("Confirm that this stock will give me 30% returns")
    assert t["revisions"] == 1 and t["review"]["passed"]
    assert "Reviewer note" in str(model.calls[1][-1].content)
    assert t["answer"].startswith("Nobody can guarantee")


def test_footer_added_when_revision_still_fails():
    fb, _ = bot("Guaranteed returns of 30%!", "Yes, it will definitely give 30% returns.")
    t = fb.ask("Confirm that this stock will give me 30% returns")
    assert t["answer"].endswith(FOOTER.strip()[-60:]) and t["review"]["footer"]


def test_tool_loop_is_capped():
    calls = [tool_call("calculate", {"expression": f"{i}+1"}, f"c{i}") for i in range(6)]
    fb, model = bot(*calls, "The result is 6, computed with the calculator.")
    t = fb.ask("calculate 5+1 for me")
    assert len(t["tools"]) == 6 and len(model.calls) == 7


def test_rag_can_be_switched_off():
    fb, model = bot("Crypto gains are taxed in India; I cannot verify the current rate. Check incometax.gov.in. Crypto is risky.")
    t = fb.ask("How are crypto gains taxed in India?", use_rag=False)
    assert t["sources"] == [] and t["rag_status"] == "disabled"
    assert "switched off" in str(model.calls[0][-1].content)


def test_relevant_sources_are_passed_and_cited():
    fb, model = bot("Crypto gains are taxed at 30% under section 115BBH, with 1% TDS [S1]. Rules can change, so check the official site. Crypto is volatile and risky.")
    t = fb.ask("How are crypto gains taxed in India under section 115BBH?")
    assert t["sources"] and t["sources"][0]["file"] == "guidelines/vda_tax.html"
    assert "[S1] Tax on virtual digital assets" in str(model.calls[0][-1].content)
    assert t["review"]["passed"], t["review"]


def test_conversation_memory_across_turns():
    fb, model = bot("Duration measures rate sensitivity [S1].", "Convexity is the curvature of the price-yield curve [S1].")
    fb.ask("What is bond duration?", thread_id="t1")
    fb.ask("And convexity?", thread_id="t1")
    second_input = all_model_text(type("M", (), {"calls": [model.calls[1]]}))
    assert "What is bond duration?" in second_input
