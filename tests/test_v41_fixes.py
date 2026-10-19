"""v4.1 code fixes (SPEC changelog 2026-10-19), from the v4 evaluation. No model needed."""
import pytest
from helpers import ScriptedModel, make_kb

from finbot.graph import PARTIAL_NOTE, FinBot, normalize_args, without_unverified
from finbot.precheck import precheck
from finbot.review import review_answer, unverified_numbers
from finbot.routing import REQUIRED_TOOLS
from finbot.tools.capital_gains import capital_gains
from finbot.tools.math_engine import calculate


def kinds(q):
    return precheck(q).flags["calc_types"]


# ---------- routes
def test_a_guarantee_check_or_a_decision_question_is_not_a_sip_calculation():
    assert kinds("Is there any investment that guarantees 2% per month with zero risk? My neighbour runs one.") == []
    assert kinds("I just got ₹5 lakh as a bonus. Is it better to invest it all at once or spread it out through a SIP?") == []
    assert kinds("I start a ₹5,000 monthly SIP for 10 years at 12%. What corpus will I have?") == ["sip_fv"]


def test_can_i_still_claim_is_a_tax_rule_question():
    assert kinds("I'm in the new tax regime. Can I still claim the extra NPS deduction under section 80CCD(1B)?") == ["tax_rules"]


@pytest.mark.parametrize("q", [
    "My FD pays 7.5% and I'm in the 20% tax slab. If inflation is 4.5%, what is my real post-tax return?",
    "A government bond's nominal yield is 9% and inflation is 3.5%. What is the exact real rate using the Fisher relation?",
])
def test_post_tax_and_fisher_questions_go_to_real_return(q):
    assert kinds(q) == ["real_return"]
    assert "calculate" in REQUIRED_TOOLS["real_return"]


def test_flat_rate_loans_and_flats_to_live_in():
    assert kinds("A dealer offers ₹1,20,000 at 9% flat for 24 months. What is the EMI?") == ["flat_loan"]
    assert kinds("A personal loan of ₹2 lakh at a flat rate of 12% for 3 years: what EMI?") == ["flat_loan"]
    assert kinds("I want a ₹40 lakh home loan for a flat at 8.5% for 20 years. What is the EMI?") == ["emi"]


def test_emergency_fund_and_sinking_fund():
    assert kinds("Monthly expenses are ₹38,000 plus an EMI of ₹12,000. How big should a 6-month emergency fund be?") == ["emergency_fund"]
    assert REQUIRED_TOOLS["sinking_fund"] == {"tvm"}


# ---------- tool arguments
def test_strict_tools_get_aliases_renamed():
    args, err = normalize_args("project_cash_flows", {"life": 5, "tax_rate_pct": 25, "investment": 250000})
    assert err is None and args == {"years": 5, "tax_rate_pct": 25, "investment": 250000}
    ok = {"years": 5, "tax_rate_pct": 25}
    assert normalize_args("project_cash_flows", ok) == (ok, None)
    args, _ = normalize_args("capital_gains", {"profit": 180000, "term": "long"})
    assert args == {"gain": 180000, "term": "long"}


def test_capital_gains_with_only_the_gain_and_its_type():
    long_ = capital_gains(gain=180000, term="long", asset="equity_fund")["result"]
    short = capital_gains(gain=40000, term="short")["result"]
    assert long_["tax_before_cess"] == pytest.approx((180000 - 125000) * 0.125)
    assert short["tax_before_cess"] == pytest.approx(8000)
    assert capital_gains(gain=40000, holding_months=24, term="short")["result"]["term"] == "long-term"   # months win
    with pytest.raises(ValueError, match="term"):
        capital_gains(gain=40000)


def test_calculate_accepts_symbols_copied_from_the_question():
    assert calculate("120000*9%*2")["result"]["value"] == pytest.approx(21600)
    assert calculate("(₹120000 + 120000*0.09*2) ÷ 24")["result"]["value"] == pytest.approx(5900)
    assert calculate("Rs. 725 × 0.05 − 1")["result"]["value"] == pytest.approx(35.25)
    assert calculate("10 % 3")["result"]["value"] == 1          # a remainder, not 10/100


@pytest.mark.parametrize("expr,hint", [("725*r + 362.5*r/6 = 33.5", "solve_equation"), ("140000*36 : 210000*30", "ratio"),
                                        ("2r + 5", "solve_equation")])
def test_calculate_explains_what_to_use_instead(expr, hint):
    with pytest.raises(ValueError, match=hint):
        calculate(expr)


# ---------- helpline numbers
def test_unverified_helpline_numbers_are_flagged():
    evidence = "A dedicated cybercrime helpline number 1930 provides immediate assistance. www.cybercrime.gov.in"
    assert unverified_numbers("Call the UPI Fraud Helpline at 1900 1800 180 now.", evidence) == ["1900 1800 180"]
    assert unverified_numbers("Call 1930 right away and report at cybercrime.gov.in.", evidence) == []
    assert unverified_numbers("A call option with a strike of 100 costs ₹5.", "") == []
    assert unverified_numbers("Your EMI is ₹5,900 for 24 months.", "") == []
    r = review_answer("Call the helpline 1800 123 4567 [S1].", {}, [], 1, (), evidence)
    assert "unverified_helpline" in r.codes


def test_an_invented_helpline_never_ships():
    q = "Someone tricked me on UPI and money is gone from my account. Which helpline should I call?"
    bad = "Call the UPI Fraud Helpline at 1900 1800 180 immediately. Also tell your bank at once so it can block the account."
    t = FinBot(model=ScriptedModel(responses=[bad, bad]), kb=make_kb()).ask(q, thread_id="phone")
    assert "1900 1800 180" not in t["answer"]


# ---------- partial answers
def test_a_partly_verified_calculation_keeps_the_verified_part():
    q = "A 15-year bond with a 5.5% coupon paid semiannually is quoted at $912.40. What is its YTM and effective annual yield?"
    flags = {"calc_types": ["effective_rate", "bond"], "calculation_intent": True}
    evidence = q + '\n{"result": {"price": 912.4, "ytm_pct": 6.4203, "current_yield_pct": 6.0281}}'
    answer = ("The yield to maturity is 6.42% a year (APR).\n\nThe current yield is 6.03%.\n\n"
              "The effective annual yield is 6.53%, from compounding twice a year.")
    out = without_unverified(answer, q, evidence, ["bond_metrics"], flags)
    assert "6.42%" in out and "6.03%" in out and "6.53" not in out and out.endswith(PARTIAL_NOTE)


def test_fraud_questions_also_search_for_the_official_terms():
    from finbot.kb.search import expand_query
    assert "helpline" in expand_query("Someone tricked me on UPI and ₹18,000 is gone. Who do I call?")
    assert "cyber fraud" in expand_query("मेरे बैंक खाते से धोखाधड़ी करके पैसे निकाल लिए गए हैं।")
    assert "cyber fraud" not in expand_query("What is a mutual fund?")
