from finbot.review import review_answer

BASE = {"guarantee_seeking": False, "personal_advice_request": False, "high_risk_topic": False,
        "calculation_intent": False, "regulatory_topic": False, "investment_topic": False}


def flags(**kw):
    return {**BASE, **kw}


def test_guarantee_language_fails():
    r = review_answer("Yes, it will definitely give you 30% returns.", flags(guarantee_seeking=True), [], 0)
    assert "guarantee_language" in r.codes


def test_negated_guarantee_passes():
    text = "No one can promise that. Returns are not guaranteed and there is a risk you could lose money."
    r = review_answer(text, flags(guarantee_seeking=True, investment_topic=True), [], 0)
    assert r.passed, r.codes


def test_risk_free_rate_is_not_a_guarantee():
    text = "The risk-free rate is the yield on government bonds; it is used in CAPM."
    assert "guarantee_language" not in review_answer(text, flags(), [], 0).codes


def test_calculation_needs_a_tool():
    r = review_answer("Your EMI is ₹17,356.", flags(calculation_intent=True), [], 0)
    assert "no_tool_for_numbers" in r.codes
    assert review_answer("Your EMI is ₹17,356.", flags(calculation_intent=True), ["loan_emi"], 0).passed


def test_crypto_needs_risk_disclosure():
    r = review_answer("Bitcoin is a digital currency traded on exchanges.", flags(high_risk_topic=True, investment_topic=True), [], 0)
    assert "no_risk_disclosure" in r.codes


def test_relevant_sources_must_be_cited_for_regulation():
    text = "Crypto gains are taxed at 30% with 1% TDS on transfers above the threshold. " * 3
    r = review_answer(text, flags(regulatory_topic=True), [], 1)
    assert "no_citation" in r.codes
    assert review_answer(text + " [S1]", flags(regulatory_topic=True), [], 1).passed


def test_personal_advice_buy_call_and_referral():
    r = review_answer("You should buy Infosys shares, they are strong.", flags(personal_advice_request=True, investment_topic=True), [], 0)
    assert {"buy_sell_call", "no_adviser_referral"} <= set(r.codes)
    good = ("I can't pick a stock for you. Compare diversification, costs and risk; equity values can fall. "
            "For a personal plan, consult a SEBI-registered investment adviser.")
    assert review_answer(good, flags(personal_advice_request=True, investment_topic=True), [], 0).passed


def test_empty_answer_fails():
    assert not review_answer("  ", flags(), [], 0).passed


def test_concept_question_does_not_need_risk_disclosure():
    r = review_answer("Duration measures how much a bond price moves when rates change.",
                      flags(investment_topic=True, concept_question=True), [], 0)
    assert r.passed, r.codes


def test_investment_decision_still_needs_risk_disclosure():
    r = review_answer("Gold funds track the price of gold.", flags(investment_topic=True), [], 0)
    assert "no_risk_disclosure" in r.codes


def test_fake_citation_without_sources():
    r = review_answer("Crypto is taxed at 20% after 36 months [S1].", flags(regulatory_topic=True), [], 0)
    assert "fake_citation" in r.codes and "unverified_regulatory" in r.codes


def test_citation_beyond_provided_sources():
    r = review_answer("See [S1] and [S4].", flags(), [], 2)
    assert "fake_citation" in r.codes and "[S4]" in r.issues[0]


def test_regulatory_without_source_must_say_cannot_verify():
    ok = "I cannot verify the current crypto tax rules; they change often. Check incometax.gov.in."
    assert review_answer(ok, flags(regulatory_topic=True), [], 0).passed


def test_unlisted_link_flagged_but_official_and_source_links_allowed():
    r = review_answer("Find advisers at investor.gov.in or sebi.gov.in.", flags(), [], 0)
    assert "unlisted_link" in r.codes and "not official: investor.gov.in." in r.issues[0]
    assert review_answer("Complain at scores.sebi.gov.in or cms.rbi.org.in.", flags(), [], 0).passed
    assert review_answer("Read the book at openstax.org [S1].", flags(), [], 1, ("https://openstax.org/books/x",)).passed


def test_invented_tax_figures_are_flagged():
    from finbot.review import unsupported_figures
    evidence = "Income from transfer of virtual digital assets is taxed at 30 per cent under section 115BBH."
    answer = "Crypto gains are taxed at 30% [S1], plus a 10% surcharge and 4% cess, about 32.2% in total."
    assert unsupported_figures(answer, evidence) == ["10%", "4%", "32.2%"]
    r = review_answer(answer, flags(regulatory_topic=True), [], 1, (), evidence)
    assert "unsupported_figures" in r.codes and "32.2%" in r.issues[-1]


def test_supported_figures_pass_including_lakh_and_tool_outputs():
    from finbot.review import unsupported_figures
    evidence = 'DICGC insures deposits up to Rs. 5,00,000 per depositor. {"formatted": {"tax": "₹1,50,000.00"}}'
    assert unsupported_figures("Cover is ₹5 lakh per depositor; tax is ₹1,50,000.", evidence) == []


def test_figures_check_only_applies_to_regulatory_questions():
    r = review_answer("If yields rise by 1%, the price falls by about 4%.", flags(), [], 0, (), "")
    assert r.passed


# check 1: a guarantee phrase in a warning is not a promise (2026-10-17 question bank)
def test_warnings_about_guaranteed_returns_are_not_guarantee_language():
    for text in ['The regulator forbids any adviser from offering "assured or guaranteed returns" [S1].',
                 'Stay cautious of anyone offering “assured returns” or asking for your login.',
                 'Be wary of unregistered channels promising assured returns; no return can be promised.',
                 'Fake platforms may falsely promise assured profits to new users.',
                 'Groups that talk about "guaranteed returns" are usually a fraud.']:
        assert "guarantee_language" not in review_answer(text, flags(), [], 0).codes, text


def test_promises_of_guaranteed_returns_are_still_caught():
    for text in ["This plan offers guaranteed returns of 9% a year.",
                 "Yes, this stock will definitely give you 30% this year.",
                 "You get assured returns with this scheme."]:
        assert "guarantee_language" in review_answer(text, flags(), [], 0).codes, text


def test_a_caution_word_in_the_next_sentence_does_not_excuse_a_promise():
    text = "This fund gives guaranteed 12% returns. Beware of fake apps that copy its name."
    assert "guarantee_language" in review_answer(text, flags(), [], 0).codes
