import pytest

from finbot.precheck import normalize_amounts, precheck, redact


@pytest.mark.parametrize("text,label,secret", [
    ("My PAN is ABCDE1234F, help", "PAN", "ABCDE1234F"),
    ("aadhaar 2345 6789 0123 please", "AADHAAR", "2345 6789 0123"),
    ("card 4111 1111 1111 1111 got charged", "CARD NUMBER", "4111 1111 1111 1111"),
    ("call me on 9876543210", "PHONE", "9876543210"),
    ("mail me at ravi.k@example.com", "EMAIL", "ravi.k@example.com"),
    ("pay to ravi@okhdfcbank", "UPI ID", "ravi@okhdfcbank"),
    ("my account no 123456789012 at SBIN0001234", "ACCOUNT NUMBER", "123456789012"),
])
def test_redaction(text, label, secret):
    out, found = redact(text)
    assert secret not in out and label in found


def test_amounts_are_not_redacted():
    out, found = redact("EMI on 2000000 at 8.5% for 240 months, SIP of 10000")
    assert found == [] and "2000000" in out


def test_non_luhn_long_number_is_not_called_a_card():
    _, found = redact("the index rose from 1234567890123 points")
    assert "CARD NUMBER" not in found


@pytest.mark.parametrize("text,value", [
    ("₹20 lakh", 2_000_000), ("20 lakhs", 2_000_000), ("1.5 crore", 15_000_000), ("2 cr", 20_000_000),
    ("50k", 50_000), ("Rs 3.5 lac", 350_000), ("20,00,000", 2_000_000), ("2,000,000", 2_000_000),
])
def test_amount_normalization(text, value):
    out, found = normalize_amounts(f"loan of {text} please")
    assert found and found[0]["value"] == value
    plain = str(int(value))
    assert f"(= {plain})" in out and text.strip() in out


def test_spike_failure_case_is_fixed():
    r = precheck("What EMI will I pay on a ₹20 lakh loan at 8.5% for 20 years?")
    assert "(= 2000000)" in r.text and r.flags["calculation_intent"]


@pytest.mark.parametrize("text,flag", [
    ("Confirm that this stock will give me 30% returns", "guarantee_seeking"),
    ("Is this fund guaranteed to double my money?", "guarantee_seeking"),
    ("Which stock should I buy with ₹50,000?", "personal_advice_request"),
    ("Should I put my savings into crypto?", "personal_advice_request"),
    ("Should I put my savings into crypto?", "high_risk_topic"),
    ("How are crypto gains taxed in India?", "regulatory_topic"),
    ("Explain F&O trading", "high_risk_topic"),
])
def test_flags_set(text, flag):
    assert precheck(text).flags[flag]


def test_concept_question_has_no_risk_flags():
    f = precheck("Explain duration vs convexity").flags
    assert not any(f[k] for k in ("guarantee_seeking", "personal_advice_request", "high_risk_topic", "calculation_intent"))


@pytest.mark.parametrize("text,expected", [
    ("What is bond duration?", True),
    ("Explain how SIPs work", True),
    ("What is the difference between FD and debt funds?", True),
    ("Should I put my savings into gold?", False),
    ("What is the best fund for my retirement?", False),
])
def test_concept_question_flag(text, expected):
    assert precheck(text).flags["concept_question"] is expected


def test_pasted_statement_is_detected_and_not_annotated():
    text = "Balance sheet (₹ lakh)\nCash\t1,20,000\nInventory\t2,50,000\nCurrent liabilities\t1,80,000\nCompute the current ratio"
    r = precheck(text)
    assert r.flags["statement_data"] and r.flags["calculation_intent"]
    assert "(= " not in r.text


def test_plain_question_with_grouped_amount_still_annotated():
    assert "(= 120000)" in precheck("Is ₹1,20,000 enough for an emergency fund?").text
