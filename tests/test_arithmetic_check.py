"""Review check 11 (SPEC 3.1): every fully numeric expression shown in the answer must evaluate, with the safe calculator,
to a figure the answer states. Mismatches are caught; normal answers, symbolic formulas, ranges and dates are not."""
import ast
import json
import re
from pathlib import Path

import pytest

from finbot.review import arithmetic_mismatches, review_answer

CALC = {"calculation_intent": True}


@pytest.mark.parametrize("answer", [
    # LaTeX: \frac, \times, \cdot, thousands separators {,} and thin spaces, \% and \$
    r"$$\text{Net income} = \frac{0.12 \times 2{,}885}{1 + 1.80} = 123.64$$",
    r"$$NI = 0.12 \cdot \frac{2\,885}{2.8} \approx \$123.64$$",
    r"\( \text{ROE} = \frac{9\%}{1 - 0.35} = 13.85\% \)",
    r"$$PV = -0.37 - 3.39 \times \frac{1 - (1.10)^{-2}}{0.10} = -6.2535$$ and $$EAC = \frac{-6.2535}{1.7355} = -3.60$$",
    # plain text: ×, ÷, *, /, −, thousands separators, currency, percent
    "Sales = 190,000 ÷ 0.091 = $2,087,912.09, so credit sales are 0.85 × 2,087,912.09 = $1,774,725.27.",
    "EBT is 8,200 / (1 − 0.24) = 10,789.47 and EBIT is 10,789.47 + 4,800 = 15,589.47.",
    "Interest: 2,000 * 0.08 = 160. The total is ₹2,160.",
    "The PV is 3.39 × 1.7355 = 5.8834, rounded.",
    # textbook-style working with a rounded intermediate (0.0197) still matches within rounding
    "PM = 0.0197, so NI = 0.0197 × 6,269 = $123.64.",
    # value stated elsewhere in the answer, not right after the working
    "Equity is 2,885 / 2.8 (about $1,030.36), so net income is $123.64.",
    # chains and percentages written either way
    "ROE = 9% / 0.65 = 13.85%, and for Firm B 0.13 / 0.75 = 17.33%.",
])
def test_consistent_working_passes(answer):
    assert arithmetic_mismatches(answer) == []
    assert "arithmetic_mismatch" not in review_answer(answer, CALC, ["calculate"], 0).codes


def test_y3k_style_mismatch_is_caught_with_both_numbers_in_the_note():
    answer = ("Using the DuPont identity, $$NI = \\frac{0.12 \\times 2885}{1 + 1.80}$$ so the net income is **$22,255.71**.")
    found = arithmetic_mismatches(answer)
    assert len(found) == 1 and found[0]["value"] == pytest.approx(123.6429, abs=1e-3) and found[0]["stated"] == "$22,255.71"
    r = review_answer(answer, CALC, ["calculate"], 0)
    assert "arithmetic_mismatch" in r.codes
    note = " ".join(r.issues)
    assert "evaluates to 123.64" in note and "$22,255.71" in note


def test_plain_text_equation_with_wrong_result_is_caught():
    found = arithmetic_mismatches("Net income is 6,100 + 2,100 = 8,000.")
    assert found and found[0]["value"] == 8200 and found[0]["stated"] == "8,000"


def test_wrong_working_whose_value_is_stated_nowhere_is_caught():
    found = arithmetic_mismatches("The yearly cost is 60 × 500 / 1000 × 0.113, which is $3.93 a year.")
    assert found and found[0]["value"] == pytest.approx(3.39) and found[0]["stated"] == "$3.93"


@pytest.mark.parametrize("answer", [
    "PV = −0.37 − 3.39 × PVIFA, where PVIFA is the annuity factor.",      # symbolic: joined to a word
    "A 60-watt bulb used 500 hours × 0.06 kW costs more.",                  # units
    "Returns of 10-12% are typical over 2024-25; markets are open 24/7.",   # ranges, financial years, 24/7
    "Rates for FY 2025–26 were 8.25%, see the 01/04/2026 circular.",       # en dash, date
    "The bill is ₹5,000 + 18% GST.",                                        # a percentage added to an amount
    "Use EAC = PV ÷ annuity factor, with r = 10% and n = 24.",              # formula in words
    "| Year | Cash flow |\n|---|---|\n| 1 | (3.39) |\n| 2 | (3.39) |",       # model-written table
    "Section 115BBH taxes VDA income at 30% [S1]; TDS is 1% under 194S.",  # section numbers, citations
    "- 5,000 is the first instalment\n- 2,000 is the second",               # markdown bullets are not minus signs
    "**Answer:** the EMI is **₹17,356.46** for *240* months.",              # markdown emphasis
    "$$EAC = \\frac{NPV}{\\text{annuity factor}}$$",                       # LaTeX with words only
    "$$\\sqrt{x} + \\frac{1}{(1+r)^{n}}$$",                                  # symbolic LaTeX
    "Split it 60:40 between equity and debt; 1/3 of the corpus stays in cash.",
])
def test_normal_text_is_not_flagged(answer):
    assert arithmetic_mismatches(answer) == []


def test_typeset_integer_fraction_is_checked_but_plain_24_7_is_not():
    found = arithmetic_mismatches("$$\\text{Life} = \\frac{12{,}000}{500}$$ so the LED lasts 20 years.")
    assert found and found[0]["value"] == 24 and found[0]["stated"] == "20"
    assert arithmetic_mismatches("$$\\text{Life} = \\frac{12{,}000}{500}$$ so the LED lasts 24 years.") == []


def test_unparseable_expressions_are_skipped_not_flagged():
    assert arithmetic_mismatches("$$\\frac{1}{0}$$ and (2 + 3 = 5") == []
    assert arithmetic_mismatches("((((") == []


def test_saved_good_answers_and_fixture_answers_are_not_flagged():
    """Every model answer string in the test fixtures plus the saved 30-question benchmark answers (FinBot's own
    calculation set, not the McGraw questions) must pass."""
    known_bad = "so net income is $18,000"          # the deliberately wrong DuPont answer in test_financial_solver
    texts = []
    for path in Path(__file__).parent.glob("test_*.py"):
        if path.name == Path(__file__).name:
            continue
        for raw in re.findall(r'"((?:[^"\\\n]|\\.){40,})"', path.read_text()):
            try:
                text = ast.literal_eval(f'"{raw}"')
            except (ValueError, SyntaxError):
                continue
            if any(ch.isdigit() for ch in text) and known_bad not in text:
                texts.append(text)
    saved = Path(__file__).parent.parent / "results" / "finbot-rag-calc.jsonl"
    if saved.exists():
        texts += [json.loads(line)["answer"] for line in saved.read_text().splitlines()]
    assert len(texts) > 40
    flagged = [(t[:80], arithmetic_mismatches(t)) for t in texts if arithmetic_mismatches(t)]
    assert flagged == []
