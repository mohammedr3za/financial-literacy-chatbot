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
    # Our own figures, worked by hand: 0.14 × 3,920 ÷ 2.35 = 233.53; 7% ÷ 0.55 = 12.73%; 10% ÷ 0.70 = 14.29%;
    # PVIFA(7%, 3) = 2.6243, so −180 − 225 × 2.6243 = −770.47 and −770.47 ÷ 2.6243 = −293.59.
    # LaTeX: \frac, \times, \cdot, thousands separators {,} and thin spaces, \% and \$
    r"$$\text{Net income} = \frac{0.14 \times 3{,}920}{1 + 1.35} = 233.53$$",
    r"$$NI = 0.14 \cdot \frac{3\,920}{2.35} \approx \$233.53$$",
    r"\( \text{ROE} = \frac{7\%}{1 - 0.45} = 12.73\% \)",
    r"$$PV = -180 - 225 \times \frac{1 - (1.07)^{-3}}{0.07} = -770.47$$ and $$EAC = \frac{-770.47}{2.6243} = -293.59$$",
    # plain text: ×, ÷, *, /, −, thousands separators, currency, percent
    "Sales = 236,000 ÷ 0.074 = $3,189,189.19, so credit sales are 0.80 × 3,189,189.19 = $2,551,351.35.",
    "EBT is 10,500 / (1 − 0.28) = 14,583.33 and EBIT is 14,583.33 + 5,600 = 20,183.33.",
    "Interest: 2,000 * 0.08 = 160. The total is ₹2,160.",
    "The PV is 225 × 2.6243 = 590.47, rounded.",
    # textbook-style working with a rounded intermediate (0.0276) still matches within rounding
    "PM = 0.0276, so NI = 0.0276 × 8,450 = $233.53.",
    # value stated elsewhere in the answer, not right after the working
    "Equity is 3,920 / 2.35 (about $1,668.09), so net income is $233.53.",
    # chains and percentages written either way
    "ROE = 7% / 0.55 = 12.73%, and for the second firm 0.10 / 0.70 = 14.29%.",
])
def test_consistent_working_passes(answer):
    assert arithmetic_mismatches(answer) == []
    assert "arithmetic_mismatch" not in review_answer(answer, CALC, ["calculate"], 0).codes


def test_dupont_working_mismatch_is_caught_with_both_numbers_in_the_note():
    # the working evaluates to 0.14 × 3,920 ÷ 2.35 = 233.5319, but the answer states a different figure
    answer = ("Using the DuPont identity, $$NI = \\frac{0.14 \\times 3920}{1 + 1.35}$$ so the net income is **$18,764.20**.")
    found = arithmetic_mismatches(answer)
    assert len(found) == 1 and found[0]["value"] == pytest.approx(233.5319, abs=1e-3) and found[0]["stated"] == "$18,764.20"
    r = review_answer(answer, CALC, ["calculate"], 0)
    assert "arithmetic_mismatch" in r.codes
    note = " ".join(r.issues)
    assert "evaluates to 233.53" in note and "$18,764.20" in note


def test_plain_text_equation_with_wrong_result_is_caught():
    found = arithmetic_mismatches("Net income is 7,400 + 3,100 = 10,000.")
    assert found and found[0]["value"] == 10500 and found[0]["stated"] == "10,000"


def test_wrong_working_whose_value_is_stated_nowhere_is_caught():
    # 750 × 2,000 / 1000 × 0.15 = 225, but the answer states $252
    found = arithmetic_mismatches("The yearly cost is 750 × 2,000 / 1000 × 0.15, which is $252 a year.")
    assert found and found[0]["value"] == pytest.approx(225) and found[0]["stated"] == "$252"


@pytest.mark.parametrize("answer", [
    "PV = −180 − 225 × PVIFA, where PVIFA is the annuity factor.",        # symbolic: joined to a word
    "A 750-watt pump run 2,000 hours × 0.75 kW costs more.",                # units
    "Returns of 10-12% are typical over 2024-25; markets are open 24/7.",   # ranges, financial years, 24/7
    "Rates for FY 2025–26 were 8.25%, see the 01/04/2026 circular.",       # en dash, date
    "The bill is ₹5,000 + 18% GST.",                                        # a percentage added to an amount
    "Use EAC = PV ÷ annuity factor, with r = 7% and n = 8.",                # formula in words
    "| Year | Cash flow |\n|---|---|\n| 1 | (225.00) |\n| 2 | (225.00) |",   # model-written table
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
    found = arithmetic_mismatches("$$\\text{Life} = \\frac{16{,}000}{2{,}000}$$ so the pump lasts 6 years.")
    assert found and found[0]["value"] == 8 and found[0]["stated"] == "6"
    assert arithmetic_mismatches("$$\\text{Life} = \\frac{16{,}000}{2{,}000}$$ so the pump lasts 8 years.") == []


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
