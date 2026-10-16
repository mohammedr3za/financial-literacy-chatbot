import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.report import kappa  # noqa: E402
from bench.scoring import calc_correct, extract_numbers  # noqa: E402


def test_extract_indian_formats():
    assert extract_numbers("EMI ₹17,356.46, corpus ₹23.23 lakh, 1.5 crore, IRR 8.90%") == [17356.46, 2323000.0, 15000000.0, 8.90]


def test_calc_correct_within_tolerance():
    assert calc_correct("Your corpus will be about ₹23.23 lakh.", "SIP of ₹10,000 for 10 years at 12%", 2323390.76)[0]
    assert not calc_correct("About ₹20 lakh.", "SIP of ₹10,000 for 10 years at 12%", 2323390.76)[0]


def test_numbers_copied_from_question_do_not_count():
    # expected 12 (YTM); the question contains 10 and 927.90; an answer that only repeats them is wrong
    assert not calc_correct("The bond has a 10% coupon and price 927.90.", "10% coupon bond at ₹927.90", 12.0)[0]
    assert calc_correct("YTM is 12.00%.", "10% coupon bond at ₹927.90", 12.0)[0]


def test_kappa():
    assert kappa([2, 1, 0, 2], [2, 1, 0, 2]) == 1.0
    assert kappa([], []) is None
