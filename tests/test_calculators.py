"""Reference values are textbook / independently computed figures, not outputs of our code.

- EMI: standard reducing-balance formula, ₹20 lakh at 8.5% for 240 months.
- SIP: ₹10,000/month at 12% for 10 years, payment at start of month (matches common Indian SIP calculators).
- Bond: 5-year 10% annual coupon at par, Macaulay duration 4.17 years (standard textbook example).
- Black-Scholes: Hull, Options Futures and Other Derivatives, S=42, K=40, r=10%, sigma=20%, T=0.5.
"""
import math

import pytest

from finbot.tools.calculators import (
    black_scholes, bond_metrics, loan_emi, npv_irr, sip_future_value, tvm,
)
from finbot.tools.fmt import format_inr, inr_words


# ---------- formatting ----------
@pytest.mark.parametrize("value,expected", [
    (2323390.76, "₹23,23,390.76"),
    (17356.46, "₹17,356.46"),
    (12345678, "₹1,23,45,678.00"),
    (999, "₹999.00"),
    (-150000, "-₹1,50,000.00"),
])
def test_format_inr(value, expected):
    assert format_inr(value) == expected


def test_inr_words():
    assert inr_words(2323390.76) == "₹23.23 lakh"
    assert inr_words(15_000_000) == "₹1.50 crore"
    assert inr_words(17356.46) == "₹17,356.46"


# ---------- loan EMI ----------
def test_loan_emi_reference():
    r = loan_emi(principal=2_000_000, annual_rate_pct=8.5, months=240)["result"]
    assert r["emi"] == pytest.approx(17356.46, abs=0.01)
    assert r["total_interest"] == pytest.approx(2165551.52, abs=1.0)
    assert r["first_year_interest"] + r["first_year_principal"] == pytest.approx(12 * r["emi"], abs=0.1)  # outputs rounded to paise


def test_loan_emi_zero_rate():
    assert loan_emi(principal=120_000, annual_rate_pct=0, months=12)["result"]["emi"] == pytest.approx(10_000)


def test_loan_emi_rejects_bad_input():
    with pytest.raises(ValueError):
        loan_emi(principal=-5, annual_rate_pct=8, months=12)
    with pytest.raises(ValueError):
        loan_emi(principal=100000, annual_rate_pct=8, months=0)


def test_loan_emi_has_formula_and_formatted():
    out = loan_emi(principal=2_000_000, annual_rate_pct=8.5, months=240)
    assert "formula" in out and out["formatted"]["emi"] == "₹17,356.46"


# ---------- SIP ----------
def test_sip_reference():
    r = sip_future_value(monthly=10_000, annual_return_pct=12, years=10)["result"]
    assert r["corpus"] == pytest.approx(2323390.76, abs=0.05)
    assert r["invested"] == pytest.approx(1_200_000)
    assert r["gain"] == pytest.approx(2323390.76 - 1_200_000, abs=0.05)


def test_sip_step_up_increases_corpus_and_invested():
    flat = sip_future_value(monthly=10_000, annual_return_pct=12, years=10)["result"]
    up = sip_future_value(monthly=10_000, annual_return_pct=12, years=10, step_up_pct=10)["result"]
    assert up["corpus"] > flat["corpus"] and up["invested"] > flat["invested"]
    # year 2 deposits are 11,000/month, so invested = 12 * sum(10000 * 1.1**k for k in 0..9)
    assert up["invested"] == pytest.approx(12 * sum(10_000 * 1.1 ** k for k in range(10)), abs=0.5)


# ---------- TVM ----------
def test_tvm_fv_lump_sum():
    assert tvm(solve_for="fv", annual_rate_pct=10, years=5, pv=1000)["result"]["fv"] == pytest.approx(1610.51, abs=0.01)


def test_tvm_pv_of_target():
    assert tvm(solve_for="pv", annual_rate_pct=10, years=5, fv=1610.51)["result"]["pv"] == pytest.approx(1000, abs=0.01)


def test_tvm_cagr():
    assert tvm(solve_for="annual_rate_pct", years=5, pv=100, fv=200)["result"]["annual_rate_pct"] == pytest.approx(14.8698, abs=1e-3)


def test_tvm_monthly_deposit_to_reach_target_round_trips():
    pmt = tvm(solve_for="pmt", annual_rate_pct=12, years=15, fv=10_000_000, periods_per_year=12)["result"]["pmt"]
    fv = tvm(solve_for="fv", annual_rate_pct=12, years=15, pmt=pmt, periods_per_year=12)["result"]["fv"]
    assert fv == pytest.approx(10_000_000, rel=1e-6)  # pmt is rounded to paise


def test_tvm_years_to_double():
    years = tvm(solve_for="years", annual_rate_pct=12, pv=1, fv=2)["result"]["years"]
    assert years == pytest.approx(math.log(2) / math.log(1.12), abs=1e-6)  # years rounded to 6 dp


def test_tvm_rate_with_deposits_round_trips():
    fv = tvm(solve_for="fv", annual_rate_pct=9, years=8, pv=5000, pmt=1000)["result"]["fv"]
    rate = tvm(solve_for="annual_rate_pct", years=8, pv=5000, pmt=1000, fv=fv)["result"]["annual_rate_pct"]
    assert rate == pytest.approx(9, abs=1e-4)  # fv is rounded to paise


def test_tvm_rejects_unknown_target():
    with pytest.raises(ValueError):
        tvm(solve_for="banana", annual_rate_pct=10, years=5, pv=1000)


# ---------- NPV / IRR ----------
def test_npv_irr_reference():
    r = npv_irr(cashflows=[-1000, 300, 400, 500], discount_rate_pct=10)["result"]
    assert r["npv"] == pytest.approx(-21.0368, abs=1e-3)
    assert r["irr_pct"] == pytest.approx(8.8963, abs=1e-3)


def test_irr_none_without_sign_change():
    r = npv_irr(cashflows=[100, 200, 300])["result"]
    assert r["irr_pct"] is None


# ---------- bonds ----------
def test_bond_at_par():
    r = bond_metrics(face=1000, coupon_pct=10, years=5, freq=1, ytm_pct=10)["result"]
    assert r["price"] == pytest.approx(1000, abs=1e-6)
    assert r["macaulay_duration"] == pytest.approx(4.1699, abs=1e-4)
    assert r["modified_duration"] == pytest.approx(3.7908, abs=1e-4)


def test_bond_price_and_ytm_round_trip():
    price = bond_metrics(face=1000, coupon_pct=10, years=5, freq=1, ytm_pct=12)["result"]["price"]
    assert price == pytest.approx(927.9045, abs=1e-3)
    ytm = bond_metrics(face=1000, coupon_pct=10, years=5, freq=1, price=price)["result"]["ytm_pct"]
    assert ytm == pytest.approx(12, abs=1e-6)


def test_bond_requires_exactly_one_of_ytm_or_price():
    with pytest.raises(ValueError):
        bond_metrics(face=1000, coupon_pct=10, years=5, freq=1)
    with pytest.raises(ValueError):
        bond_metrics(face=1000, coupon_pct=10, years=5, freq=1, ytm_pct=10, price=1000)


# ---------- Black-Scholes ----------
def test_black_scholes_hull_example():
    call = black_scholes(spot=42, strike=40, years=0.5, rate_pct=10, vol_pct=20, kind="call")["result"]
    put = black_scholes(spot=42, strike=40, years=0.5, rate_pct=10, vol_pct=20, kind="put")["result"]
    assert call["price"] == pytest.approx(4.7594, abs=1e-3)
    assert put["price"] == pytest.approx(0.8086, abs=1e-3)
    assert call["delta"] == pytest.approx(0.7791, abs=1e-3)
    # put-call parity: C - P = S - K e^{-rT}
    assert call["price"] - put["price"] == pytest.approx(42 - 40 * math.exp(-0.1 * 0.5), abs=1e-3)  # prices rounded to 4 dp


def test_black_scholes_rejects_bad_kind():
    with pytest.raises(ValueError):
        black_scholes(spot=42, strike=40, years=0.5, rate_pct=10, vol_pct=20, kind="straddle")
