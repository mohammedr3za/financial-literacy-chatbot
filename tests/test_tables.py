"""Year-wise tables returned by the calculators (SPEC 4.2). Cells are formatted strings; we parse them back."""
import pytest

from finbot.tools.calculators import black_scholes, bond_metrics, loan_emi, npv_irr, sip_future_value, tvm


def num(cell: str) -> float:
    s = cell.replace(",", "").replace("₹", "").strip()
    neg = s.startswith("(") and s.endswith(")")
    s = s.strip("()")
    return -float(s) if neg else float(s)


def test_loan_table_is_yearly_and_reconciles():
    out = loan_emi(principal=2_000_000, annual_rate_pct=8.5, months=240)
    t, r = out["table"], out["result"]
    assert t["columns"][:2] == ["Year", "Opening balance"]
    body, total = t["rows"][:-1], t["rows"][-1]
    assert len(body) == 20 and total[0] == "Total"
    assert num(body[0][3]) == pytest.approx(r["first_year_interest"], abs=1)
    assert num(body[-1][5]) == pytest.approx(0, abs=1)                       # loan fully repaid
    assert sum(num(row[3]) for row in body) == pytest.approx(r["total_interest"], abs=20)
    assert num(total[3]) == pytest.approx(r["total_interest"], abs=1)
    assert num(total[4]) == pytest.approx(2_000_000, abs=1)


def test_loan_table_partial_last_year():
    t = loan_emi(principal=100_000, annual_rate_pct=10, months=18)["table"]
    assert len(t["rows"]) == 3  # year 1, year 2 (6 months), total


def test_sip_table_ends_at_corpus():
    out = sip_future_value(monthly=10_000, annual_return_pct=12, years=10, step_up_pct=10)
    body = out["table"]["rows"]
    assert len(body) == 10
    assert num(body[-1][4]) == pytest.approx(out["result"]["corpus"], abs=1)
    assert num(body[1][1]) == pytest.approx(11_000)                          # step-up visible in year 2
    assert num(body[-1][3]) == pytest.approx(out["result"]["invested"], abs=1)


def test_tvm_table_for_fv_and_pmt():
    t = tvm(solve_for="fv", annual_rate_pct=10, years=5, pv=1000)["table"]
    assert num(t["rows"][4][4]) == pytest.approx(1610.51, abs=1) and t["rows"][-1][0] == "Total"
    out = tvm(solve_for="pmt", annual_rate_pct=12, years=15, fv=10_000_000, periods_per_year=12)
    assert num(out["table"]["rows"][14][4]) == pytest.approx(10_000_000, rel=1e-6)


def test_npv_table_sums_to_npv():
    out = npv_irr(cashflows=[-1000, 300, 400, 500], discount_rate_pct=10)
    rows = out["table"]["rows"]
    assert len(rows) == 4
    assert num(rows[-1][5]) == pytest.approx(out["result"]["npv"], abs=0.01)  # cumulative PV = NPV


def test_bond_table_reproduces_price_and_duration():
    out = bond_metrics(face=1000, coupon_pct=10, years=5, freq=1, ytm_pct=10)
    body, total = out["table"]["rows"][:-1], out["table"]["rows"][-1]
    assert len(body) == 5
    assert num(total[4]) == pytest.approx(1000, abs=0.01)
    assert num(total[5]) / num(total[4]) == pytest.approx(4.1699, abs=1e-3)


def test_black_scholes_greeks_table():
    t = black_scholes(spot=42, strike=40, years=0.5, rate_pct=10, vol_pct=20, kind="call")["table"]
    names = [r[0] for r in t["rows"]]
    assert names[:2] == ["Price", "Delta"] and "Gamma" in names


def test_tables_are_capped():
    t = loan_emi(principal=100_000, annual_rate_pct=8, months=840)["table"]  # 70 years
    assert len(t["rows"]) <= 61 and t.get("note")
