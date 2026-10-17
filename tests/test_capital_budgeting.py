"""New tool contracts (SPEC 4, 2026-10-17): plain annuities and withdrawals in tvm, annual-payment amortization in
loan_emi, payback in npv_irr, perpetuity, project_cash_flows. Scenarios are our own; expected values come from closed
forms or plain loops written here, not from the tool code. (The course-set gold values are checked outside the repo.)"""
import json

import pytest

from finbot.graph import ungrounded_values
from finbot.precheck import precheck
from finbot.tools.calculators import loan_emi, npv_irr, perpetuity, project_cash_flows, tvm
from finbot.tools.registry import TOOLS_BY_NAME


def a_factor(r, n):          # present value of 1 a period for n periods
    return (1 - (1 + r) ** -n) / r


def s_factor(r, n):          # future value of 1 a period for n periods
    return ((1 + r) ** n - 1) / r


def npv(r, flows):
    return sum(c / (1 + r) ** t for t, c in enumerate(flows))


# ---------- (a) tvm values plain annuities (ordinary and due) and withdrawals
def test_tvm_present_value_of_an_ordinary_annuity_and_all_four_values():
    out = tvm(solve_for="pv", annual_rate_pct=6, years=5, pmt=9_000)
    r = out["result"]
    assert r["pv"] == pytest.approx(9_000 * a_factor(0.06, 5), abs=0.01)
    a = r["annuity_values"]
    assert a["pv_ordinary"] == pytest.approx(9_000 * a_factor(0.06, 5), abs=0.01)
    assert a["pv_due"] == pytest.approx(9_000 * a_factor(0.06, 5) * 1.06, abs=0.01)
    assert a["fv_ordinary"] == pytest.approx(9_000 * s_factor(0.06, 5), abs=0.01)
    assert a["fv_due"] == pytest.approx(9_000 * s_factor(0.06, 5) * 1.06, abs=0.01)
    rows = out["table"]["rows"]
    assert len(rows) == 6 and float(rows[-1][-1].replace(",", "")) == pytest.approx(r["pv"], abs=0.01)   # 5 payments + total


def test_tvm_annuity_due_and_future_values():
    assert tvm(solve_for="pv", annual_rate_pct=6, years=5, pmt=9_000, when="begin")["result"]["pv"] == \
        pytest.approx(9_000 * a_factor(0.06, 5) * 1.06, abs=0.01)
    assert tvm(solve_for="fv", annual_rate_pct=6, years=5, pmt=9_000)["result"]["fv"] == pytest.approx(9_000 * s_factor(0.06, 5), abs=0.01)
    due = tvm(solve_for="fv", annual_rate_pct=6, years=5, pmt=9_000, when="begin")["result"]
    assert due["fv"] == pytest.approx(9_000 * s_factor(0.06, 5) * 1.06, abs=0.01)
    assert due["annuity_values"]["pv_ordinary"] == pytest.approx(9_000 * a_factor(0.06, 5), abs=0.01)


def test_tvm_annuity_with_explicit_zero_fv_is_not_an_error():
    assert tvm(solve_for="pv", annual_rate_pct=6, years=5, pmt=9_000, fv=0)["result"]["pv"] == \
        pytest.approx(9_000 * a_factor(0.06, 5), abs=0.01)
    out = json.loads(TOOLS_BY_NAME["tvm"].invoke({"solve_for": "pv", "annual_rate_pct": 6, "years": 5, "pmt": 9000}))
    assert "error" not in out and out["result"]["pv"] == pytest.approx(9_000 * a_factor(0.06, 5), abs=0.01)


def test_tvm_savings_semantics_are_unchanged():
    assert tvm(solve_for="pv", annual_rate_pct=10, years=5, fv=1610.51)["result"]["pv"] == pytest.approx(1000, abs=0.01)
    pmt = tvm(solve_for="pmt", annual_rate_pct=12, years=15, fv=10_000_000, periods_per_year=12)["result"]["pmt"]
    assert pmt == pytest.approx(10_000_000 / s_factor(0.01, 180), abs=0.01)


def test_tvm_level_withdrawal_from_a_balance():
    i = 0.05 / 12
    r = tvm(solve_for="pmt", annual_rate_pct=5, years=25, pv=1_500_000, periods_per_year=12)
    assert r["result"]["pmt"] == pytest.approx(1_500_000 / a_factor(i, 300), abs=0.01)
    assert r["table"]["rows"][-2][-1] in ("0", "0.00")                      # the balance is used up


# ---------- (b) loan_emi with annual payments: exact unrounded schedule
def _schedule(principal, r, n):
    pay = principal * r / (1 - (1 + r) ** -n)
    rows, b = [], principal
    for k in range(n):
        i = b * r
        rows.append([str(k + 1), f"{b:,.2f}", f"{pay:,.2f}", f"{i:,.2f}", f"{pay - i:,.2f}", f"{max(b - (pay - i), 0):,.2f}"])
        b -= pay - i
    return pay, rows


def test_annual_payment_amortization_schedule():
    pay, expected = _schedule(45_000, 0.07, 4)
    out = loan_emi(principal=45_000, annual_rate_pct=7, years=4, payments_per_year=1)
    r, rows = out["result"], out["table"]["rows"]
    assert r["emi"] == pytest.approx(pay, abs=0.005)
    assert rows[:4] == expected and len(rows) == 5                          # 4 payments + total
    total_interest = sum(float(row[3].replace(",", "")) for row in expected)
    assert r["total_interest"] == pytest.approx(total_interest, abs=0.02)
    assert r["interest_by_period"][2] == pytest.approx(float(expected[2][3].replace(",", "")), abs=0.005)


def test_annual_payments_via_months_and_rate_solve():
    pay, _ = _schedule(45_000, 0.07, 4)
    assert loan_emi(principal=45_000, annual_rate_pct=7, months=48, payments_per_year=1)["result"]["emi"] == pytest.approx(pay, abs=0.005)
    rate = loan_emi(principal=45_000, years=4, emi=round(pay, 2), payments_per_year=1)["result"]["annual_rate_pct"]
    assert rate == pytest.approx(7, abs=1e-3)
    q = loan_emi(principal=100_000, annual_rate_pct=8, years=2, payments_per_year=4)
    assert q["result"]["emi"] == pytest.approx(100_000 * 0.02 / (1 - 1.02 ** -8), abs=0.01) and len(q["table"]["rows"]) == 9
    with pytest.raises(ValueError, match="payments_per_year"):
        loan_emi(principal=45_000, annual_rate_pct=7, years=4, payments_per_year=3)


def test_monthly_loan_is_unchanged_and_tool_takes_new_args():
    assert loan_emi(principal=2_000_000, annual_rate_pct=8.5, months=240)["result"]["emi"] == pytest.approx(17356.46, abs=0.01)
    assert loan_emi(principal=2_000_000, annual_rate_pct=8.5, years=20)["result"]["emi"] == pytest.approx(17356.46, abs=0.01)
    pay, _ = _schedule(45_000, 0.07, 4)
    out = json.loads(TOOLS_BY_NAME["loan_emi"].invoke({"principal": 45000, "annual_rate_pct": 7, "years": 4, "payments_per_year": 1}))
    assert out["result"]["emi"] == pytest.approx(pay, abs=0.005)


# ---------- (c) payback and discounted payback in npv_irr
def test_payback_periods():
    assert npv_irr([-30_000, 9_000, 12_000, 10_000, 8_000])["result"]["payback_years"] == pytest.approx(2 + 9_000 / 10_000)
    assert npv_irr([-50_000, 5_000, 10_000, 15_000, 100_000])["result"]["payback_years"] == pytest.approx(3 + 20_000 / 100_000)


def test_discounted_payback_and_npv():
    flows = [-30_000, 9_000, 12_000, 10_000, 8_000]
    pv = [c / 1.1 ** t for t, c in enumerate(flows)]
    cum, expected = pv[0], None
    for t in range(1, len(pv)):
        if cum + pv[t] >= 0:
            expected = t - 1 + (-cum) / pv[t]
            break
        cum += pv[t]
    r = npv_irr(flows, 10)["result"]
    assert r["discounted_payback_years"] == pytest.approx(expected, abs=1e-6) and r["npv"] == pytest.approx(npv(0.10, flows), abs=1e-3)


def test_payback_never_reached():
    r = npv_irr([-1000, 100, 100], 10)["result"]
    assert r["payback_years"] is None and r["discounted_payback_years"] is None and "never" in r["note"].lower()


# ---------- (d) perpetuity
def test_growing_perpetuity_npv_and_break_even_growth():
    r = perpetuity(cash_flow=60_000, rate_pct=9, growth_pct=3, investment=950_000)["result"]
    assert r["pv"] == pytest.approx(1_000_000) and r["npv"] == pytest.approx(50_000)
    assert r["break_even_growth_pct"] == pytest.approx((0.09 - 60_000 / 950_000) * 100, abs=1e-6)
    assert r["irr_pct"] == pytest.approx((60_000 / 950_000 + 0.03) * 100, abs=1e-6)


def test_level_and_deferred_perpetuity_and_bad_rates():
    assert perpetuity(cash_flow=500, rate_pct=5)["result"]["pv"] == pytest.approx(10_000)
    assert perpetuity(cash_flow=500, rate_pct=5, starts_in_year=3)["result"]["pv"] == pytest.approx(10_000 / 1.05 ** 2)
    with pytest.raises(ValueError, match="greater than"):
        perpetuity(cash_flow=100, rate_pct=4, growth_pct=5)


# ---------- (e) project_cash_flows
EXPANSION = dict(years=3, tax_rate_pct=25, rate_pct=12, investment=900_000, depreciation="straight_line", sales=700_000,
                 costs=250_000, nwc_initial=60_000, salvage=90_000)


def test_project_with_straight_line_nwc_and_salvage():
    ocf = (700_000 - 250_000 - 300_000) * 0.75 + 300_000
    flows = [-960_000, ocf, ocf, ocf + 60_000 + 90_000 * 0.75]
    r = project_cash_flows(**EXPANSION)["result"]
    assert r["cash_flows"] == pytest.approx(flows) and r["npv"] == pytest.approx(npv(0.12, flows), abs=0.01)
    assert r["after_tax_salvage"] == pytest.approx(67_500) and r["irr_pct"] > 12


def test_cost_only_project():
    ocf = -80_000 * 0.79 + 0.21 * 60_000
    r = project_cash_flows(years=4, tax_rate_pct=21, rate_pct=8, investment=240_000, depreciation="straight_line", costs=80_000)
    assert r["result"]["npv"] == pytest.approx(-240_000 + ocf * a_factor(0.08, 4), abs=0.01)


BID = dict(years=4, tax_rate_pct=25, rate_pct=10, investment=1_200_000, depreciation="straight_line", units=2_000_000,
           variable_cost_per_unit=0.004, fixed_costs=250_000, nwc_initial=150_000, nwc_annual_addition=20_000,
           salvage=100_000, opportunity_cost_now=300_000, opportunity_value_end=320_000, solve_bid_price=True)


def bid_price_by_hand():
    # OCF = (2,000,000 P − 8,000 − 250,000 − 300,000) × 0.75 + 300,000 = 1,500,000 P − 118,500
    a4, a3, d4 = a_factor(0.10, 4), a_factor(0.10, 3), 1.1 ** -4
    end = 150_000 + 3 * 20_000 + 100_000 * 0.75 + 320_000                    # NWC back, after-tax salvage, land
    return (1_650_000 + 118_500 * a4 + 20_000 * a3 - end * d4) / (1_500_000 * a4)


def test_bid_price_makes_npv_zero():
    r = project_cash_flows(**BID)["result"]
    assert r["bid_price"] == pytest.approx(bid_price_by_hand(), abs=1e-7) and abs(r["npv"]) < 0.01


def test_macrs_land_and_nwc_project():
    rates, units = [33.33, 44.45, 14.81, 7.41], [1_000, 1_200, 1_100, 900]
    flows = [-1_000_000 - 50_000 - 200_000]
    for y in range(4):
        sales, dep = units[y] * 600, 1_000_000 * rates[y] / 100
        ebit = sales - 0.3 * sales - 120_000 - dep
        flows.append(ebit * 0.79 + dep)
    flows[4] += 50_000 + 80_000 * 0.79 + 220_000
    out = project_cash_flows(years=4, tax_rate_pct=21, rate_pct=10, investment=1_000_000, depreciation="macrs",
                             macrs_rates_pct=rates, units=units, price_per_unit=600, variable_cost_pct=30,
                             fixed_costs=120_000, nwc_initial=50_000, salvage=80_000, opportunity_cost_now=200_000,
                             opportunity_value_end=220_000)
    assert out["result"]["npv"] == pytest.approx(npv(0.10, flows), abs=0.01)
    assert out["result"]["book_value_at_end"] == pytest.approx(0, abs=0.01) and len(out["table"]["rows"]) == 5


def test_after_tax_salvage_uses_book_value_and_errors_are_explained():
    r = project_cash_flows(years=3, tax_rate_pct=20, rate_pct=10, investment=1000, depreciation="straight_line",
                           depreciation_years=5, ocf=300, salvage=500)["result"]
    assert r["book_value_at_end"] == pytest.approx(400) and r["after_tax_salvage"] == pytest.approx(480)
    with pytest.raises(ValueError, match="macrs_rates_pct"):
        project_cash_flows(years=3, tax_rate_pct=20, rate_pct=10, investment=1000, depreciation="macrs", sales=500)
    with pytest.raises(ValueError, match="3 values"):
        project_cash_flows(years=3, tax_rate_pct=20, rate_pct=10, units=[1, 2], price_per_unit=5)
    with pytest.raises(ValueError, match="units"):
        project_cash_flows(years=3, tax_rate_pct=20, rate_pct=10, sales=100, solve_bid_price=True)


def test_project_tool_is_registered_and_strict():
    out = json.loads(TOOLS_BY_NAME["project_cash_flows"].invoke(EXPANSION))
    assert out["result"]["npv"] == pytest.approx(project_cash_flows(**EXPANSION)["result"]["npv"])
    out = json.loads(TOOLS_BY_NAME["perpetuity"].invoke({"cash_flow": 60000, "rate_pct": 9, "growth_pct": 3, "investment": 950000}))
    assert out["result"]["npv"] == pytest.approx(50_000)
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        TOOLS_BY_NAME["project_cash_flows"].invoke({**EXPANSION, "marketing_study": 120000})


# ---------- grounding: number words, millions and cents
PROJECT_Q = ("Marlow Tools is weighing a four-year expansion that needs $1.2 million of equipment, depreciated straight-line "
             "to zero over four years. Sales will be $900,000 a year and costs $310,000. It needs $80,000 of net working "
             "capital at the start, and the equipment can be sold for $150,000 at the end. With a 21 percent tax rate and a "
             "12 percent required return, what are the yearly cash flows and the project's NPV?")
PROJECT_ARGS = dict(years=4, tax_rate_pct=21, rate_pct=12, investment=1_200_000, depreciation="straight_line", sales=900_000,
                    costs=310_000, nwc_initial=80_000, salvage=150_000)


def test_project_grounding_accepts_number_words_and_millions_but_not_invented_figures():
    assert ungrounded_values(PROJECT_ARGS, PROJECT_Q, "project_cash_flows") == []
    assert ungrounded_values({**PROJECT_ARGS, "ocf": 529_100}, PROJECT_Q, "project_cash_flows") == ["ocf=529100"]
    cents_q = "Production costs are .40 cents per unit, and fixed costs are $250,000 per year."
    assert ungrounded_values({"variable_cost_per_unit": 0.004, "fixed_costs": 250000}, cents_q, "project_cash_flows") == []


# ---------- routing
COST_Q = ("Two packaging machines: Machine X costs $210,000, lasts three years and has pretax operating costs of $64,000 a "
          "year; Machine Y costs $290,000, lasts five years with operating costs of $58,000. Both are depreciated "
          "straight-line to zero with no salvage. The tax rate is 21 percent and the discount rate 8 percent. What is each "
          "machine's NPV?")
PERP_Q = ("A water utility contract pays $60,000 at the end of the first year, growing 3 percent a year forever. It costs "
          "$950,000 today. At a 9 percent required return, what is its NPV, and what growth rate makes the NPV zero?")
ANNUITY_Q = ("You will receive $9,000 a year for five years and the interest rate is 6 percent. What is the present value "
             "as an ordinary annuity? As an annuity due? What are the matching amounts at the end of year five?")
AMORT_Q = ("Draw up an amortization schedule for a four-year loan of $45,000 at 7 percent a year with equal annual "
           "payments. How much interest is paid in year three?")


@pytest.mark.parametrize("q,kinds", [
    (PROJECT_Q, ["project"]), (COST_Q, ["project"]), (PERP_Q, ["perpetuity"]), (ANNUITY_Q, ["annuity"]),
    (AMORT_Q, ["loan"]),
    ("A project costs -$30,000 and returns 9,000, 12,000, 10,000 and 8,000. What is the payback period?", ["npv_irr"]),
])
def test_capital_budgeting_routes(q, kinds):
    f = precheck(q).flags
    assert f["calc_types"] == kinds and not f["statement_data"]


def test_project_and_annuity_hints():
    from finbot.routing import HINTS, REQUIRED_TOOLS
    assert REQUIRED_TOOLS["project"] == {"project_cash_flows"} and REQUIRED_TOOLS["perpetuity"] == {"perpetuity", "calculate"}   # two-stage dividend models need calculate
    assert "sunk" in HINTS["project"] and "solve_bid_price" in HINTS["project"]
    assert "annuity_values" in HINTS["annuity"] and "payments_per_year" in HINTS["loan"]


def test_answer_format_notes_do_not_make_a_question_a_statement():
    # several lettered parts, each followed by a formatting note that carries numbers (4+ numeric lines)
    q = ("Suppose you will receive $5,000 per year for 4 years and the rate is 6 percent.\n\na-1. What is the present value "
         "as an ordinary annuity?\nNote: Give 2 decimals (for example 45.67).\na-2. What is it as an annuity "
         "due?\nNote: Give 2 decimals (for example 45.67).\nb. What is the future value?\nNote: Give "
         "2 decimals (for example 45.67).")
    f = precheck(q).flags
    assert f["calc_types"] == ["annuity"] and not f["statement_data"]
    loan = ("Set out the repayment table for a $24,000 loan over three years at 9 percent a year, repaid with equal annual "
            "payments.\nNote: Give 2 decimals (for example 45.67).\nb. Interest in year 2?\n"
            "Note: Give 2 decimals (for example 45.67).\nc. Total interest?\nNote: Give 2 decimals (for example 45.67).")
    assert precheck(loan).flags["calc_types"] == ["loan"]
    pasted = "Balance sheet\n| Item | Amount |\n| Cash | 1,20,000 |\n| Inventory | 2,50,000 |\n| Current liabilities | 1,80,000 |"
    assert precheck(pasted + "\nCompute the current ratio").flags["statement_data"]
