"""LangChain tool wrappers around the deterministic calculators.

Each tool returns a JSON string. Bad input returns {"error": ...} so the model can
correct itself instead of the graph crashing.
"""
import json
from typing import Literal

from langchain_core.tools import tool
from pydantic import BaseModel, ConfigDict, Field, create_model, model_validator

from . import calculators as C
from . import financial_solver as F
from . import capital_gains as CG
from . import income_tax as T
from . import tax_rules as TR
from . import statements as S
from .math_engine import calculate as _calculate
from .math_engine import solve_equation as _solve_equation

AMOUNT_HINT = "Plain rupees as a number: 1 lakh = 100000, 1 crore = 10000000."


SHOWN_NOTE = ("This table is displayed to the user automatically below your answer. "
              "Do not repeat its rows or write your own table of these figures; give a short interpretation.")


def _run(fn, **kwargs) -> str:
    try:
        out = fn(**kwargs)
        if isinstance(out, dict) and out.get("table"):
            out = {**out, "table_shown_to_user": SHOWN_NOTE}
        return json.dumps(out, ensure_ascii=False)
    except (ValueError, TypeError) as exc:
        return json.dumps({"error": str(exc)}, ensure_ascii=False)


@tool(parse_docstring=True)
def sip_future_value(monthly: float, annual_return_pct: float, years: float, step_up_pct: float = 0,
                     timing: Literal["start", "end"] = "start") -> str:
    """Future value of a KNOWN monthly SIP amount (systematic investment plan). Returns corpus, amount invested and gain. To find the monthly amount NEEDED for a target, use goal_saving instead.

    Args:
        monthly: Monthly investment in plain rupees, e.g. 10000.
        annual_return_pct: Assumed yearly return in percent, e.g. 12. This is an assumption, not a guarantee.
        years: Investment period in years.
        step_up_pct: Optional yearly increase in the monthly amount, in percent. Default 0.
        timing: "start" (a SIP: deposit at the start of each month, default) or "end" (deposit at the end of each month, an ordinary annuity).
    """
    return _run(C.sip_future_value, monthly=monthly, annual_return_pct=annual_return_pct, years=years, step_up_pct=step_up_pct,
                timing=timing)


@tool(parse_docstring=True)
def tvm(solve_for: str, annual_rate_pct: float | None = None, years: float | None = None, pv: float = 0.0,
        pmt: float = 0.0, fv: float | None = None, periods_per_year: int = 1, when: str = "end") -> str:
    """Time value of money for savings and growth: future value, present value, required deposit, years, or CAGR; also plain annuities and level withdrawals.

    All amounts are positive plain rupees. Leave out the value you want to solve for.
    Examples: CAGR from 100 to 200 in 5 years -> solve_for="annual_rate_pct", years=5, pv=100, fv=200.
    Monthly saving to reach 1 crore in 15 years at 12% -> solve_for="pmt", annual_rate_pct=12, years=15, fv=10000000, periods_per_year=12.
    Annuity: PV of 9000 a year for 5 years at 6% -> solve_for="pv", pmt=9000, years=5, annual_rate_pct=6 (no fv); the result's annuity_values has PV and FV, ordinary and due.
    Withdrawal: monthly amount that uses up 2500000 over 30 years at 6% -> solve_for="pmt", pv=2500000, years=30, annual_rate_pct=6, periods_per_year=12 (no fv).

    Args:
        solve_for: One of "fv", "pv", "pmt", "years", "annual_rate_pct".
        annual_rate_pct: Yearly rate in percent, e.g. 8 for 8%.
        years: Number of years.
        pv: Starting lump sum in plain rupees.
        pmt: Regular deposit (or annuity payment) per period in plain rupees.
        fv: Target or final amount in plain rupees; leave out for an annuity or a withdrawal.
        periods_per_year: Compounding and deposit periods per year: 1 yearly, 4 quarterly, 12 monthly.
        when: "end" if payments are at the end of each period (ordinary annuity), "begin" if at the start (annuity due).
    """
    return _run(C.tvm, solve_for=solve_for, annual_rate_pct=annual_rate_pct, years=years, pv=pv, pmt=pmt, fv=fv,
                periods_per_year=periods_per_year, when=when)


@tool(parse_docstring=True)
def npv_irr(cashflows: list[float], discount_rate_pct: float | None = None) -> str:
    """Net present value and internal rate of return of a series of yearly cash flows.

    Args:
        cashflows: Cash flows in plain rupees starting at year 0, e.g. [-1000, 300, 400, 500]. Investments are negative.
        discount_rate_pct: Optional discount rate in percent for NPV, e.g. 10.
    """
    return _run(C.npv_irr, cashflows=cashflows, discount_rate_pct=discount_rate_pct)


@tool(parse_docstring=True)
def equivalent_annual_cost(purchase_cost: float, rate_pct: float, annual_cost: float | None = None,
                           life_years: float | None = None, salvage: float = 0.0, power_watts: float | None = None,
                           hours_per_year: float | None = None, price_per_kwh: float | None = None,
                           lifetime_hours: float | None = None) -> str:
    """Equivalent annual cost (EAC, equivalent annual annuity) of something bought now and run at a level yearly cost, e.g. a machine or an electric pump. Call it once per option. Pass costs as positive numbers; the tool returns the yearly cost, the life, the PV of costs, the annuity factor and the EAC (negative: a cost), with a year-by-year table. For electricity, pass power_watts, hours_per_year, price_per_kwh and lifetime_hours exactly as the question gives them instead of annual_cost and life_years; the tool works them out. Never write out the yearly cash flows yourself.

    Args:
        purchase_cost: Price paid today, e.g. 420.00.
        rate_pct: Required return or discount rate in percent, e.g. 10.
        annual_cost: Running cost per year, if the question states it. Leave out when you give the electricity figures.
        life_years: Life in whole years, if the question states it. Leave out when you give lifetime_hours.
        salvage: Resale value at the end of the life, only if the question gives one. Default 0.
        power_watts: Electricity: the power in watts, e.g. 750 for a 750-watt pump.
        hours_per_year: Hours used per year, e.g. 2000.
        price_per_kwh: Price of one kilowatt-hour, e.g. 0.15.
        lifetime_hours: Total life in hours, e.g. 16000; the life in years is lifetime_hours ÷ hours_per_year.
    """
    return _run(C.equivalent_annual_cost, purchase_cost=purchase_cost, annual_cost=annual_cost, life_years=life_years,
                rate_pct=rate_pct, salvage=salvage, power_watts=power_watts, hours_per_year=hours_per_year,
                price_per_kwh=price_per_kwh, lifetime_hours=lifetime_hours)


@tool(parse_docstring=True)
def perpetuity(cash_flow: float, rate_pct: float, growth_pct: float = 0.0, investment: float | None = None,
               starts_in_year: int = 1) -> str:
    """Present value of a perpetuity (cash flows forever), level or growing; with an investment also its NPV, IRR and the break-even growth rate.

    Args:
        cash_flow: The first yearly cash flow, received at the end of starts_in_year, e.g. 60000.
        rate_pct: Required return or discount rate in percent, e.g. 9.
        growth_pct: Constant yearly growth of the cash flow in percent, e.g. 3. Default 0 (a level perpetuity).
        investment: Initial investment today, if the question gives one; gives NPV, IRR and the break-even growth rate.
        starts_in_year: Year of the first cash flow. Default 1 (end of the first year).
    """
    return _run(C.perpetuity, cash_flow=cash_flow, rate_pct=rate_pct, growth_pct=growth_pct, investment=investment,
                starts_in_year=starts_in_year)


_PER_YEAR = float | list[float] | None


class ProjectArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    years: int = Field(description="Project life in years, e.g. 5.")
    tax_rate_pct: float = Field(description="Tax rate in percent, e.g. 30.")
    rate_pct: float | None = Field(None, description="Required return or discount rate in percent, for NPV, e.g. 12.")
    sales: _PER_YEAR = Field(None, description="Yearly sales: one number, or a list with one value per year.")
    units: _PER_YEAR = Field(None, description="Units sold per year: one number or a list per year (use with price_per_unit).")
    price_per_unit: float | None = Field(None, description="Selling price per unit. Leave out when solving for the bid price.")
    costs: _PER_YEAR = Field(None, description="Yearly cash operating costs (pretax), not depreciation.")
    fixed_costs: _PER_YEAR = Field(None, description="Fixed costs per year.")
    variable_cost_per_unit: _PER_YEAR = Field(None, description="Variable cost per unit in dollars (or rupees), e.g. 12.5.")
    variable_cost_cents_per_unit: _PER_YEAR = Field(None, description="Variable cost per unit when the question gives it in cents, as written (.40 cents -> 0.4); the tool divides by 100.")
    variable_cost_pct: float | None = Field(None, description="Variable costs as a percent of sales, e.g. 40.")
    ocf: _PER_YEAR = Field(None, description="Operating cash flow per year, only if the question gives it directly.")
    investment: float = Field(0.0, description="Fixed-asset cost at year 0, e.g. 1800000.")
    depreciation: Literal["straight_line", "macrs", "none"] = Field("straight_line", description="Depreciation method.")
    depreciation_years: int | None = Field(None, description="Straight-line tax life if it differs from years.")
    macrs_rates_pct: list[float] | None = Field(None, description="MACRS rates in percent from the question, e.g. [33.33, 44.45, 14.81, 7.41].")
    salvage: float = Field(0.0, description="Pretax market (salvage) value of the asset at the end; the tool taxes it.")
    nwc_initial: float = Field(0.0, description="Net working capital needed at year 0; recovered at the end.")
    nwc_annual_addition: float = Field(0.0, description="Extra NWC invested in each later year (years 1 to n-1); recovered at the end.")
    opportunity_cost_now: float = Field(0.0, description="After-tax value today of an asset the firm owns and will use (e.g. land it could sell).")
    opportunity_value_end: float = Field(0.0, description="After-tax value of that asset at the end of the project.")
    solve_bid_price: bool = Field(False, description="True to find the price per unit that makes NPV = 0 (needs units).")


@tool(args_schema=ProjectArgs)
def project_cash_flows(**kwargs) -> str:
    """Capital-budgeting project cash flows from year 0 to the end: OCF = EBIT + depreciation − taxes, straight-line or MACRS depreciation, after-tax salvage = salvage − t × (salvage − book value), NWC invested and recovered, opportunity costs; returns yearly cash flows, NPV, IRR and a table, or the bid price per unit that makes NPV = 0. One call per project. Pass every figure from the question; ignore sunk costs (money already spent, such as a past study or the original price of land)."""
    return _run(C.project_cash_flows, **kwargs)


@tool(parse_docstring=True)
def bond_metrics(face: float, coupon_pct: float, years: float, freq: int = 1,
                 ytm_pct: float | None = None, price: float | None = None) -> str:
    """Bond price or yield to maturity, plus current yield, Macaulay duration and modified duration.

    Give exactly one of ytm_pct or price.

    Args:
        face: Face value, e.g. 1000.
        coupon_pct: Yearly coupon rate in percent, e.g. 7.5.
        years: Years to maturity.
        freq: Coupons per year: 1, 2, 4 or 12.
        ytm_pct: Yield to maturity in percent, to compute the price.
        price: Market price, to compute the yield to maturity.
    """
    return _run(C.bond_metrics, face=face, coupon_pct=coupon_pct, years=years, freq=freq, ytm_pct=ytm_pct, price=price)


@tool(parse_docstring=True)
def black_scholes(spot: float, strike: float, years: float, rate_pct: float, vol_pct: float, kind: str = "call") -> str:
    """Black-Scholes price and Greeks of a European call or put option.

    Args:
        spot: Current price of the underlying.
        strike: Strike price.
        years: Time to expiry in years, e.g. 3 months = 0.25.
        rate_pct: Risk-free rate in percent, e.g. 7.
        vol_pct: Annual volatility in percent, e.g. 20.
        kind: "call" or "put".
    """
    return _run(C.black_scholes, spot=spot, strike=strike, years=years, rate_pct=rate_pct, vol_pct=vol_pct, kind=kind)


@tool(parse_docstring=True)
def calculate(expression: str) -> str:
    """Evaluate any other arithmetic, e.g. "245000*0.15*1.18" or "(1+0.08)^5". Use for percentages, GST, ratios and quick sums.

    Args:
        expression: Arithmetic with plain numbers (no commas), + - * / ^ % ( ), and sqrt, ln, log10, exp, abs, round, min, max.
    """
    return _run(_calculate, expression=expression)


@tool(parse_docstring=True)
def solve_equation(equation: str, solve_for: str) -> str:
    """Solve an algebraic equation for one unknown, e.g. "1000*(1+r)^5 = 2000" for r.

    Args:
        equation: The equation using numbers, variable names, + - * / ^ ( ) and one = sign.
        solve_for: The variable to solve for, e.g. "r".
    """
    return _run(_solve_equation, equation=equation, solve_for=solve_for)


@tool(parse_docstring=True)
def goal_saving(target: float, years: float, annual_return_pct: float, current_savings: float = 0.0,
                step_up_pct: float = 0.0, timing: str = "start") -> str:
    """Monthly SIP or saving NEEDED to reach a target amount (a goal) in a given number of years. Use this for "how much should I save or invest each month to reach X". Also returns the lump sum needed today instead.

    Args:
        target: The FUTURE amount to reach, in plain rupees (1 crore = 10000000). Never the monthly amount.
        years: Years until the goal.
        annual_return_pct: Assumed yearly return in percent, e.g. 12.
        current_savings: Money already saved toward the goal today, if the user says so. Default 0.
        step_up_pct: Yearly increase in the monthly amount in percent, if the user says so. Default 0.
        timing: "start" for a SIP (default), or "end" if the question says the saving happens at the end of each month.
    """
    return _run(C.goal_saving, target=target, years=years, annual_return_pct=annual_return_pct,
                current_savings=current_savings, step_up_pct=step_up_pct, timing=timing)


@tool(parse_docstring=True)
def real_return(nominal_pct: float, inflation_pct: float) -> str:
    """Real (inflation-adjusted) return using the exact Fisher formula, not nominal minus inflation.

    Args:
        nominal_pct: The stated return in percent, e.g. 8 for an FD paying 8%.
        inflation_pct: Inflation in percent, e.g. 6.
    """
    return _run(C.real_return, nominal_pct=nominal_pct, inflation_pct=inflation_pct)


@tool(parse_docstring=True)
def effective_annual_rate(nominal_pct: float, periods_per_year: int) -> str:
    """Effective annual rate (EAR) of a nominal rate compounded several times a year.

    Args:
        nominal_pct: The nominal yearly rate in percent, e.g. 12.
        periods_per_year: Compounding periods per year: 12 monthly, 4 quarterly, 2 half-yearly, 365 daily.
    """
    return _run(C.effective_annual_rate, nominal_pct=nominal_pct, periods_per_year=periods_per_year)


class _Strict(BaseModel):
    """Unknown argument names raise an error instead of being silently dropped."""
    model_config = ConfigDict(extra="forbid")


_ITEMS = (float | None)


class LoanArgs(_Strict):
    principal: float | None = Field(None, description="Loan amount in plain rupees, e.g. 20 lakh = 2000000. Leave out if it is the unknown.")
    annual_rate_pct: float | None = Field(None, description=(
        "Yearly interest rate (APR) in percent, e.g. 8.5. Leave out if it is the unknown. Never pass 0 unless the loan is interest-free."))
    months: int | None = Field(None, description="Number of monthly payments, e.g. 20 years = 240. Leave out if it is the unknown.")
    emi: float | None = Field(None, description="Payment per period (the monthly EMI, or the annual payment). Leave out if it is the unknown.")
    payments_per_year: Literal[1, 2, 4, 12] = Field(12, description="Payments per year: 12 monthly (default), 4 quarterly, 2 half-yearly, 1 annual.")
    years: float | None = Field(None, description="Loan term in years, instead of months (e.g. a five-year loan with annual payments).")


@tool(args_schema=LoanArgs)
def loan_emi(**kwargs) -> str:
    """Loan with level payments (home, car, personal loan, mortgage). Give exactly three of principal, annual_rate_pct, the term (months, or years) and emi (the payment per period); it solves the missing one: the payment, the interest rate the payments imply (APR = rate per period × payments_per_year, plus the effective annual rate), the term, or the amount that can be borrowed. Monthly by default; payments_per_year=1 for annual payments. Also returns total interest and the amortization schedule. Leave the unknown out; never pass 0 for it."""
    return _run(C.loan_emi, **kwargs)


class RatioArgs(_Strict):
    current_assets: _ITEMS = Field(None, description="Total current assets.")
    current_liabilities: _ITEMS = Field(None, description="Total current liabilities.")
    inventory: _ITEMS = Field(None, description="Inventory or stock.")
    cash: _ITEMS = Field(None, description="Cash and cash equivalents.")
    receivables: _ITEMS = Field(None, description="Trade receivables or debtors.")
    payables: _ITEMS = Field(None, description="Trade payables or creditors.")
    total_assets: _ITEMS = Field(None, description="Total assets.")
    total_liabilities: _ITEMS = Field(None, description="Total liabilities (current plus non-current).")
    total_equity: _ITEMS = Field(None, description="Shareholders' equity or net worth.")
    total_debt: _ITEMS = Field(None, description="Borrowings (short-term plus long-term debt).")
    revenue: _ITEMS = Field(None, description="Revenue from operations or sales.")
    cogs: _ITEMS = Field(None, description="Cost of goods sold.")
    gross_profit: _ITEMS = Field(None, description="Gross profit, if COGS is not given.")
    ebit: _ITEMS = Field(None, description="Operating profit, or earnings before interest and tax.")
    interest_expense: _ITEMS = Field(None, description="Finance costs or interest expense.")
    net_income: _ITEMS = Field(None, description="Net profit after tax.")
    operating_cash_flow: _ITEMS = Field(None, description="Net cash from operating activities.")
    capex: _ITEMS = Field(None, description="Capital expenditure (purchase of fixed assets), as a positive number.")
    depreciation: _ITEMS = Field(None, description="Depreciation and amortisation.")


@tool(args_schema=RatioArgs)
def financial_ratios(**items) -> str:
    """Financial ratios from balance sheet and profit-and-loss items: liquidity, leverage, profitability, efficiency, cash flow and DuPont, each with its formula. Pass every matching item that appears in the user's data, not only the ones the question mentions (for example inventory, cash, receivables and payables too), exactly as given and in the same units (do not convert lakh or crore)."""
    return _run(S.financial_ratios, **{k: v for k, v in items.items() if v is not None})


class BalanceSheetArgs(_Strict):
    assets: dict[str, float] = Field(description='Asset items and amounts, e.g. {"Cash": 50000, "Inventory": 40000}.')
    liabilities: dict[str, float] = Field(description='Liability items and amounts, e.g. {"Creditors": 30000, "Bank loan": 90000}.')
    equity: dict[str, float] = Field(description='Equity items and amounts, e.g. {"Share capital": 100000, "Reserves": 50000}.')


@tool(args_schema=BalanceSheetArgs)
def balance_sheet_check(assets: dict, liabilities: dict, equity: dict) -> str:
    """Check whether a balance sheet balances, with section totals and a common-size view (% of total assets). If the question asks for total equity (or total liabilities), pass that section as {}: it is derived as the balancing figure."""
    return _run(S.balance_sheet_check, assets=assets, liabilities=liabilities, equity=equity)


_WC = dict[str, float | list[float]]


class CashFlowArgs(_Strict):
    net_income: float = Field(description="Net profit for the year.")
    depreciation: float = Field(0, description="Depreciation and amortisation for the year (added back).")
    other_non_cash: dict[str, float] | None = Field(None, description=(
        'Other non-cash adjustments that the problem states, e.g. {"Loss on sale of machine": 2000, "Profit on sale of land": -5000}. '
        "Do not invent a profit or loss on sale."))
    current_asset_changes: _WC | None = Field(None, description=(
        'Current assets such as debtors, stock, prepaid expenses: the change (increase positive, decrease negative) '
        'or [opening, closing], e.g. {"Debtors": [40000, 55000], "Stock": -8000}.'))
    current_liability_changes: _WC | None = Field(None, description=(
        'Current liabilities such as creditors, outstanding expenses: the change or [opening, closing], e.g. {"Creditors": 6000}.'))
    investing: _WC | None = Field(None, description=(
        'Investing cash flows (purchases negative, sale proceeds positive) or an asset\'s [opening, closing] balances, e.g. '
        '{"Purchase of machinery": -60000, "Sale of furniture": 5000, "Long-term investments": [20000, 26000]}.'))
    financing: _WC | None = Field(None, description=(
        'Financing cash flows (loans, share issues +; repayments, dividends -) or [opening, closing] debt or share-capital '
        'balances, e.g. {"Dividend paid": -20000, "Long-term debt": [50000, 70000], "Common stock": [100000, 110000]}.'))
    net_fixed_assets: list[float] | None = Field(None, min_length=2, max_length=2, description=(
        "[opening, closing] net fixed assets; the tool adds capital spending (closing - opening + depreciation), so do not "
        "also list it under investing."))
    opening_cash: float | None = Field(None, description="Cash at the start of the year.")
    closing_cash: float | None = Field(None, description="Cash at the end of the year, if given, to check the statement reconciles.")


@tool(args_schema=CashFlowArgs)
def cash_flow_statement(**kwargs) -> str:
    """Prepare a cash flow statement by the indirect method (CFO, CFI, CFF, net change, closing-cash reconciliation). Include only items the problem states. Sale proceeds of an asset go under investing."""
    return _run(S.cash_flow_statement, **kwargs)


class _SolverBase(_Strict):
    """Explicit optional fields (the JSON schema lists every name for the model); unknown names get suggestions."""

    @model_validator(mode="before")
    @classmethod
    def _names(cls, data):
        if not isinstance(data, dict):
            return data
        targets = data.get("targets")
        data = F.canonical({k: v for k, v in data.items() if v is not None and k != "targets"})   # interest_expense -> interest
        F.check_names(list(data))
        if isinstance(targets, str):
            targets = [targets]
        if isinstance(targets, list):
            targets = list(dict.fromkeys(F.resolve(t) for t in targets))
            F.check_names(targets, "target")
        return {**data, "targets": targets} if targets is not None else data


SolveFinancialsArgs = create_model(
    "SolveFinancialsArgs", __base__=_SolverBase,
    targets=(list[Literal[tuple(F.VARIABLES)]], Field(min_length=1, description=(
        "The item(s) the question asks for, by the names above, e.g. ['cogs'] or ['ebit', 'net_income']."))),
    **{name: (float, Field(None, description=desc)) for name, (_, _, desc) in F.VARIABLES.items()})


@tool(args_schema=SolveFinancialsArgs)
def solve_financials(targets: list[str], **items) -> str:
    """Solve textbook corporate-finance problems from statement figures and ratios: income statement (EBIT, taxes, net income), liquidity, turnover and days' sales, leverage and DuPont, profit margin, ROA, ROE, payout, retention, sustainable or internal growth, EPS, P/E, market-to-book and EV multiples. Pass EVERY figure the question gives under its name, and the item(s) asked for in targets; the tool applies the standard formulas and returns each step. Never compute an input yourself; for statements covering several years, call it once per year. Percent items (*_pct) take the percent number (23 percent -> 23); ratios and turnovers are plain numbers (.75 -> 0.75). Example: "current liabilities $250,000, quick ratio .75, inventory turnover 4, current ratio 1.5; what is COGS?" -> current_liabilities=250000, quick_ratio=0.75, inventory_turnover=4, current_ratio=1.5, targets=["cogs"]."""
    return _run(F.solve_financials, targets=targets, **{k: v for k, v in items.items() if v is not None})


@tool(parse_docstring=True)
def income_tax(financial_year: str = "2026-27", regime: Literal["both", "new", "old"] = "both",
               gross_salary: float = 0, gross_salary_monthly: float = 0, hra_received: float = 0,
               hra_received_monthly: float = 0, basic_da: float = 0, basic_da_monthly: float = 0, rent_paid: float = 0,
               rent_paid_monthly: float = 0, city: str = "", age: int = 30, other_income: float = 0,
               section_80c: float = 0, health_insurance_self: float = 0, health_insurance_parents: float = 0,
               nps_80ccd1b: float = 0, home_loan_interest: float = 0, professional_tax: float = 0,
               employer_nps: float = 0, capital_gains_equity_stcg: float = 0, capital_gains_equity_ltcg: float = 0) -> str:
    """Indian income tax for a resident salaried person, new vs old regime: slabs, standard deduction, 87A rebate with marginal relief, surcharge, 4% cess, HRA exemption and old-regime deductions. Pass only figures the user gave, as stated: yearly amounts in the plain fields, per-month amounts in the *_monthly fields (the tool multiplies by 12). Profit from selling shares or equity mutual funds is a capital gain: call capital_gains for it, never pass it as other_income. Freelance or business income goes in other_income. Leave out anything not given; the tool lists what is missing for the old-regime HRA exemption.

    Args:
        financial_year: '2026-27' (current, default) or '2025-26' if the user names that year or AY 2026-27.
        regime: 'both' to compare (default), or 'new' / 'old' if the user chose one.
        gross_salary: Yearly gross salary incl. HRA and allowances, plain rupees (30 lakh = 3000000). Salary only: for a freelancer or business owner leave it 0 and use other_income.
        gross_salary_monthly: Monthly gross salary, if given per month instead.
        hra_received: HRA received per year.
        hra_received_monthly: HRA received per month.
        basic_da: Basic salary + DA per year (needed for the HRA exemption).
        basic_da_monthly: Basic salary + DA per month.
        rent_paid: Rent actually paid per year.
        rent_paid_monthly: Rent actually paid per month.
        city: City of the rented home, e.g. 'Mumbai' (metro cities get 50% instead of 40%).
        age: Age in years, only if the user gave it (60+ and 80+ have higher old-regime exemptions).
        other_income: Other income taxed at slab rates per year, only if given: interest, rent, and freelance, professional or business income (no standard deduction). Never share or mutual-fund profit (use capital_gains).
        section_80c: 80C investments (PF, PPF, ELSS, LIC...) per year, old regime only.
        health_insurance_self: 80D premium for self and family per year.
        health_insurance_parents: 80D premium for parents per year.
        nps_80ccd1b: Own NPS contribution under 80CCD(1B) per year.
        home_loan_interest: Interest on a self-occupied home loan per year (24(b)).
        professional_tax: Professional tax paid per year.
        employer_nps: Employer NPS contribution per year (80CCD(2)).
        capital_gains_equity_stcg: Do not use: call capital_gains for profit from selling shares or equity funds.
        capital_gains_equity_ltcg: Do not use: call capital_gains for profit from selling shares or equity funds.
    """
    return _run(T.income_tax, financial_year=financial_year, regime=regime, gross_salary=gross_salary,
                gross_salary_monthly=gross_salary_monthly, hra_received=hra_received,
                hra_received_monthly=hra_received_monthly, basic_da=basic_da, basic_da_monthly=basic_da_monthly,
                rent_paid=rent_paid, rent_paid_monthly=rent_paid_monthly, city=city, age=age, other_income=other_income,
                section_80c=section_80c, health_insurance_self=health_insurance_self,
                health_insurance_parents=health_insurance_parents, nps_80ccd1b=nps_80ccd1b,
                home_loan_interest=home_loan_interest, professional_tax=professional_tax, employer_nps=employer_nps,
                capital_gains_equity_stcg=capital_gains_equity_stcg, capital_gains_equity_ltcg=capital_gains_equity_ltcg)


@tool(parse_docstring=True)
def capital_gains(purchase_value: float = 0, sale_value: float = 0, holding_months: float = 0,
                  asset: Literal["listed_equity", "equity_fund", "debt_fund"] = "listed_equity",
                  financial_year: str = "2025-26", expenses: float = 0, ltcg_exemption_used: float = 0,
                  gain: float | None = None) -> str:
    """Capital gains tax on selling listed shares or equity mutual-fund units (India, resident): short-term (held 12 months or less) 20%; long-term 12.5% on the gain above ₹1.25 lakh a year; 4% cess; no 87A rebate. Pass the figures the user gave.

    Args:
        purchase_value: Total buying cost in rupees (2 lakh = 200000).
        sale_value: Total sale value in rupees.
        holding_months: How long it was held, in months (ten months = 10, 2 years = 24).
        asset: 'listed_equity' (shares, default), 'equity_fund' (equity mutual fund or ETF) or 'debt_fund'.
        financial_year: '2025-26' (default) or '2026-27'.
        expenses: Brokerage and other selling costs, only if given.
        ltcg_exemption_used: Other long-term equity gains already made this year, only if given.
        gain: The profit, if the user gives only the gain instead of the two values.
    """
    return _run(CG.capital_gains, purchase_value=purchase_value, sale_value=sale_value, holding_months=holding_months,
                asset=asset, financial_year=financial_year, expenses=expenses, ltcg_exemption_used=ltcg_exemption_used,
                gain=gain)


@tool(parse_docstring=True)
def tax_rules(financial_year: str = "2026-27",
              topic: Literal["all", "new_regime", "old_regime", "deductions", "capital_gains"] = "all") -> str:
    """Official Indian income-tax figures: slabs, standard deduction, 87A rebate and its limit, highest zero-tax income, cess, surcharge, 80C/80D/80CCD(1B)/24(b) limits and capital-gains rates. Use it for any question about a tax rule, limit, rate or deduction and quote its figures.

    Args:
        financial_year: '2026-27' (default) or '2025-26'.
        topic: 'new_regime', 'old_regime', 'deductions' (old-regime limits), 'capital_gains' or 'all'.
    """
    return _run(TR.tax_rules, financial_year=financial_year, topic=topic)


CALC_TOOLS = [loan_emi, sip_future_value, goal_saving, tvm, real_return, effective_annual_rate, npv_irr, perpetuity,
              project_cash_flows, equivalent_annual_cost, bond_metrics, black_scholes, calculate, solve_equation, solve_financials, financial_ratios, balance_sheet_check,
              cash_flow_statement, income_tax, capital_gains, tax_rules]
TOOLS_BY_NAME = {t.name: t for t in CALC_TOOLS}
