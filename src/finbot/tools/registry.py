"""LangChain tool wrappers around the deterministic calculators.

Each tool returns a JSON string. Bad input returns {"error": ...} so the model can
correct itself instead of the graph crashing.
"""
import json
from typing import Literal

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from . import calculators as C
from . import statements as S
from .math_engine import calculate as _calculate
from .math_engine import solve_equation as _solve_equation

AMOUNT_HINT = "Plain rupees as a number: 1 lakh = 100000, 1 crore = 10000000."


def _run(fn, **kwargs) -> str:
    try:
        return json.dumps(fn(**kwargs), ensure_ascii=False)
    except (ValueError, TypeError) as exc:
        return json.dumps({"error": str(exc)}, ensure_ascii=False)


@tool(parse_docstring=True)
def loan_emi(principal: float, annual_rate_pct: float, months: int) -> str:
    """Monthly EMI, total interest and first-year split for a home, car or personal loan.

    Args:
        principal: Loan amount in plain rupees, e.g. 20 lakh = 2000000.
        annual_rate_pct: Yearly interest rate in percent, e.g. 8.5 for 8.5%.
        months: Loan tenure in months, e.g. 20 years = 240.
    """
    return _run(C.loan_emi, principal=principal, annual_rate_pct=annual_rate_pct, months=months)


@tool(parse_docstring=True)
def sip_future_value(monthly: float, annual_return_pct: float, years: float, step_up_pct: float = 0) -> str:
    """Future value of a monthly SIP (systematic investment plan). Returns corpus, amount invested and gain.

    Args:
        monthly: Monthly investment in plain rupees, e.g. 10000.
        annual_return_pct: Assumed yearly return in percent, e.g. 12. This is an assumption, not a guarantee.
        years: Investment period in years.
        step_up_pct: Optional yearly increase in the monthly amount, in percent. Default 0.
    """
    return _run(C.sip_future_value, monthly=monthly, annual_return_pct=annual_return_pct, years=years, step_up_pct=step_up_pct)


@tool(parse_docstring=True)
def tvm(solve_for: str, annual_rate_pct: float | None = None, years: float | None = None, pv: float = 0.0,
        pmt: float = 0.0, fv: float | None = None, periods_per_year: int = 1, when: str = "end") -> str:
    """Time value of money for savings and growth: future value, present value, required deposit, years, or CAGR.

    All amounts are positive plain rupees. Leave out the value you want to solve for.
    Examples: CAGR from 100 to 200 in 5 years -> solve_for="annual_rate_pct", years=5, pv=100, fv=200.
    Monthly saving to reach 1 crore in 15 years at 12% -> solve_for="pmt", annual_rate_pct=12, years=15, fv=10000000, periods_per_year=12.

    Args:
        solve_for: One of "fv", "pv", "pmt", "years", "annual_rate_pct".
        annual_rate_pct: Yearly rate in percent, e.g. 8 for 8%.
        years: Number of years.
        pv: Starting lump sum in plain rupees.
        pmt: Regular deposit per period in plain rupees.
        fv: Target or final amount in plain rupees.
        periods_per_year: Compounding and deposit periods per year: 1 yearly, 4 quarterly, 12 monthly.
        when: "end" if deposits are made at the end of each period, "begin" if at the start.
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


class LineItem(BaseModel):
    item: str = Field(description="Line item name, e.g. Depreciation or Purchase of machinery")
    amount: float = Field(description="Amount as given. Inflows and add-backs positive, outflows negative.")


class WorkingCapitalItem(BaseModel):
    item: str = Field(description="Line item name, e.g. Trade receivables")
    type: Literal["current_asset", "current_liability"] = Field(description="current_asset or current_liability")
    change: float | None = Field(default=None, description="Closing minus opening balance. Give this, or opening and closing.")
    opening: float | None = Field(default=None, description="Opening balance")
    closing: float | None = Field(default=None, description="Closing balance")


@tool(parse_docstring=True)
def financial_ratios(current_assets: float | None = None, current_liabilities: float | None = None,
                     inventory: float | None = None, cash: float | None = None, receivables: float | None = None,
                     payables: float | None = None, total_assets: float | None = None,
                     total_liabilities: float | None = None, total_equity: float | None = None,
                     total_debt: float | None = None, revenue: float | None = None, cogs: float | None = None,
                     gross_profit: float | None = None, ebit: float | None = None, interest_expense: float | None = None,
                     net_income: float | None = None, operating_cash_flow: float | None = None,
                     capex: float | None = None, depreciation: float | None = None) -> str:
    """Financial ratios from balance sheet and profit-and-loss items: liquidity, leverage, profitability, efficiency, cash flow and DuPont, each with its formula.

    Pass only the items you have, exactly as given, in the same units (do not convert lakh or crore).

    Args:
        current_assets: Total current assets.
        current_liabilities: Total current liabilities.
        inventory: Inventory or stock.
        cash: Cash and cash equivalents.
        receivables: Trade receivables or debtors.
        payables: Trade payables or creditors.
        total_assets: Total assets.
        total_liabilities: Total liabilities (current plus non-current).
        total_equity: Shareholders' equity or net worth.
        total_debt: Borrowings (short-term plus long-term debt).
        revenue: Revenue from operations or sales.
        cogs: Cost of goods sold.
        gross_profit: Gross profit, if COGS is not given.
        ebit: Operating profit, or earnings before interest and tax.
        interest_expense: Finance costs or interest expense.
        net_income: Net profit after tax.
        operating_cash_flow: Net cash from operating activities.
        capex: Capital expenditure (purchase of fixed assets), as a positive number.
        depreciation: Depreciation and amortisation.
    """
    items = {k: v for k, v in locals().items() if v is not None}
    return _run(S.financial_ratios, **items)


@tool(parse_docstring=True)
def balance_sheet_check(assets: dict[str, float], liabilities: dict[str, float], equity: dict[str, float]) -> str:
    """Check whether a balance sheet balances, with section totals and a common-size view (% of total assets).

    Args:
        assets: Asset line items and amounts, e.g. {"Cash": 50000, "Inventory": 40000}.
        liabilities: Liability line items and amounts.
        equity: Equity line items and amounts, e.g. {"Share capital": 100000, "Reserves": 50000}.
    """
    return _run(S.balance_sheet_check, assets=assets, liabilities=liabilities, equity=equity)


@tool(parse_docstring=True)
def cash_flow_statement(net_income: float, non_cash_items: list[LineItem] | None = None,
                        working_capital: list[WorkingCapitalItem] | None = None, investing: list[LineItem] | None = None,
                        financing: list[LineItem] | None = None, opening_cash: float | None = None,
                        closing_cash: float | None = None) -> str:
    """Prepare a cash flow statement by the indirect method, with CFO, CFI, CFF, net change and a closing-cash reconciliation.

    Args:
        net_income: Net profit after tax (or before tax, if tax paid is listed under non-cash or operating items).
        non_cash_items: Add-backs and deductions, e.g. Depreciation +, Profit on sale of asset −.
        working_capital: Changes in current assets and current liabilities; give change, or opening and closing.
        investing: Investing cash flows: purchases negative, sales positive.
        financing: Financing cash flows: borrowings and share issues positive, repayments and dividends negative.
        opening_cash: Cash at the start of the period.
        closing_cash: Cash at the end of the period, if given, to check the statement reconciles.
    """
    return _run(S.cash_flow_statement, net_income=net_income, non_cash_items=non_cash_items, working_capital=working_capital,
                investing=investing, financing=financing, opening_cash=opening_cash, closing_cash=closing_cash)


CALC_TOOLS = [loan_emi, sip_future_value, tvm, npv_irr, bond_metrics, black_scholes, calculate, solve_equation,
              financial_ratios, balance_sheet_check, cash_flow_statement]
TOOLS_BY_NAME = {t.name: t for t in CALC_TOOLS}
