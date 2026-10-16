"""LangChain tool wrappers around the deterministic calculators.

Each tool returns a JSON string. Bad input returns {"error": ...} so the model can
correct itself instead of the graph crashing.
"""
import json

from langchain_core.tools import tool
from pydantic import BaseModel, ConfigDict, Field

from . import calculators as C
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


class _Strict(BaseModel):
    """Unknown argument names raise an error instead of being silently dropped."""
    model_config = ConfigDict(extra="forbid")


_ITEMS = (float | None)


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
    """Check whether a balance sheet balances, with section totals and a common-size view (% of total assets)."""
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
    investing: dict[str, float] | None = Field(None, description=(
        'Investing cash flows: purchases negative, sale proceeds positive, e.g. {"Purchase of machinery": -60000, "Sale of furniture": 5000}.'))
    financing: dict[str, float] | None = Field(None, description=(
        'Financing cash flows: loans and share issues positive, repayments and dividends negative, e.g. {"Bank loan": 25000, "Dividend paid": -20000}.'))
    opening_cash: float | None = Field(None, description="Cash at the start of the year.")
    closing_cash: float | None = Field(None, description="Cash at the end of the year, if given, to check the statement reconciles.")


@tool(args_schema=CashFlowArgs)
def cash_flow_statement(**kwargs) -> str:
    """Prepare a cash flow statement by the indirect method (CFO, CFI, CFF, net change, closing-cash reconciliation). Include only items the problem states. Sale proceeds of an asset go under investing."""
    return _run(S.cash_flow_statement, **kwargs)


CALC_TOOLS = [loan_emi, sip_future_value, tvm, npv_irr, bond_metrics, black_scholes, calculate, solve_equation,
              financial_ratios, balance_sheet_check, cash_flow_statement]
TOOLS_BY_NAME = {t.name: t for t in CALC_TOOLS}
