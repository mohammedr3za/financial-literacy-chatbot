"""solve_financials: a deterministic solver over standard corporate-finance identities (SPEC 4.1).

The model passes every figure the question gives, by canonical name, and the item(s) asked for. The solver repeatedly
finds an identity with exactly one unknown, solves it with SymPy (exact rational arithmetic), and records the step,
until nothing new can be derived. The formulas come from the library below (Ross, Westerfield & Jordan, "Fundamentals
of Corporate Finance" conventions), never from the model. Identities are built from SymPy symbols in code: no
parsing, eval or exec.

Units: amounts as plain numbers; percent items (*_pct) as the percent number (23 percent -> 23); ratios, turnovers
and multiples as plain numbers (.75 -> 0.75); days as days.
"""
import difflib
import re
from functools import lru_cache
from types import SimpleNamespace

import sympy as sp

from .fmt import format_inr, group, make_table, symbol

# name: (kind, label, description shown to the model)
VARIABLES: dict[str, tuple[str, str, str]] = {
    # income statement
    "sales": ("money", "Sales", "Sales or revenue."),
    "costs": ("money", "Costs", "Costs: operating costs other than depreciation (the 'costs' line of an income statement)."),
    "cogs": ("money", "Cost of goods sold", "Cost of goods sold (COGS)."),
    "other_expenses": ("money", "Other expenses", "Other operating expenses besides COGS and depreciation (SG&A)."),
    "gross_profit": ("money", "Gross profit", "Gross profit."),
    "depreciation": ("money", "Depreciation", "Depreciation (and amortization) expense."),
    "ebit": ("money", "EBIT", "EBIT, earnings before interest and taxes (operating income)."),
    "ebitda": ("money", "EBITDA", "EBITDA."),
    "interest": ("money", "Interest", "Interest expense."),
    "ebt": ("money", "EBT", "Earnings before taxes (taxable income)."),
    "tax_rate_pct": ("pct", "Tax rate (%)", "Tax rate in percent: 23 percent -> 23."),
    "taxes": ("money", "Taxes", "Taxes."),
    "net_income": ("money", "Net income", "Net income (profit after tax)."),
    "dividends": ("money", "Dividends", "Total dividends paid."),
    "addition_to_retained_earnings": ("money", "Addition to retained earnings", "Addition to retained earnings."),
    "operating_cash_flow": ("money", "Operating cash flow", "Operating cash flow."),
    # balance sheet
    "current_assets": ("money", "Current assets", "Total current assets."),
    "current_liabilities": ("money", "Current liabilities", "Total current liabilities."),
    "cash": ("money", "Cash", "Cash (and cash equivalents)."),
    "receivables": ("money", "Receivables", "Accounts receivable."),
    "inventory": ("money", "Inventory", "Inventory."),
    "quick_assets": ("money", "Quick assets", "Quick assets (current assets minus inventory)."),
    "net_working_capital": ("money", "Net working capital", "Net working capital."),
    "net_fixed_assets": ("money", "Net fixed assets", "Net fixed assets."),
    "total_assets": ("money", "Total assets", "Total assets."),
    "long_term_debt": ("money", "Long-term debt", "Long-term debt."),
    "total_debt": ("money", "Total debt", "Total debt = total liabilities (current liabilities + long-term debt)."),
    "total_equity": ("money", "Total equity", "Total (owners' or shareholders') equity."),
    # short-term solvency, asset use
    "current_ratio": ("ratio", "Current ratio", "Current ratio, e.g. 1.5."),
    "quick_ratio": ("ratio", "Quick ratio", "Quick (acid-test) ratio: .75 -> 0.75."),
    "cash_ratio": ("ratio", "Cash ratio", "Cash ratio."),
    "inventory_turnover": ("ratio", "Inventory turnover", "Inventory turnover (times)."),
    "days_sales_in_inventory": ("days", "Days' sales in inventory", "Days' sales in inventory (days)."),
    "credit_sales": ("money", "Credit sales", "Sales made on credit (an amount)."),
    "credit_sales_pct": ("pct", "Credit sales (% of sales)", "Share of sales made on credit, in percent: 85 percent -> 85."),
    "receivables_turnover": ("ratio", "Receivables turnover", (
        "Receivables turnover (times): credit sales ÷ receivables when credit sales are given, otherwise sales ÷ receivables.")),
    "days_sales_in_receivables": ("days", "Days' sales in receivables", "Days' sales in receivables or average collection period (days)."),
    "total_asset_turnover": ("ratio", "Total asset turnover", "Total asset turnover (times)."),
    "fixed_asset_turnover": ("ratio", "Fixed asset turnover", "Fixed asset turnover (times)."),
    "nwc_turnover": ("ratio", "NWC turnover", "Net working capital turnover (times)."),
    "capital_intensity": ("ratio", "Capital intensity", "Capital intensity ratio (total assets per 1 of sales)."),
    # long-term solvency
    "debt_ratio": ("ratio", "Total debt ratio", "Total debt ratio as a decimal: 40 percent -> 0.4."),
    "debt_equity_ratio": ("ratio", "Debt-equity ratio", "Debt-equity ratio, e.g. 0.8."),
    "equity_multiplier": ("ratio", "Equity multiplier", "Equity multiplier."),
    "long_term_debt_ratio": ("ratio", "Long-term debt ratio", "Long-term debt ratio as a decimal."),
    "times_interest_earned": ("ratio", "Times interest earned", "Times interest earned (TIE)."),
    "cash_coverage": ("ratio", "Cash coverage", "Cash coverage ratio."),
    # profitability, payout, growth
    "profit_margin_pct": ("pct", "Profit margin (%)", "Profit margin in percent."),
    "roa_pct": ("pct", "ROA (%)", "Return on assets in percent."),
    "roe_pct": ("pct", "ROE (%)", "Return on equity in percent."),
    "payout_ratio_pct": ("pct", "Payout ratio (%)", "Dividend payout ratio in percent."),
    "retention_ratio_pct": ("pct", "Retention ratio (%)", "Retention (plowback) ratio in percent."),
    "sustainable_growth_pct": ("pct", "Sustainable growth rate (%)", "Sustainable growth rate in percent."),
    "internal_growth_pct": ("pct", "Internal growth rate (%)", "Internal growth rate in percent."),
    # market value
    "shares": ("count", "Shares outstanding", "Number of shares outstanding."),
    "eps": ("per_share", "EPS", "Earnings per share."),
    "dividends_per_share": ("per_share", "Dividends per share", "Dividends per share."),
    "price": ("per_share", "Share price", "Price per share."),
    "pe_ratio": ("ratio", "P/E ratio", "Price-earnings ratio."),
    "market_cap": ("money", "Market capitalization", "Market capitalization (market value of equity)."),
    "book_value_per_share": ("per_share", "Book value per share", "Book value per share."),
    "market_to_book": ("ratio", "Market-to-book ratio", "Market-to-book ratio."),
    "price_sales_ratio": ("ratio", "Price-sales ratio", "Price-sales ratio."),
    "enterprise_value": ("money", "Enterprise value", "Enterprise value."),
    "ev_ebitda": ("ratio", "EV/EBITDA", "Enterprise value to EBITDA multiple."),
}
# alternative names a model uses for the same items; mapped to the canonical name before names are checked. Live runs sent
# interest_expense, accounts_receivable and return_on_equity_pct, were rejected, and then dropped the figure or did mental maths.
_ALIAS_GROUPS = {
    "sales": "revenue revenues net_sales total_sales sales_revenue total_revenue",
    "costs": "operating_costs operating_expenses total_costs",
    "cogs": "cost_of_goods_sold cost_of_sales",
    "other_expenses": "other_operating_expenses sga selling_general_and_administrative selling_general_administrative",
    "depreciation": "depreciation_expense depreciation_and_amortization",
    "ebit": "operating_income earnings_before_interest_and_taxes operating_profit",
    "interest": "interest_expense interest_paid",
    "ebt": "earnings_before_taxes pretax_income taxable_income profit_before_tax",
    "tax_rate_pct": "tax_rate",
    "taxes": "income_tax tax_expense taxes_paid",
    "net_income": "net_profit profit_after_tax net_earnings",
    "dividends": "dividends_paid total_dividends",
    "addition_to_retained_earnings": "addition_to_re retained_earnings_addition",
    "receivables": "accounts_receivable receivable",
    "inventory": "inventories",
    "net_fixed_assets": "fixed_assets",
    "net_working_capital": "nwc working_capital",
    "total_debt": "total_liabilities",
    "total_equity": "equity shareholders_equity stockholders_equity owners_equity total_shareholders_equity",
    "debt_equity_ratio": "debt_to_equity debt_to_equity_ratio de_ratio",
    "debt_ratio": "debt_to_assets debt_to_assets_ratio total_debt_ratio debt_to_total_assets",
    "quick_ratio": "acid_test_ratio",
    "total_asset_turnover": "asset_turnover",
    "receivables_turnover": "accounts_receivable_turnover receivable_turnover",
    "days_sales_in_receivables": "days_sales_outstanding average_collection_period dso",
    "days_sales_in_inventory": "days_in_inventory days_inventory",
    "times_interest_earned": "interest_coverage tie",
    "roe_pct": "roe return_on_equity",
    "roa_pct": "roa return_on_assets",
    "profit_margin_pct": "profit_margin net_profit_margin net_margin",
    "payout_ratio_pct": "payout_ratio dividend_payout_ratio",
    "retention_ratio_pct": "retention_ratio plowback_ratio",
    "credit_sales_pct": "percent_credit_sales credit_pct",
    "sustainable_growth_pct": "sustainable_growth_rate",
    "internal_growth_pct": "internal_growth_rate",
    "shares": "shares_outstanding number_of_shares",
    "eps": "earnings_per_share",
    "price": "price_per_share share_price stock_price",
    "pe_ratio": "pe price_earnings_ratio",
    "market_cap": "market_capitalization",
    "book_value_per_share": "bvps",
    "market_to_book": "market_to_book_ratio",
    "dividends_per_share": "dps",
}
ALIASES = {alias: name for name, words in _ALIAS_GROUPS.items() for alias in words.split()}


def resolve(name: str) -> str:
    """Canonical item name: the alias table, then *_percent / *_percentage -> *_pct (return_on_equity_percent -> roe_pct)."""
    if not isinstance(name, str) or name in VARIABLES:
        return name
    if name in ALIASES:
        return ALIASES[name]
    for suffix in ("_percentage", "_percent", "_pct"):
        if name.endswith(suffix):
            base = name[: -len(suffix)]
            for cand in (base + "_pct", ALIASES.get(base, "")):
                if cand in VARIABLES and VARIABLES[cand][0] == "pct":
                    return cand
    return name


KIND = {name: v[0] for name, v in VARIABLES.items()}
LABEL = {name: v[1] for name, v in VARIABLES.items()}
NONNEGATIVE = {"sales", "costs", "cogs", "other_expenses", "depreciation", "interest", "tax_rate_pct", "dividends", "current_assets",
               "current_liabilities", "cash", "receivables", "inventory", "quick_assets", "net_fixed_assets", "total_assets",
               "long_term_debt", "total_debt", "current_ratio", "quick_ratio", "cash_ratio", "inventory_turnover",
               "days_sales_in_inventory", "receivables_turnover", "credit_sales", "credit_sales_pct", "days_sales_in_receivables",
               "total_asset_turnover", "fixed_asset_turnover", "capital_intensity", "debt_ratio", "shares", "price", "market_cap", "dividends_per_share"}
TOLERANCE = 0.005   # inputs that disagree by more than 0.5% are reported as inconsistent

_SYM = {name: sp.Symbol(name) for name in VARIABLES}
s = SimpleNamespace(**_SYM)
_roe_b = s.roe_pct / 100 * s.retention_ratio_pct / 100
_roa_b = s.roa_pct / 100 * s.retention_ratio_pct / 100

# (defined item, expression, how the formula is written for it). Order matters only for which identity derives an
# item first; the basic definitions come before the textbook shortcuts (DuPont and friends).
IDENTITIES: list[tuple[str, sp.Expr, str]] = [
    ("ebit", s.sales - s.costs - s.depreciation, "sales − costs − depreciation"),
    ("ebit", s.sales - s.cogs - s.other_expenses - s.depreciation, "sales − cogs − other_expenses − depreciation"),
    ("ebitda", s.ebit + s.depreciation, "ebit + depreciation"),
    ("ebt", s.ebit - s.interest, "ebit − interest"),
    ("taxes", s.ebt * s.tax_rate_pct / 100, "ebt × tax_rate_pct ÷ 100"),
    ("net_income", s.ebt - s.taxes, "ebt − taxes"),
    ("addition_to_retained_earnings", s.net_income - s.dividends, "net_income − dividends"),
    ("operating_cash_flow", s.ebit + s.depreciation - s.taxes, "ebit + depreciation − taxes"),
    ("gross_profit", s.sales - s.cogs, "sales − cogs"),
    ("net_working_capital", s.current_assets - s.current_liabilities, "current_assets − current_liabilities"),
    ("current_ratio", s.current_assets / s.current_liabilities, "current_assets ÷ current_liabilities"),
    ("quick_assets", s.current_assets - s.inventory, "current_assets − inventory"),
    ("quick_ratio", s.quick_assets / s.current_liabilities, "quick_assets ÷ current_liabilities"),
    ("cash_ratio", s.cash / s.current_liabilities, "cash ÷ current_liabilities"),
    ("inventory_turnover", s.cogs / s.inventory, "cogs ÷ inventory"),
    ("days_sales_in_inventory", 365 / s.inventory_turnover, "365 ÷ inventory_turnover"),
    ("receivables_turnover", s.sales / s.receivables, "sales ÷ receivables"),               # no credit figures given
    ("credit_sales", s.sales * s.credit_sales_pct / 100, "sales × credit_sales_pct ÷ 100"),
    ("receivables_turnover", s.credit_sales / s.receivables, "credit_sales ÷ receivables"),   # credit figures given
    ("days_sales_in_receivables", 365 / s.receivables_turnover, "365 ÷ receivables_turnover"),
    ("total_assets", s.current_assets + s.net_fixed_assets, "current_assets + net_fixed_assets"),
    ("total_assets", s.total_debt + s.total_equity, "total_debt + total_equity"),
    ("total_debt", s.current_liabilities + s.long_term_debt, "current_liabilities + long_term_debt"),
    ("debt_ratio", s.total_debt / s.total_assets, "total_debt ÷ total_assets"),
    ("debt_equity_ratio", s.total_debt / s.total_equity, "total_debt ÷ total_equity"),
    ("equity_multiplier", s.total_assets / s.total_equity, "total_assets ÷ total_equity"),
    ("long_term_debt_ratio", s.long_term_debt / (s.long_term_debt + s.total_equity), "long_term_debt ÷ (long_term_debt + total_equity)"),
    ("total_asset_turnover", s.sales / s.total_assets, "sales ÷ total_assets"),
    ("fixed_asset_turnover", s.sales / s.net_fixed_assets, "sales ÷ net_fixed_assets"),
    ("nwc_turnover", s.sales / s.net_working_capital, "sales ÷ net_working_capital"),
    ("capital_intensity", s.total_assets / s.sales, "total_assets ÷ sales"),
    ("times_interest_earned", s.ebit / s.interest, "ebit ÷ interest"),
    ("cash_coverage", s.ebitda / s.interest, "ebitda ÷ interest"),
    ("profit_margin_pct", 100 * s.net_income / s.sales, "100 × net_income ÷ sales"),
    ("roa_pct", 100 * s.net_income / s.total_assets, "100 × net_income ÷ total_assets"),
    ("roe_pct", 100 * s.net_income / s.total_equity, "100 × net_income ÷ total_equity"),
    ("eps", s.net_income / s.shares, "net_income ÷ shares"),
    ("dividends_per_share", s.dividends / s.shares, "dividends ÷ shares"),
    ("pe_ratio", s.price / s.eps, "price ÷ eps"),
    ("market_cap", s.price * s.shares, "price × shares"),
    ("book_value_per_share", s.total_equity / s.shares, "total_equity ÷ shares"),
    ("market_to_book", s.price / s.book_value_per_share, "price ÷ book_value_per_share"),
    ("price_sales_ratio", s.market_cap / s.sales, "market_cap ÷ sales"),
    ("payout_ratio_pct", 100 * s.dividends / s.net_income, "100 × dividends ÷ net_income"),
    ("retention_ratio_pct", 100 - s.payout_ratio_pct, "100 − payout_ratio_pct"),
    ("sustainable_growth_pct", 100 * _roe_b / (1 - _roe_b),
     "100 × (roe_pct ÷ 100 × retention_ratio_pct ÷ 100) ÷ (1 − roe_pct ÷ 100 × retention_ratio_pct ÷ 100)"),
    ("internal_growth_pct", 100 * _roa_b / (1 - _roa_b),
     "100 × (roa_pct ÷ 100 × retention_ratio_pct ÷ 100) ÷ (1 − roa_pct ÷ 100 × retention_ratio_pct ÷ 100)"),
    ("enterprise_value", s.market_cap + s.total_debt - s.cash, "market_cap + total_debt − cash"),
    ("ev_ebitda", s.enterprise_value / s.ebitda, "enterprise_value ÷ ebitda"),
    # textbook shortcuts that follow from the definitions above; they let ratio-only questions be solved
    ("roe_pct", s.profit_margin_pct * s.total_asset_turnover * s.equity_multiplier,
     "profit_margin_pct × total_asset_turnover × equity_multiplier"),
    ("roa_pct", s.profit_margin_pct * s.total_asset_turnover, "profit_margin_pct × total_asset_turnover"),
    ("net_income", s.ebt * (1 - s.tax_rate_pct / 100), "ebt × (1 − tax_rate_pct ÷ 100)"),   # works EBT back from net income
    ("roe_pct", s.roa_pct * s.equity_multiplier, "roa_pct × equity_multiplier"),
    ("equity_multiplier", 1 + s.debt_equity_ratio, "1 + debt_equity_ratio"),
    ("debt_ratio", 1 - 1 / s.equity_multiplier, "1 − 1 ÷ equity_multiplier"),
    ("pe_ratio", s.market_cap / s.net_income, "market_cap ÷ net_income"),
    ("market_to_book", s.market_cap / s.total_equity, "market_cap ÷ total_equity"),
    ("payout_ratio_pct", 100 * s.dividends_per_share / s.eps, "100 × dividends_per_share ÷ eps"),
    ("retention_ratio_pct", 100 * s.addition_to_retained_earnings / s.net_income, "100 × addition_to_retained_earnings ÷ net_income"),
    ("capital_intensity", 1 / s.total_asset_turnover, "1 ÷ total_asset_turnover"),
]
_VARS = [frozenset({lhs} | {str(x) for x in rhs.free_symbols}) for lhs, rhs, _ in IDENTITIES]
# formula text for a rearranged identity where SymPy's form is less readable than the textbook's
_REARRANGED = {("ebt × (1 − tax_rate_pct ÷ 100)", "ebt"): "net_income ÷ (1 − tax_rate_pct ÷ 100)"}
# receivables turnover has two definitions; exactly one is active in a call, so they can never conflict
_CREDIT_NAMES = {"credit_sales", "credit_sales_pct"}
_TOTAL_SALES_ONLY = {"sales ÷ receivables"}
_CREDIT_ONLY = {"credit_sales ÷ receivables"}


def _active(credit: bool) -> list[int]:
    skip = _TOTAL_SALES_ONLY if credit else _CREDIT_ONLY
    return [i for i, (_, _, text) in enumerate(IDENTITIES) if text not in skip]


# ---------------------------------------------------------------- formula text
def _show(e: sp.Expr) -> str:
    """Readable formula text: positive terms first, × and ÷, a single fraction at the top level."""
    num, den = sp.fraction(sp.together(e))
    if den != 1:
        if num.could_extract_minus_sign():
            num, den = -num, -den
        return f"{_wrap(num)} ÷ {_wrap(den)}"
    return _term(num)


def _wrap(e: sp.Expr) -> str:
    text = _term(e)
    return f"({text})" if isinstance(e, sp.Add) or (isinstance(e, sp.Mul) and e.could_extract_minus_sign()) else text


def _term(e: sp.Expr) -> str:
    if isinstance(e, sp.Add):
        pos = [a for a in e.args if not a.could_extract_minus_sign()]
        neg = [-a for a in e.args if a.could_extract_minus_sign()]
        text = " + ".join(_term(a) for a in pos) if pos else "−" + _term(neg.pop(0))
        return text + "".join(f" − {_wrap(a) if isinstance(a, sp.Add) else _term(a)}" for a in neg)
    if isinstance(e, sp.Mul):
        coeff, rest = e.as_coeff_Mul()
        factors = sorted(sp.Mul.make_args(rest), key=lambda f: 0 if KIND.get(str(f)) in ("ratio", "pct", "days") else 1)
        parts = [_wrap(f) for f in factors]   # textbook order: ratio × amount
        if coeff == -1:
            return "−" + " × ".join(parts)
        return " × ".join(([_term(coeff)] if coeff != 1 else []) + parts)
    if isinstance(e, sp.Pow) and e.exp == -1:
        return f"1 ÷ {_wrap(e.base)}"
    return str(e)


@lru_cache(maxsize=None)
def _solved(idx: int, var: str) -> tuple[sp.Expr, str] | None:
    """The identity rearranged for one item: (expression, formula text), or None if there is no unique solution."""
    lhs, rhs, text = IDENTITIES[idx]
    if var == lhs:
        return rhs, text
    try:
        sols = sp.solve(sp.Eq(_SYM[lhs], rhs), _SYM[var])
    except (NotImplementedError, ValueError):
        return None
    return (sols[0], _REARRANGED.get((text, var)) or _show(sols[0])) if len(sols) == 1 else None


# ---------------------------------------------------------------- number text
def _trim(v: float, places: int = 4) -> str:
    text = f"{v:,.{places}f}".rstrip("0").rstrip(".")
    return "0" if text in ("-0", "") else text


def _plain(name: str, v: float) -> str:
    """A number inside a formula: grouped amounts, plain ratios, negatives in brackets."""
    body = group(abs(v), 0 if float(v).is_integer() else 2) if KIND[name] in ("money", "per_share", "count") else _trim(abs(v))
    return f"(−{body})" if v < 0 else body


def _formatted(name: str, v: float, currency: bool = True) -> str:
    kind = KIND[name]
    if kind in ("money", "per_share"):
        return format_inr(v) if currency else group(v, 0 if float(v).is_integer() else 2)
    if kind == "count":
        return group(v, 0 if float(v).is_integer() else 2)
    if kind == "pct":
        return f"{v:,.2f}%"
    if kind == "days":
        return f"{v:,.2f} days"
    return _trim(v)


def _labelled(text: str) -> str:
    return re.sub(r"\b[a-z][a-z_]*\b", lambda m: LABEL.get(m.group(0), m.group(0)), text)


def _substitute(text: str, values: dict) -> str:
    return re.sub(r"\b[a-z][a-z_]*\b", lambda m: _plain(m.group(0), values[m.group(0)]) if m.group(0) in values else m.group(0), text)


def _rational(v: float) -> sp.Rational:
    return sp.Rational(repr(float(v)))


def _evaluate(expr: sp.Expr, known: dict) -> sp.Expr | None:
    val = expr.xreplace({_SYM[k]: v for k, v in known.items()})
    if not val.is_number or val.has(sp.zoo, sp.oo, -sp.oo, sp.nan) or val.is_real is False:
        return None
    return sp.nsimplify(val) if not val.is_Rational else val


def _suggest(name: str) -> str:
    close = difflib.get_close_matches(name, list(VARIABLES), n=2, cutoff=0.5)
    return f"{name} (did you mean {' or '.join(close)}?)" if close else name


def canonical(items: dict) -> dict:
    """Alternative names mapped to canonical ones; the same item under two names with different values is an error."""
    out = {}
    for k, v in items.items():
        name = resolve(k)
        if name in out and v is not None and out[name] is not None and float(out[name]) != float(v):
            raise ValueError(f"{k} is another name for {name}, and the two were given different values ({float(v):g} and "
                             f"{float(out[name]):g}). Pass it once, as {name}.")
        out[name] = v if v is not None else out.get(name)
    return out


def check_names(names, what: str = "item") -> None:
    bad = [n for n in names if n not in VARIABLES]
    if bad:
        raise ValueError(f"Unknown {what}{'s' if len(bad) > 1 else ''}: {', '.join(_suggest(n) for n in bad)}. "
                         f"Valid names: {', '.join(VARIABLES)}.")


# ---------------------------------------------------------------- solver
NO_INTEREST = "No interest expense was given, so interest is taken as 0 and EBT = EBIT."


def solve_financials(targets: list[str], **given) -> dict:
    """Derive the targets from the given figures through the identity library. Raises ValueError on bad names."""
    if isinstance(targets, str):
        targets = [targets]
    if not targets:
        raise ValueError("targets must name at least one item to find, e.g. ['net_income'].")
    targets = list(dict.fromkeys(resolve(t) for t in targets))
    given = canonical(given)
    check_names(given)
    check_names(targets, "target")
    given = {k: float(v) for k, v in given.items() if v is not None}
    zeros = sorted(k for k, v in given.items() if v == 0)            # a zero almost always means "not given"
    inputs = {k: v for k, v in given.items() if v != 0}
    if not inputs:
        raise ValueError("Pass the figures the question gives, by name, e.g. sales=500000, tax_rate_pct=23.")

    known = {k: _rational(v) for k, v in inputs.items()}
    active = _active(credit=bool(_CREDIT_NAMES & (set(inputs) | set(targets))))
    derived: list[tuple[str, int, str]] = []                          # (item, identity index, formula text)

    def derive():
        progress = True
        while progress:
            progress = False
            for idx in active:
                names = _VARS[idx]
                unknown = [n for n in names if n not in known]
                if len(unknown) != 1:
                    continue
                solved = _solved(idx, unknown[0])
                if solved is None:
                    continue
                val = _evaluate(solved[0], known)
                if val is None:
                    continue
                known[unknown[0]] = val
                derived.append((unknown[0], idx, solved[1]))
                progress = True
                break

    derive()
    assumptions = []
    if any(t not in known for t in targets) and "ebit" in known and "ebt" not in known and "interest" not in known:
        known["interest"] = _rational(0)        # an income statement with no interest line: EBT = EBIT (SPEC 4.1)
        assumptions.append(NO_INTEREST)
        derive()

    values = {k: float(v) for k, v in known.items()}
    used = {var: idx for var, idx, _ in derived}
    warnings = []
    for idx in active:
        lhs, rhs, text = IDENTITIES[idx]
        if not _VARS[idx] <= known.keys() or used.get(lhs) == idx or any(used.get(v) == idx for v in _VARS[idx]):
            continue
        left, right = float(known[lhs]), float(_evaluate(rhs, known) or sp.nan)
        if abs(left - right) > TOLERANCE * max(abs(left), abs(right), 1e-12):
            where = "given" if lhs in inputs else "derived"
            warnings.append(f"Inconsistent figures: {lhs} = {text} gives {_trim(right)}, but {lhs} is {_trim(left)} ({where}). "
                            "Check that every figure was passed under the right name.")
    for var, _, _ in derived:
        if var in NONNEGATIVE and values[var] < 0:
            warnings.append(f"{var} came out negative ({_trim(values[var], 2)}), which is not possible: check that every "
                            "figure was passed under the right name and in the right units.")

    # keep only the steps that lead to the targets
    need, stack = set(), [t for t in targets if t in used]
    while stack:
        var = stack.pop()
        if var in need:
            continue
        need.add(var)
        stack += [v for v in _VARS[used[var]] if v != var and v in used]
    steps = []
    for var, idx, text in derived:
        if var in need:
            others = {v: values[v] for v in _VARS[idx] if v != var}
            steps.append({"variable": var, "formula": f"{var} = {text}", "substituted": f"{var} = {_substitute(text, others)}",
                          "value": round(values[var], 6), "formatted": _formatted(var, values[var])})

    result = {t: round(values[t], 6) for t in targets if t in values}
    out = {"inputs": inputs,
           "formula": "Standard textbook identities (Ross, Westerfield & Jordan); each step shows the formula used.",
           "steps": steps, "result": result, "formatted": {t: _formatted(t, values[t]) for t in result}}
    missing = {t: _missing_message(t, known, active) for t in targets if t not in values}
    if missing:
        out["missing"] = missing
    extra = {var: round(values[var], 6) for var, _, _ in derived if var not in need}
    if extra:
        out["also_derived"] = extra
    if warnings:
        out["warnings"] = warnings[:5]
    if zeros:
        out["ignored_zero_inputs"] = zeros
    if assumptions:
        out["assumptions"] = assumptions
    out["table"] = _table(inputs, steps, targets, values, warnings)
    return out


def _missing_message(target: str, known: dict, active: list[int]) -> str:
    parts = []
    for idx in active:
        names = _VARS[idx]
        if target in names:
            unknown = sorted(n for n in names if n not in known and n != target)
            lhs, _, text = IDENTITIES[idx]
            parts.append(f"{lhs} = {text} (still unknown: {', '.join(unknown)})")
    return (f"{target} cannot be derived from the figures given. Formulas that contain it: {'; '.join(parts[:6])}. "
            f"If the question states one of the unknown figures, call solve_financials again including it; otherwise say "
            f"that {target} cannot be determined from the data. Do not assume a value.")


def _table(inputs: dict, steps: list, targets: list, values: dict, warnings: list) -> dict:
    rows = [[LABEL[k], "Given", _formatted(k, v, currency=False)] for k, v in inputs.items()]
    for st in steps:
        var = st["variable"]
        rhs_text = st["formula"].split(" = ", 1)[1]
        rows.append([LABEL[var], f"{LABEL[var]} = {_labelled(rhs_text)} = {st['substituted'].split(' = ', 1)[1]}",
                     _formatted(var, values[var], currency=False)])
    bold = [i for i, r in enumerate(rows) if any(r[0] == LABEL[t] for t in targets)]
    note = "Formulas: standard textbook identities (Ross, Westerfield & Jordan)."
    if warnings:
        note += " " + " ".join(warnings[:2])
    return make_table(f"How the answer was derived ({symbol()})", ["Step", "Formula", "Value"], rows, note=note, bold_rows=bold)
