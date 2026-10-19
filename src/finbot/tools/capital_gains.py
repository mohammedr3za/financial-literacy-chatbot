"""capital_gains: tax on selling listed equity shares or equity-oriented mutual-fund units, resident individual (SPEC 4.4).

Rules for transfers on or after 23 July 2024 (Finance (No. 2) Act 2024), used for FY 2025-26 and, under the Income-tax
Act 2025, for tax year 2026-27:
- held 12 months or less: short-term, 20% (s.111A);
- held more than 12 months: long-term, 12.5% on the gain above ₹1.25 lakh in the year (s.112A);
- 4% health and education cess on the tax; no 87A rebate on these gains; surcharge on them is capped at 15%.
Debt mutual funds bought from 1 April 2023 are taxed at slab rates as other income (s.50AA): rejected here.
"""
from .fmt import format_inr, make_table
from .income_tax import CESS_PCT

STCG_RATE_PCT = 20.0
LTCG_RATE_PCT = 12.5
LTCG_EXEMPTION = 125_000
LONG_TERM_AFTER_MONTHS = 12
SOURCES = ["https://www.incometax.gov.in/iec/foportal/help/individual-business-profession",
           "https://www.indiabudget.gov.in/budget2024-25/doc/memo.pdf"]
NOTES = [
    "No 87A rebate on short- or long-term capital gains taxed at these special rates.",
    "If other taxable income is below the basic exemption limit, the shortfall can be set against these gains (residents).",
    "Surcharge on these gains, where it applies, is capped at 15%.",
    "A short-term loss can be set against short- or long-term gains; a long-term loss only against long-term gains "
    "(carried forward up to 8 years if the return is filed on time).",
]


def capital_gains(purchase_value: float = 0.0, sale_value: float = 0.0, holding_months: float = 0.0,
                  asset: str = "listed_equity", financial_year: str = "2025-26", expenses: float = 0.0,
                  ltcg_exemption_used: float = 0.0, gain: float | None = None, term: str | None = None) -> dict:
    """Capital gains tax on listed shares or equity mutual funds. gain may be given instead of purchase and sale values."""
    if asset not in ("listed_equity", "equity_fund", "debt_fund"):
        raise ValueError("asset must be 'listed_equity', 'equity_fund' or 'debt_fund'.")
    if asset == "debt_fund":
        raise ValueError("Debt mutual funds bought on or after 1 April 2023 are taxed at your slab rates as other income, "
                         "whatever the holding period: call income_tax with the gain as other_income instead.")
    fy = financial_year.replace("FY", "").replace(" ", "").strip()
    if fy not in ("2025-26", "2026-27"):
        raise ValueError("financial_year must be '2025-26' or '2026-27'.")
    if gain is None:
        if not (purchase_value and sale_value):
            raise ValueError("Give purchase_value and sale_value, or gain=<the profit> if only the profit is given "
                             "(with holding_months, or term='long' or 'short'). Several gains: one call per gain.")
        gain = sale_value - purchase_value - expenses
    if term not in (None, "long", "short"):
        raise ValueError("term must be 'long' or 'short'.")
    if holding_months <= 0 and term is None:
        raise ValueError("holding_months is how long the shares or units were held, in months (2 years = 24); if only "
                         "the type of gain is known, pass term='long' or 'short' instead.")
    long_term = holding_months > LONG_TERM_AFTER_MONTHS if holding_months > 0 else term == "long"
    rate = LTCG_RATE_PCT if long_term else STCG_RATE_PCT
    exemption = max(0.0, min(LTCG_EXEMPTION - ltcg_exemption_used, gain)) if long_term and gain > 0 else 0.0
    taxable = max(0.0, gain - exemption)
    tax = taxable * rate / 100
    cess = tax * CESS_PCT / 100
    notes = list(NOTES)
    if gain < 0:
        notes.insert(0, f"This is a {'long' if long_term else 'short'}-term capital loss of {format_inr(-gain, 0)}: no tax; "
                        "it can be set off or carried forward as described below.")
    term = "long-term" if long_term else "short-term"
    section = "s.112A" if long_term else "s.111A"
    rows = [["Sale value", format_inr(sale_value, 0)], ["Purchase value", format_inr(purchase_value, 0)]] if sale_value else []
    if expenses:
        rows.append(["Expenses (brokerage etc.)", format_inr(expenses, 0)])
    held = f"{holding_months:g} months ({term})" if holding_months > 0 else term.capitalize()
    rows += [["Capital gain", format_inr(gain, 0)], ["Holding period", held]]
    if long_term:
        rows.append(["Exemption used (₹1.25 lakh a year)", format_inr(exemption, 0)])
    rows += [["Taxable gain", format_inr(taxable, 0)], [f"Tax at {rate:g}% ({section})", format_inr(tax, 0)],
             [f"Cess ({CESS_PCT}%)", format_inr(cess, 0)]]
    table = make_table(f"Capital gains tax, FY {fy}", ["Item", "Amount"], rows, total=["Total tax", format_inr(tax + cess, 0)])
    return {"inputs": {k: v for k, v in dict(purchase_value=purchase_value, sale_value=sale_value, holding_months=holding_months,
                                              asset=asset, financial_year=fy, expenses=expenses,
                                              ltcg_exemption_used=ltcg_exemption_used).items() if v},
            "formula": ("gain = sale − purchase − expenses; held > 12 months: tax = 12.5% × (gain − unused part of ₹1.25 lakh); "
                        "otherwise tax = 20% × gain; total = tax × 1.04 (4% cess)"),
            "result": {"gain": round(gain, 2), "term": term, "section": section, "rate_pct": rate,
                       "exemption_applied": round(exemption, 2), "taxable_gain": round(taxable, 2),
                       "tax_before_cess": round(tax, 2), "cess": round(cess, 2), "total_tax": round(tax + cess, 2),
                       "notes": notes, "sources": SOURCES},
            "formatted": {"tax_before_cess": format_inr(tax), "total_tax": format_inr(tax + cess)},
            "table": table}
