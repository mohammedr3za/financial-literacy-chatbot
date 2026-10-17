"""Compact tool schemas for the OpenAI/vLLM path (SPEC 3.1, context budget).

vLLM renders every tool's JSON schema into the prompt, and the full pydantic schemas were about 7.5k of the 16,384-token
context (solve_financials alone 2.4k). The model sees these compact copies; the real pydantic schemas still validate every
call, so nothing is loosened. Dropped: "title", "default": null, anyOf [X, null] wrappers (collapsed to X), the targets enum
of solve_financials (the solver rejects unknown names with suggestions) and long descriptions (routing hints carry them).
"""
import re

from langchain_core.utils.function_calling import convert_to_openai_tool

MAX_PROP_DESC = 120          # characters: the first sentence is kept whole up to this length

TOOL_DESC = {   # short tool descriptions; the per-question routing hints carry the details
    "solve_financials": (
        "Solve textbook statement and ratio problems (income statement, liquidity, turnover, leverage, DuPont, payout, "
        "growth, market value). Pass EVERY figure the question gives under its name and the asked item(s) in targets; the "
        "tool picks the formulas. *_pct items take the percent number (23 percent -> 23); ratios are plain numbers (.75 -> "
        "0.75). Never compute an input yourself. One call per year for multi-year statements."),
    "financial_ratios": ("Ratio analysis of pasted statements (liquidity, leverage, profitability, efficiency, cash, DuPont). "
                         "Pass every item given, exactly as given and in the same units."),
    "cash_flow_statement": ("Cash flow statement, indirect method (CFO, CFI, CFF, net change, reconciliation). Only items "
                            "the problem states; sale proceeds go under investing."),
    "equivalent_annual_cost": (
        "Equivalent annual cost of an asset bought now and run at a level yearly cost; one call per option. Costs are "
        "positive. For electricity pass power_watts, hours_per_year, price_per_kwh, lifetime_hours instead of annual_cost "
        "and life_years."),
    "tvm": ("Time value of money with positive amounts: fv, pv, pmt (deposit, or withdrawal from pv), years or "
            "annual_rate_pct (CAGR). Plain annuity: solve_for='pv' or 'fv' with pmt only; returns ordinary and due values. "
            "Leave the unknown out."),
    "loan_emi": ("Loan payments: give exactly three of principal, annual_rate_pct, term (months, or years) and emi (the "
                 "payment per period); it solves the missing one, e.g. the APR the payments imply. payments_per_year 1, 2, "
                 "4 or 12. Leave the unknown out; never pass 0 for it."),
    "income_tax": ("Indian income tax for a resident individual, new vs old regime. Pass only figures the user gave; "
                   "per-month amounts go in the *_monthly fields; freelance or business income in other_income."),
    "capital_gains": ("Tax on selling listed shares or equity mutual funds: short-term (12 months or less) 20%, long-term "
                      "12.5% above ₹1.25 lakh, 4% cess. Figures as the user gave them."),
    "tax_rules": ("Official Indian tax figures (slabs, standard deduction, 87A rebate, zero-tax income, 80C/80D/80CCD(1B) "
                  "limits, capital-gains rates). Use for any rule, limit or rate question and quote them."),
    "project_cash_flows": (
        "Capital-budgeting project: yearly cash flows, OCF, after-tax salvage, NWC, NPV, IRR; or the bid price per unit "
        "that makes NPV = 0. One call per project, every figure from the question; ignore sunk costs."),
}
KEEP_PROP_DESC = {   # solve_financials: names explain themselves; only these keep a description
    "solve_financials": {"targets": "Item names asked for, e.g. ['cogs'] or ['ebit', 'net_income'].",
                         "costs": "Operating costs other than depreciation.",
                         "other_expenses": "Other expenses besides COGS and depreciation.",
                         "total_debt": "Total liabilities.",
                         "debt_ratio": "Decimal: 40 percent -> 0.4."},
    "financial_ratios": {"total_debt": "Borrowings (short- plus long-term).", "capex": "Capital expenditure, positive.",
                         "total_liabilities": "Current plus non-current liabilities."},
    "project_cash_flows": {"sales": "Number, or list per year.", "units": "Number, or list per year.",
                           "price_per_unit": "Leave out to solve the bid price.", "costs": "Cash operating costs a year.",
                           "variable_cost_per_unit": "In dollars or rupees.",
                           "variable_cost_cents_per_unit": "Cost in cents as written (.40 cents -> 0.4).",
                           "variable_cost_pct": "Percent of sales.", "ocf": "Only if given directly.",
                           "investment": "Fixed assets at year 0.", "depreciation_years": "Tax life if not years.",
                           "macrs_rates_pct": "Rates in percent as given.", "salvage": "Pretax market value at the end.",
                           "nwc_initial": "NWC at year 0.", "nwc_annual_addition": "Extra NWC each later year.",
                           "opportunity_cost_now": "After-tax value now of an owned asset used (e.g. land).",
                           "opportunity_value_end": "Its after-tax value at the end.",
                           "solve_bid_price": "True: price per unit that makes NPV = 0."},
    "income_tax": {"financial_year": "'2026-27' (default) or '2025-26' (AY 2026-27).",
                   "regime": "'both' (default), or the one the user chose.",
                   "gross_salary": "Yearly gross salary incl. HRA and allowances (0 for a freelancer).",
                   "city": "City of the rented home (metros get 50% HRA).", "age": "Only if the user gave it.",
                   "employer_nps": "Employer NPS contribution, 80CCD(2).",
                   "other_income": "Interest, rent, and freelance or business income (no standard deduction).",
                   "capital_gains_equity_stcg": "Do not use: call capital_gains.",
                   "capital_gains_equity_ltcg": "Do not use: call capital_gains."},
}


def _short(text: str, limit: int = MAX_PROP_DESC, sentences: int = 1) -> str:
    """The first sentence(s) that fit in limit characters (at least the first sentence, cut at a word if needed)."""
    parts = re.split(r"(?<!incl\.)(?<!e\.g\.)(?<!i\.e\.)(?<!etc\.)(?<=[.!?])\s+(?=[A-Z])", " ".join(text.split()))
    out = parts[0]
    for nxt in parts[1:sentences]:
        if len(out) + 1 + len(nxt) > limit:
            break
        out += " " + nxt
    return out if len(out) <= limit else out[: limit - 1].rsplit(" ", 1)[0] + "…"


def _prop(p: dict, desc: str | None = "keep") -> dict:
    p = {k: v for k, v in p.items() if k != "title"}
    if "anyOf" in p:
        options = [o for o in p.pop("anyOf") if o.get("type") != "null"]
        if len(options) == 1:
            p = {**_prop(options[0], desc=None), **p}
        else:
            p["anyOf"] = [_prop(o, desc=None) for o in options]
    if "default" in p and (p["default"] is None or p["default"] in (0, "", False)):   # "not given" defaults say nothing
        del p["default"]
    if isinstance(p.get("items"), dict):
        p["items"] = _prop(p["items"], desc=None)
    if isinstance(p.get("additionalProperties"), dict):
        p["additionalProperties"] = _prop(p["additionalProperties"], desc=None)
    if desc is None:
        p.pop("description", None)
    elif desc != "keep":
        p["description"] = desc
    elif "description" in p:
        p["description"] = _short(p["description"])
    return p


def compact_tool_schema(tool) -> dict:
    full = convert_to_openai_tool(tool)["function"]
    name = full["name"]
    params = full.get("parameters", {})
    keep = KEEP_PROP_DESC.get(name)
    props = {}
    for prop_name, spec in params.get("properties", {}).items():
        desc = "keep" if keep is None else keep.get(prop_name)
        p = _prop(spec, desc)
        if name == "solve_financials" and prop_name == "targets":
            p = {"type": "array", "items": {"type": "string"}, "description": desc}
        props[prop_name] = p
    out_params = {"type": "object", "properties": props}
    if params.get("required"):
        out_params["required"] = params["required"]
    desc = TOOL_DESC.get(name) or _short(full.get("description", ""), 220, sentences=2)
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": out_params}}


def compact_tool_schemas(tools) -> list[dict]:
    return [compact_tool_schema(t) for t in tools]
