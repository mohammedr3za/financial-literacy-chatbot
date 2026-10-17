"""tax_rules: the official Indian income-tax figures that income_tax and capital_gains use (SPEC 4.4).
Rule and limit questions ("how much is the standard deduction?") are answered from here, so every figure in the answer
comes from a tool. The figures are the calculators' own constants, so the two can never disagree."""
from . import capital_gains as CG
from . import income_tax as T
from .fmt import format_inr, make_table

TOPICS = ("all", "new_regime", "old_regime", "deductions", "capital_gains")


def _slabs(slabs) -> list[str]:
    out, lo = [], 0
    for hi, rate in slabs:
        out.append(f"{format_inr(lo, 0)} to {format_inr(hi, 0)}: {rate}%" if hi != T.INF else f"above {format_inr(lo, 0)}: {rate}%")
        lo = hi
    return out


def tax_rules(financial_year: str = "2026-27", topic: str = "all") -> dict:
    fy = financial_year.replace("FY", "").replace(" ", "").strip()
    if fy not in T.METROS:
        raise ValueError("financial_year must be '2025-26' or '2026-27'.")
    if topic not in TOPICS:
        raise ValueError(f"topic must be one of {', '.join(TOPICS)}.")
    new_limit, new_rebate = T.REBATE["new"]
    old_limit, old_rebate = T.REBATE["old"]
    sections = {
        "new_regime": {
            "slabs": _slabs(T.NEW_SLABS), "standard_deduction": T.STANDARD_DEDUCTION["new"],
            "rebate_87a_max": new_rebate, "rebate_income_limit": new_limit, "zero_tax_taxable_income": new_limit,
            "zero_tax_salary": new_limit + T.STANDARD_DEDUCTION["new"], "marginal_relief": "tax above ₹12 lakh is capped at the income above ₹12 lakh",
            "cess_pct": T.CESS_PCT, "surcharge": [f"{p}% above {format_inr(x, 0)}" for x, p in T.SURCHARGE["new"]],
            "employer_nps_80ccd2_cap_pct_of_basic_da": 14,
            "deductions_allowed": "standard deduction and employer NPS (80CCD(2)) only; no 80C, 80CCD(1B) (own NPS), 80D, HRA or home-loan interest",
        },
        "old_regime": {
            "slabs_below_60": _slabs(T.OLD_SLABS["below_60"]), "slabs_60_to_79": _slabs(T.OLD_SLABS["60_to_79"]),
            "slabs_80_plus": _slabs(T.OLD_SLABS["80_plus"]), "standard_deduction": T.STANDARD_DEDUCTION["old"],
            "rebate_87a_max": old_rebate, "rebate_income_limit": old_limit,
            "zero_tax_salary_below_60": old_limit + T.STANDARD_DEDUCTION["old"], "cess_pct": T.CESS_PCT,
            "surcharge": [f"{p}% above {format_inr(x, 0)}" for x, p in T.SURCHARGE["old"]],
        },
        "deductions": {
            "80c": T.LIMITS["80c"], "80ccd1b": T.LIMITS["80ccd1b"], "80d_self_family": 25_000, "80d_self_family_senior": 50_000,
            "80d_parents": 25_000, "80d_parents_senior": 50_000, "24b_home_loan_interest": T.LIMITS["24b"],
            "professional_tax": T.LIMITS["professional_tax"], "applies_to": "old regime only",
        },
        "capital_gains": {
            "stcg_rate_pct": CG.STCG_RATE_PCT, "ltcg_rate_pct": CG.LTCG_RATE_PCT, "ltcg_exemption": CG.LTCG_EXEMPTION,
            "long_term_after_months": CG.LONG_TERM_AFTER_MONTHS, "applies_to": "listed equity shares and equity mutual funds",
            "rebate_87a": "not available on these gains", "debt_funds": "bought from 1 April 2023: taxed at slab rates",
        },
    }
    keep = [k for k in sections if topic in ("all", k)]
    law = ("Income-tax Act 1961 (FY 2025-26, AY 2026-27)" if fy == "2025-26"
           else "Income-tax Act 2025 (tax year 2026-27): same slabs, rebate and standard deduction as FY 2025-26")
    rows = []
    for k in keep:
        for name, v in sections[k].items():
            if isinstance(v, list):
                v = "; ".join(v)
            elif isinstance(v, (int, float)) and not isinstance(v, bool):
                v = f"{v:g}%" if name.endswith("_pct") else format_inr(v, 0) if v >= 1000 else f"{v:g}"
            rows.append([k.replace("_", " "), name.replace("_", " "), str(v)])
    return {"inputs": {"financial_year": fy, "topic": topic},
            "formula": "Official rules (no calculation); the same constants income_tax and capital_gains use.",
            "result": {**{k: sections[k] for k in keep}, "financial_year": fy, "law": law, "sources": T.SOURCES + CG.SOURCES},
            "formatted": {}, "table": make_table(f"Income-tax rules, FY {fy}", ["Topic", "Rule", "Value"], rows)}
