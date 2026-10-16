"""income_tax: Indian income tax for a resident salaried individual, old vs new regime (SPEC 4.3).

Rules (verified 2026-10-16 against the Finance Act 2025 and its Memorandum, incometax.gov.in help pages, and the
eGazette texts of the Income-tax Act 2025 and Income-tax Rules 2026):
- FY 2025-26 (AY 2026-27), Income-tax Act 1961: new-regime s.115BAC slabs, standard deduction s.16(ia), rebate s.87A.
- FY 2026-27 (tax year 2026-27), Income-tax Act 2025: the same slabs (s.202), rebate (s.156) and standard deduction
  (s.19); HRA's 50% rate covers 8 cities instead of 4 (Income-tax Rules 2026).
Slab tax, rebate with marginal relief, surcharge with marginal relief, 4% cess. Special-rate income (capital gains
under 111A/112/112A, crypto) is out of scope: the tool says so instead of guessing.
"""
from .fmt import format_inr, make_table

INF = float("inf")
NEW_SLABS = [(400_000, 0), (800_000, 5), (1_200_000, 10), (1_600_000, 15), (2_000_000, 20), (2_400_000, 25), (INF, 30)]
OLD_SLABS = {
    "below_60": [(250_000, 0), (500_000, 5), (1_000_000, 20), (INF, 30)],
    "60_to_79": [(300_000, 0), (500_000, 5), (1_000_000, 20), (INF, 30)],
    "80_plus": [(500_000, 0), (1_000_000, 20), (INF, 30)],
}
STANDARD_DEDUCTION = {"new": 75_000, "old": 50_000}
REBATE = {"new": (1_200_000, 60_000), "old": (500_000, 12_500)}          # (income limit, maximum rebate)
SURCHARGE = {"new": [(5_000_000, 10), (10_000_000, 15), (20_000_000, 25)],
             "old": [(5_000_000, 10), (10_000_000, 15), (20_000_000, 25), (50_000_000, 37)]}
CESS_PCT = 4
METROS = {"2025-26": {"mumbai", "bombay", "kolkata", "calcutta", "delhi", "new delhi", "chennai", "madras"},
          "2026-27": {"mumbai", "bombay", "kolkata", "calcutta", "delhi", "new delhi", "chennai", "madras",
                      "hyderabad", "pune", "ahmedabad", "bengaluru", "bangalore"}}
LIMITS = {"80c": 150_000, "80ccd1b": 50_000, "24b": 200_000, "professional_tax": 2_500}
SOURCES = ["https://www.incometax.gov.in/iec/foportal/help/individual/return-applicable-1",
           "https://www.incometax.gov.in/iec/foportal/help/new-tax-vs-old-tax-regime-faqs",
           "https://www.indiabudget.gov.in/budget2025-26/doc/memo.pdf"]


def _slab_tax(income: float, slabs) -> float:
    tax, lower = 0.0, 0.0
    for upper, pct in slabs:
        if income > lower:
            tax += (min(income, upper) - lower) * pct / 100
        lower = upper
    return tax


def _age_band(age: int) -> str:
    return "80_plus" if age >= 80 else "60_to_79" if age >= 60 else "below_60"


def _surcharge(income: float, tax: float, regime: str, slabs) -> tuple[float, float]:
    """Surcharge after marginal relief, and the relief given."""
    rate, threshold = 0, None
    for limit, pct in SURCHARGE[regime]:
        if income > limit:
            rate, threshold = pct, limit
    if not rate:
        return 0.0, 0.0
    full = tax * rate / 100
    lower_rate = max([0] + [p for limit, p in SURCHARGE[regime] if limit < threshold])
    tax_at_threshold = _slab_tax(threshold, slabs)
    cap = tax_at_threshold * (1 + lower_rate / 100) + (income - threshold)
    relief = max(0.0, tax + full - cap)
    return full - relief, relief


def _regime(regime: str, *, salary: float, other_income: float, age: int, hra_exempt: float, basic_da: float,
            employer_nps: float, employer_type: str, ded: dict) -> dict:
    slabs = NEW_SLABS if regime == "new" else OLD_SLABS[_age_band(age)]
    std = min(STANDARD_DEDUCTION[regime], salary)
    nps_cap_pct = 14 if (regime == "new" or employer_type == "government") else 10
    nps = min(employer_nps, basic_da * nps_cap_pct / 100) if employer_nps and basic_da else 0.0
    lines = [("Gross salary", salary), ("Other income", other_income)]
    deductions = [("Standard deduction", std)]
    if nps:
        deductions.append((f"Employer NPS, 80CCD(2) (≤ {nps_cap_pct}% of basic + DA)", nps))
    if regime == "old":
        if hra_exempt:
            deductions.insert(0, ("HRA exemption", hra_exempt))
        deductions += [(name, v) for name, v in ded.items() if v]
    total_ded = sum(v for _, v in deductions)
    taxable = max(0.0, salary + other_income - total_ded)
    slab_tax = _slab_tax(taxable, slabs)
    limit, max_rebate = REBATE[regime]
    if taxable <= limit:
        rebate = min(slab_tax, max_rebate)
    elif regime == "new" and slab_tax > taxable - limit:     # marginal relief just above ₹12 lakh
        rebate = slab_tax - (taxable - limit)
    else:
        rebate = 0.0
    tax_after_rebate = slab_tax - rebate
    surcharge, surcharge_relief = _surcharge(taxable, tax_after_rebate, regime, slabs)
    cess = (tax_after_rebate + surcharge) * CESS_PCT / 100
    total = tax_after_rebate + surcharge + cess
    return {"regime": regime, "income_lines": lines, "deductions": deductions, "total_deductions": round(total_ded, 2),
            "taxable_income": round(taxable, 2), "slab_tax": round(slab_tax, 2), "rebate": round(rebate, 2),
            "surcharge": round(surcharge, 2), "surcharge_marginal_relief": round(surcharge_relief, 2),
            "cess": round(cess, 2), "total_tax": round(total, 2), "monthly_tax": round(total / 12, 2),
            "effective_rate_pct": round(100 * total / (salary + other_income), 2) if salary + other_income else 0.0}


def income_tax(gross_salary: float, financial_year: str = "2026-27", regime: str = "both", age: int = 30,
               other_income: float = 0, hra_received: float = 0, basic_da: float = 0, rent_paid: float = 0,
               city: str = "", section_80c: float = 0, health_insurance_self: float = 0,
               health_insurance_parents: float = 0, parents_senior: bool = False, nps_80ccd1b: float = 0,
               home_loan_interest: float = 0, professional_tax: float = 0, employer_nps: float = 0,
               employer_type: str = "private", gross_salary_monthly: float = 0, hra_received_monthly: float = 0,
               rent_paid_monthly: float = 0, basic_da_monthly: float = 0) -> dict:
    """Annual income tax for a resident salaried individual. Amounts are yearly rupees; *_monthly inputs are × 12."""
    gross_salary = gross_salary or 12 * gross_salary_monthly
    hra_received = hra_received or 12 * hra_received_monthly
    rent_paid = rent_paid or 12 * rent_paid_monthly
    basic_da = basic_da or 12 * basic_da_monthly
    fy = financial_year.replace("FY", "").replace(" ", "").strip()
    if fy not in METROS:
        raise ValueError("financial_year must be '2025-26' or '2026-27'.")
    if regime not in ("new", "old", "both"):
        raise ValueError("regime must be 'new', 'old' or 'both'.")
    if gross_salary <= 0:
        raise ValueError("gross_salary is the yearly gross salary in rupees (₹30 lakh = 3000000).")
    if gross_salary < 100_000 and hra_received > gross_salary:
        raise ValueError("gross_salary looks monthly; pass the yearly amount (monthly × 12).")
    missing, notes = [], []
    # HRA exemption (old regime only): least of actual HRA, rent − 10% of salary, 50%/40% of salary (Rule 2A)
    hra_exempt = 0.0
    if hra_received:
        if not (rent_paid and basic_da):
            missing += [x for x, v in (("rent_paid (yearly rent actually paid)", rent_paid),
                                       ("basic_da (yearly basic salary + DA)", basic_da)) if not v]
            if not city:
                missing.append("city (50% of basic for metros, otherwise 40%)")
            notes.append("HRA exemption not applied in the old regime until rent paid and basic + DA are known.")
        else:
            metro = city.strip().lower() in METROS[fy]
            hra_exempt = max(0.0, min(hra_received, rent_paid - 0.10 * basic_da, (0.5 if metro else 0.4) * basic_da))
            if not city:
                notes.append("City not given: used the 40% (non-metro) HRA limit.")
    senior_self = age >= 60
    ded_old = {
        "Section 80C (≤ ₹1,50,000)": min(section_80c, LIMITS["80c"]),
        "Section 80D, self/family": min(health_insurance_self, 50_000 if senior_self else 25_000),
        "Section 80D, parents": min(health_insurance_parents, 50_000 if parents_senior else 25_000),
        "Section 80CCD(1B), own NPS (≤ ₹50,000)": min(nps_80ccd1b, LIMITS["80ccd1b"]),
        "Home-loan interest, 24(b) (≤ ₹2,00,000)": min(home_loan_interest, LIMITS["24b"]),
        "Professional tax (≤ ₹2,500)": min(professional_tax, LIMITS["professional_tax"]),
    }
    if employer_nps and not basic_da:
        missing.append("basic_da (needed to cap the employer NPS deduction)")
    common = dict(salary=gross_salary, other_income=other_income, age=age, hra_exempt=hra_exempt, basic_da=basic_da,
                  employer_nps=employer_nps, employer_type=employer_type, ded=ded_old)
    results = {r: _regime(r, **common) for r in (("new", "old") if regime == "both" else (regime,))}
    best = min(results.values(), key=lambda x: x["total_tax"])["regime"] if len(results) == 2 else regime
    law = ("Income-tax Act 1961 (FY 2025-26, AY 2026-27)" if fy == "2025-26"
           else "Income-tax Act 2025 (tax year 2026-27): same slabs, rebate and standard deduction as FY 2025-26")
    order = [r for r in ("new", "old") if r in results]
    head = ["Item"] + [("New regime" if r == "new" else "Old regime") for r in order]
    def row(label, key=None, fn=None):
        return [label] + [format_inr(fn(results[r]) if fn else results[r][key], 0) for r in order]
    body = [row("Gross salary", fn=lambda x: x["income_lines"][0][1])]
    if other_income:
        body.append(row("Other income", fn=lambda x: x["income_lines"][1][1]))
    body += [row("Total deductions and exemptions", "total_deductions"), row("Taxable income", "taxable_income"),
             row("Tax on slabs", "slab_tax"), row("Rebate (87A / s.156)", "rebate"), row("Surcharge", "surcharge"),
             row("Health and education cess (4%)", "cess")]
    table = make_table(f"Income tax, FY {fy} (₹ per year)", head, body,
                       total=row("Total tax", "total_tax"), note="; ".join(notes) or None)
    out = {"inputs": {k: v for k, v in dict(gross_salary=gross_salary, financial_year=fy, regime=regime, age=age,
                                              other_income=other_income, hra_received=hra_received, basic_da=basic_da,
                                              rent_paid=rent_paid, city=city, section_80c=section_80c,
                                              health_insurance_self=health_insurance_self,
                                              health_insurance_parents=health_insurance_parents,
                                              nps_80ccd1b=nps_80ccd1b, home_loan_interest=home_loan_interest,
                                              professional_tax=professional_tax, employer_nps=employer_nps).items() if v},
           "formula": ("Taxable income = salary + other income − standard deduction − allowed deductions; tax = slab tax − "
                       "rebate (with marginal relief) + surcharge (with marginal relief); total = tax × 1.04 (4% cess)"),
           "result": {"by_regime": results, "lower_tax_regime": best, "law": law,
                      "saving_with_lower_regime": round(abs(results["new"]["total_tax"] - results["old"]["total_tax"]), 2)
                      if len(results) == 2 else None,
                      "missing_inputs": missing, "notes": notes,
                      "out_of_scope": "Capital gains, crypto and other special-rate income are taxed separately and are not included.",
                      "sources": SOURCES},
           "formatted": {f"{r}_total_tax": format_inr(results[r]["total_tax"]) for r in order},
           "table": table}
    return out
