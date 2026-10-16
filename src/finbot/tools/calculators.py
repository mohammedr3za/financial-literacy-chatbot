"""Deterministic finance calculators. Every function returns
{"inputs": ..., "formula": ..., "result": {...}, "formatted": {...}} and raises ValueError on bad input.
No eval/exec anywhere.
"""
import math

from .fmt import format_inr, group, inr_words, make_table, symbol


def _out(inputs: dict, formula: str, result: dict, money_keys=()) -> dict:
    formatted = {k: format_inr(result[k]) for k in money_keys if result.get(k) is not None}
    formatted.update({f"{k}_short": inr_words(result[k]) for k in money_keys if result.get(k) is not None and abs(result[k]) >= 1e5})
    return {"inputs": inputs, "formula": formula, "result": result, "formatted": formatted}


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def _bisect(f, lo: float, hi: float, tol: float = 1e-12, max_iter: int = 300) -> float:
    flo, fhi = f(lo), f(hi)
    _require(flo * fhi <= 0, "No solution in the search range.")
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        fm = f(mid)
        if abs(fm) < tol or (hi - lo) / 2 < tol:
            return mid
        if flo * fm < 0:
            hi, fhi = mid, fm
        else:
            lo, flo = mid, fm
    return (lo + hi) / 2


# ---------------------------------------------------------------- loan EMI
_LOAN_KEYS = ("principal", "annual_rate_pct", "term", "emi")
_FREQ = {1: "Year", 2: "Half-year", 4: "Quarter", 12: "Month"}


def _pv_payments(emi: float, r: float, n: float) -> float:
    return emi * n if r == 0 else emi * (1 - (1 + r) ** -n) / r


def _loan_term(months, years, m: int) -> int | None:
    """Number of payments from months (monthly count) or years; None if neither is given. Must be whole."""
    months = None if months in (None, 0) else months
    years = None if years in (None, 0) else years
    counts = []
    if months is not None:
        _require(months > 0, "months must be positive.")
        counts.append(months * m / 12)
    if years is not None:
        _require(years > 0, "years must be positive.")
        counts.append(years * m)
    if not counts:
        return None
    _require(len(counts) == 1 or abs(counts[0] - counts[1]) < 1e-9, "months and years disagree; give only one of them.")
    _require(abs(counts[0] - round(counts[0])) < 1e-9, f"The term must be a whole number of payments ({counts[0]:g} with "
                                                      f"{m} payments a year).")
    return int(round(counts[0]))


def loan_emi(principal: float | None = None, annual_rate_pct: float | None = None, months: int | None = None,
             emi: float | None = None, payments_per_year: int = 12, years: float | None = None) -> dict:
    """Reducing-balance loan with level payments, payments_per_year times a year (12 by default). Give exactly three of
    principal, annual_rate_pct, the term (months or years) and emi (the payment per period); the missing one is solved.
    A zero principal, term or payment means "not given" (none of them can really be zero)."""
    _require(payments_per_year in (1, 2, 4, 12), "payments_per_year must be 1 (annual), 2, 4 or 12 (monthly).")
    m = int(payments_per_year)
    vals = {"principal": principal or None, "annual_rate_pct": annual_rate_pct, "term": _loan_term(months, years, m),
            "emi": emi or None}
    given = [k for k in _LOAN_KEYS if vals[k] is not None]
    names = {"term": "months" if months else "years" if years else "months/years"}
    if len(given) == 4:
        raise ValueError("All four of principal, annual_rate_pct, the term and emi were given. Leave out the one to solve for; "
                         "never pass 0 for an unknown value.")
    _require(len(given) == 3, f"Give exactly three of principal, annual_rate_pct, months (or years) and emi; the missing one is "
                              f"solved. Only {', '.join(names.get(k, k) for k in given) or 'none'} given.")
    solve = next(k for k in _LOAN_KEYS if vals[k] is None)
    P, rate, n, pay = vals["principal"], vals["annual_rate_pct"], vals["term"], vals["emi"]
    _require(P is None or P > 0, "principal must be positive (plain rupees, e.g. 20 lakh = 2000000).")
    _require(rate is None or rate >= 0, "annual_rate_pct must be zero or positive.")
    _require(pay is None or pay > 0, "emi (the payment per period) must be positive.")
    per = "month" if m == 12 else _FREQ[m].lower()
    extra, final_payment = {}, None

    if solve == "emi":
        r = rate / (100 * m)
        pay = P / n if r == 0 else P * r * (1 + r) ** n / ((1 + r) ** n - 1)
        formula = (f"Payment = P·r·(1+r)^n / ((1+r)^n − 1), r = annual rate / {m}, n = number of payments")
    elif solve == "principal":
        r = rate / (100 * m)
        P = _pv_payments(pay, r, n)
        extra["principal"] = round(P, 2)
        formula = f"P = payment·(1 − (1+r)^−n) / r, r = annual rate / {m}, n = number of payments (present value of the payments)"
    elif solve == "annual_rate_pct":
        total = pay * n
        _require(total >= P - 0.005, f"The payments ({n} × {format_inr(pay)} = {format_inr(total)}) do not even repay the "
                                     f"principal of {format_inr(P)}, so no rate of zero or more fits.")
        if total - P < 0.005:
            r = 0.0
        else:
            hi = 0.01
            while _pv_payments(pay, hi, n) > P:
                hi *= 2
                _require(hi <= 100, "No realistic interest rate fits these figures.")
            r = _bisect(lambda x: _pv_payments(pay, x, n) - P, 1e-15, hi, tol=1e-14)
        rate = r * 100 * m
        extra.update({"annual_rate_pct": round(rate, 4), f"rate_per_{per}_pct": round(r * 100, 6),
                      "effective_annual_rate_pct": round(((1 + r) ** m - 1) * 100, 4)})
        if m == 12:
            extra["monthly_rate_pct"] = round(r * 100, 6)
        formula = (f"Solve P = payment·(1 − (1+r)^−n) / r for the rate per {per} r (bisection); APR = r × {m}; "
                   f"effective annual rate = (1+r)^{m} − 1")
    else:  # the term
        r = rate / (100 * m)
        _require(pay > P * r + 1e-9, f"The payment of {format_inr(pay)} does not cover the first period's interest of "
                                     f"{format_inr(P * r)}, so the loan would never be repaid.")
        exact = P / pay if r == 0 else -math.log(1 - P * r / pay) / math.log(1 + r)
        n = max(1, math.ceil(exact - 1e-9))
        bal = P
        for _ in range(n - 1):
            bal = bal * (1 + r) - pay
        final_payment = bal * (1 + r)
        extra.update({"payments": n, "payments_exact": round(exact, 4), "years": round(n / m, 2),
                      "final_payment": round(final_payment, 2)})
        if m == 12:
            extra.update({"months": n, "months_exact": round(exact, 4)})
        formula = ("n = −ln(1 − P·r/payment) / ln(1+r), r = annual rate / payments per year; rounded up to whole payments, "
                   "the last one smaller (final_payment)")

    # schedule with the exact unrounded payment; the last one is smaller when the term was rounded up
    def paid(k: int) -> float:
        return final_payment if final_payment is not None and k == n - 1 else pay

    yi = yp = 0.0
    body, b, t_paid, t_int, t_prin = [], P, 0.0, 0.0, 0.0
    if m == 12:   # year-wise view of a monthly loan
        year = 0
        for start in range(0, n, 12):
            year += 1
            opening, paid_y, interest_y, prin_y = b, 0.0, 0.0, 0.0
            for k in range(start, min(start + 12, n)):
                i_amt, amt = b * r, paid(k)
                paid_y, interest_y, prin_y, b = paid_y + amt, interest_y + i_amt, prin_y + amt - i_amt, b - (amt - i_amt)
            if year == 1:
                yi, yp = interest_y, prin_y
            t_paid, t_int, t_prin = t_paid + paid_y, t_int + interest_y, t_prin + prin_y
            body.append([str(year), group(opening), group(paid_y), group(interest_y), group(prin_y), group(b if abs(b) >= 0.5 else 0)])
        table = make_table(f"Year-wise loan schedule ({symbol()})", ["Year", "Opening balance", "EMI paid", "Interest", "Principal",
                                                                     "Closing balance"],
                           body, total=["Total", "", group(t_paid), group(t_int), group(t_prin), ""])
    else:         # one row per payment, 2 decimals (textbook amortization schedule)
        for k in range(n):
            opening, i_amt, amt = b, b * r, paid(k)
            b = b - (amt - i_amt)
            if k < m:
                yi, yp = yi + i_amt, yp + amt - i_amt
            t_paid, t_int, t_prin = t_paid + amt, t_int + i_amt, t_prin + amt - i_amt
            body.append([str(k + 1), group(opening, 2), group(amt, 2), group(i_amt, 2), group(amt - i_amt, 2),
                         group(b if abs(b) >= 0.005 else 0, 2)])
        table = make_table(f"Amortization schedule ({symbol()})", [_FREQ[m], "Beginning balance", "Total payment", "Interest payment",
                                                                   "Principal payment", "Ending balance"],
                           body, total=["Total", "", group(t_paid, 2), group(t_int, 2), group(t_prin, 2), ""],
                           note="Computed with the exact unrounded payment.")
    total = pay * n if final_payment is None else pay * (n - 1) + final_payment
    result = {"emi": round(pay, 2), "total_payment": round(total, 2), "total_interest": round(total - P, 2),
              "first_year_interest": round(yi, 2), "first_year_principal": round(yp, 2),
              "solved_for": ("months" if m == 12 else "years") if solve == "term" else solve,
              "payments_per_year": m, **extra}
    if m != 12:
        result["payment"] = round(pay, 2)
        result["interest_by_period"] = [round(float(row[3].replace(",", "")), 2) for row in body] if n <= 40 else None
    inputs = {names.get(k, k): (months if k == "term" and months else years if k == "term" else vals[k]) for k in given}
    if m != 12:
        inputs["payments_per_year"] = m
    out = _out(inputs, formula, result,
               money_keys=tuple(k for k in ("emi", "total_payment", "total_interest", "first_year_interest",
                                            "first_year_principal", "principal", "final_payment") if k in result))
    out["table"] = table
    return out


# ---------------------------------------------------------------- SIP
def sip_future_value(monthly: float, annual_return_pct: float, years: float, step_up_pct: float = 0,
                     timing: str = "start") -> dict:
    _require(monthly > 0, "monthly must be positive (plain rupees).")
    _require(timing in ("start", "end"), "timing must be 'start' (SIP, start of each month) or 'end' (ordinary annuity).")
    start = timing == "start"
    _require(years > 0 and abs(years * 12 - round(years * 12)) < 1e-9, "years must be positive and a whole number of months.")
    _require(step_up_pct >= 0, "step_up_pct must be zero or positive.")
    i = annual_return_pct / 1200
    n = round(years * 12)
    bal, invested, pay = 0.0, 0.0, monthly
    body, year_inv = [], 0.0
    for m in range(n):
        if m and m % 12 == 0:
            pay *= 1 + step_up_pct / 100
        bal = (bal + pay) * (1 + i) if start else bal * (1 + i) + pay   # start: deposit, then a month of growth
        invested += pay
        year_inv += pay
        if (m + 1) % 12 == 0 or m == n - 1:
            body.append([str((m // 12) + 1), group(pay), group(year_inv), group(invested), group(bal), group(bal - invested)])
            year_inv = 0.0
    table = make_table(f"Year-wise SIP growth ({symbol()})", ["Year", "Monthly SIP", "Invested in year", "Total invested",
                                                    "Value at year end", "Gain"], body,
                       note="Assumes the stated return every year; actual returns vary and are not guaranteed.")
    out = _out(
        {"monthly": monthly, "annual_return_pct": annual_return_pct, "years": years, "step_up_pct": step_up_pct, "timing": timing},
        ("FV = P·((1+i)^n − 1)/i·(1+i), i = annual return / 12, n = months (deposit at start of month); " if start else
         "FV = P·((1+i)^n − 1)/i, i = annual return / 12, n = months (deposit at end of month, ordinary annuity); ")
        + 
        "with step-up, the monthly amount rises by step_up_pct each year",
        {"corpus": round(bal, 2), "invested": round(invested, 2), "gain": round(bal - invested, 2)},
        money_keys=("corpus", "invested", "gain"),
    )
    out["table"] = table
    return out


# ---------------------------------------------------------------- TVM
_TVM_TARGETS = ("fv", "pv", "pmt", "years", "annual_rate_pct")


def _tvm_fv(rate_per: float, n: float, pv: float, pmt: float, due: int) -> float:
    if rate_per == 0:
        return pv + pmt * n
    try:
        g = (1 + rate_per) ** n
        return pv * g + pmt * (1 + rate_per * due) * (g - 1) / rate_per
    except OverflowError:
        return math.inf


def _tvm_table(rate_pct: float, years: float, pv: float, pmt: float, m: int, due: int, flow_label: str = "Deposits") -> dict | None:
    if years is None or years <= 0 or years > 200:
        return None
    i, n = rate_pct / 100 / m, int(round(years * m))
    body, b, t_dep, t_growth = [], pv, 0.0, 0.0
    opening, dep_y = b, 0.0
    for k in range(n):
        b = (b + pmt) * (1 + i) if due else b * (1 + i) + pmt
        dep_y += pmt
        if (k + 1) % m == 0 or k == n - 1:
            growth = b - opening - dep_y
            body.append([str(k // m + 1), group(opening, 2), group(dep_y, 2), group(growth, 2), group(b, 2)])
            t_dep, t_growth, opening, dep_y = t_dep + dep_y, t_growth + growth, b, 0.0
    return make_table(f"Year-wise growth ({symbol()})", ["Year", "Opening", flow_label, "Growth", "Closing"], body,
                      total=["Total", group(pv, 2), group(t_dep, 2), group(t_growth, 2), group(b, 2)])


def _annuity_pv_table(rate_pct: float, years: float, pmt: float, m: int, due: int) -> dict | None:
    if years is None or years <= 0 or years * m > 600:
        return None
    i, n = rate_pct / 100 / m, int(round(years * m))
    body, total_pv = [], 0.0
    for k in range(n):
        t = k if due else k + 1
        df = 1 / (1 + i) ** t
        total_pv += pmt * df
        body.append([str(t), group(pmt, 2), f"{df:.6f}", group(pmt * df, 2)])
    return make_table(f"Present value of each payment ({symbol()}, {'annuity due' if due else 'ordinary annuity'})",
                      ["Period", "Payment", "Discount factor", "Present value"], body,
                      total=["Total", group(pmt * n, 2), "", group(total_pv, 2)])


def _annuity_values(pmt: float, i: float, n: float) -> dict:
    a = n if i == 0 else (1 - (1 + i) ** -n) / i
    s = n if i == 0 else ((1 + i) ** n - 1) / i
    return {"pv_ordinary": round(pmt * a, 2), "pv_due": round(pmt * a * (1 + i), 2),
            "fv_ordinary": round(pmt * s, 2), "fv_due": round(pmt * s * (1 + i), 2)}


def tvm(*args, **kwargs) -> dict:
    out = _tvm_solve(*args, **kwargs)
    kind, exact_pmt = out.pop("_kind", None), out.pop("_exact_pmt", None)
    inp, res = out["inputs"], out["result"]
    rate = res.get("annual_rate_pct", inp["annual_rate_pct"])
    years = res.get("years", inp["years"])
    m, due = inp["periods_per_year"], 1 if inp["when"] == "begin" else 0
    if kind == "annuity_pv":
        table = _annuity_pv_table(rate, years, inp["pmt"], m, due)
    elif kind == "withdrawal":
        table = _tvm_table(rate, years, inp["pv"], -exact_pmt, m, due, "Withdrawals")   # exact, so it ends at 0
    elif rate is not None:
        table = _tvm_table(rate, years, res.get("pv", inp["pv"]), res.get("pmt", inp["pmt"]), m, due)
    else:
        table = None
    if table:
        out["table"] = table
    return out


def _tvm_solve(solve_for: str, annual_rate_pct: float | None = None, years: float | None = None,
        pv: float = 0.0, pmt: float = 0.0, fv: float | None = None,
        periods_per_year: int = 1, when: str = "end") -> dict:
    _require(solve_for in _TVM_TARGETS, f"solve_for must be one of {_TVM_TARGETS}.")
    _require(int(periods_per_year) == periods_per_year and periods_per_year >= 1, "periods_per_year must be a positive whole number.")
    _require(when in ("end", "begin"), "when must be 'end' or 'begin'.")
    _require(pv >= 0 and pmt >= 0 and (fv is None or fv >= 0), "Use positive amounts: pv = starting amount, pmt = deposit per period, fv = target.")
    m, due = int(periods_per_year), 1 if when == "begin" else 0
    inputs = {"solve_for": solve_for, "annual_rate_pct": annual_rate_pct, "years": years, "pv": pv, "pmt": pmt,
              "fv": fv, "periods_per_year": m, "when": when}
    formula = "FV = PV·(1+i)^n + PMT·(1+i·d)·((1+i)^n − 1)/i, i = annual rate / periods_per_year, n = years × periods_per_year, d = 1 if paid at period start"

    annuity_pv = solve_for == "pv" and pmt > 0 and not fv      # value today of receiving pmt each period
    withdrawal = solve_for == "pmt" and pv > 0 and not fv       # level withdrawal that uses up pv
    if solve_for != "annual_rate_pct":
        _require(annual_rate_pct is not None, "annual_rate_pct is required.")
    if solve_for != "years":
        _require(years is not None and years > 0, "years is required and must be positive.")
    if solve_for not in ("fv",) and not (annuity_pv or withdrawal):
        _require(fv is not None, "fv (the target amount) is required. For the present value of a series of payments "
                                 "(an annuity) give pmt and leave fv out; for a withdrawal from a balance give pv and leave fv out.")

    i = (annual_rate_pct or 0) / 100 / m
    n = (years or 0) * m
    plain = pmt > 0 and pv == 0 and not fv

    if solve_for == "fv":
        val = _tvm_fv(i, n, pv, pmt, due)
        res = {"fv": round(val, 2)}
        if plain:
            res["annuity_values"] = _annuity_values(pmt, i, n)
        return _out(inputs, formula, res, ("fv",))
    if annuity_pv:
        val = _tvm_fv(i, n, 0, pmt, due) / (1 + i) ** n
        out = _out(inputs, "PV = PMT·(1 − (1+i)^−n)/i·(1+i·d), i = annual rate / periods_per_year, n = years × "
                   "periods_per_year, d = 1 for an annuity due (payments at the start)",
                   {"pv": round(val, 2), "annuity": "annuity due" if due else "ordinary annuity",
                    "annuity_values": _annuity_values(pmt, i, n)}, ("pv",))
        out["_kind"] = "annuity_pv"
        return out
    if solve_for == "pv":
        val = (fv - _tvm_fv(i, n, 0, pmt, due)) / (1 + i) ** n
        _require(val >= 0, "The deposits alone already exceed the target; no starting amount is needed. (pmt and fv are "
                           "both treated as savings toward fv. For the value today of receiving pmt each period, leave fv out.)")
        return _out(inputs, formula, {"pv": round(val, 2)}, ("pv",))
    if withdrawal:
        unit_pv = _tvm_fv(i, n, 0, 1, due) / (1 + i) ** n
        val = pv / unit_pv
        out = _out(inputs, "Withdrawal = PV ÷ [(1 − (1+i)^−n)/i·(1+i·d)], i = annual rate / periods_per_year, "
                   "n = years × periods_per_year (the balance is used up)", {"pmt": round(val, 2), "withdrawal": round(val, 2)},
                   ("withdrawal",))
        out["_kind"], out["_exact_pmt"] = "withdrawal", val
        return out
    if solve_for == "pmt":
        unit = _tvm_fv(i, n, 0, 1, due)
        val = (fv - pv * (1 + i) ** n) / unit
        _require(val >= 0, "The starting amount alone already reaches the target; no deposits are needed. (For a level "
                           "withdrawal that uses up pv, leave fv out.)")
        return _out(inputs, formula, {"pmt": round(val, 2)}, ("pmt",))
    if solve_for == "years":
        _require(fv > pv, "fv must be larger than pv to solve for years.")
        _require(i > 0 or pmt > 0, "With a 0% rate and no deposits the target is never reached.")
        hi = 1.0
        while _tvm_fv(i, hi, pv, pmt, due) < fv:
            hi *= 2
            _require(hi <= 2000 * m, "The target is not reached within 1,000 years.")
        n_per = _bisect(lambda k: _tvm_fv(i, k, pv, pmt, due) - fv, 0.0, hi)
        return _out(inputs, formula, {"years": round(n_per / m, 6)})
    # annual_rate_pct
    _require(fv > 0, "fv must be positive to solve for the rate.")
    _require(pv > 0 or pmt > 0, "Provide pv and/or pmt to solve for the rate.")
    f = lambda rp: _tvm_fv(rp, n, pv, pmt, due) - fv  # noqa: E731
    hi = 0.01
    while f(hi) < 0:
        hi *= 2
        _require(hi <= 100, "No realistic growth rate reaches this target.")
    rp = _bisect(f, -0.9999, hi)
    return _out(inputs, formula, {"annual_rate_pct": round(rp * m * 100, 6)})


# ---------------------------------------------------------------- NPV / IRR
def _payback(flows: list[float]) -> float | None:
    """Years until the cumulative flow turns non-negative: the year before, plus the unrecovered amount ÷ that year's flow."""
    cum = flows[0]
    if cum >= 0:
        return 0.0
    for t in range(1, len(flows)):
        if flows[t] > 0 and cum + flows[t] >= 0:
            return round(t - 1 + (-cum) / flows[t], 6)
        cum += flows[t]
    return None


def npv_irr(cashflows: list[float], discount_rate_pct: float | None = None) -> dict:
    _require(isinstance(cashflows, list) and len(cashflows) >= 2, "cashflows needs at least two values; cashflows[0] is time 0.")
    cf = [float(c) for c in cashflows]
    npv_at = lambda rate: sum(c / (1 + rate) ** t for t, c in enumerate(cf))  # noqa: E731
    result = {"npv": None, "irr_pct": None}
    if discount_rate_pct is not None:
        r = discount_rate_pct / 100
        npv = npv_at(r)
        result["npv"] = round(npv, 4)
        n = len(cf) - 1
        if cf[0] < 0:   # profitability index = PV of the later cash flows / initial investment
            pv_future = npv - cf[0]
            result["pv_of_future_cash_flows"] = round(pv_future, 4)
            result["profitability_index"] = round(pv_future / -cf[0], 6)
        # equivalent annual amount (EAC when the flows are costs): the level annuity over n years with the same NPV
        annuity_factor = n if r == 0 else (1 - (1 + r) ** -n) / r
        result["annuity_factor"] = round(annuity_factor, 6)
        result["equivalent_annual_amount"] = round(npv / annuity_factor, 4)
    has_sign_change = any(a * b < 0 for a, b in zip(cf, cf[1:]) if a and b) or (min(cf) < 0 < max(cf))
    note = None
    if has_sign_change:
        grid = [-0.99 + k * 0.01 for k in range(0, 1100)]  # -99% .. +1000%
        roots = []
        for a, b in zip(grid, grid[1:]):
            if npv_at(a) * npv_at(b) <= 0:
                roots.append(_bisect(npv_at, a, b))
        if roots:
            result["irr_pct"] = round(roots[0] * 100, 4)
            if len(roots) > 1:
                note = f"Multiple IRRs exist: {[round(r * 100, 4) for r in roots]}%. IRR is unreliable here; use NPV."
    else:
        note = "IRR is undefined because the cash flows never change sign."
    result["payback_years"] = _payback(cf)
    if discount_rate_pct is not None:
        result["discounted_payback_years"] = _payback([c / (1 + discount_rate_pct / 100) ** t for t, c in enumerate(cf)])
    if result["payback_years"] is None:
        note = (note + " " if note else "") + "The investment is never paid back within the cash flows given."
    elif result.get("discounted_payback_years", 0) is None:
        note = (note + " " if note else "") + "On a discounted basis the investment is never paid back within the cash flows given."
    if note:
        result["note"] = note
    out = _out({"cashflows": cf, "discount_rate_pct": discount_rate_pct},
               ("NPV = Σ CF_t/(1+r)^t ; IRR is the r where NPV = 0 ; PI = (NPV − CF_0)/(−CF_0) ; "
                "equivalent annual amount = NPV / ((1 − (1+r)^−n)/r) ; payback = years before the cumulative (discounted) "
                "flow turns non-negative + unrecovered amount ÷ that year's (discounted) flow"), result, ("npv",))
    rate = discount_rate_pct if discount_rate_pct is not None else result["irr_pct"]
    body, cum, cum_pv = [], 0.0, 0.0
    for t, c in enumerate(cf):
        cum += c
        if rate is None:
            body.append([str(t), group(c, 2), group(cum, 2), "–", "–", "–"])
            continue
        df = 1 / (1 + rate / 100) ** t
        cum_pv += c * df
        body.append([str(t), group(c, 2), group(cum, 2), f"{df:.4f}", group(c * df, 2), group(cum_pv, 2)])
    label = (f"discounted at {discount_rate_pct}%" if discount_rate_pct is not None
             else f"discounted at the IRR, {result['irr_pct']}%" if result["irr_pct"] is not None else "undiscounted")
    out["table"] = make_table(f"Cash flows by year ({symbol()}, {label})",
                              ["Year", "Cash flow", "Cumulative cash flow", "Discount factor", "Present value", "Cumulative PV"], body)
    return out


# ---------------------------------------------------------------- equivalent annual cost
def _given(v) -> bool:
    return v is not None and v != 0


def equivalent_annual_cost(purchase_cost: float, annual_cost: float | None = None, life_years: float | None = None,
                           rate_pct: float | None = None, salvage: float = 0.0, power_watts: float | None = None,
                           hours_per_year: float | None = None, price_per_kwh: float | None = None,
                           lifetime_hours: float | None = None) -> dict:
    """EAC of an asset bought now and run at a level yearly cost (paid at each year end) for a whole number of years.
    Costs are magnitudes; following the textbook, they are shown as negative cash flows, so the EAC is negative.
    For electricity, power_watts, hours_per_year and price_per_kwh give the yearly cost and lifetime_hours ÷ hours_per_year
    the life, so the model only copies figures from the question. A zero means "not given"."""
    _require(rate_pct is not None and rate_pct > -100, "rate_pct must be given and above -100 (e.g. 10 for 10%).")
    _require(salvage is None or salvage >= 0, "salvage must be zero or positive (the resale value at the end of the life).")
    working = []
    usage = {"power_watts": power_watts, "hours_per_year": hours_per_year, "price_per_kwh": price_per_kwh}
    if any(_given(usage[k]) for k in ("power_watts", "price_per_kwh")):
        missing = [k for k, v in usage.items() if not _given(v)]
        _require(not missing, f"For an electricity cost give power_watts, hours_per_year and price_per_kwh; missing: {', '.join(missing)}.")
        derived_cost = abs(power_watts) / 1000 * abs(hours_per_year) * abs(price_per_kwh)
        working.append(f"Yearly running cost = {group(abs(power_watts), 2).removesuffix('.00')} ÷ 1000 × "
                       f"{group(abs(hours_per_year), 2).removesuffix('.00')} × {abs(price_per_kwh):g} = {derived_cost:.6g}")
        if _given(annual_cost):
            _require(abs(abs(annual_cost) - derived_cost) <= 0.005 * derived_cost,
                     f"annual_cost={annual_cost:g} does not match power_watts ÷ 1000 × hours_per_year × price_per_kwh = "
                     f"{derived_cost:.6g}. Pass one or the other.")
        annual_cost = derived_cost
    _require(_given(annual_cost), "annual_cost (the yearly running cost) is missing. Give it, or for electricity give "
                                  "power_watts, hours_per_year and price_per_kwh from the question.")
    if _given(lifetime_hours) and not _given(life_years):
        _require(_given(hours_per_year), "To get the life from lifetime_hours, also give hours_per_year.")
        life_years = abs(lifetime_hours) / abs(hours_per_year)
        working.append(f"Life = {group(abs(lifetime_hours))} ÷ {group(abs(hours_per_year))} = {life_years:g} years")
    _require(_given(life_years) and life_years > 0, "life_years is missing. Give it, or lifetime_hours and hours_per_year "
                                                    "(e.g. 16,000 hours ÷ 2,000 hours a year = 8 years).")
    _require(abs(life_years - round(life_years)) < 1e-9, f"life_years must be a whole number of years (got {life_years:g}).")
    purchase, yearly, n, r = abs(float(purchase_cost)), abs(float(annual_cost)), int(round(life_years)), rate_pct / 100
    salvage = float(salvage or 0.0)
    annuity_factor = float(n) if r == 0 else (1 - (1 + r) ** -n) / r
    flows = [-purchase] + [-yearly] * n
    flows[-1] += salvage
    pv = sum(c / (1 + r) ** t for t, c in enumerate(flows))
    eac = pv / annuity_factor
    body, cum = [], 0.0
    for t, c in enumerate(flows):
        df = 1 / (1 + r) ** t
        cum += c * df
        body.append([str(t), group(c, 2), f"{df:.4f}", group(c * df, 2), group(cum, 2)])
    note = " ".join(w + "." for w in working)
    minus = lambda v: f"{v:,.4f}".replace("-", "−")  # noqa: E731
    note += (f" EAC = PV of costs ÷ annuity factor = {minus(pv)} ÷ {annuity_factor:.4f} = {minus(eac)} "
             "(costs are negative).")
    table = make_table(f"Equivalent annual cost ({symbol()}, discounted at {rate_pct:g}%)",
                       ["Year", "Cash flow", "Discount factor", "Present value", "Cumulative PV"], body,
                       total=["Total", group(sum(flows), 2), "", group(pv, 2), ""], note=note.strip())
    inputs = {"purchase_cost": purchase, "annual_cost": yearly, "life_years": n, "rate_pct": rate_pct, "salvage": salvage}
    inputs.update({k: v for k, v in (("power_watts", power_watts), ("hours_per_year", hours_per_year),
                                     ("price_per_kwh", price_per_kwh), ("lifetime_hours", lifetime_hours)) if _given(v)})
    out = _out(inputs,
               "PV of costs = −purchase − annual cost × annuity factor + salvage ÷ (1+r)^n ; "
               "annuity factor = (1 − (1+r)^−n) ÷ r ; EAC = PV of costs ÷ annuity factor (costs negative)",
               {"annual_cost": round(yearly, 6), "life_years": n, "pv_of_costs": round(pv, 4),
                "annuity_factor": round(annuity_factor, 6), "eac": round(eac, 4)},
               ("pv_of_costs", "eac"))
    if working:
        out["working"] = working
    out["table"] = table
    return out


# ---------------------------------------------------------------- bonds
def _bond_price_dur(face, coupon_pct, n_periods, freq, y):
    c = face * coupon_pct / 100 / freq
    yp = y / freq
    pv_flows = [(c + (face if t == n_periods else 0)) / (1 + yp) ** t for t in range(1, n_periods + 1)]
    price = sum(pv_flows)
    mac = sum((t / freq) * v for t, v in zip(range(1, n_periods + 1), pv_flows)) / price
    return price, mac, mac / (1 + yp)


def bond_metrics(face: float, coupon_pct: float, years: float, freq: int = 1,
                 ytm_pct: float | None = None, price: float | None = None) -> dict:
    _require((ytm_pct is None) != (price is None), "Give exactly one of ytm_pct or price.")
    _require(face > 0 and coupon_pct >= 0 and years > 0, "face and years must be positive; coupon_pct zero or positive.")
    _require(freq in (1, 2, 4, 12), "freq must be 1, 2, 4 or 12 coupons per year.")
    n = years * freq
    _require(abs(n - round(n)) < 1e-9, "years × freq must be a whole number of coupon periods.")
    n = int(round(n))
    if price is None:
        y = ytm_pct / 100
        p, mac, mod = _bond_price_dur(face, coupon_pct, n, freq, y)
    else:
        _require(price > 0, "price must be positive.")
        y = _bisect(lambda yy: _bond_price_dur(face, coupon_pct, n, freq, yy)[0] - price, -0.5, 5.0)
        p, mac, mod = _bond_price_dur(face, coupon_pct, n, freq, y)
    c, yp = face * coupon_pct / 100 / freq, y / freq
    body, t_cf, t_pv, t_tpv = [], 0.0, 0.0, 0.0
    for t in range(1, n + 1):
        cf = c + (face if t == n else 0)
        df = 1 / (1 + yp) ** t
        pv_t = cf * df
        t_cf, t_pv, t_tpv = t_cf + cf, t_pv + pv_t, t_tpv + (t / freq) * pv_t
        body.append([str(t), f"{t / freq:g}", group(cf, 2), f"{df:.4f}", group(pv_t, 2), group((t / freq) * pv_t, 2)])
    table = make_table("Bond cash flows and duration", ["Period", "Time (years)", "Cash flow", "Discount factor",
                                                        "Present value", "Time × PV"], body,
                       total=["Total", "", group(t_cf, 2), "", group(t_pv, 2), group(t_tpv, 2)],
                       note="Price = total present value. Macaulay duration = total Time × PV ÷ Price.")
    out = _out(
        {"face": face, "coupon_pct": coupon_pct, "years": years, "freq": freq, "ytm_pct": ytm_pct, "price": price},
        "Price = Σ C/(1+y/f)^t + F/(1+y/f)^N ; Macaulay D = Σ (t/f)·PV_t / Price ; Modified D = D/(1+y/f)",
        {"price": round(p, 4), "ytm_pct": round(y * 100, 6), "current_yield_pct": round(face * coupon_pct / p, 4),
         "macaulay_duration": round(mac, 4), "modified_duration": round(mod, 4)},
        ("price",),
    )
    out["table"] = table
    return out


# ---------------------------------------------------------------- Black-Scholes
def _ncdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _npdf(x: float) -> float:
    return math.exp(-x * x / 2) / math.sqrt(2 * math.pi)


def black_scholes(spot: float, strike: float, years: float, rate_pct: float, vol_pct: float, kind: str = "call") -> dict:
    _require(kind in ("call", "put"), "kind must be 'call' or 'put'.")
    _require(spot > 0 and strike > 0 and years > 0 and vol_pct > 0, "spot, strike, years and vol_pct must be positive.")
    r, s, T = rate_pct / 100, vol_pct / 100, years
    d1 = (math.log(spot / strike) + (r + s * s / 2) * T) / (s * math.sqrt(T))
    d2 = d1 - s * math.sqrt(T)
    disc = math.exp(-r * T)
    if kind == "call":
        price = spot * _ncdf(d1) - strike * disc * _ncdf(d2)
        delta = _ncdf(d1)
        theta = -spot * _npdf(d1) * s / (2 * math.sqrt(T)) - r * strike * disc * _ncdf(d2)
        rho = strike * T * disc * _ncdf(d2) / 100
    else:
        price = strike * disc * _ncdf(-d2) - spot * _ncdf(-d1)
        delta = _ncdf(d1) - 1
        theta = -spot * _npdf(d1) * s / (2 * math.sqrt(T)) + r * strike * disc * _ncdf(-d2)
        rho = -strike * T * disc * _ncdf(-d2) / 100
    gamma = _npdf(d1) / (spot * s * math.sqrt(T))
    vega = spot * _npdf(d1) * math.sqrt(T) / 100
    table = make_table(f"Black-Scholes {kind}: price and Greeks", ["Measure", "Value", "Meaning"], [
        ["Price", f"{price:.4f}", "Fair value of the option"],
        ["Delta", f"{delta:.4f}", "Change in option price per 1 unit move in the spot price"],
        ["Gamma", f"{gamma:.6f}", "Change in delta per 1 unit move in the spot price"],
        ["Vega", f"{vega:.4f}", "Change in price per 1 percentage point of volatility"],
        ["Theta", f"{theta / 365:.4f}", "Change in price per day as expiry approaches"],
        ["Rho", f"{rho:.4f}", "Change in price per 1 percentage point of interest rate"],
    ])
    out = _out(
        {"spot": spot, "strike": strike, "years": years, "rate_pct": rate_pct, "vol_pct": vol_pct, "kind": kind},
        "d1 = (ln(S/K) + (r + σ²/2)T)/(σ√T), d2 = d1 − σ√T ; Call = S·N(d1) − K·e^(−rT)·N(d2) ; Put = K·e^(−rT)·N(−d2) − S·N(−d1)",
        {"price": round(price, 4), "delta": round(delta, 4), "gamma": round(gamma, 6), "vega_per_1pct": round(vega, 4),
         "theta_per_year": round(theta, 4), "theta_per_day": round(theta / 365, 4), "rho_per_1pct": round(rho, 4),
         "d1": round(d1, 4), "d2": round(d2, 4)},
    )
    out["table"] = table
    return out


# ---------------------------------------------------------------- goal-based saving (SPEC 4.0)
def _sip_value(monthly: float, i: float, n: int, step_up_pct: float, start: bool) -> tuple[float, list]:
    """Simulate a monthly SIP; returns final value and per-year rows (year, monthly, invested_year, invested_total, value)."""
    bal, invested, pay, rows, year_inv = 0.0, 0.0, monthly, [], 0.0
    for m in range(n):
        if m and m % 12 == 0:
            pay *= 1 + step_up_pct / 100
        bal = (bal + pay) * (1 + i) if start else bal * (1 + i) + pay
        invested += pay
        year_inv += pay
        if (m + 1) % 12 == 0 or m == n - 1:
            rows.append((m // 12 + 1, pay, year_inv, invested, bal))
            year_inv = 0.0
    return bal, rows


def goal_saving(target: float, years: float, annual_return_pct: float, current_savings: float = 0.0,
                step_up_pct: float = 0.0, timing: str = "start") -> dict:
    _require(target > 0, "target must be positive: the future amount to reach, in plain rupees (1 crore = 10000000).")
    _require(years > 0 and abs(years * 12 - round(years * 12)) < 1e-9, "years must be positive and a whole number of months.")
    _require(annual_return_pct > -100, "annual_return_pct must be above -100.")
    _require(current_savings >= 0 and step_up_pct >= 0, "current_savings and step_up_pct must be zero or positive.")
    _require(timing in ("start", "end"), "timing must be 'start' (SIP, default) or 'end' of each month.")
    i, n, start = annual_return_pct / 1200, int(round(years * 12)), timing == "start"
    grown_savings = current_savings * (1 + i) ** n
    remaining = max(0.0, target - grown_savings)
    unit, _ = _sip_value(1.0, i, n, step_up_pct, start)          # value of ₹1/month; the answer is linear in the amount
    monthly = remaining / unit if remaining > 0 else 0.0
    final, rows = _sip_value(monthly, i, n, step_up_pct, start)
    check = final + grown_savings
    _require(abs(check - target) <= max(1.0, target * 1e-9) or remaining == 0, "Internal check failed: the plan does not reach the target.")
    lump_today = target / (1 + i) ** n
    body = [[str(y), group(p), group(iy), group(it), group(v + current_savings * (1 + i) ** (y * 12 if y * 12 <= n else n)),
             group(v + current_savings * (1 + i) ** min(y * 12, n) - it - current_savings)] for y, p, iy, it, v in rows]
    table = make_table(f"Year-wise plan to reach the goal ({symbol()})", ["Year", "Monthly SIP", "Invested in year", "Total invested",
                                                                "Value at year end", "Gain"], body,
                       note="Assumes the stated return every year; actual returns vary and are not guaranteed.")
    out = _out(
        {"target": target, "years": years, "annual_return_pct": annual_return_pct, "current_savings": current_savings,
         "step_up_pct": step_up_pct, "timing": timing},
        "Monthly SIP = (Target − Savings·(1+i)^n) ÷ [((1+i)^n − 1)/i · (1+i if paid at month start)], i = annual return / 12, "
        "n = months; with step-up, solved on the simulated schedule",
        {"monthly_required": round(monthly, 2), "total_invested": round(rows[-1][3], 2) if rows else 0.0,
         "check_target_reached": round(check, 2), "lump_sum_needed_today_instead": round(lump_today, 2),
         "current_savings_grows_to": round(grown_savings, 2)},
        money_keys=("monthly_required", "total_invested", "check_target_reached", "lump_sum_needed_today_instead"),
    )
    out["table"] = table
    return out


def real_return(nominal_pct: float, inflation_pct: float) -> dict:
    _require(nominal_pct > -100 and inflation_pct > -100, "Rates must be above -100%.")
    real = ((1 + nominal_pct / 100) / (1 + inflation_pct / 100) - 1) * 100
    return _out({"nominal_pct": nominal_pct, "inflation_pct": inflation_pct},
                "Real return = (1 + nominal) ÷ (1 + inflation) − 1  (Fisher equation)",
                {"real_return_pct": round(real, 4), "approximation_pct": round(nominal_pct - inflation_pct, 4)})


def effective_annual_rate(nominal_pct: float, periods_per_year: int) -> dict:
    _require(int(periods_per_year) == periods_per_year and periods_per_year >= 1, "periods_per_year must be a whole number ≥ 1.")
    ear = ((1 + nominal_pct / 100 / periods_per_year) ** periods_per_year - 1) * 100
    return _out({"nominal_pct": nominal_pct, "periods_per_year": periods_per_year},
                "EAR = (1 + nominal ÷ m)^m − 1", {"effective_pct": round(ear, 4)})


# ---------------------------------------------------------------- perpetuity (SPEC 4)
def perpetuity(cash_flow: float, rate_pct: float, growth_pct: float = 0.0, investment: float | None = None,
               starts_in_year: int = 1) -> dict:
    """Level or growing perpetuity whose first cash flow comes at the end of year starts_in_year."""
    _require(cash_flow > 0, "cash_flow (the first yearly cash flow) must be positive.")
    _require(rate_pct > growth_pct, "rate_pct must be greater than growth_pct, or the perpetuity has no finite value.")
    _require(int(starts_in_year) == starts_in_year and starts_in_year >= 1, "starts_in_year must be a whole number of 1 or more.")
    r, g, t = rate_pct / 100, growth_pct / 100, int(starts_in_year)
    value_before_first = cash_flow / (r - g)
    pv = value_before_first / (1 + r) ** (t - 1)
    working = [f"Value one year before the first cash flow = C ÷ (r − g) = {cash_flow:,.2f} ÷ ({r:g} − {g:g}) = {value_before_first:,.2f}"]
    if t > 1:
        working.append(f"PV today = {value_before_first:,.2f} ÷ (1 + {r:g})^{t - 1} = {pv:,.2f}")
    result = {"pv": round(pv, 2), "value_one_year_before_first_flow": round(value_before_first, 2)}
    if investment:
        _require(investment > 0, "investment must be positive (the cost today).")
        result["npv"] = round(pv - investment, 2)
        g_star = r - cash_flow / (investment * (1 + r) ** (t - 1))
        result["break_even_growth_pct"] = round(g_star * 100, 6)
        result["irr_pct"] = round((cash_flow / investment + g) * 100, 6) if t == 1 else None
        working.append(f"NPV = PV − investment = {pv:,.2f} − {investment:,.2f} = {pv - investment:,.2f}")
        working.append(f"Break-even growth g* = r − C ÷ (I·(1+r)^{t - 1}) = {g_star * 100:.4f}%")
    out = _out({"cash_flow": cash_flow, "rate_pct": rate_pct, "growth_pct": growth_pct, "investment": investment,
                "starts_in_year": t},
               "PV = C ÷ (r − g), discounted (starts_in_year − 1) years ; NPV = PV − investment ; IRR = C ÷ I + g ; "
               "break-even growth g* = r − C ÷ I (first flow in year 1)", result,
               tuple(k for k in ("pv", "npv") if k in result))
    out["working"] = working
    return out


# ---------------------------------------------------------------- capital-budgeting project (SPEC 4)
def _per_year(x, n: int, name: str) -> list[float]:
    if x is None:
        return [0.0] * n
    if isinstance(x, (list, tuple)):
        _require(len(x) == n, f"{name} needs {n} values, one per year (got {len(x)}); or give one number for every year.")
        return [float(v) for v in x]
    return [float(x)] * n


def project_cash_flows(years: int, tax_rate_pct: float, rate_pct: float | None = None, sales=None, units=None,
                       price_per_unit: float | None = None, costs=None, fixed_costs=None, variable_cost_per_unit=None,
                       variable_cost_pct: float | None = None, ocf=None, investment: float = 0.0,
                       variable_cost_cents_per_unit=None,
                       depreciation: str = "straight_line", depreciation_years: int | None = None,
                       macrs_rates_pct: list[float] | None = None, salvage: float = 0.0, nwc_initial: float = 0.0,
                       nwc_annual_addition: float = 0.0, opportunity_cost_now: float = 0.0,
                       opportunity_value_end: float = 0.0, solve_bid_price: bool = False) -> dict:
    """Yearly project cash flows (year 0 to years), NPV, IRR and optionally the bid price per unit that makes NPV = 0."""
    _require(int(years) == years and 1 <= years <= 50, "years must be a whole number from 1 to 50.")
    _require(0 <= tax_rate_pct < 100, "tax_rate_pct must be from 0 to 100 (e.g. 22 for 22 percent).")
    _require(depreciation in ("straight_line", "macrs", "none"), "depreciation must be 'straight_line', 'macrs' or 'none'.")
    n, t = int(years), tax_rate_pct / 100
    for name, v in (("investment", investment), ("salvage", salvage), ("nwc_initial", nwc_initial),
                    ("nwc_annual_addition", nwc_annual_addition), ("opportunity_cost_now", opportunity_cost_now),
                    ("opportunity_value_end", opportunity_value_end)):
        _require((v or 0) >= 0, f"{name} must be zero or positive (the tool applies the signs).")
    investment, salvage = investment or 0.0, salvage or 0.0
    operating_given = any(v is not None for v in (sales, units, price_per_unit, costs, fixed_costs, variable_cost_per_unit,
                                                  variable_cost_pct, variable_cost_cents_per_unit))
    _require(not (ocf is not None and operating_given), "Give ocf, or the sales and cost lines, not both.")
    if solve_bid_price:
        _require(units is not None, "solve_bid_price needs units (the quantity sold each year).")
        _require(price_per_unit is None and sales is None, "Leave price_per_unit and sales out when solving for the bid price.")
        _require(rate_pct is not None, "solve_bid_price needs rate_pct (the required return).")
    if sales is not None:
        _require(units is None and price_per_unit is None, "Give sales, or units and price_per_unit, not both.")
    if price_per_unit is not None:
        _require(units is not None, "price_per_unit needs units (the quantity sold each year).")
    units_y = _per_year(units, n, "units")
    if variable_cost_cents_per_unit is not None:   # ".40 cents per unit": the tool converts cents to dollars (÷ 100)
        cents = _per_year(variable_cost_cents_per_unit, n, "variable_cost_cents_per_unit")
        variable_cost_per_unit = [d + c / 100 for d, c in zip(_per_year(variable_cost_per_unit, n, "variable_cost_per_unit"), cents)]

    # depreciation and book value
    if depreciation == "straight_line":
        life = depreciation_years or n
        _require(life > 0, "depreciation_years must be positive.")
        dep = [investment / life if y < life else 0.0 for y in range(n)]
        dep_text = f"straight-line to zero over {life:g} years: {investment:,.2f} ÷ {life:g} = {investment / life:,.2f} a year"
    elif depreciation == "macrs":
        _require(bool(macrs_rates_pct), "depreciation='macrs' needs macrs_rates_pct, the MACRS rates in percent from the question "
                                        "(e.g. [33.33, 44.45, 14.81, 7.41]).")
        rates = [float(x) for x in macrs_rates_pct]
        dep = [investment * (rates[y] / 100 if y < len(rates) else 0.0) for y in range(n)]
        dep_text = "MACRS: " + ", ".join(f"year {y + 1} {rates[y]:g}%" for y in range(min(n, len(rates))))
    else:
        dep, dep_text = [0.0] * n, "none"
    book_end = investment - sum(dep)
    after_tax_salvage = salvage - t * (salvage - book_end)
    nwc_total = (nwc_initial or 0.0) + (nwc_annual_addition or 0.0) * (n - 1)

    def flows(price: float | None):
        if ocf is not None:
            ocf_y = _per_year(ocf, n, "ocf")
            lines = None
        else:
            sales_y = _per_year(sales, n, "sales") if sales is not None else [u * (price or 0.0) for u in units_y]
            costs_y = [c + f + vu * u + (variable_cost_pct or 0) / 100 * sv for c, f, vu, u, sv in zip(
                _per_year(costs, n, "costs"), _per_year(fixed_costs, n, "fixed_costs"),
                _per_year(variable_cost_per_unit, n, "variable_cost_per_unit"), units_y, sales_y)]
            ebit = [sv - c - d for sv, c, d in zip(sales_y, costs_y, dep)]
            taxes = [e * t for e in ebit]
            ocf_y = [e - tx + d for e, tx, d in zip(ebit, taxes, dep)]
            lines = (sales_y, costs_y, ebit, taxes)
        nwc = [-(nwc_initial or 0.0)] + [-(nwc_annual_addition or 0.0)] * (n - 1) + [0.0]
        nwc[n] += nwc_total
        capital = [-(investment + (opportunity_cost_now or 0.0))] + [0.0] * n
        capital[n] += after_tax_salvage + (opportunity_value_end or 0.0)
        cf = [capital[0] + nwc[0]] + [ocf_y[y] + nwc[y + 1] + capital[y + 1] for y in range(n)]
        return cf, ocf_y, nwc, capital, lines

    def npv_of(cf):
        return sum(c / (1 + rate_pct / 100) ** k for k, c in enumerate(cf))

    bid = None
    if solve_bid_price:
        base, slope = npv_of(flows(0.0)[0]), npv_of(flows(1.0)[0]) - npv_of(flows(0.0)[0])
        _require(slope > 0, "The NPV does not rise with the price, so no bid price can be solved.")
        bid = -base / slope
    cf, ocf_y, nwc, capital, lines = flows(bid if solve_bid_price else price_per_unit)
    summary = npv_irr([round(c, 10) for c in cf], rate_pct)["result"]
    result = {"cash_flows": [round(c, 2) for c in cf], "ocf": [round(o, 2) for o in ocf_y],
              "depreciation": [round(d, 2) for d in dep], "book_value_at_end": round(book_end, 2),
              "after_tax_salvage": round(after_tax_salvage, 2), "nwc_recovered": round(nwc_total, 2),
              "npv": summary["npv"] if rate_pct is not None else None, "irr_pct": summary["irr_pct"],
              "payback_years": summary.get("payback_years")}
    if rate_pct is not None:
        result["equivalent_annual_amount"] = summary.get("equivalent_annual_amount")
        result["profitability_index"] = summary.get("profitability_index")
    if bid is not None:
        result["bid_price"] = round(bid, 8)
    sym = symbol()
    formatted = {k: format_inr(result[k]) for k in ("npv", "after_tax_salvage") if result.get(k) is not None}
    formatted["cash_flows"] = [format_inr(c) for c in cf]
    if bid is not None:
        formatted["bid_price"] = f"{sym}{bid:.5f}"
    working = [f"Depreciation: {dep_text}.",
               f"After-tax salvage = salvage − t × (salvage − book value) = {salvage:,.2f} − {t:g} × ({salvage:,.2f} − "
               f"{book_end:,.2f}) = {after_tax_salvage:,.2f}."]
    if ocf is None:
        working.append("OCF = (sales − costs − depreciation) × (1 − t) + depreciation.")
    if nwc_total:
        working.append(f"NWC: {nwc_initial or 0:,.2f} at year 0" + (f", {nwc_annual_addition:,.2f} in years 1 to {n - 1}"
                                                                    if nwc_annual_addition and n > 1 else "")
                       + f"; all {nwc_total:,.2f} recovered in year {n}.")
    if bid is not None:
        working.append(f"Bid price: NPV is linear in the price; the price that makes NPV = 0 is {bid:.6f} per unit.")
    # year-by-year table
    if lines is not None:
        cols = ["Year", "Sales", "Costs", "Depreciation", "EBIT", "Taxes", "OCF", "NWC", "Capital spending", "Total cash flow"]
        body = [["0", "", "", "", "", "", "", group(nwc[0], 2), group(capital[0], 2), group(cf[0], 2)]]
        for y in range(n):
            body.append([str(y + 1), group(lines[0][y], 2), group(-lines[1][y], 2), group(-dep[y], 2), group(lines[2][y], 2),
                         group(-lines[3][y], 2), group(ocf_y[y], 2), group(nwc[y + 1], 2), group(capital[y + 1], 2),
                         group(cf[y + 1], 2)])
    else:
        cols = ["Year", "OCF", "NWC", "Capital spending", "Total cash flow"]
        body = [["0", "", group(nwc[0], 2), group(capital[0], 2), group(cf[0], 2)]] + [
            [str(y + 1), group(ocf_y[y], 2), group(nwc[y + 1], 2), group(capital[y + 1], 2), group(cf[y + 1], 2)] for y in range(n)]
    label = f", NPV at {rate_pct:g}% = {group(result['npv'], 2)}" if result["npv"] is not None else ""
    inputs = {k: v for k, v in dict(years=n, tax_rate_pct=tax_rate_pct, rate_pct=rate_pct, sales=sales, units=units,
                                    price_per_unit=price_per_unit, costs=costs, fixed_costs=fixed_costs,
                                    variable_cost_per_unit=variable_cost_per_unit, variable_cost_pct=variable_cost_pct,
                                    variable_cost_cents_per_unit=variable_cost_cents_per_unit,
                                    ocf=ocf, investment=investment, depreciation=depreciation,
                                    depreciation_years=depreciation_years, macrs_rates_pct=macrs_rates_pct, salvage=salvage,
                                    nwc_initial=nwc_initial, nwc_annual_addition=nwc_annual_addition,
                                    opportunity_cost_now=opportunity_cost_now, opportunity_value_end=opportunity_value_end,
                                    solve_bid_price=solve_bid_price or None).items() if v not in (None, 0, 0.0)}
    out = {"inputs": inputs,
           "formula": ("OCF = EBIT + depreciation − taxes, taxes = EBIT × t ; year 0 = −investment − NWC − opportunity cost ; "
                       "last year adds NWC recovery, after-tax salvage = salvage − t × (salvage − book value) and the "
                       "opportunity value at the end ; NPV = Σ CF_t/(1+r)^t"),
           "result": result, "formatted": formatted, "working": working,
           "table": make_table(f"Project cash flows ({sym}{label})", cols, body, note=" ".join(working))}
    return out
