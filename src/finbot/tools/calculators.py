"""Deterministic finance calculators. Every function returns
{"inputs": ..., "formula": ..., "result": {...}, "formatted": {...}} and raises ValueError on bad input.
No eval/exec anywhere.
"""
import math

from .fmt import format_inr, group, inr_words, make_table


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
def loan_emi(principal: float, annual_rate_pct: float, months: int) -> dict:
    _require(principal > 0, "principal must be positive (plain rupees, e.g. 20 lakh = 2000000).")
    _require(annual_rate_pct >= 0, "annual_rate_pct must be zero or positive.")
    _require(int(months) == months and months > 0, "months must be a positive whole number.")
    months = int(months)
    r = annual_rate_pct / 1200
    emi = principal / months if r == 0 else principal * r * (1 + r) ** months / ((1 + r) ** months - 1)
    bal, yi, yp = principal, 0.0, 0.0
    for _ in range(min(12, months)):
        interest = bal * r
        yi += interest
        yp += emi - interest
        bal -= emi - interest
    total = emi * months
    body, b, t_paid, t_int, t_prin, year = [], principal, 0.0, 0.0, 0.0, 0
    for start in range(0, months, 12):
        year += 1
        opening, paid, interest_y, prin_y = b, 0.0, 0.0, 0.0
        for _ in range(min(12, months - start)):
            i_amt = b * r
            paid, interest_y, prin_y, b = paid + emi, interest_y + i_amt, prin_y + emi - i_amt, b - (emi - i_amt)
        t_paid, t_int, t_prin = t_paid + paid, t_int + interest_y, t_prin + prin_y
        body.append([str(year), group(opening), group(paid), group(interest_y), group(prin_y), group(b if abs(b) >= 0.5 else 0)])
    table = make_table("Year-wise loan schedule (₹)", ["Year", "Opening balance", "EMI paid", "Interest", "Principal", "Closing balance"],
                       body, total=["Total", "", group(t_paid), group(t_int), group(t_prin), ""])
    out = _out(
        {"principal": principal, "annual_rate_pct": annual_rate_pct, "months": months},
        "EMI = P·r·(1+r)^n / ((1+r)^n − 1), r = annual rate / 12, n = months",
        {"emi": round(emi, 2), "total_payment": round(total, 2), "total_interest": round(total - principal, 2),
         "first_year_interest": round(yi, 2), "first_year_principal": round(yp, 2)},
        money_keys=("emi", "total_payment", "total_interest", "first_year_interest", "first_year_principal"),
    )
    out["table"] = table
    return out


# ---------------------------------------------------------------- SIP
def sip_future_value(monthly: float, annual_return_pct: float, years: float, step_up_pct: float = 0) -> dict:
    _require(monthly > 0, "monthly must be positive (plain rupees).")
    _require(years > 0 and abs(years * 12 - round(years * 12)) < 1e-9, "years must be positive and a whole number of months.")
    _require(step_up_pct >= 0, "step_up_pct must be zero or positive.")
    i = annual_return_pct / 1200
    n = round(years * 12)
    bal, invested, pay = 0.0, 0.0, monthly
    body, year_inv = [], 0.0
    for m in range(n):
        if m and m % 12 == 0:
            pay *= 1 + step_up_pct / 100
        bal = (bal + pay) * (1 + i)   # deposit at the start of the month, then a month of growth
        invested += pay
        year_inv += pay
        if (m + 1) % 12 == 0 or m == n - 1:
            body.append([str((m // 12) + 1), group(pay), group(year_inv), group(invested), group(bal), group(bal - invested)])
            year_inv = 0.0
    table = make_table("Year-wise SIP growth (₹)", ["Year", "Monthly SIP", "Invested in year", "Total invested",
                                                    "Value at year end", "Gain"], body,
                       note="Assumes the stated return every year; actual returns vary and are not guaranteed.")
    out = _out(
        {"monthly": monthly, "annual_return_pct": annual_return_pct, "years": years, "step_up_pct": step_up_pct},
        "FV = P·((1+i)^n − 1)/i·(1+i), i = annual return / 12, n = months (deposit at start of month); "
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


def _tvm_table(rate_pct: float, years: float, pv: float, pmt: float, m: int, due: int) -> dict | None:
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
    return make_table("Year-wise growth (₹)", ["Year", "Opening", "Deposits", "Growth", "Closing"], body,
                      total=["Total", group(pv, 2), group(t_dep, 2), group(t_growth, 2), group(b, 2)])


def tvm(*args, **kwargs) -> dict:
    out = _tvm_solve(*args, **kwargs)
    inp, res = out["inputs"], out["result"]
    rate = res.get("annual_rate_pct", inp["annual_rate_pct"])
    years = res.get("years", inp["years"])
    pv = res.get("pv", inp["pv"])
    pmt = res.get("pmt", inp["pmt"])
    if rate is not None:
        table = _tvm_table(rate, years, pv, pmt, inp["periods_per_year"], 1 if inp["when"] == "begin" else 0)
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

    if solve_for != "annual_rate_pct":
        _require(annual_rate_pct is not None, "annual_rate_pct is required.")
    if solve_for != "years":
        _require(years is not None and years > 0, "years is required and must be positive.")
    if solve_for not in ("fv",):
        _require(fv is not None, "fv (the target amount) is required.")

    i = (annual_rate_pct or 0) / 100 / m
    n = (years or 0) * m

    if solve_for == "fv":
        val = _tvm_fv(i, n, pv, pmt, due)
        return _out(inputs, formula, {"fv": round(val, 2)}, ("fv",))
    if solve_for == "pv":
        val = (fv - _tvm_fv(i, n, 0, pmt, due)) / (1 + i) ** n
        _require(val >= 0, "The deposits alone already exceed the target; no starting amount is needed.")
        return _out(inputs, formula, {"pv": round(val, 2)}, ("pv",))
    if solve_for == "pmt":
        unit = _tvm_fv(i, n, 0, 1, due)
        val = (fv - pv * (1 + i) ** n) / unit
        _require(val >= 0, "The starting amount alone already reaches the target; no deposits are needed.")
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
def npv_irr(cashflows: list[float], discount_rate_pct: float | None = None) -> dict:
    _require(isinstance(cashflows, list) and len(cashflows) >= 2, "cashflows needs at least two values; cashflows[0] is time 0.")
    cf = [float(c) for c in cashflows]
    npv_at = lambda rate: sum(c / (1 + rate) ** t for t, c in enumerate(cf))  # noqa: E731
    result = {"npv": None, "irr_pct": None}
    if discount_rate_pct is not None:
        result["npv"] = round(npv_at(discount_rate_pct / 100), 4)
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
    if note:
        result["note"] = note
    out = _out({"cashflows": cf, "discount_rate_pct": discount_rate_pct},
               "NPV = Σ CF_t/(1+r)^t ; IRR is the r where NPV = 0", result, ("npv",))
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
    out["table"] = make_table(f"Cash flows by year (₹, {label})",
                              ["Year", "Cash flow", "Cumulative cash flow", "Discount factor", "Present value", "Cumulative PV"], body)
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
    table = make_table("Year-wise plan to reach the goal (₹)", ["Year", "Monthly SIP", "Invested in year", "Total invested",
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
