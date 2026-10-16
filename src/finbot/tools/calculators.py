"""Deterministic finance calculators. Every function returns
{"inputs": ..., "formula": ..., "result": {...}, "formatted": {...}} and raises ValueError on bad input.
No eval/exec anywhere.
"""
import math

from .fmt import format_inr, inr_words


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
    return _out(
        {"principal": principal, "annual_rate_pct": annual_rate_pct, "months": months},
        "EMI = P·r·(1+r)^n / ((1+r)^n − 1), r = annual rate / 12, n = months",
        {"emi": round(emi, 2), "total_payment": round(total, 2), "total_interest": round(total - principal, 2),
         "first_year_interest": round(yi, 2), "first_year_principal": round(yp, 2)},
        money_keys=("emi", "total_payment", "total_interest", "first_year_interest", "first_year_principal"),
    )


# ---------------------------------------------------------------- SIP
def sip_future_value(monthly: float, annual_return_pct: float, years: float, step_up_pct: float = 0) -> dict:
    _require(monthly > 0, "monthly must be positive (plain rupees).")
    _require(years > 0 and abs(years * 12 - round(years * 12)) < 1e-9, "years must be positive and a whole number of months.")
    _require(step_up_pct >= 0, "step_up_pct must be zero or positive.")
    i = annual_return_pct / 1200
    n = round(years * 12)
    bal, invested, pay = 0.0, 0.0, monthly
    for m in range(n):
        if m and m % 12 == 0:
            pay *= 1 + step_up_pct / 100
        bal = (bal + pay) * (1 + i)   # deposit at the start of the month, then a month of growth
        invested += pay
    return _out(
        {"monthly": monthly, "annual_return_pct": annual_return_pct, "years": years, "step_up_pct": step_up_pct},
        "FV = P·((1+i)^n − 1)/i·(1+i), i = annual return / 12, n = months (deposit at start of month); "
        "with step-up, the monthly amount rises by step_up_pct each year",
        {"corpus": round(bal, 2), "invested": round(invested, 2), "gain": round(bal - invested, 2)},
        money_keys=("corpus", "invested", "gain"),
    )


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


def tvm(solve_for: str, annual_rate_pct: float | None = None, years: float | None = None,
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
    return _out({"cashflows": cf, "discount_rate_pct": discount_rate_pct},
                "NPV = Σ CF_t/(1+r)^t ; IRR is the r where NPV = 0", result, ("npv",))


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
    return _out(
        {"face": face, "coupon_pct": coupon_pct, "years": years, "freq": freq, "ytm_pct": ytm_pct, "price": price},
        "Price = Σ C/(1+y/f)^t + F/(1+y/f)^N ; Macaulay D = Σ (t/f)·PV_t / Price ; Modified D = D/(1+y/f)",
        {"price": round(p, 4), "ytm_pct": round(y * 100, 6), "current_yield_pct": round(face * coupon_pct / p, 4),
         "macaulay_duration": round(mac, 4), "modified_duration": round(mod, 4)},
        ("price",),
    )


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
    return _out(
        {"spot": spot, "strike": strike, "years": years, "rate_pct": rate_pct, "vol_pct": vol_pct, "kind": kind},
        "d1 = (ln(S/K) + (r + σ²/2)T)/(σ√T), d2 = d1 − σ√T ; Call = S·N(d1) − K·e^(−rT)·N(d2) ; Put = K·e^(−rT)·N(−d2) − S·N(−d1)",
        {"price": round(price, 4), "delta": round(delta, 4), "gamma": round(gamma, 6), "vega_per_1pct": round(vega, 4),
         "theta_per_year": round(theta, 4), "theta_per_day": round(theta / 365, 4), "rho_per_1pct": round(rho, 4),
         "d1": round(d1, 4), "d2": round(d2, 4)},
    )
