"""Deterministic question-type router (SPEC 4.0): detects the calculation type, names the tool, and lists
which tools may answer it. Used by precheck (detection), prompts (hints) and review (enforcement)."""
import re

I = re.I
_ASK_AMOUNT = re.compile(r"\bhow much\b|\bwhat (monthly |yearly |annual )?(amount|sip|saving|investment)\b|"
                         r"\bmonthly (sip|saving|investment|amount)s?\b.{0,20}\b(need|needed|required|should)\b", I)
_TARGET = re.compile(r"\b(reach|accumulate|corpus|target|goal|build up|save up|end up with|to have|to become|to get)\b", I)
_LUMP = re.compile(r"\b(today|now|lump[- ]?sum|one[- ]time|at once|upfront)\b", I)
_PERIODIC = re.compile(r"\b(sip|every month|each month|per month|a month|monthly|every year|each year|per year|regularly)\b", I)

ROUTES = [  # (type, test) in priority order
    ("goal_saving", lambda q: bool(_ASK_AMOUNT.search(q) and _TARGET.search(q) and _PERIODIC.search(q))),
    ("present_value", lambda q: bool(_ASK_AMOUNT.search(q) and _TARGET.search(q) and _LUMP.search(q) and not _PERIODIC.search(q))),
    ("years", lambda q: bool(re.search(r"\b(how many years|how long|years? to (double|triple|reach)|doubling time|rule of 72)\b", q, I)
                             or re.search(r"\b(double|triple)\b.{0,15}\bmoney\b", q, I))),
    ("real_return", lambda q: bool(re.search(r"\breal (rate of )?return\b|\binflation[- ]adjusted\b|\bafter inflation\b|\bbeat(ing)? inflation\b", q, I))),
    ("effective_rate", lambda q: bool(re.search(r"\beffective (annual )?(rate|yield|interest)\b|\bear\b|\bapy\b|annual percentage yield", q, I))),
    ("emi", lambda q: bool(re.search(r"\bemi\b|\bequated monthly\b|\bmonthly instal+ments?\b", q, I))),
    ("cagr", lambda q: bool(re.search(r"\bcagr\b|compound(ed)? annual growth|annuali[sz]ed return", q, I))),
    ("npv_irr", lambda q: bool(re.search(r"\bnpv\b|\birr\b|net present value|internal rate of return|\bpayback\b", q, I))),
    ("bond", lambda q: bool(re.search(r"\bbond\b|\bytm\b|yield to maturity|\bcoupon\b", q, I))),
    ("option", lambda q: bool(re.search(r"black[- ]?scholes|\b(call|put) options?\b|\boption (price|premium)\b", q, I))),
    ("sip_fv", lambda q: bool(re.search(r"\bsip\b|\b(invest|put|save|deposit)\w*\b.{0,30}\b(every|each|per|a) month\b|\bmonthly (investment|deposit)\b", q, I))),
    ("lump_fv", lambda q: bool(re.search(r"\b(grow to|grow into|be worth|maturity (value|amount)|future value|will (it|i|my money) (become|grow|have))\b", q, I)
                               or re.search(r"\bwhat will .{0,30}\b(grow|become)\b", q, I))),
    ("percent", lambda q: bool(re.search(r"\bgst\b|\d+(\.\d+)?\s*%\s*of\b|\bper ?cent of\b|\bpercentage of\b|\bdiscount of\b|\bmarkup\b", q, I))),
]
EXCLUSIVE = {"goal_saving": {"sip_fv", "lump_fv", "present_value"}, "present_value": {"lump_fv"}, "years": {"lump_fv"},
             "emi": {"sip_fv"}, "cagr": {"lump_fv"}, "sip_fv": {"lump_fv"}}

REQUIRED_TOOLS = {
    "goal_saving": {"goal_saving"}, "present_value": {"tvm", "goal_saving"}, "years": {"tvm"},
    "real_return": {"real_return"}, "effective_rate": {"effective_annual_rate"}, "emi": {"loan_emi"},
    "cagr": {"tvm"}, "npv_irr": {"npv_irr"}, "bond": {"bond_metrics"}, "option": {"black_scholes"},
    "sip_fv": {"sip_future_value", "goal_saving"}, "lump_fv": {"tvm"}, "percent": {"calculate"},
    "statement": {"financial_ratios", "balance_sheet_check", "cash_flow_statement"},
}

HINTS = {
    "goal_saving": ("Goal question: call goal_saving(target=<the amount to reach>, years=..., annual_return_pct=...). The target is the FUTURE "
                    "amount the user wants (₹1 crore = 10000000), never the monthly amount. Use timing='end' only if the question says end of each "
                    "month; add current_savings or step_up_pct only if stated."),
    "present_value": "Lump sum needed today: call tvm(solve_for='pv', annual_rate_pct=..., years=..., fv=<amount to reach>).",
    "years": ("Time to reach a multiple or a target: call tvm(solve_for='years', annual_rate_pct=..., pv=..., fv=...). For doubling use pv=1, fv=2. "
              "Never use the Rule of 72."),
    "real_return": "Call real_return(nominal_pct=..., inflation_pct=...). Do not subtract the rates yourself.",
    "effective_rate": "Call effective_annual_rate(nominal_pct=..., periods_per_year=12 for monthly, 4 for quarterly, 2 for half-yearly).",
    "emi": "Call loan_emi(principal=..., annual_rate_pct=..., months=years × 12).",
    "cagr": "Call tvm(solve_for='annual_rate_pct', years=..., pv=<starting value>, fv=<ending value>).",
    "npv_irr": "Call npv_irr(cashflows=[-investment, inflow year 1, ...], discount_rate_pct only if a rate is given).",
    "bond": "Call bond_metrics(face, coupon_pct, years, freq, and either ytm_pct or price).",
    "option": "Call black_scholes(spot, strike, years, rate_pct, vol_pct, kind).",
    "sip_fv": "The monthly amount is known: call sip_future_value(monthly=..., annual_return_pct=..., years=...).",
    "lump_fv": "Lump-sum growth: call tvm(solve_for='fv', annual_rate_pct=..., years=..., pv=<amount invested>, periods_per_year as stated).",
    "percent": ("Use calculate. Apply each percentage to the right base: 'X% of A, plus Y% GST on that amount' means A × X/100 × (1 + Y/100). "
                "Write the whole expression in one call. If you also show an intermediate amount (such as the GST amount alone), compute it with "
                "calculate too; never work out a figure yourself."),
    "statement": "Use financial_ratios, balance_sheet_check or cash_flow_statement with the figures exactly as given.",
}


def detect(question: str, has_number: bool, statement: bool) -> list[str]:
    if not has_number:
        return []
    found = [kind for kind, test in ROUTES if test(question)]
    for kind in list(found):
        found = [k for k in found if k not in EXCLUSIVE.get(kind, set()) or k == kind]
    if statement:
        found.append("statement")
    return found


def missing_tools(calc_types: list[str], tools_used: list[str]) -> list[str]:
    """Types whose required tool was not called."""
    used = set(tools_used)
    return [k for k in calc_types if not (REQUIRED_TOOLS.get(k, set()) & used)]
