"""Deterministic question-type router (SPEC 4.0): detects the calculation type, names the tool, and lists
which tools may answer it. Used by precheck (detection), prompts (hints) and review (enforcement)."""
import re

I = re.I
_ASK_AMOUNT = re.compile(r"\bhow much\b|\bwhat (monthly |yearly |annual )?(amount|sip|saving|investment)\b|"
                         r"\bmonthly (sip|saving|investment|amount)s?\b.{0,20}\b(need|needed|required|should)\b", I)
_TARGET = re.compile(r"\b(reach|accumulate|corpus|target|goal|build up|save up|end up with|to have|to become|to get)\b", I)
_LUMP = re.compile(r"\b(today|now|lump[- ]?sum|one[- ]time|at once|upfront)\b", I)
# Hindi and Tamil words (no \b: Indic vowel signs are not word characters): लोन/ऋण/कर्ज = loan, கடன்/லோன் = loan,
# किस्त/தவணை = instalment, कितना/எவ்வளவு = how much, हर महीने/மாதம் = every month, ईएमआई/இஎம்ஐ = EMI
_LOAN = re.compile(r"\b(loans?|borrow\w*|mortgages?|lenders?)\b|लोन|ऋण|कर्ज|क़र्ज़|கடன்|லோன்", I)
_LOAN_ASK = re.compile(r"\b(payments?|instal+ments?|emi|apr|repa(y|id|ying|yment)|pay (it )?(back|off)|afford\w*|"
                       r"how (much|many)\b.{0,25}\b(borrow|months|payments|years|instal+ments)|"
                       r"(what|which|highest|maximum|max|implied) (annual )?(interest )?rate)\b|"
                       r"कितनी|कितना|कितने|किस्त|क़िस्त|எவ்வளவு|தவணை", I)
_RATE_ASK = re.compile(r"\b(what|which|highest|maximum|max|implied|lowest|find|solve for|calculate|compute|determine)\b"
                       r"(?:\s+(?:is|was|the|an?|annual|nominal|monthly|effective|interest|percentage))*\s+(rate|apr)\b|"
                       r"\brate\b[^.?]{0,30}\bafford\b|\bafford\b[^.?]{0,40}\brate\b", I)
_PERIODIC = re.compile(r"\b(sip|every month|each month|per month|a month|monthly|every year|each year|per year|regularly)\b|"
                       r"हर महीने|प्रति माह|हर साल|மாதந்தோறும்|ஒவ்வொரு மாதமும்|மாதம்", I)
# a loan calculation needs an amount plus a rate or a second amount; "Should I pay off a 9% loan or invest?" is a decision
_MONEY = re.compile(r"(₹|\$|\brs\.?\s?|\binr\s?)\s*\d|\d[\d,]*(\.\d+)?\s*(lakhs?|lacs?|crores?|cr\b|k\b|thousand|million|"
                    r"लाख|करोड|லட்சம்|கோடி)|(?<![\d.,])\d{1,3}(,\d{2,3})+(?![\d,])|(?<![\d.,])\d{4,}(?![\d,])", I)
_TERM = re.compile(r"\d+\s*(years?|yrs?|months?|saal|varusham|साल|வருட|ஆண்டு)", I)
_PCT = re.compile(r"\d\s*(%|per ?cent\b|percent\b|प्रतिशत|சதவீத)", I)


def _loan_inputs(q: str) -> bool:
    money = len(_MONEY.findall(q))
    return money >= 2 or (money >= 1 and bool(_PCT.search(q)))

_TAX_ASK = re.compile(r"\bincome[- ]?tax\b|\btax (liability|payable|outgo)\b|\bhow much (income )?tax\b|\btax (will|would|do|should|must) i\b|"
                     r"\b(my|the) tax\b|\btax on (my )?(salary|income)\b|\b(old|new) (tax )?regime\b|"
                     r"\bhow much\b[^.?]{0,40}\btax(es)?\b|\b(pay|owe)\b[^.?]{0,20}\btax(es)?\b", I)
_TAX_CTX = re.compile(r"\bsalary\b|\bctc\b|\bregime\b|\bhra\b|\b80c\b|\b80d\b|\blakhs?\b|\bcrores?\b|₹|\brs\.?\s|\binr\b|per annum|\bp\.?a\.?\b", I)

# capital gains: selling shares, stocks, equity funds or ETFs, asking about the tax (SPEC 4.4)
_CG_ASSET = re.compile(r"\b(shares?|stocks?|equit(y|ies)|mutual funds?|\bmf\b|units|etfs?|demat)\b", I)
_CG_EVENT = re.compile(r"\b(sell|sold|selling|sale|redeem\w*|redemption|profit|gains?)\b", I)
_CG_TAX = re.compile(r"\btax\w*\b|\bltcg\b|\bstcg\b|capital gains?", I)
_SALARY = re.compile(r"\b(salary|ctc|earn(s|ing)?|per month|a month|monthly income)\b", I)
# tax rules: a limit, rate or deduction question, not a calculation on a salary (SPEC 4.4)
_RULE_ITEM = re.compile(r"\bstandard deduction\b|\brebate\b|\b87a\b|\b80c+d?\b|\b80ccd|\b80d\b|\b80tta\b|\bsection \d+|"
                        r"\bdeductions?\b|\bexempt(ion)?\b|\bslabs?\b|\bcess\b|\bsurcharge\b|\bzero (income )?tax\b|\bno tax\b", I)
_RULE_ASK = re.compile(r"\bhow much\b|\bwhat('s| is| are)\b|\bmaximum\b|\bmax\b|\blimit\b|\ballowed\b|\beligible\b|\brate\b|"
                       r"\bthreshold\b|\bcan (i|we|one) (still |also )?(claim|get|avail|use)\b|\bhighest\b", I)
_TAX_WORD = re.compile(r"\btax\w*\b|\bregime\b|\bitr\b|\bsection\b|\b80c|\b80d|\b87a\b", I)


_INCOME_FIGURE = re.compile(  # the user's own income to compute on: "salary is ₹15 lakh", "I earn 45k a month", "18 LPA"
    r"\b(salary|income|ctc|package|earnings?)\b[^.?\d]{0,25}(₹|rs\.?|inr)?\s*\d|"
    r"\d[\d,.]*\s*(lakhs?|lacs?|crores?|k|lpa)?\s*(salary|income|ctc|package|lpa|per annum|p\.?a\.?|a month|per month|a year)\b|"
    r"\b(earn|make|get|draw)\w*\s+(₹|rs\.?)?\s*\d", I)


def _capital_gains(q: str) -> bool:
    return bool(_CG_ASSET.search(q) and _CG_EVENT.search(q) and _CG_TAX.search(q))


def _tax_rule(q: str) -> bool:
    if re.search(r"\b(post|after)[- ]tax (rate of )?(return|yield)\b|\breal (post[- ]tax |after[- ]tax )?return\b", q, I):
        return False
    return bool(_RULE_ITEM.search(q) and _RULE_ASK.search(q) and _TAX_WORD.search(q))


_PROJECT_ITEMS = re.compile(r"\b(depreciat\w*|salvage|net working capital|nwc|macrs|working capital)\b", I)
_PROJECT_ASK = re.compile(r"\b(npv|net present value|bid price|required return|discount rate|project)\b", I)
_PERPETUITY = re.compile(r"\bperpetuit\w*|\bperpetual\b|\bforever\b|\bindefinitely\b", I)
_ANNUITY = re.compile(r"(?<!equivalent annual )\bannuit(y|ies)\b", I)

ROUTES = [  # (type, test) in priority order
    ("goal_saving", lambda q: bool(_ASK_AMOUNT.search(q) and _TARGET.search(q) and _PERIODIC.search(q))),
    ("present_value", lambda q: bool(_ASK_AMOUNT.search(q) and _TARGET.search(q) and _LUMP.search(q) and not _PERIODIC.search(q))),
    ("years", lambda q: bool(re.search(r"\b(how many years|how long|years? to (double|triple|reach)|doubling time|rule of 72)\b", q, I)
                             or re.search(r"\b(double|triple)\b.{0,15}\bmoney\b", q, I))),
    ("real_return", lambda q: bool(re.search(r"\breal (post[- ]tax |after[- ]tax )?(rate of )?(return|yield)\b|\breal (interest )?rate\b|\bfisher\b|"
                                             r"\b(post|after)[- ]tax (rate of )?(return|yield)\b|\binflation[- ]adjusted\b|\bafter inflation\b|"
                                             r"\bbeat(ing)? inflation\b", q, I))),
    ("effective_rate", lambda q: bool(re.search(r"\beffective (annual )?(rate|yield|interest)\b|\bear\b|\bapy\b|annual percentage yield", q, I))),
    ("emi", lambda q: bool(re.search(r"\bemi\b|\bequated monthly\b|\bmonthly instal+ments?\b|ईएमआई|इएमआई|இஎம்ஐ", q, I))
     and bool(_MONEY.search(q))),
    ("loan_rate", lambda q: bool(_LOAN.search(q) and _RATE_ASK.search(q)) and _loan_inputs(q)),
    ("loan", lambda q: bool(_LOAN.search(q) and _LOAN_ASK.search(q)) and _loan_inputs(q)),
    ("cagr", lambda q: bool(re.search(r"\bcagr\b|compound(ed)? annual growth|annuali[sz]ed return", q, I))),
    ("income_tax", lambda q: bool(_TAX_ASK.search(q) and _TAX_CTX.search(q))),
    ("capital_gains", _capital_gains),
    ("flat_loan", lambda q: bool(re.search(r"\d\s*%\s*['\"]?\s*flat\b|\bflat (interest )?rate\b|\bflat interest\b|\bflat basis\b", q, I)
                                 and _PCT.search(q)
                                 and re.search(r"\b(loan|emi|finance|financing|instal+ments?)\b", q, I))),
    ("emergency_fund", lambda q: bool(re.search(r"\bemergency (fund|corpus|savings?|kitty)\b|आपातकालीन|अपात", q, I))),
    ("simple_interest", lambda q: bool(re.search(r"\bsimple interest\b|\bflat (interest|rate)\b|साधारण ब्याज", q, I))),
    ("sinking_fund", lambda q: bool(re.search(r"\bsinking fund\b", q, I))),
    ("ppf", lambda q: bool(re.search(r"\bppf\b|public provident fund", q, I))),
    ("growing_annuity", lambda q: bool(re.search(r"\bgrow\w*\b[^.?]{0,40}\d+(\.\d+)?\s*%[^.?]{0,60}\bfor\s+\d+\s+(years|periods|payments)\b", q, I)
                                       and not _PERPETUITY.search(q))),
    ("tax_rules", _tax_rule),
    ("eac", lambda q: bool(re.search(r"equivalent annual (cost|annuity)|\beac\b|\beaa\b", q, I))),
    ("project", lambda q: bool(_PROJECT_ITEMS.search(q) and re.search(r"\btax", q, I) and _PROJECT_ASK.search(q))),
    ("perpetuity", lambda q: bool(_PERPETUITY.search(q))),
    ("annuity", lambda q: bool(_ANNUITY.search(q))),
    ("withdrawal", lambda q: bool(re.search(r"\bwithdraw\w*\b", q, I) and _PERIODIC.search(q))),
    ("npv_irr", lambda q: bool(re.search(r"\bnpv\b|\birr\b|net present value|internal rate of return|\bpayback\b|profitability index", q, I))),
    ("bond", lambda q: bool(re.search(r"\bbonds?\b(?!\s+(account|fund|funds|portfolio|mutual fund|etf)s?\b)|\bytm\b|"
                                      r"yield to maturity|\bcoupon\b", q, I))),
    ("option", lambda q: bool(re.search(r"black[- ]?scholes|\b(call|put) options?\b|\boption (price|premium)\b", q, I))),
    ("sip_fv", lambda q: bool(_MONEY.search(q) and (_PCT.search(q) or _TERM.search(q))) and bool(re.search(r"एसआईपी|எஸ்ஐபி|\bsip\b|\b(invest|put|save|deposit)\w*\b.{0,30}\b(every|each|per|a) month\b|\bmonthly (investment|deposit)\b", q, I))),
    ("lump_fv", lambda q: bool(re.search(r"\b(grow to|grow into|be worth|maturity (value|amount)|future value|will (it|i|my money) (become|grow|have))\b", q, I)
                               or re.search(r"\bwhat will .{0,30}\b(grow|become)\b", q, I))),
    ("percent", lambda q: bool(re.search(r"\bgst\b|\d+(\.\d+)?\s*%\s*of\b|\bper ?cent of\b|\bpercentage of\b|\bdiscount of\b|\bmarkup\b", q, I))),
]
EXCLUSIVE = {"goal_saving": {"sip_fv", "lump_fv", "present_value"}, "present_value": {"lump_fv"}, "years": {"lump_fv"},
             "emi": {"sip_fv", "loan", "annuity"}, "loan": {"years", "present_value", "lump_fv", "sip_fv", "annuity"},
             "cagr": {"lump_fv"},
             "loan_rate": {"loan", "emi", "years", "present_value", "lump_fv", "sip_fv", "goal_saving", "annuity"},
             "project": {"npv_irr", "eac", "lump_fv", "percent", "years", "present_value", "perpetuity", "annuity"},
             "perpetuity": {"npv_irr", "lump_fv", "present_value", "years", "annuity"},
             "annuity": {"lump_fv", "present_value", "sip_fv", "goal_saving", "years"},
             "eac": {"annuity"}, "withdrawal": {"goal_saving", "present_value"},
             "sip_fv": {"lump_fv"},
             "capital_gains": {"percent", "lump_fv", "cagr", "years", "tax_rules"},
             "simple_interest": {"lump_fv", "percent", "years", "loan", "emi"},
             "sinking_fund": {"goal_saving", "present_value", "lump_fv", "annuity", "sip_fv"},
             "ppf": {"lump_fv", "sip_fv", "goal_saving", "annuity", "years"},
             "growing_annuity": {"annuity", "present_value", "lump_fv", "years", "perpetuity"},
             "real_return": {"bond", "tax_rules", "percent"},
             "flat_loan": {"simple_interest", "emi", "loan", "loan_rate", "percent"},
             "emergency_fund": {"emi", "loan", "percent", "sip_fv", "goal_saving", "lump_fv"}}

REQUIRED_TOOLS = {
    "goal_saving": {"goal_saving"}, "present_value": {"tvm", "goal_saving"}, "years": {"tvm"},
    "real_return": {"real_return", "calculate"}, "effective_rate": {"effective_annual_rate"}, "emi": {"loan_emi"}, "loan": {"loan_emi"},
    "loan_rate": {"loan_emi"},
    "cagr": {"tvm"}, "npv_irr": {"npv_irr"}, "eac": {"equivalent_annual_cost", "npv_irr"}, "income_tax": {"income_tax"}, "bond": {"bond_metrics", "calculate"},
    "capital_gains": {"capital_gains"}, "tax_rules": {"tax_rules", "income_tax"},
    "simple_interest": {"calculate"}, "sinking_fund": {"tvm"}, "ppf": {"tvm"},
    "flat_loan": {"calculate", "loan_emi"}, "emergency_fund": {"calculate"},
    "growing_annuity": {"perpetuity", "calculate"},
    "option": {"black_scholes"}, "project": {"project_cash_flows"}, "perpetuity": {"perpetuity", "calculate"}, "annuity": {"tvm"},
    "withdrawal": {"tvm", "loan_emi"},
    "sip_fv": {"sip_future_value"}, "lump_fv": {"tvm"}, "percent": {"calculate"},
    "statement": {"solve_financials", "financial_ratios", "balance_sheet_check", "cash_flow_statement", "calculate", "solve_equation"},
}

_LOAN_HINT = ("Loan: call loan_emi with exactly the three values the question gives among principal, annual_rate_pct, "
              "months (years × 12) and emi (the monthly payment). It solves the missing one: the EMI, the rate the payments imply "
              "(APR = monthly rate × 12), the number of months, or the amount that can be borrowed. Leave the unknown out; never pass 0 "
              "for it. For annual, half-yearly or quarterly payments pass payments_per_year (1, 2 or 4) and years instead of months; "
              "the schedule then has one row per payment, and its interest column answers 'interest paid in year k'.")

HINTS = {
    "goal_saving": ("Goal question: call goal_saving(target=<the amount to reach>, years=..., annual_return_pct=...). The target is the FUTURE "
                    "amount the user wants (₹1 crore = 10000000), never the monthly amount. Use timing='end' only if the question says end of each "
                    "month; add current_savings or step_up_pct only if stated."),
    "present_value": "Lump sum needed today: call tvm(solve_for='pv', annual_rate_pct=..., years=..., fv=<amount to reach>).",
    "years": ("Time to reach a multiple or a target: call tvm(solve_for='years', annual_rate_pct=..., pv=..., fv=...). For doubling use pv=1, fv=2. "
              "Never use the Rule of 72."),
    "real_return": ("Call real_return(nominal_pct=..., inflation_pct=...). Do not subtract the rates yourself. For a post-tax "
                    "return first compute rate × (1 − tax rate) with calculate, then pass that as nominal_pct for the real post-tax return."),
    "flat_loan": ("Flat-rate loan: interest = principal × flat rate × years on the full amount. Compute the EMI = (principal + "
                  "interest) ÷ months with calculate. The true reducing-balance rate is loan_emi(principal=..., months=..., "
                  "emi=<that EMI>) with annual_rate_pct left out; say the flat rate understates the real cost."),
    "emergency_fund": ("Emergency fund: (monthly expenses + monthly EMIs and other fixed payments) × the number of months, "
                       "computed with calculate."),
    "effective_rate": "Call effective_annual_rate(nominal_pct=..., periods_per_year=12 for monthly, 4 for quarterly, 2 for half-yearly).",
    "loan_rate": ("Loan rate: the interest rate is the UNKNOWN. Call loan_emi(principal=<amount borrowed>, months=<number of "
                  "monthly payments>, emi=<the payment>) and leave annual_rate_pct out. It returns annual_rate_pct (the APR = "
                  "monthly rate × 12) and effective_annual_rate_pct; quote the APR to 2 decimals. Never estimate, assume or guess "
                  "a rate, and never pass 0 for it."),
    "emi": _LOAN_HINT,
    "loan": _LOAN_HINT,
    "cagr": "Call tvm(solve_for='annual_rate_pct', years=..., pv=<starting value>, fv=<ending value>).",
    "npv_irr": ("Call npv_irr(cashflows=[-investment, inflow year 1, ...], discount_rate_pct only if a rate is given), once per project. "
                "It also returns profitability_index. For an equivalent annual cost use equivalent_annual_cost instead. For two projects "
                "call it once per project; the crossover rate is the irr of the year-by-year difference of their cash flows (one more call)."),
    "income_tax": ("Indian salary tax: call income_tax with only the figures the user gave (per-month amounts in the *_monthly "
                   "fields), regime='both' unless they chose one. Report both regimes' totals and the lower one. If the old-regime "
                   "HRA exemption needs rent paid, basic + DA or the city, say so and ask for them; never estimate tax yourself."),
    "capital_gains": ("Profit from selling shares or equity funds: call capital_gains with purchase_value, sale_value and "
                      "holding_months exactly as given (ten months = 10, 2 years = 24; asset='equity_fund' for a mutual fund). If only the gain "
                      "is given, pass gain=<the profit> and term='long' or 'short' (or holding_months); one call per gain. "
                      "Never use income_tax other_income for it. If the user also gives a salary, call income_tax for the salary too."),
    "simple_interest": ("Simple interest: call calculate with I = P × R × T ÷ 100 (T in years: months ÷ 12, days ÷ 365); the "
                        "amount is P + I. For a missing P, R or T rearrange the same formula inside calculate."),
    "sinking_fund": ("Sinking fund: call tvm(solve_for='pmt', fv=<amount needed>, annual_rate_pct=..., years=..., "
                     "periods_per_year=<deposits per year>), with when='begin' only if deposits are at the start."),
    "ppf": ("PPF: deposits at the start of each financial year, compounded yearly. Call tvm(solve_for='fv', pmt=<yearly "
            "deposit>, annual_rate_pct=<rate, e.g. 7.1>, years=<15 or as given>, when='begin')."),
    "growing_annuity": ("Growing annuity (payments growing for a fixed number of years): call perpetuity(cash_flow=<first "
                        "payment>, rate_pct=..., growth_pct=..., periods=<number of payments>)."),
    "tax_rules": ("A tax rule, limit, rate or deduction question: call tax_rules (topic new_regime, old_regime, deductions or "
                  "capital_gains; the financial year if given) and quote its figures. If the user gives their own salary, use "
                  "income_tax instead."),
    "eac": ("Call equivalent_annual_cost once per option with purchase_cost and rate_pct, costs as positive numbers (salvage only if "
            "given). For electricity (watts, kWh) pass power_watts, hours_per_year, price_per_kwh and lifetime_hours exactly as the "
            "question gives them; otherwise pass annual_cost and life_years. Report its eac, which is negative because it is a cost, "
            "and its annual_cost and life_years. Never write out the yearly cash flows."),
    "project": ("Capital-budgeting project: call project_cash_flows once per project (or system) with every figure from the "
                "question: years, tax_rate_pct, rate_pct, investment, depreciation ('straight_line', with depreciation_years if the "
                "tax life differs; or 'macrs' with macrs_rates_pct as given), salvage (pretax market value), sales or units with "
                "price_per_unit, costs / fixed_costs / variable_cost_per_unit / variable_cost_pct, nwc_initial, "
                "nwc_annual_addition, opportunity_cost_now and opportunity_value_end (after-tax values of land or another asset "
                "already owned). For a bid price pass units and solve_bid_price=true and leave the price out. Ignore sunk costs "
                "(money already spent). Quote its yearly cash flows, NPV and IRR; never build the cash flows yourself."),
    "perpetuity": ("Perpetuity (cash flows forever): call perpetuity(cash_flow=<first year's cash flow>, rate_pct=..., "
                   "growth_pct=<growth, 0 if level>, investment=<cost today, if given>). It returns pv, npv, irr_pct and "
                   "break_even_growth_pct (the growth rate where NPV = 0)."),
    "annuity": ("Annuity: call tvm(solve_for='pv' for a present value or 'fv' for a future value, pmt=<payment per period>, "
                "annual_rate_pct=..., years=..., periods_per_year=...) with no pv and no fv. Its annuity_values gives the PV and "
                "FV of the ordinary annuity and of the annuity due (when='begin'); quote the ones asked."),
    "withdrawal": ("Level withdrawal from a balance: call tvm(solve_for='pmt', pv=<the balance at the start>, annual_rate_pct=..., "
                   "years=<withdrawal period>, periods_per_year=12 for monthly) with no fv. If the balance is built up first, "
                   "compute it with a tool and pass that exact result."),
    "bond": ("Call bond_metrics(face, coupon_pct, years, freq, and either ytm_pct or price). For the effective annual yield "
             "also call effective_annual_rate(nominal_pct=<its ytm_pct>, periods_per_year=<freq>)."),
    "option": "Call black_scholes(spot, strike, years, rate_pct, vol_pct, kind).",
    "sip_fv": "The monthly amount is known: call sip_future_value(monthly=..., annual_return_pct=..., years=...).",
    "lump_fv": "Lump-sum growth: call tvm(solve_for='fv', annual_rate_pct=..., years=..., pv=<amount invested>, periods_per_year as stated).",
    "percent": ("Use calculate. Apply each percentage to the right base: 'X% of A, plus Y% GST on that amount' means A × X/100 × (1 + Y/100). "
                "Write the whole expression in one call. If you also show an intermediate amount (such as the GST amount alone), compute it with "
                "calculate too; never work out a figure yourself."),
    "statement": ("Call solve_financials FIRST whenever the question gives statement items or ratios and asks for another item "
                  "(income statement, ratio algebra, DuPont, growth rates, market value ratios): pass every figure from the question "
                  "under its name (percent items in percent: 23 percent -> tax_rate_pct=23; ratios as plain numbers: .75 -> 0.75) and "
                  "targets=[the item(s) asked]. It picks the formulas; never write your own formula, and quote its step values. "
                  "For statements covering two or more years, call it once per year. If COGS and other expenses are listed "
                  "separately, pass cogs and other_expenses (not costs); if dividends are given, pass them and add "
                  "addition_to_retained_earnings to targets. "
                  "For a full ratio analysis of pasted statements use financial_ratios; to check a balance sheet use "
                  "balance_sheet_check; for a cash flow statement use cash_flow_statement."),
}


def detect(question: str, has_number: bool, statement: bool) -> list[str]:
    if not has_number:
        return []
    found = [kind for kind, test in ROUTES if test(question)]
    if "capital_gains" in found and not _SALARY.search(question):     # share-profit tax alone is not salary tax
        found = [k for k in found if k != "income_tax"]
    if "tax_rules" in found and "income_tax" in found:                 # a rule question unless the user's own income is given
        drop = "tax_rules" if _INCOME_FIGURE.search(question) else "income_tax"
        found = [k for k in found if k != drop]
    for kind in list(found):
        found = [k for k in found if k not in EXCLUSIVE.get(kind, set()) or k == kind]
    if statement and "project" not in found:   # "85 percent of sales are on credit" is a statement input; the statement tools include calculate
        found = [k for k in found if k != "percent"] + ["statement"]
    return found


def missing_tools(calc_types: list[str], tools_used: list[str]) -> list[str]:
    """Types whose required tool was not called."""
    used = set(tools_used)
    return [k for k in calc_types if not (REQUIRED_TOOLS.get(k, set()) & used)]
