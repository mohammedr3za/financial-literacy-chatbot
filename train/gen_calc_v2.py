"""Calculation questions for fine-tune v2, with gold answers computed by FinBot's own deterministic tools.

uv run python train/gen_calc_v2.py --out train/data/questions_v2/calc_v2.jsonl --n 700
Then: train/make_dataset.py --questions train/data/questions_v2/calc_v2.jsonl --out train/data/sft_v2_calc.jsonl
A transcript is kept only if its answer contains every gold figure (within 1%) and passes review on the first draft.
Figures are random and phrasings original, so nothing copies a test question (prepare_sft also decontaminates).
"""
import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from finbot.tools import calculators as C, income_tax as T, financial_solver as F  # noqa: E402

TONES = {
    "plain": lambda q: q,
    "eager": lambda q: q + " I'm really excited to get started!",
    "worried": lambda q: "I'm a bit worried about my finances. " + q,
    "short": lambda q: q.replace("Could you tell me ", "").replace("Please ", ""),
}


def lakh(x: float) -> str:
    """₹ amount as people say it: '₹20 lakh', '₹1.5 crore', '₹75,000'."""
    if x >= 1e7 and x % 1e5 == 0:
        return f"₹{x / 1e7:g} crore"
    if x >= 1e5 and x % 1e4 == 0:
        return f"₹{x / 1e5:g} lakh"
    return f"₹{x:,.0f}"


def families(rng: random.Random):
    p = rng.choice
    def emi():
        P, r, y = p([5, 8, 10, 15, 20, 25, 30, 40, 50, 60, 75]) * 1e5, p([7.5, 8, 8.4, 8.75, 9, 9.25, 9.5, 10.5, 11, 12]), p([5, 10, 15, 20, 25, 30])
        q = p([f"What will my EMI be on a home loan of {lakh(P)} at {r}% for {y} years?",
               f"Could you tell me the monthly EMI for a {lakh(P)} loan at {r}% interest over {y} years?",
               f"Mujhe {lakh(P)} ka loan {r}% par {y} saal ke liye lena hai. EMI kitni hogi?"])
        return q, [C.loan_emi(principal=P, annual_rate_pct=r, months=y * 12)["result"]["emi"]]
    def loan_rate():
        P, n = p([3, 5, 6, 8, 10, 12, 15]) * 1e5, p([24, 36, 48, 60, 72, 84])
        emi_ = round(C.loan_emi(principal=P, annual_rate_pct=rng.uniform(7, 16), months=n)["result"]["emi"], -2)
        q = p([f"A bank offers me {lakh(P)} for {n} months and I can pay ₹{emi_:,.0f} a month. What annual interest rate does that work out to?",
               f"If I borrow {lakh(P)} and repay ₹{emi_:,.0f} every month for {n} months, what is the APR on the loan?"])
        return q, [C.loan_emi(principal=P, emi=emi_, months=n)["result"]["annual_rate_pct"]]
    def loan_months():
        P, r = p([2, 4, 5, 8, 10, 20]) * 1e5, p([9, 10.5, 11, 12, 13, 14])
        base = C.loan_emi(principal=P, annual_rate_pct=r, months=p([36, 48, 60, 84]))["result"]["emi"]
        emi_ = round(base * rng.uniform(1.05, 1.3), -2)
        q = f"I have a {lakh(P)} loan at {r}% and want to pay ₹{emi_:,.0f} a month. How many months will it take to repay?"
        res = C.loan_emi(principal=P, annual_rate_pct=r, emi=emi_)["result"]
        return q, [res.get("months") or res.get("months_exact")]
    def sip():
        m, r, y = p([2000, 3000, 5000, 7500, 10000, 15000, 25000]), p([8, 10, 11, 12, 13, 14]), p([5, 7, 10, 12, 15, 20, 25])
        q = p([f"If I invest ₹{m:,} a month in a SIP for {y} years at an expected {r}% return, how much will I have?",
               f"₹{m:,} monthly SIP for {y} years at {r}% annual return. What could the corpus be?"])
        return q, [C.sip_future_value(monthly=m, annual_return_pct=r, years=y)["result"]["corpus"]]
    def goal():
        tgt, y, r = p([10, 25, 50]) * 1e5 if rng.random() < 0.5 else p([1, 2, 5]) * 1e7, p([5, 8, 10, 15, 20]), p([8, 10, 12])
        q = p([f"How much should I save every month to build {lakh(tgt)} in {y} years if I earn {r}% a year?",
               f"I want {lakh(tgt)} in {y} years. What monthly investment do I need at {r}% expected return?"])
        return q, [C.goal_saving(target=tgt, years=y, annual_return_pct=r)["result"]["monthly_required"]]
    def lump():
        pv, r, y = p([50000, 100000, 200000, 500000, 1000000]), p([6.5, 7, 7.5, 8, 9, 12]), p([3, 5, 7, 10, 15])
        q = f"If I put {lakh(pv)} in an investment growing at {r}% a year compounded annually, what will it be worth after {y} years?"
        return q, [C.tvm(solve_for="fv", annual_rate_pct=r, years=y, pv=pv)["result"]["fv"]]
    def years():
        r = p([6, 7, 8, 9, 10, 12, 15])
        q = f"At {r}% annual compound interest, how many years does it take for money to double?"
        return q, [C.tvm(solve_for="years", annual_rate_pct=r, pv=1, fv=2)["result"]["years"]]
    def cagr():
        a, b, y = p([1, 2, 5, 10]) * 1e5, None, p([3, 4, 5, 6, 8, 10])
        b = round(a * rng.uniform(1.3, 3.5), -3)
        q = f"My investment grew from {lakh(a)} to ₹{b:,.0f} in {y} years. What is the CAGR?"
        return q, [C.tvm(solve_for="annual_rate_pct", years=y, pv=a, fv=b)["result"]["annual_rate_pct"]]
    def pv_needed():
        fv, r, y = p([5, 10, 20, 50]) * 1e5, p([6, 7, 8, 10]), p([5, 8, 10, 15])
        q = f"How much do I need to invest today as a lump sum to have {lakh(fv)} in {y} years at {r}% a year?"
        return q, [C.tvm(solve_for="pv", annual_rate_pct=r, years=y, fv=fv)["result"]["pv"]]
    def ear():
        r, m, name = p([8, 9, 10, 12, 14, 18, 24, 36]), *p([(12, "monthly"), (4, "quarterly"), (2, "half-yearly"), (365, "daily")])
        q = f"A {r}% annual rate is compounded {name}. What is the effective annual rate?"
        return q, [C.effective_annual_rate(nominal_pct=r, periods_per_year=m)["result"]["effective_pct"]]
    def real():
        n, i = p([6, 7, 7.5, 8, 9, 10, 12]), p([4, 4.5, 5, 5.5, 6, 7])
        q = f"My FD gives {n}% and inflation is {i}%. What is my real rate of return?"
        return q, [C.real_return(nominal_pct=n, inflation_pct=i)["result"]["real_return_pct"]]
    def npv():
        c0 = p([50000, 100000, 250000, 500000]); k = p([3, 4, 5]); r = p([8, 10, 12, 14])
        flows = [round(c0 * rng.uniform(0.2, 0.45), -3) for _ in range(k)]
        res = C.npv_irr(cashflows=[-c0] + flows, discount_rate_pct=r)["result"]
        cf = ", ".join(f"₹{x:,.0f}" for x in flows)
        q = p([f"A project costs ₹{c0:,} and returns {cf} over the next {k} years. At a {r}% discount rate, what is the NPV and the IRR?",
               f"Should we take a project with an upfront cost of ₹{c0:,} and inflows of {cf} in years 1 to {k}? Use {r}% as the required return and give the NPV and profitability index."])
        return q, ([res["npv"], res["irr_pct"]] if "IRR" in q else [res["npv"], res["profitability_index"]])
    def bond():
        f_, c, y, ytm = 1000, p([6, 7, 7.5, 8, 9]), p([3, 5, 7, 10]), p([6.5, 7, 8, 8.5, 9.5])
        q = f"A ₹{f_:,} bond pays a {c}% annual coupon for {y} years. If the market yield is {ytm}%, what is its price?"
        return q, [C.bond_metrics(face=f_, coupon_pct=c, years=y, freq=1, ytm_pct=ytm)["result"]["price"]]
    def tax_new():
        sal = p([6, 8, 10, 12, 12.75, 13, 15, 18, 20, 25, 30, 40, 60]) * 1e5
        q = p([f"My salary is {lakh(sal)} a year. How much income tax will I pay under the new regime for FY 2026-27?",
               f"How much tax on a {lakh(sal)} salary in the new tax regime?",
               f"Meri salary {lakh(sal)} per annum hai. New regime mein kitna tax lagega?"])
        return q, [T.income_tax(gross_salary=sal, regime="new")["result"]["by_regime"]["new"]["total_tax"]]
    def tax_both():
        sal, basic = p([12, 15, 18, 24, 30]) * 1e5, None
        basic = sal * 0.4; hra = basic * 0.5; rent = p([15000, 20000, 25000, 35000]) * 12; c80 = p([0, 50000, 150000])
        city = p(["Mumbai", "Pune", "Bengaluru", "Jaipur", "Lucknow"])
        res = T.income_tax(gross_salary=sal, hra_received=hra, basic_da=basic, rent_paid=rent, city=city, section_80c=c80)["result"]
        q = (f"I earn {lakh(sal)} a year including HRA of ₹{hra:,.0f}. My basic is ₹{basic:,.0f}, I pay rent of ₹{rent / 12:,.0f} a month in "
             f"{city}" + (f" and invest ₹{c80:,} under 80C" if c80 else "") + ". Which tax regime is better for FY 2026-27 and how much tax would I pay?")
        return q, [res["by_regime"]["new"]["total_tax"], res["by_regime"]["old"]["total_tax"]]
    def stmt():
        s, c, d, i, t = p([4, 6, 8, 12]) * 1e5, None, p([30000, 50000, 80000]), p([10000, 20000, 40000]), p([22, 25, 30])
        c = round(s * rng.uniform(0.45, 0.65), -3)
        q = f"A company has sales of ₹{s:,.0f}, costs of ₹{c:,.0f}, depreciation of ₹{d:,}, interest of ₹{i:,} and a {t}% tax rate. What is its net income?"
        return q, [F.solve_financials(sales=s, costs=c, depreciation=d, interest=i, tax_rate_pct=t, targets=["net_income"])["result"]["net_income"]]
    def dupont():
        pm, tat, de = p([4, 5, 6, 8, 10]), p([0.8, 1.2, 1.5, 2.0]), p([0.5, 0.8, 1.0, 1.5])
        q = f"A firm has a profit margin of {pm}%, total asset turnover of {tat} and a debt-equity ratio of {de}. What is its ROE?"
        return q, [F.solve_financials(profit_margin_pct=pm, total_asset_turnover=tat, debt_equity_ratio=de, targets=["roe_pct"])["result"]["roe_pct"]]
    def gst():
        price, g = p([1200, 2500, 4999, 15000, 45000]), p([5, 12, 18, 28])
        disc = p([10, 15, 20])
        q = f"An item costs ₹{price:,} and has a {disc}% discount, then {g}% GST is added on the discounted price. What do I pay?"
        return q, [round(price * (1 - disc / 100) * (1 + g / 100), 2)]
    return {"emi": (emi, 60), "loan_rate": (loan_rate, 40), "loan_months": (loan_months, 25), "sip": (sip, 50), "goal": (goal, 50),
            "lump": (lump, 30), "years": (years, 15), "cagr": (cagr, 30), "pv": (pv_needed, 25), "ear": (ear, 25), "real": (real, 25),
            "npv": (npv, 55), "bond": (bond, 30), "tax_new": (tax_new, 55), "tax_both": (tax_both, 45), "statement": (stmt, 35),
            "dupont": (dupont, 25), "gst": (gst, 20)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "train/data/questions_v2/calc_v2.jsonl"))
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--prefix", default="calc2", help="id prefix, so several seeds can be generated without id clashes")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    out, seen = [], set()
    for name, (fn, n) in families(rng).items():
        made = tries = 0
        while made < n and tries < n * 5:
            tries += 1
            try:
                q, gold = fn()
            except Exception:
                continue
            tone = rng.choice(list(TONES))
            q = TONES[tone](q)
            if q in seen or any(g is None for g in gold):
                continue
            seen.add(q)
            made += 1
            out.append({"id": f"{a.prefix}-{name}-{made}", "question": q, "type": "numeric", "topic": name,
                        "gold_answer": "; ".join(f"{g:.4f}" for g in gold), "source": f"calc_v2:{name}", "synthetic": True})
    Path(a.out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out), encoding="utf-8")
    print(f"{len(out)} questions -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
