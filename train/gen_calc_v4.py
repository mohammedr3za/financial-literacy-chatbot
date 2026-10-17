"""v4 training questions for v3's gaps (docs/finetune_v4_plan.md, step B): the tools v3 never saw (tax_rules,
capital_gains, freelance income_tax), weak calculation families (step-up SIP, PPF, quarterly FD, annuity due, sinking
fund, growing annuity, uneven cash flows, credit card, CA-style SI/CI/depreciation) and Indian-language calculations.
Wording and numbers are our own; gold values come from FinBot's tools. prepare_sft drops anything close to a test question.
usage: uv run python train/gen_calc_v4.py [--seed N] [--out path]"""
import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from finbot.tools import calculators as C, capital_gains as CG, income_tax as T  # noqa: E402


def lakh(x: float) -> str:
    if x >= 1e7:
        return f"₹{x / 1e7:g} crore"
    return f"₹{x / 1e5:g} lakh" if x >= 1e5 else f"₹{x:,.0f}"


def families(rng: random.Random):
    p, u = rng.choice, rng.uniform

    def tax_rule():
        y = p(["2025-26", "2026-27"])
        items = [
            (f"For FY {y}, what standard deduction does a salaried person get under the new tax regime?", T.STANDARD_DEDUCTION["new"]),
            (f"How big is the standard deduction in the old regime for a salaried employee in FY {y}?", T.STANDARD_DEDUCTION["old"]),
            (f"Under the new regime for FY {y}, up to what taxable income is the tax fully rebated under section 87A?", T.REBATE["new"][0]),
            (f"What is the most a salaried person can earn in FY {y} under the new regime and still owe no income tax?",
             T.REBATE["new"][0] + T.STANDARD_DEDUCTION["new"]),
            (f"What is the maximum rebate under section 87A in the new regime for FY {y}?", T.REBATE["new"][1]),
            (f"How much can I claim under section 80C in the old regime for FY {y}?", T.LIMITS["80c"]),
            (f"What extra deduction does section 80CCD(1B) give for my own NPS contribution in the old regime, FY {y}?", T.LIMITS["80ccd1b"]),
            (f"What is the 80D limit for health insurance for myself and family if we are all below 60, FY {y}?", 25_000),
            (f"For senior-citizen parents, how much 80D deduction is allowed for their health insurance in FY {y}?", 50_000),
            (f"How much home-loan interest on a self-occupied house can I deduct under section 24(b) in the old regime, FY {y}?", T.LIMITS["24b"]),
            (f"Up to what amount are long-term capital gains on listed shares tax-free each year in FY {y}?", CG.LTCG_EXEMPTION),
            (f"At what rate are long-term gains on listed shares taxed in FY {y}, above the yearly exemption?", CG.LTCG_RATE_PCT),
            (f"What is the tax rate on short-term capital gains from listed equity shares in FY {y}?", CG.STCG_RATE_PCT),
            (f"What health and education cess is added on income tax in FY {y}?", T.CESS_PCT),
        ]
        q, gold = p(items)
        return q, [gold]

    def cap_gains():
        buy = round(u(0.5, 8), 1) * 1e5; sell = round(buy * u(1.05, 1.9), -3); m = p([3, 6, 8, 10, 11, 12, 14, 18, 24, 30, 36])
        asset = p(["shares", "equity mutual fund units"])
        used = p([0, 0, 0, 50_000, 100_000]) if m > 12 else 0
        q = p([f"I bought {asset} for {lakh(buy)} and sold them after {m} months for ₹{sell:,.0f}. How much capital gains tax do I pay?",
               f"Sold {asset} purchased {m} months ago: cost {lakh(buy)}, sale value ₹{sell:,.0f}. What is the tax on the gain?"])
        if used:
            q += f" I have already booked ₹{used:,} of long-term gains on shares earlier this year."
        r = CG.capital_gains(purchase_value=buy, sale_value=sell, holding_months=m, ltcg_exemption_used=used,
                             asset="equity_fund" if "fund" in asset else "listed_equity")["result"]
        return q, [r["tax_before_cess"]] if r["tax_before_cess"] else [r["gain"]]

    def freelance():
        inc = p([12.5, 13, 14, 15, 16.5, 18, 20, 22, 25, 30]) * 1e5     # above the zero-tax line, so the gold is a real tax
        q = p([f"I'm a freelance designer earning {lakh(inc)} a year (no salary). How much income tax do I pay under the new regime for FY 2025-26?",
               f"My consulting income is {lakh(inc)} this year and I have no salaried job. What is my new-regime tax for FY 2025-26?"])
        r = T.income_tax(gross_salary=0, other_income=inc, financial_year="2025-26", regime="new")["result"]["by_regime"]["new"]
        return q, [r["total_tax"]] if r["total_tax"] else [r["taxable_income"]]

    def sip_stepup():
        m, r, y, s = p([3000, 5000, 8000, 10000, 15000]), p([10, 11, 12, 13]), p([10, 12, 15, 20]), p([5, 10, 15])
        q = f"I start a ₹{m:,} monthly SIP and raise it by {s}% every year for {y} years. At {r}% a year, what corpus will I have?"
        return q, [C.sip_future_value(monthly=m, annual_return_pct=r, years=y, step_up_pct=s)["result"]["corpus"]]

    def ppf():
        d, y = p([50000, 75000, 100000, 120000, 150000]), p([15, 20])
        q = f"If I put ₹{d:,} into PPF at the beginning of every financial year for {y} years at 7.1%, what will the account be worth?"
        return q, [C.tvm(solve_for="fv", pmt=d, annual_rate_pct=7.1, years=y, when="begin")["result"]["fv"]]

    def fd_quarterly():
        pv, r, y = p([1, 2, 3, 5, 10]) * 1e5, p([6.5, 6.8, 7, 7.25, 7.5, 7.75]), p([1, 2, 3, 5])
        q = f"A {y}-year cumulative FD of {lakh(pv)} pays {r}% compounded quarterly. What is the maturity amount?"
        return q, [C.tvm(solve_for="fv", pv=pv, annual_rate_pct=r, years=y, periods_per_year=4)["result"]["fv"]]

    def annuity_due():
        pmt, r, n = round(u(5, 50), 0) * 1000, p([6, 7, 8, 9, 10]), p([5, 8, 10, 12])
        q = f"A lease needs ₹{pmt:,.0f} at the start of each year for {n} years. At {r}% a year, what is its present value?"
        return q, [C.tvm(solve_for="pv", pmt=pmt, annual_rate_pct=r, years=n)["result"]["annuity_values"]["pv_due"]]

    def sinking():
        fv, r, y = p([5, 10, 20, 50]) * 1e5, p([6, 7, 8]), p([5, 8, 10])
        q = f"A school wants {lakh(fv)} in {y} years to replace its buses. How much must go into a sinking fund at the end of each year at {r}%?"
        return q, [C.tvm(solve_for="pmt", fv=fv, annual_rate_pct=r, years=y)["result"]["pmt"]]

    def growing_annuity():
        c, g, r, n = round(u(20, 200), 0) * 1000, p([2, 3, 4, 5]), p([7, 8, 9, 10, 12]), p([10, 15, 20, 25, 30])
        q = f"A pension pays ₹{c:,.0f} next year and then grows {g}% a year for {n} years in total. At {r}% a year, what is it worth today?"
        return q, [C.perpetuity(cash_flow=c, rate_pct=r, growth_pct=g, periods=n)["result"]["pv"]]

    def uneven_pv():
        r, k = p([8, 9, 10, 12]), p([3, 4, 5])
        flows = [round(u(10, 90), 0) * 1000 for _ in range(k)]
        q = f"At {r}% a year, what is the present value of receiving " + ", ".join(f"₹{x:,.0f}" for x in flows) + f" at the end of years 1 to {k}?"
        return q, [C.npv_irr(cashflows=[-1] + flows, discount_rate_pct=r)["result"]["pv_of_future_cash_flows"]]

    def credit_card():
        m = p([3, 3.25, 3.5, 3.75, 3.99])
        q = f"My credit card charges {m}% a month on unpaid balances. What is the effective annual interest rate?"
        return q, [C.effective_annual_rate(nominal_pct=m * 12, periods_per_year=12)["result"]["effective_pct"]]

    def ca_si():
        pr, r, t = round(u(10, 200), 0) * 1000, p([6, 7, 8, 9, 10, 12]), p([2, 3, 4, 5])
        i = pr * r * t / 100
        if rng.random() < 0.5:
            return f"Find the simple interest on ₹{pr:,.0f} at {r}% per annum for {t} years.", [i]
        return f"₹{pr:,.0f} is lent at {r}% simple interest a year. How much interest is earned in {t} years, and what is the amount?", [i, pr + i]

    def ca_ci():
        pr, r, t, (k, name) = round(u(10, 200), 0) * 1000, p([6, 8, 10, 12]), p([2, 3]), p([(2, "half-yearly"), (4, "quarterly")])
        fv = C.tvm(solve_for="fv", pv=pr, annual_rate_pct=r, years=t, periods_per_year=k)["result"]["fv"]
        return f"Find the compound interest on ₹{pr:,.0f} at {r}% per annum for {t} years, compounded {name}.", [fv - pr]

    def ca_depr():
        cost, r, n = round(u(50, 500), 0) * 1000, p([10, 15, 20]), p([2, 3, 4])
        if rng.random() < 0.5:
            return (f"A machine costs ₹{cost:,.0f} and depreciates at {r}% a year on the reducing balance. What is its value after {n} years?",
                    [cost * (1 - r / 100) ** n])
        now = round(cost * (1 - r / 100) ** n, 2)
        return f"After {n} years of {r}% reducing-balance depreciation a vehicle is worth ₹{now:,.2f}. What did it cost?", [now / (1 - r / 100) ** n]

    def language():
        kind = p(["emi_hi", "sip_hi", "fd_ta", "emi_ta", "sip_hinglish", "tax_hinglish", "ci_hi", "sip_tanglish"])
        if kind in ("emi_hi", "emi_ta"):
            P, r, y = p([10, 15, 20, 30, 40]) * 1e5, p([8, 8.5, 9, 9.5]), p([10, 15, 20])
            emi = C.loan_emi(principal=P, annual_rate_pct=r, months=y * 12)["result"]["emi"]
            q = (f"₹{P / 1e5:g} लाख का होम लोन {r}% ब्याज पर {y} साल के लिए लिया है। मासिक EMI कितनी होगी?" if kind == "emi_hi" else
                 f"₹{P / 1e5:g} லட்சம் வீட்டுக் கடன் {r}% வட்டியில் {y} ஆண்டுகளுக்கு. மாதாந்திர EMI எவ்வளவு?")
            return q, [emi]
        if kind in ("sip_hi", "sip_hinglish", "sip_tanglish"):
            m, r, y = p([2000, 5000, 10000]), p([10, 12]), p([10, 15, 20])
            fv = C.sip_future_value(monthly=m, annual_return_pct=r, years=y)["result"]["corpus"]
            q = {"sip_hi": f"हर महीने ₹{m:,} की SIP {y} साल तक {r}% सालाना रिटर्न पर करूँ तो कितना पैसा बनेगा?",
                 "sip_hinglish": f"Agar main har mahine ₹{m:,} ki SIP {y} saal tak karun aur {r}% return mile, toh kitna corpus banega?",
                 "sip_tanglish": f"Maasam ₹{m:,} SIP {y} varusham pannina, {r}% return-la evlo corpus varum?"}[kind]
            return q, [fv]
        if kind == "fd_ta":
            pv, r, y = p([1, 2, 5]) * 1e5, p([7, 7.5]), p([2, 3, 5])
            fv = C.tvm(solve_for="fv", pv=pv, annual_rate_pct=r, years=y, periods_per_year=4)["result"]["fv"]
            return f"₹{pv / 1e5:g} லட்சம் FD {r}% வட்டியில் {y} ஆண்டுகள், காலாண்டுக்கு ஒருமுறை வட்டி சேர்க்கப்படுகிறது. முதிர்வுத் தொகை எவ்வளவு?", [fv]
        if kind == "ci_hi":
            pr, r, t = p([50000, 100000, 200000]), p([8, 10]), p([2, 3])
            fv = C.tvm(solve_for="fv", pv=pr, annual_rate_pct=r, years=t)["result"]["fv"]
            return f"₹{pr:,} पर {r}% सालाना चक्रवृद्धि ब्याज से {t} साल में कुल कितनी रकम होगी?", [fv]
        sal = p([13, 14, 15, 16, 18, 20, 24, 25, 30]) * 1e5        # above ₹12.75 lakh, so the tax is not zero
        r = T.income_tax(gross_salary=sal, financial_year="2025-26", regime="new")["result"]["by_regime"]["new"]
        return f"Meri salary ₹{sal / 1e5:g} lakh saalana hai. FY 2025-26 new regime mein kitna income tax lagega?", \
            [r["total_tax"]] if r["total_tax"] else [r["taxable_income"]]

    return {"tax_rule": (tax_rule, 60), "cap_gains": (cap_gains, 60), "freelance": (freelance, 30), "sip_stepup": (sip_stepup, 40),
            "ppf": (ppf, 30), "fd_quarterly": (fd_quarterly, 25), "annuity_due": (annuity_due, 20), "sinking": (sinking, 20),
            "growing_annuity": (growing_annuity, 20), "uneven_pv": (uneven_pv, 25), "credit_card": (credit_card, 15),
            "ca_si": (ca_si, 25), "ca_ci": (ca_ci, 20), "ca_depr": (ca_depr, 20), "language": (language, 150)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "train/data/questions_v2/calc_v5_v4gaps.jsonl"))
    ap.add_argument("--seed", type=int, default=5151)
    ap.add_argument("--prefix", default="calc5")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    out, seen = [], set()
    for name, (fn, n) in families(rng).items():
        made = tries = 0
        while made < n and tries < n * 8:
            tries += 1
            try:
                q, gold = fn()
            except Exception:
                continue
            if q in seen or not gold or any(g is None for g in gold):
                continue
            seen.add(q)
            made += 1
            out.append({"id": f"{a.prefix}-{name}-{made}", "question": q, "type": "numeric", "topic": name,
                        "gold_answer": "; ".join(f"{g:.4f}" for g in gold), "source": f"calc_v5:{name}", "synthetic": True})
    Path(a.out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out), encoding="utf-8")
    print(f"{len(out)} questions -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
