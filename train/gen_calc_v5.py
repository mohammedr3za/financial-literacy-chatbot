"""v5 training questions for what v4.1 still fails (docs/finetune_v4_plan.md follow-up, SPEC changelog 2026-10-19):
statement ratios from raw items, CA-style business mathematics (partnership ratios, bills of exchange, average due date,
depreciation, finding a rate or a time, perpetuity rate) and multi-step Indian goals and loans (inflate then save,
prepayment, payoff time, flat-rate loans, post-tax real return, emergency fund), plus Indian-language versions.
Wording and numbers are our own; gold values come from FinBot's tools or first-principles formulas. prepare_sft drops
anything close to a test question. usage: uv run python train/gen_calc_v5.py [--seed N] [--out path]"""
import argparse
import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from finbot.tools import calculators as C, financial_solver as F  # noqa: E402
from finbot.tools.fmt import format_inr  # noqa: E402


def usd(x: float) -> str:
    return f"${x:,.0f}"


def inr(x: float, rng: random.Random) -> str:
    """₹1,20,000 or ₹120,000 or ₹1.2 lakh: people write all three."""
    style = rng.random()
    if style < 0.5:
        return format_inr(x, 0)
    if style < 0.8 or x < 1e5:
        return f"₹{x:,.0f}"
    return f"₹{x / 1e7:g} crore" if x >= 1e7 else f"₹{x / 1e5:g} lakh"


def balance_after(P: float, r: float, n: int, k: int) -> tuple[float, float]:
    """EMI and outstanding balance after k of n monthly payments."""
    i = r / 1200
    emi = P * i * (1 + i) ** n / ((1 + i) ** n - 1)
    return emi, P * (1 + i) ** k - emi * ((1 + i) ** k - 1) / i


def families(rng: random.Random):
    p, u = rng.choice, rng.uniform
    rs = lambda x: inr(x, rng)  # noqa: E731

    # ---------------------------------------------------------------- statements and ratios
    def liquidity():
        cash, ar, inv = round(u(20, 90)) * 1000, round(u(40, 160)) * 1000, round(u(60, 220)) * 1000
        pre = p([0, 0, round(u(3, 15)) * 1000]); cl = round((cash + ar + inv + pre) / u(1.2, 2.4), -3)
        extra = f", prepaid expenses {usd(pre)}" if pre else ""
        q = p([f"A company's balance sheet shows cash {usd(cash)}, accounts receivable {usd(ar)}, inventory {usd(inv)}{extra} and "
               f"current liabilities of {usd(cl)}. Calculate the current ratio and the quick ratio.",
               f"Current assets: cash {usd(cash)}, debtors {usd(ar)}, stock {usd(inv)}{extra}. Current liabilities total {usd(cl)}. "
               f"What are the current and acid-test ratios?"])
        return q, [(cash + ar + inv + pre) / cl, (cash + ar) / cl]

    def turnover():
        sales = round(u(500, 3000)) * 1000; cogs = round(sales * u(0.55, 0.8), -3)
        inv, ar = round(cogs / u(3, 9), -3), round(sales / u(5, 12), -3)
        r = F.solve_financials(targets=["inventory_turnover", "days_sales_in_inventory", "receivables_turnover",
                                        "days_sales_in_receivables"], sales=sales, cogs=cogs, inventory=inv, receivables=ar)["result"]
        q = (f"A wholesaler had sales of {usd(sales)} and cost of goods sold of {usd(cogs)}. Inventory is {usd(inv)} and receivables "
             f"are {usd(ar)}. Find the inventory turnover, days' sales in inventory, receivables turnover and days' sales in "
             f"receivables (365-day year).")
        return q, [r["inventory_turnover"], r["days_sales_in_inventory"], r["receivables_turnover"], r["days_sales_in_receivables"]]

    def dupont():
        sales = round(u(800, 5000)) * 1000; ni = round(sales * u(0.03, 0.12), -2)
        ta = round(sales / u(0.6, 2.2), -3); te = round(ta * u(0.3, 0.7), -3)
        r = F.solve_financials(targets=["profit_margin_pct", "total_asset_turnover", "equity_multiplier", "roe_pct"],
                               sales=sales, net_income=ni, total_assets=ta, total_equity=te)["result"]
        q = p([f"Net income {usd(ni)}, sales {usd(sales)}, total assets {usd(ta)} and total equity {usd(te)}. Use the DuPont identity: "
               f"what are the profit margin, total asset turnover, equity multiplier and ROE?",
               f"Break down the return on equity of a firm with sales of {usd(sales)}, net income of {usd(ni)}, assets of {usd(ta)} "
               f"and equity of {usd(te)} into margin, turnover and leverage."])
        return q, [r["profit_margin_pct"], r["total_asset_turnover"], r["equity_multiplier"], r["roe_pct"]]

    def coverage():
        ebit = round(u(200, 2000)) * 1000; intr = round(ebit / u(2, 12), -3); dep = round(ebit * u(0.1, 0.5), -3)
        r = F.solve_financials(targets=["times_interest_earned", "cash_coverage"], ebit=ebit, interest=intr, depreciation=dep)["result"]
        q = (f"A company reports EBIT of {usd(ebit)}, interest expense of {usd(intr)} and depreciation of {usd(dep)}. "
             f"What are its times interest earned and cash coverage ratios?")
        return q, [r["times_interest_earned"], r["cash_coverage"]]

    def market():
        ni = round(u(1, 30)) * 100000; sh = round(u(2, 40)) * 100000; pe = u(8, 30); price = round(ni / sh * pe, 2)
        div = round(ni * u(0.2, 0.6), -3)
        r = F.solve_financials(targets=["eps", "pe_ratio", "dividends_per_share", "payout_ratio_pct"], net_income=ni, shares=sh,
                               price=price, dividends=div)["result"]
        q = (f"A listed company earned {usd(ni)} this year, has {sh:,.0f} shares outstanding trading at ${price:,.2f}, and paid total "
             f"dividends of {usd(div)}. Find EPS, the P/E ratio, dividends per share and the payout ratio.")
        return q, [r["eps"], r["pe_ratio"], r["dividends_per_share"], r["payout_ratio_pct"]]

    def growth():
        roe, roa, po = p([10, 12, 14, 15, 18, 20, 22]), p([5, 6, 7, 8, 9, 11]), p([20, 25, 30, 40, 50, 60])
        if rng.random() < 0.5:
            q = f"A firm has an ROE of {roe}% and pays out {po}% of its earnings as dividends. What is its sustainable growth rate?"
            return q, [F.solve_financials(targets=["sustainable_growth_pct"], roe_pct=roe, payout_ratio_pct=po)["result"]["sustainable_growth_pct"]]
        q = f"With a return on assets of {roa}% and a dividend payout ratio of {po}%, what is the company's internal growth rate?"
        return q, [F.solve_financials(targets=["internal_growth_pct"], roa_pct=roa, payout_ratio_pct=po)["result"]["internal_growth_pct"]]

    # ---------------------------------------------------------------- CA-style business mathematics
    def partnership():
        caps = [round(u(50, 500)) * 1000 for _ in range(p([2, 3]))]
        months = [12] + [p([3, 4, 6, 8, 9, 10]) for _ in caps[1:]]
        profit = round(u(30, 300)) * 1000
        names = ["Asha", "Bharat", "Chitra"][:len(caps)]
        parts = [f"{n} invested {rs(c)} for {m} months" for n, c, m in zip(names, caps, months)]
        q = (f"In a partnership, " + "; ".join(parts) + f". The year's profit is {rs(profit)}, shared in the ratio of capital × time. "
             f"How much does each partner get?")
        w = [c * m for c, m in zip(caps, months)]
        return q, [profit * x / sum(w) for x in w]

    def ratio_divide():
        kind = p(["divide", "mean", "fourth"])
        if kind == "divide":
            a = [p([2, 3, 4, 5, 7]) for _ in range(3)]; tot = round(u(10, 200)) * 1000
            q = f"Divide {rs(tot)} among three friends in the ratio {a[0]}:{a[1]}:{a[2]}. How much does each get?"
            return q, [tot * x / sum(a) for x in a]
        if kind == "mean":
            x, y = p([4, 9, 16, 25, 2, 8]), p([36, 49, 64, 81, 18, 50])
            return f"Find the mean proportional between {x} and {y}.", [math.sqrt(x * y)]
        a, b, c = p([3, 4, 5, 6]), p([8, 10, 12, 15]), p([9, 14, 21, 27])
        return f"Find the fourth proportional to {a}, {b} and {c}.", [b * c / a]

    def bills_discount():
        A, r, m = round(u(10, 200)) * 1000, p([6, 8, 9, 10, 12]), p([3, 4, 6, 9])
        t = m / 12; td = A * r * t / (100 + r * t); bd = A * r * t / 100
        q = p([f"A bill of exchange for {rs(A)} is due in {m} months. It is discounted at {r}% per annum. Find the true discount, "
               f"the banker's discount and the banker's gain.",
               f"A trader holds a bill of {rs(A)} payable after {m} months and discounts it with a bank at {r}% p.a. What are the "
               f"banker's discount, the true discount and the banker's gain?"])
        return q, [td, bd, bd - td]

    def avg_due_date():
        k = p([3, 4]); amts = [round(u(5, 80)) * 1000 for _ in range(k)]; days = sorted(rng.sample(range(15, 150, 5), k))
        lines = ", ".join(f"{rs(a)} due in {d} days" for a, d in zip(amts, days))
        q = (f"A customer owes {lines}. If all of it is paid in one lump sum without loss of interest to either side, after how "
             f"many days from today is the average due date?")
        return q, [sum(a * d for a, d in zip(amts, days)) / sum(amts)]

    def depr_slm():
        cost = round(u(100, 1500)) * 1000; sal = round(cost * u(0.05, 0.2), -3); n = p([5, 8, 10]); k = p([2, 3, 4])
        q = (f"A machine costs {rs(cost)}, has a useful life of {n} years and a scrap value of {rs(sal)}. Using the straight-line "
             f"method, find the yearly depreciation and the book value at the end of year {k}.")
        d = (cost - sal) / n
        return q, [d, cost - d * k]

    def depr_rate():
        cost = round(u(100, 1500)) * 1000; n = p([3, 4, 5, 6]); r_true = p([10, 12.5, 15, 18, 20, 25])
        val = round(cost * (1 - r_true / 100) ** n, -2)
        q = (f"An asset bought for {rs(cost)} is worth {rs(val)} after {n} years of reducing-balance depreciation. What yearly rate "
             f"of depreciation was used?")
        return q, [(1 - (val / cost) ** (1 / n)) * 100]

    def ci_rate():
        P, n = round(u(10, 300)) * 1000, p([2, 3, 4, 5, 6]); A = round(P * (1 + p([5, 6, 8, 9, 10, 12]) / 100) ** n * u(0.98, 1.02), -1)
        q = p([f"At what annual rate of compound interest will {rs(P)} grow to {rs(A)} in {n} years?",
               f"{rs(P)} invested today becomes {rs(A)} after {n} years, interest compounded yearly. Find the rate of interest."])
        return q, [C.tvm(solve_for="annual_rate_pct", pv=P, fv=A, years=n)["result"]["annual_rate_pct"]]

    def ci_time():
        P, r = round(u(10, 300)) * 1000, p([6, 7, 8, 9, 10, 12]); mult = p([1.5, 2, 2.5, 3])
        A = P * mult
        q = f"In how many years will {rs(P)} become {rs(A)} at {r}% per annum compounded annually?"
        return q, [math.log(mult) / math.log(1 + r / 100)]

    def perp_rate():
        if rng.random() < 0.5:
            pv = round(u(5, 200)) * 1000
            c = round(pv * p([4, 5, 6, 7.5, 8, 10]) / 100, 2)
            q = f"An investment of {rs(pv)} pays {rs(c)} at the end of every year forever. What annual rate of return does it give?"
            return q, [c / pv * 100]
        price, d, g = round(u(20, 200), 2), round(u(1, 8), 2), p([3, 4, 5, 6])
        q = (f"A share trades at ${price:,.2f}. Next year's dividend is expected to be ${d:,.2f}, growing {g}% a year forever. What "
             f"return are investors requiring?")
        return q, [d / price * 100 + g]

    # ---------------------------------------------------------------- multi-step Indian goals and loans
    def edu_goal():
        cost, i, n, r = p([10, 15, 20, 25, 30, 40]) * 1e5, p([6, 7, 8, 10]), p([8, 10, 12, 15, 18]), p([10, 11, 12])
        when = p(["start", "end"])
        fut = C.tvm(solve_for="fv", pv=cost, annual_rate_pct=i, years=n)["result"]["fv"]
        sip = C.goal_saving(target=fut, years=n, annual_return_pct=r, timing=when)["result"]["monthly_required"]
        goal = p(["an engineering degree", "an MBA", "my son's medical college", "my daughter's wedding", "a study-abroad year"])
        q = (f"{goal[0].upper() + goal[1:]} costs about {rs(cost)} today. Costs rise {i}% a year and we need the money in {n} years. "
             f"What will it cost then, and what monthly SIP (invested at the {when} of each month, {r}% a year) gets us there?")
        return q, [fut, sip]

    def retire_expense():
        e, i, n = round(u(25, 150)) * 1000, p([5, 6, 7]), p([15, 20, 25, 28, 30])
        q = (f"Our household spends {rs(e)} a month now. With {i}% inflation, what will the same lifestyle cost per month when I "
             f"retire in {n} years?")
        return q, [C.tvm(solve_for="fv", pv=e, annual_rate_pct=i, years=n)["result"]["fv"]]

    def prepay():
        P, r, yrs, k = p([20, 30, 40, 50, 60]) * 1e5, p([8.25, 8.5, 8.75, 9, 9.5]), p([15, 20, 25]), p([24, 36, 48, 60])
        X = p([2, 3, 5, 8]) * 1e5
        emi, bal = balance_after(P, r, yrs * 12, k)
        if rng.random() < 0.5:
            q = (f"I took a {rs(P)} home loan at {r}% for {yrs} years and have paid {k} EMIs. If I prepay {rs(X)} now and keep the "
                 f"EMI unchanged, how many months will be left?")
            return q, [C.loan_emi(principal=bal - X, annual_rate_pct=r, emi=emi)["result"]["months_exact"]]
        q = (f"Home loan of {rs(P)} at {r}% for {yrs} years; {k} EMIs are paid. I'll prepay {rs(X)} and ask the bank to keep the "
             f"tenure and cut the EMI instead. What is the current EMI and what will the new EMI be?")
        return q, [emi, C.loan_emi(principal=bal - X, annual_rate_pct=r, months=yrs * 12 - k)["result"]["emi"]]

    def payoff_time():
        B, r = p([5, 8, 12, 18, 25, 35]) * 1e5, p([8.5, 9, 10.5, 11, 12])
        m = round(B * (r / 1200) * u(1.3, 3.0), -3)
        q = p([f"I still owe {rs(B)} on my loan at {r}% interest and can pay {rs(m)} every month. How many months until it is fully "
               f"repaid?", f"Outstanding loan {rs(B)}, rate {r}% p.a., monthly payment {rs(m)}. How long will it take to close the loan?"])
        return q, [C.loan_emi(principal=B, annual_rate_pct=r, emi=m)["result"]["months_exact"]]

    def flat_loan():
        P, f, n = round(u(50, 500)) * 1000, p([7, 8, 9, 10, 12]), p([12, 18, 24, 36])
        emi = (P + P * f / 100 * n / 12) / n
        q = p([f"A dealer offers {rs(P)} of finance at {f}% flat for {n} months. What is the EMI, and what is the real "
               f"reducing-balance interest rate?", f"{rs(P)} personal loan, '{f}% flat rate', {n} monthly instalments: EMI and true "
               f"annual rate please."])
        return q, [emi, C.loan_emi(principal=P, months=n, emi=emi)["result"]["annual_rate_pct"]]

    def post_tax_real():
        r, slab, i = p([6.5, 7, 7.25, 7.5, 7.75, 8]), p([5, 20, 30]), p([4, 4.5, 5, 5.5, 6])
        eff = slab * 1.04; post = r * (1 - eff / 100)
        q = (f"My FD pays {r}% and I'm in the {slab}% slab ({eff:g}% with cess). With inflation at {i}%, what is my post-tax return and "
             f"my real post-tax return?")
        return q, [post, C.real_return(nominal_pct=post, inflation_pct=i)["result"]["real_return_pct"]]

    def emergency():
        e, emi, k = round(u(20, 120)) * 1000, p([0, 8000, 12000, 15000, 25000]), p([3, 6, 9, 12])
        extra = f" plus EMIs of {rs(emi)}" if emi else ""
        q = f"Monthly household expenses are {rs(e)}{extra}. How much should a {k}-month emergency fund hold?"
        return q, [(e + emi) * k]

    # ---------------------------------------------------------------- the same in Indian languages
    def language():
        kind = p(["flat_hinglish", "prepay_hi", "edu_hinglish", "emergency_hi", "ci_rate_ta", "post_tax_hinglish", "payoff_tanglish"])
        if kind == "flat_hinglish":
            P, f, n = round(u(50, 300)) * 1000, p([8, 9, 10, 12]), p([12, 24, 36])
            emi = (P + P * f / 100 * n / 12) / n
            return (f"Dealer ₹{P:,.0f} ka loan {f}% flat pe {n} mahine ke liye de raha hai. EMI kitni hogi aur asli reducing rate kya hai?",
                    [emi, C.loan_emi(principal=P, months=n, emi=emi)["result"]["annual_rate_pct"]])
        if kind == "prepay_hi":
            P, r, yrs, k, X = p([20, 30, 40]) * 1e5, p([8.5, 9]), p([20]), p([36, 60]), p([3, 5]) * 1e5
            emi, bal = balance_after(P, r, yrs * 12, k)
            return (f"₹{P / 1e5:g} लाख का होम लोन {r}% पर {yrs} साल के लिए है और {k} EMI भर चुका हूँ। अब ₹{X / 1e5:g} लाख प्रीपे करूँ और "
                    f"EMI वही रखूँ तो कितने महीने बचेंगे?", [C.loan_emi(principal=bal - X, annual_rate_pct=r, emi=emi)["result"]["months_exact"]])
        if kind == "edu_hinglish":
            cost, i, n, r = p([10, 15, 20]) * 1e5, p([6, 8]), p([10, 12, 15]), p([12])
            fut = C.tvm(solve_for="fv", pv=cost, annual_rate_pct=i, years=n)["result"]["fv"]
            sip = C.goal_saving(target=fut, years=n, annual_return_pct=r)["result"]["monthly_required"]
            return (f"Bachche ki padhai aaj ₹{cost / 1e5:g} lakh ki hai, kharcha {i}% saal badhta hai, {n} saal baad chahiye. Tab kitna "
                    f"lagega aur {r}% return pe har mahine (shuru mein) kitni SIP karni hogi?", [fut, sip])
        if kind == "emergency_hi":
            e, emi, k = round(u(20, 80)) * 1000, p([5000, 10000, 15000]), p([6])
            return (f"घर का मासिक खर्च ₹{e:,} है और ₹{emi:,} की EMI भी जाती है। {k} महीने का इमरजेंसी फंड कितना होना चाहिए?",
                    [(e + emi) * k])
        if kind == "ci_rate_ta":
            P, n = p([50000, 100000, 200000]), p([3, 5]); A = round(P * (1 + p([8, 10]) / 100) ** n, -1)
            return (f"₹{P:,} தொகை {n} ஆண்டுகளில் கூட்டு வட்டியில் ₹{A:,.0f} ஆகிறது. ஆண்டு வட்டி விகிதம் என்ன?",
                    [C.tvm(solve_for="annual_rate_pct", pv=P, fv=A, years=n)["result"]["annual_rate_pct"]])
        if kind == "post_tax_hinglish":
            r, slab, i = p([7, 7.5]), p([20, 30]), p([5, 6])
            post = r * (1 - slab * 1.04 / 100)
            return (f"FD {r}% de raha hai, main {slab}% slab mein hoon (cess milake {slab * 1.04:g}%). Inflation {i}% hai toh post-tax "
                    f"aur real post-tax return kitna hai?", [post, C.real_return(nominal_pct=post, inflation_pct=i)["result"]["real_return_pct"]])
        B, r = p([5, 10, 15]) * 1e5, p([9, 10]); m = round(B * (r / 1200) * u(1.5, 2.5), -3)
        return (f"Loan ₹{B / 1e5:g} lakh baaki irukku, {r}% vatti, maasam ₹{m:,.0f} kattuven. Evlo maasam-la loan mudiyum?",
                [C.loan_emi(principal=B, annual_rate_pct=r, emi=m)["result"]["months_exact"]])

    return {"liquidity": (liquidity, 25), "turnover": (turnover, 25), "dupont": (dupont, 25), "coverage": (coverage, 15),
            "market": (market, 15), "growth": (growth, 15), "partnership": (partnership, 25), "ratio_divide": (ratio_divide, 15),
            "bills_discount": (bills_discount, 25), "avg_due_date": (avg_due_date, 15), "depr_slm": (depr_slm, 15),
            "depr_rate": (depr_rate, 15), "ci_rate": (ci_rate, 15), "ci_time": (ci_time, 15), "perp_rate": (perp_rate, 15),
            "edu_goal": (edu_goal, 30), "retire_expense": (retire_expense, 15), "prepay": (prepay, 30),
            "payoff_time": (payoff_time, 15), "flat_loan": (flat_loan, 15), "post_tax_real": (post_tax_real, 15),
            "emergency": (emergency, 10), "language": (language, 50)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "train/data/questions_v2/calc_v6_v5gaps.jsonl"))
    ap.add_argument("--seed", type=int, default=6161)
    ap.add_argument("--prefix", default="calc6")
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
            if q in seen or not gold or any(g is None or not math.isfinite(g) for g in gold):
                continue
            seen.add(q)
            made += 1
            out.append({"id": f"{a.prefix}-{name}-{made}", "question": q, "type": "numeric", "topic": name,
                        "gold_answer": "; ".join(f"{g:.4f}" for g in gold), "source": f"calc_v6:{name}", "synthetic": True})
    Path(a.out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out), encoding="utf-8")
    print(f"{len(out)} questions -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
