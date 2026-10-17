"""Textbook-style calculation questions for fine-tune v3 (multi-part statement, cash-flow, capital-budgeting and annuity
problems, mostly in US dollars). v2 was weak exactly here: it skipped the tool on 15 of 34 textbook problems.
Wording and numbers are our own; gold values come from FinBot's tools. prepare_sft still drops anything close to a test
question. usage: uv run python train/gen_calc_v3.py [--seed N] [--out path]"""
import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from finbot.tools import calculators as C, financial_solver as F, statements as S  # noqa: E402


def usd(x: float) -> str:
    return f"${x:,.0f}"


def families(rng: random.Random):
    p, u = rng.choice, rng.uniform

    def income():
        s = round(u(300, 900), 0) * 1000; c = round(s * u(0.5, 0.7), -2); d = round(s * u(0.06, 0.15), -2); t = p([21, 22, 24, 25, 30, 35])
        q = (f"Last year a company had revenue of {usd(s)}, operating costs (excluding depreciation) of {usd(c)} and depreciation of "
             f"{usd(d)}. It has no debt and its tax rate is {t} percent. Work out EBIT, the tax bill, net income and operating cash flow.")
        r = F.solve_financials(targets=["ebit", "taxes", "net_income", "operating_cash_flow"], sales=s, costs=c, depreciation=d, tax_rate_pct=t)["result"]
        return q, [r["ebit"], r["taxes"], r["net_income"], r["operating_cash_flow"]]

    def income_interest():
        s = round(u(400, 1200), 0) * 1000; c = round(s * u(0.45, 0.65), -2); d = round(s * u(0.05, 0.12), -2)
        i = round(s * u(0.02, 0.06), -2); t = p([21, 23, 25, 34]); div = round(u(0.2, 0.5) * 0.1 * s, -2)
        q = (f"Prepare the income statement: sales {usd(s)}, costs {usd(c)}, depreciation {usd(d)}, interest paid {usd(i)}, tax rate "
             f"{t}%. The firm paid {usd(div)} in dividends. What are net income and the addition to retained earnings?")
        r = F.solve_financials(targets=["net_income", "addition_to_retained_earnings"], sales=s, costs=c, depreciation=d, interest=i,
                               tax_rate_pct=t, dividends=div)["result"]
        return q, [r["net_income"], r["addition_to_retained_earnings"]]

    def receivable_days():
        ni = round(u(80, 400), 0) * 1000; pm = p([5.5, 6.2, 7.5, 8.4, 9.6, 11]); ar = round(ni / (pm / 100) * u(0.06, 0.12), -1)
        cr = p([60, 70, 75, 80, 90])
        q = (f"A retailer earned net income of {usd(ni)} on a profit margin of {pm}%. Its accounts receivable stand at {usd(ar)} and "
             f"{cr}% of its sales are on credit. How many days' sales are tied up in receivables? Use a 365-day year.")
        r = F.solve_financials(targets=["days_sales_in_receivables"], net_income=ni, profit_margin_pct=pm, receivables=ar, credit_sales_pct=cr)
        return q, [r["result"]["days_sales_in_receivables"]]

    def roe_from_ratios():
        dr, roa = p([0.35, 0.4, 0.45, 0.5, 0.55, 0.6]), p([5.2, 6, 7.4, 8.5, 9.1, 10.5])
        q = f"A firm's total debt ratio is {dr} and its return on assets is {roa}%. What is its return on equity?"
        return q, [F.solve_financials(targets=["roe_pct"], debt_ratio=dr, roa_pct=roa)["result"]["roe_pct"]]

    def balance_equity():
        ca, nfa = round(u(20, 90), 0) * 1000, round(u(100, 400), 0) * 1000
        cl = round(ca * u(0.5, 0.95), -2); ltd = round(nfa * u(0.2, 0.5), -2)
        q = (f"From the balance sheet: current assets {usd(ca)}, net fixed assets {usd(nfa)}, current liabilities {usd(cl)} and "
             f"long-term debt {usd(ltd)}. What is shareholders' equity, and what is net working capital?")
        r = F.solve_financials(targets=["total_equity", "net_working_capital"], current_assets=ca, net_fixed_assets=nfa,
                               current_liabilities=cl, long_term_debt=ltd)["result"]
        return q, [r["total_equity"], r["net_working_capital"]]

    def cash_flow_stmt():
        ni, dep, div = round(u(40, 150), 0) * 1000, round(u(20, 80), 0) * 1000, round(u(5, 30), 0) * 1000
        ar = [round(u(20, 60), 0) * 1000]; ar.append(round(ar[0] * u(0.85, 1.3), -2))
        inv = [round(u(30, 90), 0) * 1000]; inv.append(round(inv[0] * u(0.85, 1.3), -2))
        ap = [round(u(15, 50), 0) * 1000]; ap.append(round(ap[0] * u(0.85, 1.3), -2))
        nfa = [round(u(200, 500), 0) * 1000]; nfa.append(round(nfa[0] * u(1.0, 1.3), -2))
        ltd = [round(u(80, 200), 0) * 1000]; ltd.append(round(ltd[0] * u(0.8, 1.3), -2))
        cs = [round(u(50, 150), 0) * 1000]; cs.append(cs[0] + p([0, 0, 10000, 20000]))
        cash0 = round(u(10, 40), 0) * 1000
        base = S.cash_flow_statement(net_income=ni, depreciation=dep, current_asset_changes={"Accounts receivable": ar, "Inventory": inv},
                                     current_liability_changes={"Accounts payable": ap}, investing={},
                                     financing={"Long-term debt": ltd, "Common stock": cs, "Dividends paid": -div},
                                     net_fixed_assets=nfa)["result"]
        cash1 = cash0 + base["net_change"]
        rows = [("Cash", cash0, cash1), ("Accounts receivable", *ar), ("Inventory", *inv), ("Net fixed assets", *nfa),
                ("Accounts payable", *ap), ("Long-term debt", *ltd), ("Common stock", *cs)]
        table = "\n".join(f"| {n} | {usd(a)} | {usd(b)} |" for n, a, b in rows)
        q = (f"Using the two balance sheets below, net income of {usd(ni)}, depreciation of {usd(dep)} and dividends of {usd(div)}, "
             f"prepare the statement of cash flows. What are cash flow from operations, from investing and from financing, and the "
             f"change in cash?\n\n| Item | Start of year | End of year |\n|---|---:|---:|\n{table}")
        return q, [base["cfo"], base["cfi"], base["cff"], base["net_change"]]

    def annuity_due():
        pmt, r, n = round(u(2, 20), 1) * 1000, p([4.5, 5, 6, 6.5, 7, 8, 9]), p([4, 5, 6, 8, 10, 12])
        q = (f"You will receive {usd(pmt)} a year for {n} years and the discount rate is {r}% a year. Find the present value and the "
             f"future value if the payments come at the end of each year, and again if they come at the start of each year.")
        v = C.tvm(solve_for="pv", pmt=pmt, annual_rate_pct=r, years=n)["result"]["annuity_values"]
        return q, [v["pv_ordinary"], v["fv_ordinary"], v["pv_due"], v["fv_due"]]

    def growing_perp():
        cost = round(u(0.6, 3), 2) * 1e6; cf = round(cost * u(0.05, 0.09), -3); g = p([2, 2.5, 3, 4, 5]); r = p([9, 10, 11, 12, 13])
        q = (f"An investment costs {usd(cost)} today and pays {usd(cf)} next year, with that cash flow growing at {g}% a year forever. "
             f"With a required return of {r}%, what is the NPV, and at what growth rate would the NPV be zero?")
        res = C.perpetuity(cash_flow=cf, rate_pct=r, growth_pct=g, investment=cost)["result"]
        return q, [res["npv"], res["break_even_growth_pct"]]

    def pi_two_projects():
        r = p([8, 9, 10, 11, 12, 14])
        def proj():
            c0 = round(u(20, 120), 0) * 1000
            return c0, [round(c0 * u(0.25, 0.6), -2) for _ in range(3)]
        (a0, af), (b0, bf) = proj(), proj()
        q = (f"Two mutually exclusive projects, required return {r}%. Project A costs {usd(a0)} and returns "
             f"{', '.join(usd(x) for x in af)} over three years; Project B costs {usd(b0)} and returns {', '.join(usd(x) for x in bf)}. "
             f"Give the NPV and profitability index of each project.")
        ra, rb = (C.npv_irr(cashflows=[-c] + f, discount_rate_pct=r)["result"] for c, f in ((a0, af), (b0, bf)))
        return q, [ra["npv"], ra["profitability_index"], rb["npv"], rb["profitability_index"]]

    def payback():
        c0, r, n = round(u(50, 400), 0) * 1000, p([8, 10, 12, 15]), p([4, 5, 6])
        flows = [round(c0 * u(0.2, 0.4), -2) for _ in range(n)]
        q = (f"A machine costs {usd(c0)} and saves {', '.join(usd(x) for x in flows)} in years 1 to {n}. What are the payback period "
             f"and the discounted payback period at {r}%?")
        res = C.npv_irr(cashflows=[-c0] + flows, discount_rate_pct=r)["result"]
        return q, [res["payback_years"], res["discounted_payback_years"]]

    def tax_shield():
        s = round(u(200, 800), 0) * 1000; c = round(s * u(0.4, 0.65), -2); d = round(s * u(0.08, 0.2), -2); t = p([21, 25, 30, 34])
        q = (f"A project has annual sales of {usd(s)}, cash costs of {usd(c)} and depreciation of {usd(d)}, taxed at {t}%. Compute the "
             f"operating cash flow and the depreciation tax shield.")
        ocf = F.solve_financials(targets=["operating_cash_flow"], sales=s, costs=c, depreciation=d, tax_rate_pct=t)["result"]["operating_cash_flow"]
        return q, [ocf, d * t / 100]

    def eac_two():
        r = p([8, 10, 12, 14])
        def m():
            return round(u(20, 90), 0) * 1000, round(u(3, 15), 0) * 1000, p([3, 4, 5, 6, 8])
        (c1, a1, l1), (c2, a2, l2) = m(), m()
        q = (f"Machine X costs {usd(c1)}, costs {usd(a1)} a year to run and lasts {l1} years. Machine Y costs {usd(c2)}, costs "
             f"{usd(a2)} a year and lasts {l2} years. At a {r}% discount rate, what is the equivalent annual cost of each?")
        e1, e2 = (abs(C.equivalent_annual_cost(purchase_cost=c, annual_cost=a, life_years=n, rate_pct=r)["result"]["eac"])
                  for c, a, n in ((c1, a1, l1), (c2, a2, l2)))
        return q, [e1, e2]

    def bond_semi():
        c, y, ytm = p([4, 5, 5.5, 6, 7, 8]), p([5, 8, 10, 15, 20]), p([4.5, 5.2, 6, 6.8, 7.5, 9])
        q = f"A $1,000 bond pays a {c}% coupon semiannually and matures in {y} years. If the yield to maturity is {ytm}%, what is the price?"
        return q, [C.bond_metrics(face=1000, coupon_pct=c, years=y, freq=2, ytm_pct=ytm)["result"]["price"]]

    def growth_coverage():
        roe, pay = p([10, 12, 14, 15.5, 18]), p([25, 30, 40, 50, 60])
        ebit, i, d = round(u(100, 500), 0) * 1000, round(u(10, 60), 0) * 1000, round(u(20, 80), 0) * 1000
        q = (f"A company's ROE is {roe}% and it pays out {pay}% of earnings as dividends. Its EBIT is {usd(ebit)}, interest expense "
             f"{usd(i)} and depreciation {usd(d)}. What are its sustainable growth rate and its times interest earned ratio?")
        g = F.solve_financials(targets=["sustainable_growth_pct"], roe_pct=roe, payout_ratio_pct=pay)["result"]["sustainable_growth_pct"]
        return q, [g, F.solve_financials(targets=["times_interest_earned"], ebit=ebit, interest=i)["result"]["times_interest_earned"]]

    return {"income": (income, 30), "income_interest": (income_interest, 25), "receivable_days": (receivable_days, 20),
            "roe_ratios": (roe_from_ratios, 15), "balance_equity": (balance_equity, 20), "cash_flow_stmt": (cash_flow_stmt, 40),
            "annuity_due": (annuity_due, 30), "growing_perp": (growing_perp, 25), "pi_two": (pi_two_projects, 30),
            "payback": (payback, 25), "tax_shield": (tax_shield, 25), "eac_two": (eac_two, 20), "bond_semi": (bond_semi, 20),
            "growth_coverage": (growth_coverage, 20)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "train/data/questions_v2/calc_v4_textbook.jsonl"))
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--prefix", default="calc4")
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
            if q in seen or any(g is None for g in gold):
                continue
            seen.add(q)
            made += 1
            out.append({"id": f"{a.prefix}-{name}-{made}", "question": q, "type": "numeric", "topic": name,
                        "gold_answer": "; ".join(f"{g:.4f}" for g in gold), "source": f"calc_v4:{name}", "synthetic": True})
    Path(a.out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out), encoding="utf-8")
    print(f"{len(out)} questions -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
