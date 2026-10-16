"""Writes bench/heldout_problems.yaml: 20 fresh problems (new wording and numbers) for judging fixes and the fine-tune.
Answers use standalone formulas (not finbot.tools). These questions are excluded from all training data."""
import math
from pathlib import Path

import yaml

from make_calc_answers import bond_price, bs, emi, irr, npv, sip  # same independent formulas


def annuity_due(i, n):
    return ((1 + i) ** n - 1) / i * (1 + i)


i11, i10 = 11 / 1200, 10 / 1200
P = [
    ("h01", "goal", "I want ₹50 lakh for my daughter's education in 12 years. How much should I invest every month if I can earn 11% a year?",
     5e6 / annuity_due(i11, 144), "INR"),
    ("h02", "years", "At 8% a year, how long will it take for my money to triple?", math.log(3) / math.log(1.08), "YEARS"),
    ("h03", "real", "My savings account pays 3.5% and inflation is 5.5%. What is my real return?", abs((1.035 / 1.055 - 1) * 100), "PCT"),
    ("h04", "ear", "A bank offers 7.5% compounded quarterly. What is the effective annual yield?", ((1 + 0.075 / 4) ** 4 - 1) * 100, "PCT"),
    ("h05", "emi", "I want to buy a car worth ₹8 lakh with a 5-year loan at 9.75%. What's my monthly EMI?", emi(8e5, 9.75, 60), "INR"),
    ("h06", "emi", "How much interest will I pay in total on a ₹35 lakh home loan at 8.9% for 25 years?", emi(35e5, 8.9, 300) * 300 - 35e5, "INR"),
    ("h07", "sip", "If I invest ₹7,500 every month for 12 years and earn 11% a year, what will it grow to?", sip(7500, 11, 12), "INR"),
    ("h08", "sip", "I start a ₹5,000 monthly SIP and raise it by 10% every year. At 12% a year for 20 years, how much will I have?",
     sip(5000, 12, 20, 10), "INR"),
    ("h09", "fv", "I put ₹2 lakh in an FD at 7.1% compounded quarterly for 5 years. What will I get at maturity?", 2e5 * (1 + 0.071 / 4) ** 20, "INR"),
    ("h10", "pv", "I need ₹25 lakh in 8 years for a house down payment. How much should I invest today at 9% a year?", 25e5 / 1.09 ** 8, "INR"),
    ("h11", "cagr", "A fund's NAV went from ₹42 to ₹118 in 7 years. What is its CAGR?", ((118 / 42) ** (1 / 7) - 1) * 100, "PCT"),
    ("h12", "npv", "A machine costs ₹5,00,000 and saves ₹1,40,000 a year for 5 years. At a 10% discount rate, what is the NPV?",
     npv(0.10, [-5e5] + [1.4e5] * 5), "INR"),
    ("h13", "irr", "I invest ₹1,00,000 and get back ₹25,000, ₹35,000, ₹45,000 and ₹30,000 over the next four years. What's the IRR?",
     irr([-1e5, 2.5e4, 3.5e4, 4.5e4, 3e4]), "PCT"),
    ("h14", "bond", "What is the price of a ₹1,000 bond with a 9% annual coupon maturing in 6 years if similar bonds yield 10%?",
     bond_price(1000, 9, 6, 1, 10), "PRICE"),
    ("h15", "bond", "A 10-year ₹1,000 bond pays 8% coupons twice a year. If market yields are 7%, what's its price?",
     bond_price(1000, 8, 10, 2, 7), "PRICE"),
    ("h16", "option", "Using Black-Scholes, what is a 3-month call worth if the stock is at ₹500, the strike is ₹520, the risk-free rate is 6.5% and volatility is 30%?",
     bs(500, 520, 0.25, 6.5, 30, "call"), "PRICE"),
    ("h17", "percent", "A laptop costs ₹62,000 before tax. With 18% GST, what's the final price?", 62000 * 1.18, "INR"),
    ("h18", "percent", "A ₹4,000 jacket has a 25% discount, and then 12% GST is charged on the discounted price. How much do I pay?",
     4000 * 0.75 * 1.12, "INR"),
    ("h19", "goal", "I have ₹3 lakh saved and want ₹40 lakh in 10 years. At 10% a year, how much must I add each month?",
     (4e6 - 3e5 * (1 + i10) ** 120) / annuity_due(i10, 120), "INR"),
    ("h20", "ratio", "A company has current assets of ₹450 crore and current liabilities of ₹300 crore. What is its current ratio?", 1.5, "X"),
]
out = {"about": "Held-out set written after the 27/30 baseline, to judge fixes and the fine-tune fairly. Never use for training.",
       "tolerance_pct": 0.5, "problems": [{"id": i, "topic": t, "question": q, "expected": round(v, 4), "unit": u} for i, t, q, v, u in P]}
Path("bench/heldout_problems.yaml").write_text(yaml.safe_dump(out, sort_keys=False, allow_unicode=True, width=160))
for i, t, q, v, u in P:
    print(f"{i} {t:7s} {v:>16,.4f} {u}")
