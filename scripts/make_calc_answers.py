"""Writes bench/calc_problems.yaml. Expected answers use standalone textbook formulas written
here, independently of finbot.tools, so the benchmark does not grade the bot with its own code."""
import math
from pathlib import Path

import yaml


def emi(p, annual, months):
    r = annual / 1200
    return p * r * (1 + r) ** months / ((1 + r) ** months - 1)


def sip(monthly, annual, years, step=0.0):
    i, bal, pay = annual / 1200, 0.0, monthly
    for m in range(int(years * 12)):
        if m and m % 12 == 0:
            pay *= 1 + step / 100
        bal = (bal + pay) * (1 + i)
    return bal


def npv(rate, cfs):
    return sum(c / (1 + rate) ** t for t, c in enumerate(cfs))


def irr(cfs):
    lo, hi = -0.99, 1.0
    for _ in range(300):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if npv(mid, cfs) > 0 else (lo, mid)
    return mid * 100


def bond_price(face, cpn, years, freq, ytm):
    c, y, n = face * cpn / 100 / freq, ytm / 100 / freq, int(years * freq)
    return sum(c / (1 + y) ** t for t in range(1, n + 1)) + face / (1 + y) ** n


def macaulay(face, cpn, years, ytm):
    c, y = face * cpn / 100, ytm / 100
    flows = [(t, c + (face if t == years else 0)) for t in range(1, years + 1)]
    price = sum(f / (1 + y) ** t for t, f in flows)
    return sum(t * f / (1 + y) ** t for t, f in flows) / price


def bs(S, K, T, r, sig, kind):
    N = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))  # noqa: E731
    r, sig = r / 100, sig / 100
    d1 = (math.log(S / K) + (r + sig * sig / 2) * T) / (sig * math.sqrt(T))
    d2 = d1 - sig * math.sqrt(T)
    return S * N(d1) - K * math.exp(-r * T) * N(d2) if kind == "call" else K * math.exp(-r * T) * N(-d2) - S * N(-d1)


def ytm_from_price(face, cpn, years, price):
    lo, hi = 0.0, 50.0
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if bond_price(face, cpn, years, 1, mid) > price else (lo, mid)
    return mid


P = [
    ("c01", "emi", "What is the monthly EMI on a ₹20 lakh home loan at 8.5% a year for 20 years?", emi(2e6, 8.5, 240), "INR"),
    ("c02", "emi", "What is the EMI on a ₹5 lakh car loan at 9.25% for 5 years?", emi(5e5, 9.25, 60), "INR"),
    ("c03", "emi", "On a ₹50 lakh home loan at 8.75% for 25 years, how much total interest will I pay?", emi(5e6, 8.75, 300) * 300 - 5e6, "INR"),
    ("c04", "emi", "What is the EMI on a ₹2,00,000 personal loan at 14% for 36 months?", emi(2e5, 14, 36), "INR"),
    ("c05", "emi", "What is the EMI on a ₹10 lakh education loan at 10.5% for 7 years?", emi(1e6, 10.5, 84), "INR"),
    ("c06", "sip", "If I invest ₹10,000 every month in a SIP for 10 years at 12% a year, what will the corpus be?", sip(1e4, 12, 10), "INR"),
    ("c07", "sip", "What will a ₹5,000 monthly SIP grow to in 20 years at 10% a year?", sip(5e3, 10, 20), "INR"),
    ("c08", "sip", "What will a ₹2,500 monthly SIP become after 15 years at 8% a year?", sip(2.5e3, 8, 15), "INR"),
    ("c09", "sip", "With a ₹15,000 monthly SIP at 12% for 5 years, how much gain will I make over what I invested?", sip(1.5e4, 12, 5) - 1.5e4 * 60, "INR"),
    ("c10", "sip", "A ₹10,000 monthly SIP at 12% for 10 years, increased by 10% every year: what is the final corpus?", sip(1e4, 12, 10, 10), "INR"),
    ("c11", "tvm", "What will ₹1,00,000 grow to in 5 years at 7% compounded annually?", 1e5 * 1.07 ** 5, "INR"),
    ("c12", "tvm", "How much must I invest today at 8% a year to have ₹10 lakh in 10 years?", 1e6 / 1.08 ** 10, "INR"),
    ("c13", "tvm", "An investment grew from ₹50,000 to ₹1,20,000 in 6 years. What is its CAGR?", ((1.2e5 / 5e4) ** (1 / 6) - 1) * 100, "PCT"),
    ("c14", "tvm", "How many years will money take to double at 9% a year compounded annually?", math.log(2) / math.log(1.09), "YEARS"),
    ("c15", "tvm", "How much must I save at the end of each month to reach ₹1 crore in 15 years at 12% a year compounded monthly?",
     1e7 * (0.01) / ((1.01) ** 180 - 1), "INR"),
    ("c16", "npv", "What is the NPV at 10% of a project costing ₹1,00,000 that returns ₹30,000, ₹40,000, ₹50,000 and ₹20,000 over four years?",
     npv(0.10, [-1e5, 3e4, 4e4, 5e4, 2e4]), "INR"),
    ("c17", "irr", "What is the IRR of investing ₹50,000 and receiving ₹15,000, ₹20,000 and ₹25,000 over the next three years?",
     irr([-5e4, 1.5e4, 2e4, 2.5e4]), "PCT"),
    ("c18", "irr", "A project costs ₹1,000 and returns ₹300, ₹400 and ₹500 over three years. What is its IRR?", irr([-1e3, 300, 400, 500]), "PCT"),
    ("c19", "npv", "What is the NPV at 12% of paying ₹2,00,000 today and receiving ₹60,000 a year for 5 years?",
     npv(0.12, [-2e5] + [6e4] * 5), "INR"),
    ("c20", "bond", "What is the price of a ₹1,000 face value bond with an 8% annual coupon, 10 years to maturity, if the yield is 9%?",
     bond_price(1000, 8, 10, 1, 9), "PRICE"),
    ("c21", "bond", "Price a ₹1,000 bond paying a 7% coupon semi-annually with 5 years to maturity at a 6% yield.",
     bond_price(1000, 7, 5, 2, 6), "PRICE"),
    ("c22", "bond", "A 5-year bond with a 10% annual coupon and ₹1,000 face value trades at ₹927.90. What is its yield to maturity?",
     ytm_from_price(1000, 10, 5, 927.90), "PCT"),
    ("c23", "bond", "What is the Macaulay duration of a 5-year, 10% annual coupon bond priced at par?", macaulay(1000, 10, 5, 10), "YEARS"),
    ("c24", "option", "Using Black-Scholes, price a European call with spot 42, strike 40, 6 months to expiry, 10% risk-free rate and 20% volatility.",
     bs(42, 40, 0.5, 10, 20, "call"), "PRICE"),
    ("c25", "option", "Using Black-Scholes, price a European put with spot 42, strike 40, 6 months to expiry, 10% risk-free rate and 20% volatility.",
     bs(42, 40, 0.5, 10, 20, "put"), "PRICE"),
    ("c26", "option", "Black-Scholes price of a 1-year at-the-money European call on a stock at ₹100 with 7% risk-free rate and 25% volatility?",
     bs(100, 100, 1, 7, 25, "call"), "PRICE"),
    ("c27", "percent", "A phone costs ₹45,000 before GST. What is the price including 18% GST?", 45000 * 1.18, "INR"),
    ("c28", "percent", "What is 15% of ₹2,45,000, plus 18% GST on that amount?", 245000 * 0.15 * 1.18, "INR"),
    ("c29", "percent", "If my FD pays 8% and inflation is 6%, what is my real return?", (1.08 / 1.06 - 1) * 100, "PCT"),
    ("c30", "percent", "What is the effective annual rate of 12% a year compounded monthly?", ((1 + 0.01) ** 12 - 1) * 100, "PCT"),
]

if __name__ == "__main__":
    out = {
        "about": "30 calculation problems. Expected answers come from standalone formulas in scripts/make_calc_answers.py, "
                 "not from FinBot's tools. Team: spot-check at least 5 in a spreadsheet and record who checked them.",
        "tolerance_pct": 0.5,
        "spot_checked_by": [],
        "problems": [{"id": i, "topic": t, "question": q, "expected": round(v, 4), "unit": u} for i, t, q, v, u in P],
    }
    Path("bench/calc_problems.yaml").write_text(yaml.safe_dump(out, sort_keys=False, allow_unicode=True, width=140))
    for i, t, q, v, u in P:
        print(f"{i} {t:7s} {v:>16,.4f} {u}")
