"""Follow-up chips under each answer (SPEC 7.1), chosen by code from the tools used and the pre-check flags."""

BY_TOOL = {
    "financial_ratios": ["Is the company healthy? Give me the key ratios", "Which ratio is the weakest, and why?", "Should I make this investment?"],
    "balance_sheet_check": ["Is the company healthy? Give me the key ratios", "Should I make this investment?"],
    "cash_flow_statement": ["Is the company healthy? Give me the key ratios", "Explain each section of this cash flow statement",
                            "Should I make this investment?"],
    "loan_emi": ["What if the interest rate rises to 9.5%?", "What would the EMI be over 15 years instead?",
                 "How much interest do I pay in total?"],
    "sip_future_value": ["What if I step up the SIP by 10% every year?", "How much should I save each month to reach ₹1 crore?",
                         "Is a 12% return realistic? What are the risks?"],
    "goal_saving": ["What if I step up the SIP by 10% every year?", "What if returns are only 10% a year?",
                    "What if I already have ₹5 lakh saved?"],
    "tvm": ["What is the real return after 6% inflation?", "How long would it take to double at this rate?"],
    "npv_irr": ["Should I make this investment?", "What if the discount rate is 12%?"],
    "bond_metrics": ["What happens to the price if yields rise by 1%?", "Explain duration in simple words"],
    "black_scholes": ["Price the put option too", "What does delta mean here?"],
    "real_return": ["How much would ₹1 lakh be worth in real terms after 10 years?", "Which investments usually beat inflation?"],
    "effective_annual_rate": ["Compare this with quarterly compounding", "What is the real return after 6% inflation?"],
}
BY_FLAG = {
    "statement_data": ["Is the company healthy? Give me the key ratios", "Should I make this investment?"],
    "guarantee_seeking": ["What returns are realistic for this kind of investment?", "How do I check if an adviser is SEBI-registered?"],
    "personal_advice_request": ["How do I check if an adviser is SEBI-registered?", "What are the risks?"],
    "high_risk_topic": ["What are the risks?", "How is this taxed in India?"],
    "regulatory_topic": ["Where can I verify this officially?", "Explain this in simple words"],
}
DEFAULT = ["Explain this in simple words", "What are the risks?", "Should I make this investment?"]


def suggest(trace: dict, limit: int = 3) -> list[str]:
    out: list[str] = []
    for t in trace.get("tools", []):
        out += BY_TOOL.get(t["name"], [])
    for flag, items in BY_FLAG.items():
        if trace.get("flags", {}).get(flag):
            out += items
    out += DEFAULT
    seen, picked = set(), []
    for s in out:
        if s not in seen:
            seen.add(s)
            picked.append(s)
    return picked[:limit]
