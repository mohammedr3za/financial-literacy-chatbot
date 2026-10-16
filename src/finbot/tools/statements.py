"""Financial-statement tools (SPEC 4.1): ratios, balance-sheet check, cash flow statement (indirect method).

Amounts are in whatever unit the user gives (₹, ₹ lakh, ₹ crore); results use the same unit.
Every function returns {"inputs", "formula", "result", "formatted", "table"} and raises ValueError on bad input.
"""
from .fmt import group, make_table

_NUM_FIELDS = ("current_assets", "current_liabilities", "inventory", "cash", "receivables", "payables", "total_assets",
               "total_liabilities", "total_equity", "total_debt", "revenue", "cogs", "gross_profit", "ebit",
               "interest_expense", "net_income", "operating_cash_flow", "capex", "depreciation")

# key, group, label, formula, needed inputs, function, display kind
_RATIOS = [
    ("current_ratio", "Liquidity", "Current ratio", "Current assets ÷ Current liabilities",
     ("current_assets", "current_liabilities"), lambda d: d["current_assets"] / d["current_liabilities"], "x"),
    ("quick_ratio", "Liquidity", "Quick ratio", "(Current assets − Inventory) ÷ Current liabilities",
     ("current_assets", "inventory", "current_liabilities"), lambda d: (d["current_assets"] - d["inventory"]) / d["current_liabilities"], "x"),
    ("cash_ratio", "Liquidity", "Cash ratio", "Cash ÷ Current liabilities",
     ("cash", "current_liabilities"), lambda d: d["cash"] / d["current_liabilities"], "x"),
    ("working_capital", "Liquidity", "Working capital", "Current assets − Current liabilities",
     ("current_assets", "current_liabilities"), lambda d: d["current_assets"] - d["current_liabilities"], "amt"),
    ("debt_to_equity", "Leverage", "Debt to equity", "Total debt ÷ Total equity",
     ("total_debt", "total_equity"), lambda d: d["total_debt"] / d["total_equity"], "x"),
    ("debt_ratio", "Leverage", "Debt ratio", "Total liabilities ÷ Total assets",
     ("total_liabilities", "total_assets"), lambda d: d["total_liabilities"] / d["total_assets"], "x"),
    ("equity_multiplier", "Leverage", "Equity multiplier", "Total assets ÷ Total equity",
     ("total_assets", "total_equity"), lambda d: d["total_assets"] / d["total_equity"], "x"),
    ("interest_coverage", "Leverage", "Interest coverage", "EBIT ÷ Interest expense",
     ("ebit", "interest_expense"), lambda d: d["ebit"] / d["interest_expense"], "x"),
    ("gross_margin_pct", "Profitability", "Gross margin", "(Revenue − COGS) ÷ Revenue",
     ("revenue", "cogs"), lambda d: (d["revenue"] - d["cogs"]) / d["revenue"] * 100, "pct"),
    ("operating_margin_pct", "Profitability", "Operating margin", "EBIT ÷ Revenue",
     ("ebit", "revenue"), lambda d: d["ebit"] / d["revenue"] * 100, "pct"),
    ("net_margin_pct", "Profitability", "Net margin", "Net income ÷ Revenue",
     ("net_income", "revenue"), lambda d: d["net_income"] / d["revenue"] * 100, "pct"),
    ("roa_pct", "Profitability", "Return on assets", "Net income ÷ Total assets",
     ("net_income", "total_assets"), lambda d: d["net_income"] / d["total_assets"] * 100, "pct"),
    ("roe_pct", "Profitability", "Return on equity", "Net income ÷ Total equity",
     ("net_income", "total_equity"), lambda d: d["net_income"] / d["total_equity"] * 100, "pct"),
    ("asset_turnover", "Efficiency", "Asset turnover", "Revenue ÷ Total assets",
     ("revenue", "total_assets"), lambda d: d["revenue"] / d["total_assets"], "x"),
    ("inventory_turnover", "Efficiency", "Inventory turnover", "COGS ÷ Inventory",
     ("cogs", "inventory"), lambda d: d["cogs"] / d["inventory"], "x"),
    ("days_inventory", "Efficiency", "Days inventory", "Inventory × 365 ÷ COGS",
     ("inventory", "cogs"), lambda d: d["inventory"] * 365 / d["cogs"], "days"),
    ("days_sales_outstanding", "Efficiency", "Days sales outstanding", "Receivables × 365 ÷ Revenue",
     ("receivables", "revenue"), lambda d: d["receivables"] * 365 / d["revenue"], "days"),
    ("days_payables", "Efficiency", "Days payables", "Payables × 365 ÷ COGS",
     ("payables", "cogs"), lambda d: d["payables"] * 365 / d["cogs"], "days"),
    ("free_cash_flow", "Cash", "Free cash flow", "Operating cash flow − Capex",
     ("operating_cash_flow", "capex"), lambda d: d["operating_cash_flow"] - abs(d["capex"]), "amt"),
    ("ebitda", "Cash", "EBITDA", "EBIT + Depreciation",
     ("ebit", "depreciation"), lambda d: d["ebit"] + d["depreciation"], "amt"),
]


def _show(v: float, kind: str) -> str:
    return {"x": f"{v:.2f}", "pct": f"{v:.2f}%", "days": f"{v:.1f} days", "amt": group(v, 2 if abs(v) < 1000 else 0)}[kind]


def financial_ratios(**items) -> dict:
    unknown = set(items) - set(_NUM_FIELDS)
    if unknown:
        raise ValueError(f"Unknown items: {sorted(unknown)}. Use: {', '.join(_NUM_FIELDS)}.")
    d = {k: float(v) for k, v in items.items() if v is not None}
    if len(d) < 2:
        raise ValueError("Give at least two statement items, e.g. current_assets and current_liabilities.")
    if "cogs" not in d and "gross_profit" in d and "revenue" in d:
        d["cogs"] = d["revenue"] - d["gross_profit"]
    result, rows = {}, []
    for key, grp, label, formula, needs, fn, kind in _RATIOS:
        if all(n in d for n in needs):
            try:
                v = fn(d)
            except ZeroDivisionError:
                continue
            result[key] = round(v, 4)
            rows.append([grp, label, formula, _show(v, kind)])
    if all(k in result for k in ("net_margin_pct", "asset_turnover", "equity_multiplier")):
        dupont = result["net_margin_pct"] * result["asset_turnover"] * result["equity_multiplier"]
        result["dupont_roe_pct"] = round(d["net_income"] / d["revenue"] * d["revenue"] / d["total_assets"]
                                         * d["total_assets"] / d["total_equity"] * 100, 4)
        rows.append(["Profitability", "DuPont ROE", "Net margin × Asset turnover × Equity multiplier", f"{dupont:.2f}%"])
    if not result:
        raise ValueError("These items do not allow any standard ratio. Add matching items, e.g. revenue with net_income.")
    table = make_table("Financial ratios", ["Group", "Ratio", "Formula", "Value"], rows,
                       note="Amounts are in the same unit as the inputs.")
    return {"inputs": d, "formula": "see table", "result": result, "formatted": {}, "table": table}


def _as_dict(x) -> dict:
    return x.model_dump() if hasattr(x, "model_dump") else dict(x)


def balance_sheet_check(assets: dict, liabilities: dict, equity: dict) -> dict:
    if not assets:
        raise ValueError("assets must list at least one item with its amount.")
    sections = [("Assets", {k: float(v) for k, v in assets.items()}),
                ("Liabilities", {k: float(v) for k, v in (liabilities or {}).items()}),
                ("Equity", {k: float(v) for k, v in (equity or {}).items()})]
    ta, tl, te = (sum(sec.values()) for _, sec in sections)
    if ta == 0:
        raise ValueError("Total assets are zero.")
    pct = lambda v: f"{v / ta * 100:.1f}%"  # noqa: E731
    rows, bold = [], []
    for name, sec in sections:
        for j, (item, v) in enumerate(sec.items()):
            rows.append([name if j == 0 else "", item, group(v), pct(v)])
        bold.append(len(rows))
        rows.append(["", f"Total {name.lower()}", group(sum(sec.values())), pct(sum(sec.values()))])
    diff = round(ta - (tl + te), 2)
    total = ["", "Total liabilities + equity", group(tl + te), pct(tl + te)]
    note = "Balanced: assets equal liabilities plus equity." if abs(diff) < 0.5 else \
        f"Not balanced: assets exceed liabilities plus equity by {group(diff)}." if diff > 0 else \
        f"Not balanced: liabilities plus equity exceed assets by {group(-diff)}."
    table = make_table("Balance sheet (common-size, % of total assets)", ["Section", "Item", "Amount", "% of total assets"],
                       rows, total=total, note=note, bold_rows=bold)
    return {"inputs": {"assets": sections[0][1], "liabilities": sections[1][1], "equity": sections[2][1]},
            "formula": "Assets = Liabilities + Equity",
            "result": {"total_assets": ta, "total_liabilities": tl, "total_equity": te, "balanced": abs(diff) < 0.5,
                       "difference": diff},
            "formatted": {}, "table": table}


def cash_flow_statement(net_income: float, non_cash_items=None, working_capital=None, investing=None, financing=None,
                        opening_cash: float | None = None, closing_cash: float | None = None) -> dict:
    rows, bold = [["Operating activities", "Net income", group(net_income)]], []
    cfo = float(net_income)
    for it in map(_as_dict, non_cash_items or []):
        cfo += float(it["amount"])
        rows.append(["", it["item"], group(float(it["amount"]))])
    for it in map(_as_dict, working_capital or []):
        kind = it.get("type")
        if kind not in ("current_asset", "current_liability"):
            raise ValueError(f"{it.get('item')}: type must be current_asset or current_liability.")
        if it.get("change") is not None:
            change = float(it["change"])
        elif it.get("opening") is not None and it.get("closing") is not None:
            change = float(it["closing"]) - float(it["opening"])
        else:
            raise ValueError(f"{it.get('item')}: give either change, or both opening and closing balances.")
        effect = -change if kind == "current_asset" else change
        cfo += effect
        word = "Increase" if change > 0 else "Decrease" if change < 0 else "No change"
        rows.append(["", f"{word} in {it['item']}", group(effect)])
    bold.append(len(rows))
    rows.append(["", "Net cash from operating activities", group(cfo)])
    totals = {}
    for section, items in (("Investing activities", investing), ("Financing activities", financing)):
        sub = 0.0
        for j, it in enumerate(map(_as_dict, items or [])):
            sub += float(it["amount"])
            rows.append([section if j == 0 else "", it["item"], group(float(it["amount"]))])
        bold.append(len(rows))
        rows.append(["" if items else section, f"Net cash from {section.lower()}", group(sub)])
        totals[section] = sub
    cfi, cff = totals["Investing activities"], totals["Financing activities"]
    net = cfo + cfi + cff
    bold.append(len(rows))
    rows.append(["", "Net change in cash", group(net)])
    result = {"cfo": round(cfo, 2), "cfi": round(cfi, 2), "cff": round(cff, 2), "net_change": round(net, 2)}
    note = None
    if opening_cash is not None:
        computed = float(opening_cash) + net
        result["computed_closing_cash"] = round(computed, 2)
        rows.append(["", "Opening cash", group(float(opening_cash))])
        bold.append(len(rows))
        rows.append(["", "Closing cash", group(computed)])
        if closing_cash is not None:
            gap = round(float(closing_cash) - computed, 2)
            result["reconciles"] = abs(gap) < 0.5
            result["unexplained_difference"] = gap
            note = ("Reconciles with the given closing cash." if abs(gap) < 0.5 else
                    f"Does not reconcile: the given closing cash is {group(float(closing_cash))}, a gap of {group(gap)}. "
                    "An item is missing or misclassified.")
    table = make_table("Cash flow statement (indirect method)", ["Section", "Item", "Amount"], rows, note=note, bold_rows=bold)
    return {"inputs": {"net_income": net_income, "opening_cash": opening_cash, "closing_cash": closing_cash},
            "formula": "CFO = Net income + non-cash items − increase in current assets + increase in current liabilities; "
                       "Net change = CFO + CFI + CFF",
            "result": result, "formatted": {}, "table": table}
