"""Money formatting. Default: Indian rupees with lakh/crore grouping. For questions in US dollars the graph switches
the style to "USD": $ sign and Western 3-digit grouping (SPEC 4.2)."""
import contextvars
import re

MONEY_STYLE: contextvars.ContextVar[str] = contextvars.ContextVar("money_style", default="INR")


def symbol() -> str:
    return "$" if MONEY_STYLE.get() == "USD" else "₹"


def _western(whole: str) -> str:
    return f"{int(whole):,}"


def format_inr(value: float, decimals: int = 2) -> str:
    """Format money: '₹23,23,390.76' (Indian grouping) by default, '$2,323,390.76' in USD style."""
    sign = "-" if value < 0 else ""
    text = f"{abs(value):.{decimals}f}"
    whole, _, frac = text.partition(".")
    if MONEY_STYLE.get() == "USD":
        return f"{sign}${_western(whole)}" + (f".{frac}" if decimals else "")
    if len(whole) > 3:
        head, tail = whole[:-3], whole[-3:]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        whole = ",".join(groups + [tail])
    return f"{sign}₹{whole}" + (f".{frac}" if decimals else "")


def inr_words(value: float) -> str:
    """Short human form: '₹23.23 lakh', '₹1.50 crore', or the full amount below one lakh."""
    a = abs(value)
    sign = "-" if value < 0 else ""
    if MONEY_STYLE.get() == "USD":
        return format_inr(value)
    if a >= 1e7:
        return f"{sign}₹{a / 1e7:.2f} crore"
    if a >= 1e5:
        return f"{sign}₹{a / 1e5:.2f} lakh"
    return format_inr(value)


def group(value: float, decimals: int = 0) -> str:
    """Indian digit grouping for table cells, accounting style: negatives in brackets, no currency sign."""
    if value is None:
        return ""
    body = format_inr(abs(value), decimals).lstrip("$₹")
    return f"({body})" if round(value, decimals) < 0 else body


MAX_TABLE_ROWS = 60


def make_table(title: str, columns: list[str], body: list[list[str]], total: list[str] | None = None,
               note: str | None = None, bold_rows: list[int] | None = None) -> dict:
    """A table for the UI and the markdown answer. Cells are already formatted strings."""
    n = len(body)
    if n > MAX_TABLE_ROWS:
        body = body[:MAX_TABLE_ROWS]
        note = (note + " " if note else "") + f"Showing the first {MAX_TABLE_ROWS} of {n} rows."
    rows = body + ([total] if total else [])
    bold = sorted(set((bold_rows or []) + ([len(rows) - 1] if total else [])))
    numeric = re.compile(r"^[\(\-−]?[₹]?[\d,]+(\.\d+)?\)?(%| days|×)?$")
    align = []
    for j in range(len(columns)):
        cells = [r[j] for r in body if j < len(r) and str(r[j]).strip() not in ("", "–")]
        align.append("right" if j > 0 and cells and sum(bool(numeric.match(str(c))) for c in cells) >= 0.8 * len(cells) else "left")
    return {"title": title, "columns": columns, "rows": rows, "bold_rows": bold, "note": note, "align": align}
