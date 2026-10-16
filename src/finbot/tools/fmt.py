"""Indian-rupee formatting: lakh/crore digit grouping."""


def format_inr(value: float, decimals: int = 2) -> str:
    """Format a number as rupees with Indian grouping, e.g. 2323390.76 -> '₹23,23,390.76'."""
    sign = "-" if value < 0 else ""
    text = f"{abs(value):.{decimals}f}"
    whole, _, frac = text.partition(".")
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
    if a >= 1e7:
        return f"{sign}₹{a / 1e7:.2f} crore"
    if a >= 1e5:
        return f"{sign}₹{a / 1e5:.2f} lakh"
    return format_inr(value)
