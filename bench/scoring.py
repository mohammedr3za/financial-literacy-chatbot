"""Numeric answer extraction and grading for calculation problems (same rule for every bot)."""
import re

_NUM = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{2,3})+|\d+)(\.\d+)?\s*(lakhs?|lacs?|crores?|cr\b|%)?", re.I)
_MULT = {"lakh": 1e5, "lakhs": 1e5, "lac": 1e5, "lacs": 1e5, "crore": 1e7, "crores": 1e7, "cr": 1e7}


def extract_numbers(text: str) -> list[float]:
    out = []
    for whole, frac, unit in _NUM.findall(text or ""):
        v = float(whole.replace(",", "") + (frac or ""))
        out.append(v * _MULT.get((unit or "").lower().strip(), 1.0))
    return out


def calc_correct(answer: str, question: str, expected: float, tol_pct: float = 0.5) -> tuple[bool, float | None]:
    """True if any number in the answer is within tol_pct of expected. Numbers that merely repeat the
    question are ignored unless they equal the expected value."""
    def close(a, b):
        return abs(a - b) <= abs(b) * tol_pct / 100 + 1e-9
    from_question = extract_numbers(question)
    for v in extract_numbers(answer):
        if any(close(v, q) for q in from_question) and not close(v, expected):
            continue
        if close(v, expected):
            return True, v
    return False, None
