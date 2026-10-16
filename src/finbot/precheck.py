"""Deterministic pre-check: personal-data redaction, Indian amount normalization, and risk flags.

Runs before anything reaches the model (SPEC 3.1).
"""
import re
import unicodedata
from dataclasses import dataclass, field

from .routing import detect as detect_calc_types

# ---- personal data -------------------------------------------------------------
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_UPI = re.compile(r"\b[\w.-]{2,}@[a-zA-Z]{2,}\b")
_PAN = re.compile(r"\b[A-Za-z]{5}\d{4}[A-Za-z]\b")
_IFSC = re.compile(r"\b[A-Za-z]{4}0[A-Za-z0-9]{6}\b")
_CARD = re.compile(r"\b(?:\d[ -]?){12,18}\d\b")
_AADHAAR = re.compile(r"\b[2-9]\d{3}[ -]?\d{4}[ -]?\d{4}\b")
_PHONE = re.compile(r"(?:\+91[ -]?|\b0)?\b[6-9]\d{9}\b")
_ACCOUNT = re.compile(r"(?i)\b(?:a/?c|acct|account)(?:\s*(?:no\.?|number|#))?\s*[:\-]?\s*(\d[\d ]{7,19}\d)")


def _luhn_ok(digits: str) -> bool:
    nums = [int(d) for d in digits][::-1]
    total = sum(n if i % 2 == 0 else (n * 2 - 9 if n * 2 > 9 else n * 2) for i, n in enumerate(nums))
    return total % 10 == 0


def redact(text: str) -> tuple[str, list[str]]:
    found: list[str] = []

    def sub(pattern, label, s, check=None, group=0):
        def repl(m):
            if check and not check(m.group(group)):
                return m.group(0)
            found.append(label)
            if group:
                return m.group(0).replace(m.group(group), f"[REDACTED {label}]")
            return f"[REDACTED {label}]"
        return pattern.sub(repl, s)

    text = sub(_ACCOUNT, "ACCOUNT NUMBER", text, group=1)
    text = sub(_EMAIL, "EMAIL", text)
    text = sub(_UPI, "UPI ID", text)
    text = sub(_CARD, "CARD NUMBER", text, check=lambda s: _luhn_ok(re.sub(r"\D", "", s)))
    text = sub(_AADHAAR, "AADHAAR", text)
    text = sub(_PHONE, "PHONE", text)
    text = sub(_PAN, "PAN", text)
    text = sub(_IFSC, "IFSC", text)
    return text, sorted(set(found))


# ---- Indian amounts --------------------------------------------------------------
_UNITS = {"lakh": 1e5, "lakhs": 1e5, "lac": 1e5, "lacs": 1e5, "crore": 1e7, "crores": 1e7, "cr": 1e7, "k": 1e3}
_WORD_AMOUNT = re.compile(r"(?i)(?<![\w.])(\d+(?:\.\d+)?)\s*(lakhs?|lacs?|crores?|cr|k)\b\.?")
# Hindi and Tamil amount words (after NFC, "ड़" is "ड" + nukta): लाख/लाखों -> lakh, करोड़/करोड़ों -> crore,
# லட்சம் -> lakh, கோடி -> crore. No \b: Indic vowel signs are not word characters for the regex engine.
_INDIC_AMOUNT = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*(लाखों|लाख|करोड\u093c?ों|करोड\u093c?|லட்சம்|லட்சங்கள்|கோடி)")


def _indic_unit(word: str) -> float:
    return 1e5 if word.startswith(("लाख", "லட்ச")) else 1e7


_GROUPED = re.compile(r"(?<![\d,])(\d{1,3}(?:,\d{2})+,\d{3}|\d{1,3}(?:,\d{3})+)(?:\.\d+)?(?![\d,])")


def _plain(v: float) -> str:
    return str(int(v)) if float(v).is_integer() else f"{v:.2f}"


_QUANTITY = re.compile(r"(?i)(?<![a-z\d])(₹\s*\d|rs\.?\s*\d|\$\s*\d|\d[\d,]*(\.\d+)?\s*(%|per ?cent|lakhs?|lacs?|crores?|cr\b|k\b|"
                       r"लाख|करोड|லட்சம்|கோடி)|\d[\d,]*\.\d+|\d{2,})")


def has_quantity(text: str) -> bool:
    """A real quantity: ₹ amount, percentage, lakh/crore/k, a decimal, or a number of 2+ digits. Digits inside words
    ("eli5", "ITR2") and lone single digits do not count, so concept questions are not forced into calculations."""
    return bool(_QUANTITY.search(text.replace("[REDACTED", "")))


def is_tabular(text: str) -> bool:
    """Pasted statements: several lines carrying numbers, or table separators."""
    numeric_lines = [ln for ln in text.splitlines() if re.search(r"\d", ln)]
    return len(numeric_lines) >= 3 and ("|" in text or "\t" in text or len(numeric_lines) >= 4)


def normalize_amounts(text: str) -> tuple[str, list[dict]]:
    found: list[dict] = []

    def word(m):
        value = float(m.group(1)) * _UNITS[m.group(2).lower()]
        found.append({"text": m.group(0).strip(), "value": value})
        return f"{m.group(0).rstrip()} (= {_plain(value)})"

    def grouped(m):
        value = float(m.group(0).replace(",", ""))
        found.append({"text": m.group(0), "value": value})
        return f"{m.group(0)} (= {_plain(value)})"

    def indic(m):
        value = float(m.group(1)) * _indic_unit(m.group(2))
        found.append({"text": m.group(0).strip(), "value": value})
        return f"{m.group(0).rstrip()} (= {_plain(value)})"

    text = _WORD_AMOUNT.sub(word, text)
    text = _INDIC_AMOUNT.sub(indic, text)
    if not is_tabular(text):  # in a pasted table, annotating every cell only adds noise
        text = _GROUPED.sub(grouped, text)
    return text, found


# ---- flags -------------------------------------------------------------------------
_GUARANTEE = re.compile(r"(?i)\b(guarantee[sd]?|assured|sure[- ]?shot|risk[- ]?free (return|profit|investment)|double my money|"
                        r"definitely (give|make|earn|return)|confirm (that|this|it)|100% safe|no risk|can'?t lose|will (it )?(surely|definitely))\b")
_ADVICE = re.compile(r"(?i)\b(should i (buy|sell|invest|put|hold|exit|switch)|which (stock|share|fund|mutual fund|scheme|coin|crypto|etf|ipo)s? (should|to|can|is best)|"
                     r"best (stock|share|fund|mutual fund|coin|crypto|ipo|etf)s? (to|for)|where should i (invest|put)|"
                     r"(tell|suggest|recommend) (me )?(what|which|a|some|good)\s*(stock|share|fund|scheme|coin|to buy)|what should i (buy|invest))")
_HIGH_RISK = re.compile(r"(?i)\b(crypto\w*|bitcoin|ethereum|btc|eth|nfts?|f&o|futures? and options?|options? trading|intraday|leveraged (trading|positions?|etfs?)|trading on leverage|"
                        r"margin trading|penny stocks?|forex|binary options?|grey market|gmp|day trading|derivatives? trading)\b")
_CALC = re.compile(r"(?i)\b(calculate|compute|how much|what (will|would) (be|i|my)|emi|sip|future value|present value|maturity|"
                   r"cagr|irr|npv|ytm|yield|bond price|price of (a|the) bond|option price|black[- ]scholes|interest on|returns? on|"
                   r"total interest|corpus|duration of|break[- ]?even|percent(age)? of|% of|gst)\b"
                   r"|कितनी|कितना|कितने|गणना|எவ்வளவு|கணக்கிடு")      # Hindi / Tamil "how much", "calculate"
_STATEMENT = re.compile(r"(?i)\b(balance sheet|profit (and|&) loss|p&l|income statement|cash flow statement|cash flows?|"
                        r"ratios?|working capital|current assets?|current liabilit\w+|reserves and surplus|net worth|ebitda|ebit|"
                        r"debtors|creditors|depreciation|dupont|roe|roa|liquidity|leverage)\b")
_STATEMENT_DOC = re.compile(r"(?i)\b(balance sheet|income statement|profit (and|&) loss|p&l|cash flow statement)\b")
# corporate-finance items and ratios (textbook problems); "sales" counts only next to income-statement words
_CORPORATE = re.compile(r"(?i)\b(earnings per share|eps|p/e|price[- ](to[- ])?earnings|market[- ]to[- ]book|book value per share|"
                        r"enterprise value|times interest earned|cash coverage|equity multiplier|debt[- ](to[- ])?equity|"
                        r"total debt ratio|asset turnover|inventory turnover|receivables turnover|days'? sales|cost of goods sold|"
                        r"cogs|gross profit|profit margin|payout ratio|retention ratio|plowback|retained earnings|sustainable growth|"
                        r"internal growth|operating cash flow|ebt|taxable income|net fixed assets|long[- ]term debt|quick ratio|"
                        r"current ratio|net working capital)\b"
                        r"|\bsales\b.*\b(costs?|net income|depreciation|tax rate|profit|turnover|ebit)\b"
                        r"|\b(costs?|net income|depreciation|profit)\b.*\bsales\b")
_REGULATORY = re.compile(r"(?i)\b(sebi|rbi|amfi|dicgc|tax\w*|tds|section \d+\w*|rule|regulation|regulat\w+|law|legal|illegal|allowed|"
                         r"ombudsman|complaint|insur\w+|kyc|registered|licen[cs]\w*|circular|guideline|compliance)\b")
_GIVEN_RATE = re.compile(r"(?i)\b(tax|gst|tds)\s+rate\s+((of|is|at|were|was|to|becomes|=)\s+)?\d+(\.\d+)?\s*(%|percent|per cent)?|\d+(\.\d+)?\s*(%|percent|per cent)\s+(tax|gst|tds)(\s+rate)?")
_INVEST = re.compile(r"(?i)\b(invest\w*|stocks?|shares?|equity|mutual funds?|sip|returns?|portfolio|ipo|bonds?|fd|fixed deposit|"
                     r"gold|real estate|nps|ppf|elss|etf|trading|crypto\w*|savings?)\b")


_CONCEPT = re.compile(r"(?i)^\s*(what (is|are|does|do)( an?| the)?|explain|define|describe|how (does|do|is|are)|"
                      r"(what is the )?difference between|meaning of|why (is|are|do|does)|tell me about)\b")
_PERSONAL = re.compile(r"(?i)\b(my|i|me|mine|should i)\b")


@dataclass
class PrecheckResult:
    text: str                        # what the model is allowed to see
    redactions: list[str] = field(default_factory=list)
    amounts: list[dict] = field(default_factory=list)
    flags: dict = field(default_factory=dict)


_FOLLOWUP_OPENER = re.compile(r"(?i)^\s*(what if|and if|how about|what about|instead|also|same|then|ok|okay|and|but|only|"
                              r"would it|could i|can i|what would)\b")
_FOLLOWUP_MARKER = re.compile(r"(?i)\b(instead|already|only|as well|too|same)\b")
_PRONOUN = re.compile(r"(?i)\b(this|that|it|these|those)\b")


def is_followup(text: str) -> bool:
    words = len(text.split())
    return words <= 14 and bool(_FOLLOWUP_OPENER.search(text) or _FOLLOWUP_MARKER.search(text)
                                or (words <= 8 and _PRONOUN.search(text)))


def precheck(user_text: str, previous_question: str | None = None) -> PrecheckResult:
    clean, redactions = redact(unicodedata.normalize("NFC", user_text or ""))
    normalized, amounts = normalize_amounts(clean)
    has_number = has_quantity(clean)
    statement = bool(_STATEMENT.search(clean) or _CORPORATE.search(clean)) and has_number or is_tabular(clean)
    followup = bool(previous_question) and is_followup(clean)
    routed_text = f"{redact(previous_question)[0]} {clean}" if followup else clean
    calc_types = detect_calc_types(routed_text, has_quantity(routed_text), statement)
    if "project" in calc_types or (
            {"npv_irr", "eac", "perpetuity"} & set(calc_types) and not (_CORPORATE.search(clean) or _STATEMENT_DOC.search(clean))):
        statement = False                                  # a table of project cash flows (or a cost comparison) is not a statement
        calc_types = [k for k in calc_types if k != "statement"]
    elif set(calc_types) - {"statement", "percent"} and not (_STATEMENT.search(clean) or _CORPORATE.search(clean)
                                                             or _STATEMENT_DOC.search(clean)):
        statement = False          # several numbered lines ("Note: round to 2 decimals, e.g., 32.16") are not a pasted statement
        calc_types = [k for k in calc_types if k != "statement"]
    # a follow-up keeps the previous question's currency ("what if the rate were 9%?" after a $ question)
    usd = bool(re.search(r"\$\s?\d|\bUSD\b|\bdollars?\b", routed_text, re.I)) and not re.search(
        r"₹|\brs\.?\s?\d|\binr\b|rupee|lakh|crore", routed_text, re.I)
    flags = {
        "currency": "USD" if usd else "INR",
        "followup": followup and bool(calc_types),   # only calculation follow-ups get the "reuse the numbers" note
        "calc_types": calc_types,
        "statement_data": statement,
        "guarantee_seeking": bool(_GUARANTEE.search(clean)),
        "personal_advice_request": bool(_ADVICE.search(clean)),
        "high_risk_topic": bool(_HIGH_RISK.search(clean)),
        "calculation_intent": (bool(_CALC.search(clean)) and has_number) or statement or bool(calc_types),
        # a tax rate the user states is an input, not an Indian rule to verify; dollar textbook problems are not Indian regulation
        "regulatory_topic": bool(_REGULATORY.search(_GIVEN_RATE.sub(" ", clean))) and not usd,
        "investment_topic": bool(_INVEST.search(clean) or _HIGH_RISK.search(clean)),
        "concept_question": bool(_CONCEPT.search(clean)) and not _PERSONAL.search(clean),
    }
    return PrecheckResult(text=normalized, redactions=redactions, amounts=amounts, flags=flags)
