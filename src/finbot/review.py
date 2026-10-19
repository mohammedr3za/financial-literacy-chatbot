"""Deterministic answer review against the trust checklist (SPEC 3.1, review node)."""
import re
import unicodedata
from dataclasses import dataclass, field

from .routing import HINTS, REQUIRED_TOOLS, missing_tools
from .tools.registry import TOOLS_BY_NAME

FOOTER = ("\n\n---\n*This is general information, not investment advice. Verify with official sources "
          "or a SEBI-registered investment adviser.*")

_NEG = r"(?:not|no|never|cannot|can't|can not|isn't|aren't|don't|doesn't|won't|nobody|none|without)"
_GUARANTEE_PHRASES = re.compile(
    r"(?i)\b(guaranteed?|assured)\s+(returns?|profits?|gains?|income|growth|"
    r"\d+(?:\.\d+)?\s*%(?!\s*(?:saving|savings|interest saving|cost|reduction)))"   # paying off a 9% loan saves 9% for sure
    r"|\b(will|shall)\s+(definitely|surely|certainly)\s+(give|make|earn|return|grow|double)"
    r"|\b(risk[- ]free)\s+(returns?|profits?|investment)"
    r"|\b100%\s+(safe|sure|guaranteed)"
    r"|\bsure[- ]?shot\b"
    r"|\b(yes|confirmed)[,!.]?\s+(it|this|the stock|you)\s+will\s+(give|make|earn|return)")
_RISK_WORDS = re.compile(r"(?i)\b(risks?|risky|not guaranteed|no guarantee|cannot be guaranteed|can't be guaranteed|"
                         r"volatil\w+|could lose|may lose|can lose|loss(es)?|market[- ]linked|subject to market|fluctuat\w+|uncertain)\b")
_RISK_WORDS_INDIC = re.compile("जोखिम|नुकसान|उतार-चढ़ाव|गारंटी नहीं|ஆபத்து|அபாய|நஷ்ட|இழப்பு|ரிஸ்க்")   # Hindi, Tamil
# tools that carry official, source-checked Indian tax rules: a regulatory answer backed by one is verified (SPEC 4.4)
RULE_TOOLS = {"tax_rules", "income_tax", "capital_gains"}
_CITATION = re.compile(r"\[S\d+\]")
_REFERRAL = re.compile(r"(?i)(sebi[- ]registered|registered investment advis[eo]r|investment advis[eo]r|financial advis[eo]r|"
                       r"certified financial planner|registered advis[eo]r)")
_CANNOT_VERIFY = re.compile(r"(?i)(cannot|can't|could not|couldn't|unable to|not able to)\s+(verify|confirm)|no (official )?source")
_DOMAIN = re.compile(r"(?i)\b(?:https?://)?((?:[a-z0-9-]+\.)+(?:gov\.in|org\.in|co\.in|nic\.in|com|in|org|net|io))\b")
ALLOWED_DOMAINS = ("sebi.gov.in", "rbi.org.in", "incometax.gov.in", "incometaxindia.gov.in", "amfiindia.com",
                   "mutualfundssahihai.com", "dicgc.org.in", "irdai.gov.in", "pfrda.org.in", "nseindia.com", "bseindia.com",
                   "nsdl.co.in", "cdslindia.com", "cybercrime.gov.in")
# a helpline or phone number the answer gives: digits right after a calling word (v4 invented "1900 1800 180" for 1930)
_HELPLINE = re.compile(r"(?i)(?:helplines?|help line|toll[- ]free|hotline|dial|(?:phone|contact|helpline|mobile) (?:number|no\.?)|"
                       r"\bcall(?:ing)?\b(?!\s+(?:options?|money|price|premium|auction))|हेल्पलाइन|कॉल|फ़ोन|फोन|உதவி எண்|அழைக்க)"
                       r"[^\d\n₹$]{0,30}?(?<![\d.,₹$])((?:\+91[\s-]?)?\d(?:[\d\s-]{1,14}\d)?)(?![\d.,%]|\s*(?:%|lakh|crore|years?|months?|days?))")
_FIGURE = re.compile(r"(?i)(?:(₹|rs\.?|inr|\$|usd|€|£)\s*)?(?<![\w.])(\d+(?:,\d+)*(?:\.\d+)?)\s*(%|per ?cent|lakhs?|lacs?|crores?|cr\b)?")
_UNIT = {"lakh": 1e5, "lakhs": 1e5, "lac": 1e5, "lacs": 1e5, "crore": 1e7, "crores": 1e7, "cr": 1e7}


def _figures(text: str, claims_only: bool) -> set[float]:
    """Numbers in text. With claims_only, only percentages and rupee amounts (the figures that matter)."""
    out = set()
    for cur, num, unit in _FIGURE.findall(text or ""):
        unit = (unit or "").lower().replace(" ", "")
        if claims_only and not (cur or unit):
            continue
        v = float(num.replace(",", ""))
        out.add(round(v, 4))
        if unit in _UNIT:
            out.add(round(v * _UNIT[unit], 4))
    return out


def _big_plain(num: str) -> bool:
    digits = num.replace(",", "").split(".")[0].lstrip("0")
    return len(digits) >= 4 or "." in num


def unverified_numbers(answer: str, evidence: str) -> list[str]:
    """Helpline or phone numbers in the answer that the evidence (question, sources, tool results) does not contain."""
    known = re.sub(r"[\s-]", "", evidence or "")
    out = []
    for m in _HELPLINE.finditer(answer or ""):
        num = m.group(1).strip()
        digits = re.sub(r"\D", "", num)
        if len(digits) >= 3 and digits not in known and num not in out:
            out.append(num)
    return out


def unsupported_figures(answer: str, evidence: str, rel_tol: float = 0.0, plain_numbers: bool = False,
                        scale_variants: bool = True) -> list[str]:
    allowed = _figures(evidence, claims_only=False)

    allowed_ext = allowed | ({round(a / 100, 6) for a in allowed} | {round(a * 100, 4) for a in allowed}  # 23 % <-> 0.23
                             if scale_variants else set())

    def close(c: float, a: float) -> bool:
        return abs(c - a) <= max(abs(a) * rel_tol, 0.005)

    def exact(c: float, v: float) -> bool:   # recomputed figures must match almost exactly (display rounding only)
        return abs(c - v) <= max(abs(v) * 1e-4, 0.006)

    # operands for recomputation: only verified numbers the answer itself shows (its visible working), plus 1, 12, 100, 365
    shown = {round(float(n.replace(",", "")), 4) for n in re.findall(r"(?<![\w.])\d+(?:,\d+)*(?:\.\d+)?", answer or "")}
    operands = [a for a in shown if a and (a in allowed_ext or any(close(a, b) for b in allowed if b))] + [1.0, 12.0, 100.0, 365.0]

    def one_step(c: float) -> bool:
        """The checker recomputes simple arithmetic the answer shows: c = x+y, x-y, x*y or x/y for two shown, verified numbers."""
        nums = operands[:200]
        for i, x in enumerate(nums):
            for y in nums[i:]:
                for v in (x + y, abs(x - y), x * y, x / y if y else None, y / x if x else None):
                    if v is not None and exact(c, v):
                        return True
        return False

    def supported(cands: set) -> bool:
        if cands & allowed_ext:
            return True
        if rel_tol > 0 and any(close(c, a) for c in cands for a in allowed if a):
            return True
        return rel_tol > 0 and any(one_step(c) for c in cands)
    bad = []
    for m in _FIGURE.finditer(answer or ""):
        cur, num, unit = m.groups()
        unit_l = (unit or "").lower().replace(" ", "")
        if not (cur or unit_l or (plain_numbers and _big_plain(num))):
            continue
        v = float(num.replace(",", ""))
        candidates = {round(v, 4)} | ({round(v * _UNIT[unit_l], 4)} if unit_l in _UNIT else set())
        if not supported(candidates):
            bad.append(m.group(0).strip())
    return list(dict.fromkeys(bad))


# internal tool names must not reach the user. Names that are also English words or acronyms ("calculate", "TVM") count
# only when written like code: name(, `name`, or "name function/tool".
_CODE_NAMES = sorted(n for n in TOOLS_BY_NAME if "_" in n)
_WORD_NAMES = sorted(n for n in TOOLS_BY_NAME if "_" not in n)
_TOOL_NARRATION = re.compile(
    rf"\b({'|'.join(_CODE_NAMES)})\b"
    rf"|\b({'|'.join(_WORD_NAMES)})\(|`({'|'.join(_WORD_NAMES)})`|\b({'|'.join(_WORD_NAMES)})\s+(function|tool)\b"
    r"|\b(function|tool)s?\s+(was|were|is|has been|have been)\s+(called|invoked|used|run|executed)\b"
    r"|\b(called|invoked|calling|invoking|ran)\s+the\s+[\w ]{0,25}?\b(function|tool)\b|\btool calls?\b", re.I)

_BUY_CALL = re.compile(r"(?i)\b(you should|i (would |'d )?recommend|i suggest|go ahead and|my (top )?pick is|the best (stock|fund|share) (to buy )?is)"
                       r"\s+(buy(ing)?|sell(ing)?|invest(ing)? in|put(ting)? (your|all)|go(ing)? for)\b")


# a guarantee phrase in a warning is not a promise (SPEC 3.1 check 1): quoted, or in a sentence with a caution word
_CAUTION = re.compile(r"(?i)\b(prohibit\w*|forbid\w*|not allowed|illegal|bann?ed|beware|wary|cautious|careful|caution\w*|"
                      r"avoid\w*|fraud\w*|scam\w*|ponzi|falsely|fake|lure\w*|red[- ]flags?|too good to be true|unregistered|"
                      r"warning signs?|mis-?sell\w*)\b")


def _quoted(text: str, m: re.Match) -> bool:
    return bool(re.match(r'\s?["”]', text[m.end():m.end() + 2]) and re.search(r'["“][^"”]*$', text[max(0, m.start() - 40):m.start()]))


def _has_unnegated_guarantee(text: str) -> bool:
    for m in _GUARANTEE_PHRASES.finditer(text):
        window = text[max(0, m.start() - 40):m.start()]
        ends = [i for i in (text.find(". ", m.end()), text.find("\n", m.end())) if i >= 0]
        sentence = text[max(text.rfind(". ", 0, m.start()), text.rfind("\n", 0, m.start()), 0):min(ends, default=len(text))]
        if not (re.search(rf"(?i)\b{_NEG}\b", window) or _quoted(text, m) or _CAUTION.search(sentence)):
            return True
    return False


@dataclass
class ReviewResult:
    passed: bool
    codes: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)


def _strip_math_markup(text: str) -> str:
    """Drop LaTeX commands and $$ so formula markup is not read as figures (e.g. 10^{-3})."""
    return re.sub(r"\\[a-zA-Z]+", " ", text).replace("$$", " ")


# ---------------------------------------------------------------- check 11: shown working must add up
# Every fully numeric expression in the answer is evaluated with the safe calculator and must equal a figure the answer
# states. Only clear working is checked: anything joined to words or units, ranges, dates and unparseable text is skipped.
_DISPLAY_MATH = re.compile(r"\$\$(.+?)\$\$|\\\[(.+?)\\\]", re.S)
_INLINE_MATH = re.compile(r"\\\((.+?)\\\)", re.S)
_FRAC = re.compile(r"\\[dt]?frac\s*")
_TEXT_CMD = re.compile(r"\\(?:text|mathrm|textbf|mathbf|textit|mathit|operatorname|mbox)\s*\{([^{}]*)\}")
_LATEX_SUBS = [
    (re.compile(r"\\(?:begin|end)\s*\{[^{}]*\}"), "\n"), (re.compile(r"\\\\"), "\n"), (re.compile(r"&"), " "),
    (re.compile(r"\\(?:times|cdot|ast)(?![A-Za-z])"), "×"), (re.compile(r"\\div(?![A-Za-z])"), "÷"),
    (re.compile(r"\\(?:approx|simeq|sim)(?![A-Za-z])"), "≈"),
    (re.compile(r"\\(?:Rightarrow|rightarrow|Longrightarrow|implies|therefore|to)(?![A-Za-z])"), "\n"),
    (re.compile(r"\\(?:left|right|bigl|bigr|Bigl|Bigr|big|Big)(?![A-Za-z])"), ""),
    (re.compile(r"\\%"), "%"), (re.compile(r"\\\$"), "$"), (re.compile(r"(?<=\d)\{,\}(?=\d)"), ","),
    (re.compile(r"(?<=\d)\\,(?=\d)"), ""), (re.compile(r"\\(?:quad|qquad|[,;:! ])"), " "),
    (re.compile(r"_\{[^{}]*\}|_[A-Za-z0-9]+"), ""),                   # subscripts: PVIFA_{10%,2}
    (re.compile(r"\\\{"), "("), (re.compile(r"\\\}"), ")"),
    (re.compile(r"\\([A-Za-z]+)"), r" \1 "),                         # any other command reads as a word (\sqrt, \ln)
]
_SCALE_WORDS = {"lakh": 1e5, "lakhs": 1e5, "lac": 1e5, "lacs": 1e5, "crore": 1e7, "crores": 1e7, "cr": 1e7,
                "thousand": 1e3, "million": 1e6, "mn": 1e6, "billion": 1e9, "bn": 1e9}
_NUM_AT = re.compile(r"([$₹]\s?)?(\d{1,3}(?:,\d{2,3})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+)(\s?%)?")
_OP_CHARS = {"×": "*", "·": "*", "*": "*", "÷": "/", "/": "/", "∕": "/", "+": "+", "−": "-", "-": "-", "^": "**"}


def _brace_group(s: str, i: int):
    depth = 0
    for j in range(i, len(s)):
        depth += {"{": 1, "}": -1}.get(s[j], 0)
        if depth == 0:
            return s[i + 1:j], j + 1
    return None


def _fracs(s: str) -> str:
    """\\frac{a}{b} -> ((a)∕(b)), nested fractions included (∕ marks a typeset fraction, unlike a plain-text 24/7)."""
    out, i = [], 0
    while (m := _FRAC.search(s, i)) is not None:
        out.append(s[i:m.start()])
        a = _brace_group(s, m.end()) if s[m.end():m.end() + 1] == "{" else None
        k = a[1] if a else 0
        while a and k < len(s) and s[k] == " ":
            k += 1
        b = _brace_group(s, k) if a and s[k:k + 1] == "{" else None
        if not (a and b):
            out.append(" frac ")      # malformed: reads as a word, so the expression is skipped
            i = m.end()
            continue
        out.append(f"(({_fracs(a[0])})∕({_fracs(b[0])}))")
        i = b[1]
    return "".join(out) + s[i:]


def _latex_to_plain(s: str) -> str:
    s = _TEXT_CMD.sub(r" \1 ", s)
    s = _fracs(s)
    for pattern, repl in _LATEX_SUBS:
        s = pattern.sub(repl, s)
    return s.replace("{", "(").replace("}", ")").replace("[", "(").replace("]", ")").replace("-", "−")


def _normalize_for_working(text: str) -> str:
    text = _DISPLAY_MATH.sub(lambda m: "\n" + _latex_to_plain(m.group(1) or m.group(2)) + "\n", text)
    text = _INLINE_MATH.sub(lambda m: " " + _latex_to_plain(m.group(1)) + " ", text)
    text = re.sub(r"(?m)^\s*(?:[-*+]|\d+\.)\s+", "\n", text)            # markdown bullets and numbered lists
    text = text.replace("**", "").replace("__", "").replace("`", "")
    return re.sub(r"(?<![\w*])\*(?=\S)([^*\n]+?)(?<=\S)\*(?![\w*])", r"\1", text)   # *italic*


@dataclass
class _Tok:
    kind: str                 # NUM OP LP RP WORD EQ SEP
    text: str
    space: bool = False       # whitespace right before the token
    value: float = 0.0
    decimals: int = 0
    pct: bool = False
    unary: bool = False


def _tokenize(text: str) -> list[_Tok]:
    toks, i, n, space = [], 0, len(text), False
    while i < n:
        ch, prev = text[i], text[i - 1] if i else " "
        if ch.isspace():
            j = i
            while j < n and text[j].isspace():
                j += 1
            if "\n" in text[i:j]:
                toks.append(_Tok("SEP", "\n", True))
            space, i = True, j
            continue
        m = _NUM_AT.match(text, i) if (ch.isdigit() or ch in "$₹.") and not (prev.isalnum() or prev in "._") else None
        if m and not (m.end() < n and (text[m.end()].isalpha() or text[m.end()] == "_") and not m.group(3)):
            num = m.group(2)
            toks.append(_Tok("NUM", m.group(0).strip(), space, float(num.replace(",", "")),
                             len(num.split(".")[1]) if "." in num else 0, bool(m.group(3))))
            i, space = m.end(), False
            continue
        if ch.isalnum() or ch == "_":                                  # words, and "2nd", "115BBH", "60W"
            j = i
            while j < n and (text[j].isalnum() or text[j] in "_'"):
                j += 1
            toks.append(_Tok("WORD", text[i:j], space))
            i, space = j, False
            continue
        nxt = text[i + 1] if i + 1 < n else " "
        if ch == "-":   # a minus only when spaced (" - ") or a sign ("-4.25"); otherwise a range, date or hyphenated word
            sign = (nxt.isdigit() or nxt in "$₹.(") and not prev.isalnum() and (
                not toks or space or toks[-1].kind in ("LP", "EQ", "OP", "SEP"))
            kind = "OP" if (space and nxt.isspace()) or sign else "SEP"
        else:
            kind = ("OP" if ch in _OP_CHARS else "LP" if ch == "(" else "RP" if ch == ")"
                    else "EQ" if ch in "=≈≃" else "SEP")
        toks.append(_Tok(kind, ch, space))
        i, space = i + 1, False
    return toks


def _expr(run: list[_Tok], pct_as_fraction: bool, bump: int = -1) -> str:
    """Python arithmetic for the safe calculator; bump perturbs one operand by half a unit of its last shown digit."""
    parts, last = [], None
    for k, t in enumerate(run):
        if t.kind in ("NUM", "LP") and last in ("NUM", "RP"):
            parts.append("*")                                          # implicit product: (1 + 0.5)(2,000)
        if t.kind == "NUM":
            v = t.value + (0.5 * 10 ** -t.decimals if k == bump else 0)
            parts.append(f"({v!r}/100)" if t.pct and pct_as_fraction else repr(v))
        else:
            parts.append(_OP_CHARS.get(t.text, t.text))
        last = "NUM" if t.kind == "NUM" else t.kind
    return " ".join(parts)


def _display(run: list[_Tok]) -> str:
    out = []
    for k, t in enumerate(run):
        if t.kind == "LP" and k + 2 < len(run) and run[k + 1].kind == "NUM" and run[k + 2].kind == "RP":
            continue                                                    # (2000) -> 2000
        if t.kind == "RP" and k >= 2 and run[k - 1].kind == "NUM" and run[k - 2].kind == "LP":
            continue
        text = {"*": "×", "·": "×", "/": "÷", "∕": "÷", "-": "−"}.get(t.text, t.text) if t.kind == "OP" else t.text
        out.append(text if t.kind in ("NUM", "LP", "RP") or t.unary else f" {text} ")
    s = re.sub(r"\(\s*\(([^()]*)\)\s*\)", r"(\1)", "".join(out)).strip()
    while s.startswith("(") and s.endswith(")") and _balanced(s[1:-1]):
        s = s[1:-1].strip()
    return re.sub(r"\s+", " ", s)


def _balanced(s: str) -> bool:
    depth = 0
    for ch in s:
        depth += {"(": 1, ")": -1}.get(ch, 0)
        if depth < 0:
            return False
    return depth == 0


def _runs(toks: list[_Tok]):
    """Candidate expressions: lists of token indices (numbers, operators, brackets) inside one clause. A clause ends at
    "=", "≈" or punctuation; a clause where an operator touches a word ("250 × PVIFA") is symbolic and skipped whole."""
    start = 0
    for end in range(len(toks) + 1):
        if end < len(toks) and toks[end].kind not in ("SEP", "EQ"):
            continue
        clause = list(range(start, end))
        start = end + 1
        symbolic = False
        for j in clause:
            t = toks[j]
            prv = toks[j - 1] if j > clause[0] else None
            nxt = toks[j + 1] if j + 1 < end else None
            if t.kind == "OP":
                t.unary = t.text in "−-+" and (prv is None or prv.kind in ("LP", "OP", "WORD")) and nxt is not None \
                    and nxt.kind in ("NUM", "LP")
                if (nxt is not None and nxt.kind == "WORD") or (not t.unary and prv is not None and prv.kind == "WORD"):
                    symbolic = True                                     # "250 × PVIFA", "hours × 0.75"
            elif t.kind == "LP" and prv is not None and prv.kind == "WORD" and not t.space:
                symbolic = True                                         # PVIFA(10%, 2)
        if symbolic:
            continue
        span = []
        for j in clause + [None]:
            if j is not None and toks[j].kind in ("NUM", "OP", "LP", "RP") and not (
                    span and toks[j].kind == "NUM" and toks[span[-1]].kind == "NUM"):
                span.append(j)
                continue
            if span:
                yield span
            span = [j] if j is not None and toks[j].kind in ("NUM", "OP", "LP", "RP") else []


def _trim(toks: list[_Tok], span: list[int]) -> list[int] | None:
    while span and toks[span[-1]].kind == "LP":
        span = span[:-1]
    while span and toks[span[0]].kind == "RP":
        span = span[1:]
    for _ in range(4):                                                  # brackets opened or closed in the prose around it
        text = "".join(toks[j].text for j in span if toks[j].kind in ("LP", "RP"))
        if _balanced(text):
            break
        if span and toks[span[0]].kind == "LP" and text.count("(") > text.count(")"):
            span = span[1:]
        elif span and toks[span[-1]].kind == "RP" and text.count(")") > text.count("("):
            span = span[:-1]
        else:
            return None
    if not span or not _balanced("".join(toks[j].text for j in span if toks[j].kind in ("LP", "RP"))):
        return None
    first, last = toks[span[0]], toks[span[-1]]
    if (first.kind == "OP" and not first.unary) or last.kind == "OP":
        return None                                                     # joined to something outside the expression
    return span


def _evaluate(run: list[_Tok], pct_as_fraction: bool) -> tuple[float, float] | None:
    from .tools.math_engine import calculate
    try:
        v = calculate(_expr(run, pct_as_fraction))["result"]["value"]
    except ValueError:
        return None
    err = 0.0
    for k, t in enumerate(run):                                         # rounding of the operands shown
        if t.kind == "NUM" and t.decimals:
            try:
                err += abs(calculate(_expr(run, pct_as_fraction, bump=k))["result"]["value"] - v)
            except ValueError:
                return None
    return v, err


def _fmt_value(v: float) -> str:
    if abs(v) >= 100:
        return f"{v:,.2f}".removesuffix(".00")
    return f"{v:,.4f}".rstrip("0").rstrip(".") or "0"


def arithmetic_mismatches(answer: str) -> list[dict]:
    """Shown working that evaluates to no figure the answer states: [{"expression", "value", "stated"}]."""
    toks = _tokenize(_normalize_for_working(answer or ""))
    stated = []                                                         # (token index, |value|, half a unit of the last digit)
    for j, t in enumerate(toks):
        if t.kind == "NUM":
            half = 0.5 * 10 ** -t.decimals
            stated.append((j, abs(t.value), half))
            unit = toks[j + 1].text.lower() if j + 1 < len(toks) and toks[j + 1].kind == "WORD" else ""
            if unit in _SCALE_WORDS:
                stated.append((j, abs(t.value) * _SCALE_WORDS[unit], half * _SCALE_WORDS[unit]))
    found, spans = [], []
    for span in _runs(toks):
        span = _trim(toks, span)
        if span:
            spans.append(span)
    for span in spans:
        run = [toks[j] for j in span]
        nums = [t for t in run if t.kind == "NUM"]
        ops = [t for t in run if t.kind == "OP" and not t.unary]
        if len(nums) < 2 or not ops:
            continue
        before = toks[span[0] - 1] if span[0] else None
        after = toks[span[-1] + 1] if span[-1] + 1 < len(toks) else None
        if after is not None and after.kind in ("WORD", "NUM"):
            continue                                                    # "$180 + $225 a year": a unit follows
        if before is not None and before.kind == "WORD" and span[0] >= 2 and toks[span[0] - 2].kind == "NUM":
            continue                                                    # "15% of 2,000 × 1.18": joined by a word
        if (all(t.text == "/" for t in ops) and all(not t.decimals and not t.pct for t in nums)
                and not (after is not None and after.kind == "EQ")):
            continue                                                    # plain-text 24/7, 1/3, 01/04/2026
        if any(t.pct for t in nums) and any(t.text in "+−-" for t in ops) and any(
                not t.pct and t.value not in (1, 100) for t in nums):
            continue                                                    # ₹5,000 + 18% (GST): not arithmetic
        variants = [r for r in (_evaluate(run, False), _evaluate(run, True) if any(t.pct for t in nums) else None) if r]
        if not variants:
            continue
        own = set(span)
        targets = [(s, h) for j, s, h in stated if j not in own]
        if after is not None and after.kind == "EQ":                    # a chain: A = B, where B is another expression
            nxt = next((sp for sp in spans if sp[0] == span[-1] + 2), None)
            if nxt is not None:
                v2 = _evaluate([toks[j] for j in nxt], False)
                if v2:
                    targets.append((abs(v2[0]), v2[1]))

        def agrees(v: float, err: float) -> bool:
            return any(abs(vv - s) <= max(0.005 * s, h, ee, 1e-9) for s, h in targets
                       for vv, ee in ((abs(v), err), (abs(v) * 100, err * 100), (abs(v) / 100, err / 100)))
        if any(agrees(v, err) for v, err in variants):
            continue
        later = [toks[j] for j in range(span[-1] + 1, len(toks)) if toks[j].kind == "NUM"]
        found.append({"expression": _display(run), "value": round(variants[0][0], 6),
                      "stated": later[0].text if later else None})
    return found


# check 12: a calculation that lacks inputs gets a request for them, not a made-up example
_HYPOTHETICAL = re.compile(r"(?i)\b(hypothetical(ly)?|for illustration|illustrative|let'?s (say|assume)|suppose|"
                           r"for example,? if|as an example|example calculation|made[- ]up example|assuming (that )?(the|a|an)\b)")
_ASKS_INPUT = re.compile(r"(?i)\b(please (provide|share|give|tell)|could you (provide|share|give|tell)|i need (the|to know|more)|"
                         r"need the|missing|not (given|provided|specified|stated)|without (the|knowing))\b")


# a direct request to the user for figures; stricter than _ASKS_INPUT ("the risk of missing the goal" is no request)
_REQUESTS_INPUT = re.compile(r"(?i)\b((please|could you|can you) (provide|share|give|tell|confirm|specify)|"
                             r"i need (the|to know|more|your)|let me know|if you (share|provide|give|tell))\b")


def asks_for_inputs(text: str) -> bool:
    """The answer asks the user for missing figures; such a draft is not forced to call a tool, and keeps its text."""
    return bool(_REQUESTS_INPUT.search(text or ""))


def has_hypothetical_example(text: str) -> bool:
    m = _HYPOTHETICAL.search(text or "")
    return bool(m and _ASKS_INPUT.search(text) and re.search(r"\d", text[m.start():]))


def cut_hypothetical(text: str) -> str:
    """Drop everything from the sentence that starts a made-up example."""
    m = _HYPOTHETICAL.search(text or "")
    if not m:
        return text
    start = max(text.rfind(". ", 0, m.start()) + 1, text.rfind("\n", 0, m.start()) + 1, 0)
    return text[:start].strip()


_HEADING = re.compile(r"^\s*(#{1,6}\s+\S.*|\*\*[^*\n]+\*\*:?)\s*$")
_BULLET = re.compile(r"^(\s*(?:[-*•]|\d+[.)])\s+)?(.*)$")


def prune_figures(text: str, figures: list[str]) -> str:
    """Drop every sentence, bullet or table row that states one of figures, and any heading left with nothing under it;
    keep the rest of the answer (SPEC 3.1)."""
    if not figures:
        return text

    def hit(s: str) -> bool:
        return any(f in s for f in figures)

    kept = []
    for line in (text or "").splitlines():
        if not hit(line):
            kept.append(line)
        elif not line.lstrip().startswith("|"):          # a table row with a bad figure goes entirely
            lead, body = _BULLET.match(line).groups()
            rest = [p for p in re.split(r"(?<=[.!?])\s+", body) if not hit(p)]
            if any(re.search(r"[^\W\d_]{3}", p) for p in rest):
                kept.append((lead or "") + " ".join(rest))
    out = [line for i, line in enumerate(kept)
           if not (_HEADING.match(line) and _HEADING.match(next((x for x in kept[i + 1:] if x.strip()), "## end")))]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()


def review_answer(answer: str, flags: dict, tools_used: list[str], num_sources: int, source_urls: tuple = (), evidence: str | None = None) -> ReviewResult:
    codes, issues = [], []
    text = answer or ""
    if not text.strip():
        return ReviewResult(False, ["empty"], ["The answer is empty. Write a complete answer."])
    if _has_unnegated_guarantee(text):
        codes.append("guarantee_language")
        issues.append("It promises or confirms guaranteed or certain returns. No investment return can be guaranteed; say so plainly.")
    missing = missing_tools(flags.get("calc_types", []), tools_used)
    if missing:
        need = sorted({t for k in missing for t in REQUIRED_TOOLS[k]})
        codes.append("wrong_tool" if tools_used else "no_tool_for_numbers")
        how = " ".join(f"How: {HINTS[k]}" for k in missing if k in HINTS)
        issues.append(f"This question must be answered with {' or '.join(need)}. Call it with the numbers from the question and use its exact "
                      "result; do not compute anything yourself." + (f" {how}" if how else ""))
    elif flags.get("calculation_intent") and not tools_used:
        codes.append("no_tool_for_numbers")
        issues.append("The question needs a calculation but no calculator tool was called. Call the right tool and use its exact result.")
    needs_risk = (flags.get("high_risk_topic") or flags.get("guarantee_seeking") or flags.get("personal_advice_request") or (
        flags.get("investment_topic") and not flags.get("regulatory_topic") and not flags.get("calculation_intent")
        and not flags.get("concept_question") and not flags.get("statement_data")))
    if needs_risk and not (_RISK_WORDS.search(text) or _RISK_WORDS_INDIC.search(text)):
        codes.append("no_risk_disclosure")
        issues.append("It discusses an investment without stating the risks. Add a short, specific risk disclosure.")
    sources_relevant = num_sources > 0
    needs_citation = sources_relevant and (flags.get("regulatory_topic") or not flags.get("calculation_intent"))
    if needs_citation and len(text) > 200 and not _CITATION.search(text):
        codes.append("no_citation")
        issues.append("Relevant sources were provided but none is cited. Cite the sources you used as [S1], [S2].")
    if flags.get("personal_advice_request"):
        if _BUY_CALL.search(text):
            codes.append("buy_sell_call")
            issues.append("It tells the user what to buy or sell. Explain how to evaluate options instead of picking one.")
        if not _REFERRAL.search(text):
            codes.append("no_adviser_referral")
            issues.append("For a personal investment decision, suggest consulting a SEBI-registered investment adviser.")
    cited = {int(n) for n in re.findall(r"\[S(\d+)\]", text)}
    fake = sorted(n for n in cited if n < 1 or n > num_sources)
    if fake:
        codes.append("fake_citation")
        have = f"only [S1] to [S{num_sources}] were provided" if num_sources else "no sources were provided"
        issues.append(f"It cites {', '.join(f'[S{n}]' for n in fake)} but {have}. Remove every citation that does not match a provided source.")
    if (flags.get("regulatory_topic") and not sources_relevant and not (RULE_TOOLS & set(tools_used))
            and not _CANNOT_VERIFY.search(text)):
        codes.append("unverified_regulatory")
        issues.append("No official source was found for this Indian rule or tax question. Say clearly that you cannot verify the details "
                      "and that rules change, then point to the official website. Do not state specific rates or sections as facts.")
    narration = _TOOL_NARRATION.search(text)
    if narration:
        codes.append("tool_narration")
        issues.append(f"It mentions an internal tool or tool call (\"{narration.group(0)}\"). Give the result in plain words; "
                      "do not describe tool calls or name tools.")
    mismatches = arithmetic_mismatches(text)
    if mismatches:
        parts = [f"{m['expression']} evaluates to {_fmt_value(m['value'])}, but the answer states {m['stated']}" if m["stated"]
                 else f"{m['expression']} evaluates to {_fmt_value(m['value'])}, which is not a figure the answer states"
                 for m in mismatches[:3]]
        codes.append("arithmetic_mismatch")
        issues.append(f"The working shown does not match the result: {'; '.join(parts)}. Recompute with the right tool "
                      "(for statement items, solve_financials) and make every step you show agree with the result you state.")
    if flags.get("calculation_intent") and has_hypothetical_example(text):
        codes.append("hypothetical_example")
        issues.append("Required inputs are missing, and the answer adds a made-up example. Ask for the missing inputs and "
                      "stop: do not compute an example or a hypothetical case.")
    allowed = ALLOWED_DOMAINS + tuple(_DOMAIN.findall(" ".join(source_urls)))
    unknown = sorted({d.lower() for d in _DOMAIN.findall(text)
                      if not any(d.lower() == a or d.lower().endswith("." + a) for a in allowed)})
    if unknown:
        codes.append("unlisted_link")
        issues.append(f"It mentions websites that may not exist or are not official: {', '.join(unknown)}. "
                      "Only mention sebi.gov.in, rbi.org.in, incometax.gov.in, amfiindia.com, dicgc.org.in or scores.sebi.gov.in.")
    phones = unverified_numbers(text, evidence) if evidence is not None else []
    if phones:
        codes.append("unverified_helpline")
        issues.append(f"It gives phone or helpline numbers that are not in the sources: {', '.join(phones)}. Never give a number "
                      "the sources do not state; point to the official website from the sources, or say you cannot verify it.")
    if evidence is not None and flags.get("calculation_intent") and tools_used:
        bad = unsupported_figures(_strip_math_markup(text), evidence, rel_tol=0.005, plain_numbers=True)
        if bad:
            codes.append("unsupported_figures")
            issues.append(f"These figures do not match any calculator result or the question: {', '.join(bad)}. "
                          "Use only the exact numbers the tools returned (rounding is fine); do not calculate anything yourself.")
    elif evidence is not None and flags.get("regulatory_topic"):
        bad = unsupported_figures(text, evidence)
        if bad:
            codes.append("unsupported_figures")
            issues.append(f"These figures are not in the sources, the question or a calculator result: {', '.join(bad)}. "
                          "Remove them, compute them with a calculator tool, or say you cannot verify them.")
    return ReviewResult(not codes, codes, issues)


def reviewer_note(result: ReviewResult) -> str:
    bullet = "\n".join(f"- {i}" for i in result.issues)
    return ("Reviewer note (automatic quality check, not from the user). Your previous answer has these problems:\n"
            f"{bullet}\nRewrite your complete answer for the user, fixing every problem. Do not mention this note.")


# a Hindi or Tamil rewrite that lost its vowel signs and viramas (SPEC 3.1, 2026-10-17 live 27B failure)
_INDIC = re.compile(r"[\u0900-\u097F\u0B80-\u0BFF]")


def _mark_ratio(text: str) -> float | None:
    letters = _INDIC.findall(text or "")
    if len(letters) < 50:
        return None
    return sum(unicodedata.category(c) in ("Mn", "Mc") for c in letters) / len(letters)


def script_damaged(draft: str, revision: str) -> bool:
    """The revision kept under 70% of the vowel-sign and virama density of a Hindi or Tamil draft."""
    a, b = _mark_ratio(draft), _mark_ratio(revision)
    return a is not None and b is not None and b < 0.7 * a
