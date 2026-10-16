"""Deterministic answer review against the trust checklist (SPEC 3.1, review node)."""
import re
from dataclasses import dataclass, field

from .routing import REQUIRED_TOOLS, missing_tools
from .tools.registry import TOOLS_BY_NAME

FOOTER = ("\n\n---\n*This is general information, not investment advice. Verify with official sources "
          "or a SEBI-registered investment adviser.*")

_NEG = r"(?:not|no|never|cannot|can't|can not|isn't|aren't|don't|doesn't|won't|nobody|none|without)"
_GUARANTEE_PHRASES = re.compile(
    r"(?i)\b(guaranteed?|assured)\s+(returns?|profits?|gains?|income|growth|\d+(?:\.\d+)?\s*%)"
    r"|\b(will|shall)\s+(definitely|surely|certainly)\s+(give|make|earn|return|grow|double)"
    r"|\b(risk[- ]free)\s+(returns?|profits?|investment)"
    r"|\b100%\s+(safe|sure|guaranteed)"
    r"|\bsure[- ]?shot\b"
    r"|\b(yes|confirmed)[,!.]?\s+(it|this|the stock|you)\s+will\s+(give|make|earn|return)")
_RISK_WORDS = re.compile(r"(?i)\b(risks?|risky|not guaranteed|no guarantee|cannot be guaranteed|can't be guaranteed|"
                         r"volatil\w+|could lose|may lose|can lose|loss(es)?|market[- ]linked|subject to market|fluctuat\w+|uncertain)\b")
_CITATION = re.compile(r"\[S\d+\]")
_REFERRAL = re.compile(r"(?i)(sebi[- ]registered|registered investment advis[eo]r|investment advis[eo]r|financial advis[eo]r|"
                       r"certified financial planner|registered advis[eo]r)")
_CANNOT_VERIFY = re.compile(r"(?i)(cannot|can't|could not|couldn't|unable to|not able to)\s+(verify|confirm)|no (official )?source")
_DOMAIN = re.compile(r"(?i)\b(?:https?://)?((?:[a-z0-9-]+\.)+(?:gov\.in|org\.in|co\.in|nic\.in|com|in|org|net|io))\b")
ALLOWED_DOMAINS = ("sebi.gov.in", "rbi.org.in", "incometax.gov.in", "incometaxindia.gov.in", "amfiindia.com",
                   "mutualfundssahihai.com", "dicgc.org.in", "irdai.gov.in", "pfrda.org.in", "nseindia.com", "bseindia.com",
                   "nsdl.co.in", "cdslindia.com", "cybercrime.gov.in")
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


def unsupported_figures(answer: str, evidence: str, rel_tol: float = 0.0, plain_numbers: bool = False) -> list[str]:
    allowed = _figures(evidence, claims_only=False)

    allowed_ext = allowed | {round(a / 100, 6) for a in allowed} | {round(a * 100, 4) for a in allowed}  # 23 % <-> 0.23

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
    rf"|\b({'|'.join(_WORD_NAMES)})\s*\(|`({'|'.join(_WORD_NAMES)})`|\b({'|'.join(_WORD_NAMES)})\s+(function|tool)\b"
    r"|\b(function|tool)s?\s+(was|were|is|has been|have been)\s+(called|invoked|used|run|executed)\b"
    r"|\b(called|invoked|calling|invoking|ran)\s+the\s+[\w ]{0,25}?\b(function|tool)\b|\btool calls?\b", re.I)

_BUY_CALL = re.compile(r"(?i)\b(you should|i (would |'d )?recommend|i suggest|go ahead and|my (top )?pick is|the best (stock|fund|share) (to buy )?is)"
                       r"\s+(buy(ing)?|sell(ing)?|invest(ing)? in|put(ting)? (your|all)|go(ing)? for)\b")


def _has_unnegated_guarantee(text: str) -> bool:
    for m in _GUARANTEE_PHRASES.finditer(text):
        window = text[max(0, m.start() - 40):m.start()]
        if not re.search(rf"(?i)\b{_NEG}\b", window):
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
        issues.append(f"This question must be answered with {' or '.join(need)}. Call it with the numbers from the question and use its exact "
                      "result; do not compute anything yourself.")
    elif flags.get("calculation_intent") and not tools_used:
        codes.append("no_tool_for_numbers")
        issues.append("The question needs a calculation but no calculator tool was called. Call the right tool and use its exact result.")
    needs_risk = (flags.get("high_risk_topic") or flags.get("guarantee_seeking") or flags.get("personal_advice_request") or (
        flags.get("investment_topic") and not flags.get("regulatory_topic") and not flags.get("calculation_intent")
        and not flags.get("concept_question") and not flags.get("statement_data")))
    if needs_risk and not _RISK_WORDS.search(text):
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
    if flags.get("regulatory_topic") and not sources_relevant and not _CANNOT_VERIFY.search(text):
        codes.append("unverified_regulatory")
        issues.append("No official source was found for this Indian rule or tax question. Say clearly that you cannot verify the details "
                      "and that rules change, then point to the official website. Do not state specific rates or sections as facts.")
    narration = _TOOL_NARRATION.search(text)
    if narration:
        codes.append("tool_narration")
        issues.append(f"It mentions an internal tool or tool call (\"{narration.group(0)}\"). Give the result in plain words; "
                      "do not describe tool calls or name tools.")
    allowed = ALLOWED_DOMAINS + tuple(_DOMAIN.findall(" ".join(source_urls)))
    unknown = sorted({d.lower() for d in _DOMAIN.findall(text)
                      if not any(d.lower() == a or d.lower().endswith("." + a) for a in allowed)})
    if unknown:
        codes.append("unlisted_link")
        issues.append(f"It mentions websites that may not exist or are not official: {', '.join(unknown)}. "
                      "Only mention sebi.gov.in, rbi.org.in, incometax.gov.in, amfiindia.com, dicgc.org.in or scores.sebi.gov.in.")
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
