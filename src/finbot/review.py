"""Deterministic answer review against the trust checklist (SPEC 3.1, review node)."""
import re
from dataclasses import dataclass, field

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


def review_answer(answer: str, flags: dict, tools_used: list[str], sources_relevant: bool) -> ReviewResult:
    codes, issues = [], []
    text = answer or ""
    if not text.strip():
        return ReviewResult(False, ["empty"], ["The answer is empty. Write a complete answer."])
    if _has_unnegated_guarantee(text):
        codes.append("guarantee_language")
        issues.append("It promises or confirms guaranteed or certain returns. No investment return can be guaranteed; say so plainly.")
    if flags.get("calculation_intent") and not tools_used:
        codes.append("no_tool_for_numbers")
        issues.append("The question needs a calculation but no calculator tool was called. Call the right tool and use its exact result.")
    needs_risk = (flags.get("high_risk_topic") or flags.get("guarantee_seeking") or flags.get("personal_advice_request") or (
        flags.get("investment_topic") and not flags.get("regulatory_topic") and not flags.get("calculation_intent")
        and not flags.get("concept_question")))
    if needs_risk and not _RISK_WORDS.search(text):
        codes.append("no_risk_disclosure")
        issues.append("It discusses an investment without stating the risks. Add a short, specific risk disclosure.")
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
    return ReviewResult(not codes, codes, issues)


def reviewer_note(result: ReviewResult) -> str:
    bullet = "\n".join(f"- {i}" for i in result.issues)
    return ("Reviewer note (automatic quality check, not from the user). Your previous answer has these problems:\n"
            f"{bullet}\nRewrite your complete answer for the user, fixing every problem. Do not mention this note.")
