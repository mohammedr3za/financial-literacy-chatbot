"""M2 acceptance: known-answer queries must find the right source in the top 5.
Also prints scores for off-topic queries to calibrate FINBOT_MIN_DENSE_SCORE.

Usage: uv run python scripts/check_retrieval.py
"""
import sys

from finbot.config import get_settings
from finbot.kb.search import KnowledgeBase
from finbot.llm import get_embedder

KNOWN = [
    ("How do I file a complaint against my bank with the RBI ombudsman?", ("rbi_integrated_ombudsman_scheme",)),
    ("How much of my bank deposit is insured if the bank fails?", ("dicgc_information_booklet",)),
    ("What does the riskometer on a mutual fund mean?", ("riskometer", "amfi_mutual_funds", "sebi_mf_investor_faqs")),
    ("Can a SEBI-registered broker work with an unregistered finfluencer?", ("finfluencer",)),
    ("Does an investment adviser need to be registered with SEBI?", ("sebi_ia_regulations", "investor_charter_investment_advisers")),
    ("How is income from virtual digital assets like crypto taxed?", ("incometax_itr2_faq_vda",)),
    ("What is vishing and how do fraudsters steal OTPs?", ("rbi_beaware_fraud_booklet",)),
    ("What did the RBI committee recommend for responsible AI in finance?", ("rbi_free_ai_committee_report",)),
    ("What is the expense ratio of a mutual fund scheme?", ("amfi_mutual_funds", "sebi_mf_investor_faqs")),
    ("Explain bond duration and convexity", ("principles_of_finance",)),
]
OFF_TOPIC = ["Who won the cricket world cup?", "Give me a biryani recipe", "How do I fix my laptop's wifi?", "Tell me a joke about cats"]


def main() -> int:
    s = get_settings()
    kb = KnowledgeBase.load(s.kb_dir / "index", get_embedder(s), s.min_dense_score)
    ok = 0
    for q, expected in KNOWN:
        res = kb.search(q, k=5)
        files = [h["file"] for h in res["hits"]]
        rank = next((i + 1 for i, f in enumerate(files) if any(e in f for e in expected)), None)
        ok += rank is not None
        print(f"{'PASS' if rank else 'FAIL'} rank={rank} best_dense={res['best_dense']:.3f} relevant={res['relevant']}  {q}")
        if not rank:
            for f in files:
                print("      got:", f)
    print(f"\n{ok}/{len(KNOWN)} known-answer queries found the right source in the top 5")
    print(f"\nOff-topic calibration (threshold {s.min_dense_score}):")
    for q in OFF_TOPIC:
        res = kb.search(q, k=1)
        print(f"  best_dense={res['best_dense']:.3f} relevant={res['relevant']}  {q}")
    return 0 if ok == len(KNOWN) else 1


if __name__ == "__main__":
    sys.exit(main())
