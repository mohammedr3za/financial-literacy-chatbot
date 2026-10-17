"""Benchmark sheet for testing other finance chatbots by hand (SPEC 8).
uv run python bench/make_bot_sheet.py
Reads bench/questions.yaml and picks a balanced core of 24 prompts: every topic in all three framings, with the persona
rotated so each persona appears 8 times and with each framing. Writes bench/bot_test_sheet.md (printable, for testers)
and pre-fills bench/bank_responses.csv with one row per bot and prompt (response_text empty; make_score_sheet.py skips
rows until they are filled). Re-run after questions.yaml changes or is frozen; existing filled rows are kept."""
import csv
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
FRAMINGS = ["neutral", "eager", "leading"]
PERSONAS = ["novice", "greedy_literate", "anxious_saver"]
# shortlist from the 2026-10-17 survey (results/benchmark_targets.md): name, owner, where, access, generative per owner
BOTS = [
    ("PNB PIHU", "Punjab National Bank", "https://pnb.bank.in/ (PIHU chat icon; type Hi)", "anyone", "yes"),
    ("Federal Bank Feddy", "Federal Bank", "https://www.federal.bank.in/ (Feddy, bottom right); WhatsApp +91 96337 00800", "anyone", "yes (Gemini, per Google)"),
    ("HDFC Bank EVA", "HDFC Bank", "https://www.hdfc.bank.in/ (EVA chat widget)", "anyone", "not disclosed"),
    ("SBI SIA", "State Bank of India", "https://sbi.bank.in/ (Ask Sia widget)", "anyone", "not disclosed"),
    ("ICICI iPal", "ICICI Bank", "https://www.icici.bank.in/ (Ask iPal, top header)", "anyone", "not disclosed"),
    ("SEBI SEVA", "SEBI", "https://investor.sebi.gov.in/ or the Saa₹thi app", "anyone", "yes"),
    ("Kar Saathi", "Income Tax Department", "https://www.incometax.gov.in/ (chat widget; do not log in)", "anyone", "not disclosed"),
    ("Policybazaar Finova AI", "Policybazaar", "https://www.policybazaar.com/ (AI chat; location unconfirmed)", "anyone?", "yes"),
    ("ClearTax AI", "ClearTax", "https://cleartax.in/ (free team account, File ITR > Chat Now)", "free account", "not disclosed"),
    ("Paytm Ask AI", "Paytm", "Paytm app search bar > Ask AI (Perplexity)", "app login", "yes (Perplexity)"),
    ("ChatGPT (baseline)", "OpenAI", "https://chatgpt.com/ logged out, new chat per question", "anyone", "yes"),
]
COLUMNS = ["bot", "channel_url", "tested_on", "tester", "question_id", "persona", "framing", "question", "response_text",
           "screenshot_path", "outcome", "notes"]
OUTCOMES = "answered / menu_or_faq / handed_to_human / refused / requires_login / not_available / error"


def core_set(data: dict) -> list[dict]:
    rows = []
    for i, q in enumerate(data["questions"]):
        for j, framing in enumerate(FRAMINGS):
            persona = PERSONAS[(i + j) % 3]
            rows.append({"question_id": q["id"], "topic": q["topic"], "persona": persona, "framing": framing,
                         "question": q["variants"][persona][framing]})
    return rows


def main() -> int:
    data = yaml.safe_load((ROOT / "bench/questions.yaml").read_text(encoding="utf-8"))
    core = core_set(data)
    status = "FROZEN on " + str(data.get("frozen_on")) if data.get("frozen") else "DRAFT (questions.yaml is not frozen yet)"

    # pre-filled response sheet; keep anything testers already entered
    path = ROOT / "bench/bank_responses.csv"
    old = {}
    if path.exists():
        with path.open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if (r.get("response_text") or "").strip() or (r.get("outcome") or "").strip():
                    old[(r["bot"], r["question_id"], r["persona"], r["framing"])] = r
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for bot, _, where, _, _ in BOTS:
            for c in core:
                key = (bot, c["question_id"], c["persona"], c["framing"])
                row = {k: "" for k in COLUMNS} | {"bot": bot, "channel_url": where, "question_id": c["question_id"],
                                                    "persona": c["persona"], "framing": c["framing"], "question": c["question"]}
                w.writerow({**row, **{k: v for k, v in old.get(key, {}).items() if k in COLUMNS and v}})

    lines = [f"# Benchmark sheet: other finance chatbots", "",
             f"Question set: `bench/questions.yaml` ({data.get('version')}), **{status}**. Regenerate this sheet with "
             f"`uv run python bench/make_bot_sheet.py` after any change. Every bot gets exactly these {len(core)} prompts; FinBot "
             f"answers the same prompts automatically (`bench/run_finbot.py --set questions`).", "",
             "## Rules for every tester", "",
             "1. Team device, fresh incognito window. **Never log in or type a real name, PAN, Aadhaar, account number, OTP or card "
             "number.** If a bot insists on login, KYC or OTP, stop and record `requires_login`.",
             "2. **One prompt per fresh chat** (reload the page or start a new chat). Paste the prompt exactly as written, in English.",
             "3. For every prompt record, in `bench/bank_responses.csv` (import it into a shared Google Sheet, then download it "
             "back as CSV with the same columns):",
             "   - `tested_on`: date and time (IST), `tester`: your name",
             "   - `response_text`: the bot's full reply, copied as text (include any links it gave)",
             "   - `screenshot_path`: `screens/<bot>/<question_id>_<persona>_<framing>.png`, a full screenshot showing the bot's name",
             f"   - `outcome`: one of {OUTCOMES}",
             "   - `notes`: anything odd (disclaimer shown, language switched, asked for a phone number, answer cut off)",
             "4. If a bot shows terms to accept before chatting, the tester decides; save a screenshot of the terms.",
             "5. WhatsApp bots see your phone number: use a team phone if possible and blur the number in screenshots.",
             "6. Do not score while testing. Scoring is blind, from `bench/make_score_sheet.py`, with the rubric in `bench/RUBRIC.md`.",
             "", "## Bots (shortlist from the 2026-10-17 survey; steps in results/benchmark_targets.md)", "",
             "| # | Bot | Owner | Where | Access | Generative AI (as stated by the owner) | Tester | Done |", "|---|---|---|---|---|---|---|---|"]
    lines += [f"| {i} | {b} | {o} | {w} | {a} | {g} |  |  |" for i, (b, o, w, a, g) in enumerate(BOTS, 1)]
    lines += ["", "Reserves if a bot fails: Bank of Baroda ADI, Kotak Keya, HDFC ChatBanking (WhatsApp).", "",
              f"## The {len(core)} prompts (same order for every bot)", "",
              "| # | Question id | Topic | Persona | Framing | Prompt (paste exactly) |", "|---|---|---|---|---|---|"]
    lines += [f"| {k} | {c['question_id']} | {c['topic']} | {c['persona']} | {c['framing']} | {c['question']} |"
              for k, c in enumerate(core, 1)]
    lines += ["", "## Notes for the analysis", "",
              "- Bank bots are mostly built for account servicing, so many will deflect investment questions. Record the deflection "
              "as the answer (`outcome` = menu_or_faq or refused); that is itself a finding for the poster.",
              "- Only PNB, Federal Bank, SEBI, Policybazaar and Paytm say publicly that their bot uses generative AI. Do not "
              "describe the others as \"AI like ChatGPT\".",
              "- Generative bots can vary between runs. If time allows, ask the leading prompts twice and note any difference.",
              "- Check with the supervisor before publishing screenshots (some bots' terms claim ownership of the bot's replies).", ""]
    (ROOT / "bench/bot_test_sheet.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"{len(core)} prompts x {len(BOTS)} bots = {len(core) * len(BOTS)} rows -> bench/bank_responses.csv; sheet -> bench/bot_test_sheet.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
