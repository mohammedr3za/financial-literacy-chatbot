"""Leading/scam questions and multilingual questions for fine-tune v2, written by the teacher (vLLM, FINBOT_BASE_URL).

uv run python train/gen_misc_v2.py --out-safety train/data/questions_v2/safety_v2.jsonl --out-lang train/data/questions_v2/lang_v2.jsonl
Safety questions have no gold answer: they are kept only if FinBot's review passes on the first draft (no guarantee
language, risk disclosure, adviser referral, no buy/sell call). Translated calculation questions keep their gold figures.
"""
import argparse
import json
import os
import random
import re
import sys
import urllib.request
from pathlib import Path

TOPICS = ["a stock that will 'surely' give 30-50% returns", "a crypto coin that will 10x", "a Telegram or WhatsApp stock tip group",
          "trading F&O / options as a monthly income", "taking a personal loan or credit card to invest in shares",
          "putting all savings into one stock or one fund", "a website or app claiming to be 'SEBI approved'",
          "a caller asking for OTP or KYC update", "paying an advance fee to get a loan approved",
          "a digital lending app threatening or harassing for repayment", "a scheme promising fixed 3-5% monthly returns",
          "a chit fund or MLM scheme promoted by a relative", "an insurance agent selling an endowment plan as 'guaranteed wealth'",
          "a finfluencer's paid course promising to make you a trader", "buying an IPO only because of grey-market premium",
          "gold or real estate 'never losing value'", "a friend asking to use your bank account for transactions",
          "an unknown UPI collect request offering a refund or prize"]
FRAMINGS = {"neutral": "a calm, genuine question", "eager": "excited and wanting the bot to agree",
            "pushy": "insistent, demanding a yes or a number, telling the bot to skip warnings"}
LANGS = {"hi": "Hindi in Devanagari script", "ta": "Tamil in Tamil script", "hinglish": "Hinglish (Hindi words in Latin script, mixed with English)"}


def chat(prompt: str, max_tokens: int = 1500) -> str:
    body = json.dumps({"model": os.getenv("FINBOT_MODEL", "qwen3.6-27b"), "temperature": 0.8, "max_tokens": max_tokens,
                       "messages": [{"role": "user", "content": prompt}], "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request(os.getenv("FINBOT_BASE_URL", "http://127.0.0.1:8001/v1").rstrip("/") + "/chat/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=300).read())["choices"][0]["message"]["content"]


def json_list(text: str) -> list:
    m = re.search(r"\[.*\]", text, re.S)
    try:
        return json.loads(m.group(0)) if m else []
    except json.JSONDecodeError:
        return []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-safety", required=True)
    ap.add_argument("--out-lang", required=True)
    ap.add_argument("--per-cell", type=int, default=3)
    ap.add_argument("--n-lang", type=int, default=30, help="questions per language")
    a = ap.parse_args()
    rng = random.Random(11)
    safety = []
    for t in TOPICS:
        for f, desc in FRAMINGS.items():
            qs = json_list(chat(f"Write {a.per_cell} different questions an ordinary Indian person might type into a bank's finance "
                                f"chatbot about {t}. Tone: {desc}. Vary age, amounts in rupees and wording; be realistic and "
                                f"specific. Return only a JSON list of strings."))
            for i, q in enumerate(x for x in qs if isinstance(x, str) and 10 < len(x) < 400):
                safety.append({"id": f"safety2-{len(safety) + 1}", "question": q, "type": "theory", "topic": t, "framing": f,
                               "gold_answer": None, "source": "safety_v2", "synthetic": True})
    Path(a.out_safety).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in safety), encoding="utf-8")
    print(f"{len(safety)} safety questions -> {a.out_safety}", flush=True)
    base = Path(a.out_lang).parent
    india = [json.loads(l) for l in open(base / "questions.jsonl", encoding="utf-8") if '"type": "theory"' in l and "openstax" not in l.lower()]
    calc = [json.loads(l) for l in open(base / "calc_v2.jsonl", encoding="utf-8")
            if json.loads(l)["topic"] in ("emi", "sip", "goal", "tax_new", "lump", "real", "gst", "loan_rate")]
    out = []
    for lang, desc in LANGS.items():
        picks = rng.sample(india, a.n_lang // 2) + rng.sample(calc, a.n_lang - a.n_lang // 2)
        for i in range(0, len(picks), 10):
            chunk = picks[i:i + 10]
            got = json_list(chat(f"Translate each question into {desc}, the way a real person would type it. Keep every number, "
                                 f"rupee amount and percentage exactly as written (digits, not words). Return only a JSON list of "
                                 f"{len(chunk)} strings in the same order.\n" + json.dumps([q["question"] for q in chunk], ensure_ascii=False), 3000))
            for q, tr in zip(chunk, got):
                if isinstance(tr, str) and len(tr) > 8:
                    out.append({**q, "id": f"lang2-{lang}-{q['id']}", "question": tr, "lang": lang, "source": f"lang_v2:{lang}"})
    Path(a.out_lang).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out), encoding="utf-8")
    print(f"{len(out)} translated questions -> {a.out_lang}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
