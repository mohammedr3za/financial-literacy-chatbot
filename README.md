# FinBot

FinBot is an open-source finance chatbot for the exhibition project *"Can you trust a financial chatbot?"*. It runs entirely on a laptop with no internet. It solves finance problems with exact calculators, answers from SEBI, RBI, and finance-book sources with citations, and checks every answer against the team's trust checklist.

The source of truth is [`SPEC.md`](SPEC.md). Progress is tracked in [`PLAN.md`](PLAN.md).

## Run the booth demo
```bash
./run_demo.sh
```
This starts Ollama, warms up the models, and opens the chat in your browser. Once set up, it works with Wi-Fi off.

## First-time setup on a new Mac
```bash
brew install ollama && brew services start ollama
ollama pull qwen3:8b               # the chat model (see SPEC for the current choice)
ollama pull bge-m3                # the embedding model for document search
uv sync                           # Python 3.12 environment
uv run python -m finbot.kb.ingest # build the document index from kb/raw/
```

## How it works
```
question → precheck → retrieve → agent ⇄ tools → review → answer
```
- **precheck** removes personal data (PAN, Aadhaar, card, account, phone, email, UPI) and rewrites "₹20 lakh" as 2000000.
- **retrieve** finds the 3 most relevant pages from the books and guidelines.
- **agent** is the local model. It calls a calculator for every number.
- **tools** are exact Python calculators: EMI, SIP, time value of money, NPV/IRR, bonds, Black-Scholes, arithmetic, and equations.
- **review** checks for guarantee language, missing risk warnings, fake citations, unofficial links, and buy or sell calls. It sends the answer back for one rewrite if needed.

Each answer has a **"How I got this"** panel showing the calculations, sources, and checks.

## Adding books or guidelines
1. Put text PDFs in `kb/raw/books/` or `kb/raw/guidelines/`. Scanned PDFs are skipped.
2. Add a row to `kb/sources.csv` with the file, title, publisher, URL, date, and license.
3. Re-run `uv run python -m finbot.kb.ingest`.

Never commit anything in `kb/raw/` or `kb/index/`. Books may be copyrighted, so keep them private.

## Testing
```bash
uv run pytest                             # all unit tests; no model needed
uv run python scripts/check_retrieval.py  # retrieval check on the real index
uv run python -m finbot.cli --acceptance  # the 6 acceptance prompts on the real model
```

## Benchmark against bank chatbots
1. Member 2 finalizes `bench/questions.yaml` and sets `frozen: true`.
2. Testers ask each bank bot every prompt (free versions, no personal data) and paste the replies into `bench/bank_responses.csv`, with screenshot paths.
3. Run FinBot on the same prompts, with and without document search:
   ```bash
   uv run python bench/run_finbot.py --config rag
   uv run python bench/run_finbot.py --config base
   ```
4. Build the blind score sheets: `uv run python bench/make_score_sheet.py results/finbot-*.jsonl`.
5. Members 1 and 2 score their sheets in `bench/sheets/` using `bench/RUBRIC.md`.
6. Build the report: `uv run python bench/report.py results/finbot-*.jsonl`. It goes to `bench/report/report.md`.

## Licenses
- FinBot's code: MIT License, see [`LICENSE`](LICENSE). The documents in `kb/raw/` are not included in this repository; `kb/sources.csv` lists where to download each one.
- OpenStax *Principles of Finance 2e*: CC BY-NC-SA 4.0 (attribution, non-commercial).
- Regulator documents: public government publications.
- Models: Qwen3-8B (Apache-2.0) and bge-m3 (MIT).
