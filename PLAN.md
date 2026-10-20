# FinBot — Plan (milestones)

Work on the **first milestone that still has unchecked boxes**. Tick a box only after its acceptance check has actually passed, and add the evidence (a command and its result) under the milestone. Scope changes go to `SPEC.md` first.

## M0 — Setup and tool-calling spike (Fri Oct 16)
- [x] `SPEC.md`, `PLAN.md`, and `CLAUDE.md` written; `git init`
- [x] `uv` project on Python 3.12 with dependencies installed
- [x] Ollama installed as a Homebrew service; `qwen3.5:9b` and `bge-m3` pulled
- [x] Spike: 5 tool-calling prompts through `ChatOllama.bind_tools`, with thinking on and off
- **Acceptance:** at least 4 of 5 correct tool calls, and a median answer time of 30 s or less. Otherwise switch to the fallback model and log the change in SPEC.

## M1 — Tools, test-first (Fri–Sat)
- [x] `calculators.py`: `tvm`, `loan_emi`, `sip_future_value`, `npv_irr`, `bond_metrics`, `black_scholes`
- [x] `math_engine.py`: `calculate` (AST whitelist) and `solve_equation` (sanitized SymPy)
- [x] `registry.py`: LangChain tool wrappers with ₹ lakh and crore formatting
- **Acceptance:** `uv run pytest tests/test_calculators.py tests/test_math_engine.py` passes. The EMI for ₹20 lakh at 8.5% over 240 months is ≈ ₹17,356, and malicious expressions are rejected.

## M2 — Knowledge base (Sat AM)
- [x] Download the guideline corpus and the OpenStax book; fill in `kb/sources.csv`
- [x] `ingest.py`: PDF and HTML to chunks and embeddings
- [x] `search.py`: hybrid BM25 and dense search with reciprocal-rank fusion
- **Acceptance:** 10 known-answer queries find the correct source in the top 5 (`tests/test_search.py`).

## M3 — Graph and CLI (Sat)
- [x] `precheck.py` and `review.py`, with unit tests
- [x] `prompts.py` (trust policy) and `graph.py` (LangGraph)
- [x] `cli.py`
- [x] `tests/test_graph.py` with a fake chat model, so no model is needed to run it
- **Acceptance:** all 6 acceptance prompts in SPEC section 10 behave as specified through the CLI.

## M4 — UI and demo script (Sat PM)
- [x] Streamlit `app.py` with the "How I got this" panel, banner, and session cap
- [x] `run_demo.sh`
- **Acceptance:** works end to end with Wi-Fi off.

## M4.5 — Tables, financial statements and the cream UI (user request, 2026-10-16)
- [x] Year-wise `table` output on `loan_emi`, `sip_future_value`, `tvm`, `npv_irr`, `bond_metrics`, `black_scholes` (test-first)
- [x] `financial_ratios`, `balance_sheet_check`, `cash_flow_statement` (test-first, textbook examples)
- [x] `present` graph step that appends tool tables to the answer; prompt and precheck updates for pasted statements
- [x] Streamlit redesign per SPEC 7.1, with CSV/Excel upload
- **Acceptance:**
  - `uv run pytest` passes.
  - In the browser: the EMI question shows a 20-row year-wise table with a total row that matches the calculator.
  - A pasted balance sheet gets a ratio table.
  - A cash flow problem gets a statement table that reconciles.
  - The new theme is checked by screenshot.

## M4.6 — Calculation reliability and the GPU model (user request, 2026-10-16)
- [ ] `goal_saving`, `real_return`, `effective_annual_rate` tools (test-first)
- [ ] Question-type router, routing enforcement, no-unsourced-figures check for calculations (test-first)
- [ ] Follow-up chips and the goal example on the homepage
- [ ] A100 + vLLM serving Qwen3.6-27B-FP8 (restored: the user wants local and GPU), SSH tunnel for the Mac app with automatic local fallback
- [ ] Fresh held-out calculation set (different wording and numbers from the 30)
- **Acceptance:**
  - `uv run pytest` passes.
  - The ₹1 crore goal question returns about ₹20,017 a month, with a table.
  - On the fresh held-out set, every calculation is correct or the miss is explained.
  - The 30-problem set is re-run and reported separately as "after fixes" (the baseline stays 27/30).

## M4.7 — Public web frontend (Vercel → API on the A100)
- [ ] `api.py` (FastAPI, SSE streaming, token, rate limit, size limits) with tests
- [ ] Deploy the API on the A100 (public documents only), plus a Cloudflare HTTPS tunnel
- [ ] Static cream frontend and a Vercel proxy function; user logs in to Vercel once
- **Acceptance:**
  - From a phone, the Vercel URL answers the EMI question with the year-wise table.
  - A request without the token is refused.
  - The 11th request in a minute is rate-limited.
  - With the GPU off, the page shows "offline".

### ✅ Chatbot deliverable complete (target Sat Oct 17 night)

## M5 — Benchmark harness (Sun AM)
- [ ] `questions.yaml` (frozen by member 2 by Sat noon) and `calc_problems.yaml`
- [ ] `run_finbot.py`, `bank_responses.csv`, `make_score_sheet.py`, `report.py`
- **Acceptance:** the full grid runs for `base` and `rag`, and the blind score sheet is generated.

## M6 — Fine-tune (Sun, rented GPU, optional)
- [x] GPU rented (user action) and SSH access shared (Lambda A100 40 GB: finbot-serve + finbot-train-v2)
- [x] Teacher served by vLLM; `make_dataset.py` produces filtered transcripts (Qwen3.6-27B teacher, 1,414 examples after fact-check, cleaning and decontamination)
- [x] Unsloth LoRA, GGUF export, `ollama create finbot-8b-v2` (Qwen3-8B base; the 9B was dropped in M0)
- **Acceptance:** `rag` vs `rag_ft` compared on calculation accuracy and the rubric sample; the winner ships; both are recorded.

## M8 — Fine-tune v4 (from the v3 evaluation)
- [ ] FinBot code fixes: unknown tool arguments rejected, with aliases; routes for simple interest, sinking fund, PPF and SIP; KB sources for SCSS, SEBI's F&O study and 1930
- [ ] v4 data: new-tool, weak-family and Indian-language questions (27B teacher, fixed code, no test-set items)
- [ ] Train v4 (v3 recipe and format), test it on the same sets
- **Acceptance:** v4 reaches at least 85% on the test bank and on CA, at least 24/34 McGraw problems and at least 7/10 Indian-language questions, with no loss on concept, safety or held-out. Plan: `docs/finetune_v4_plan.md`.

## M7 — Booth hardening (Sun PM)
- [ ] Offline test, backup screen recording, README
- **Acceptance:** a cold start to the first answer takes 2 minutes or less.

## Evidence log
(Add dated entries here as milestones close.)

### 2026-10-16 — M0 closed
- Environment: Python 3.12.13, langgraph 1.2.15, langchain-ollama 1.1.0, langchain-core 1.6.11, streamlit 1.66.0. Ollama runs `qwen3.5:9b` on the MLX runner (nvfp4, 9.65B parameters).
- `uv run python scripts/spike_toolcalling.py`:

| Thinking | Correct tool calls | Median time | Max time | Speed |
|---|---|---|---|---|
| off | 4/5 | 7.8 s | 30.5 s (includes model load) | 11–14 tok/s |
| on | 4/5 | 26.3 s | 37.6 s | 11–16 tok/s |

- Decision: keep `qwen3.5:9b` with thinking off.
- Failure found: for "₹20 lakh", both modes passed `principal=200000` (10× too small). Fix: precheck normalizes lakh/crore amounts (SPEC changelog 2026-10-16). Must be covered by a test in M3.

### 2026-10-16 — M1 closed
- `uv run pytest -q` → **60 passed**.
- Reference values come from textbooks, not our code: EMI ₹17,356.46 (₹20 lakh, 8.5%, 240 months); SIP ₹23,23,390.76 (₹10,000/month, 12%, 10 years); bond Macaulay duration 4.1699 (5-year 10% par); Black-Scholes call 4.7594 / put 0.8086 (Hull's example).
- Rejected inputs include `__import__('os')`, attribute access, lists, `10**10**10`, unknown names, and `20,00,000` (with a hint to drop separators).
- Bug found by the tests and fixed: solving `tvm` for years overflowed because the search started at 10,000 periods; the search range now grows step by step.

### 2026-10-16 — Blockers and parallel work
- M2 acceptance is waiting on indexing the 15 downloaded documents (629-page book included). The corpus itself is in `kb/raw/` with a 15-row `kb/sources.csv`.
- M3 acceptance is waiting on the index and on a speed fix. In the first end-to-end run (no index), answers took 32–66 s. The Mac was swapping about 7.9 GB because the `qwen3.5:9b` default variant (nvfp4 on MLX) uses 10.6 GB. `qwen3.5:9b-q4_K_M` is being downloaded to compare.
- The first end-to-end run also found fake `[S1]` citations and a wrong crypto tax rule. Review checks 6–8 were added (SPEC changelog).
- M4 code (`app.py`, `run_demo.sh`) and M5 code are being written in parallel, per the CLAUDE.md exception for blocked milestones.
- Corpus gaps: the official Income Tax VDA PDF blocked automated download (a partial ITR-2 FAQ page was saved instead). The Income-tax Act, 2025 has applied since 2026-04-01, so section numbers such as 194S changed.

### 2026-10-16 — M2 closed
- Corpus: 15 official documents in `kb/raw/` (OpenStax *Principles of Finance 2e*, 629 pages; 13 SEBI/RBI/DICGC/AMFI PDFs; 1 Income Tax HTML page), listed in `kb/sources.csv`.
- `uv run python -m finbot.kb.ingest` → 1,724 chunks from 15 files in 208 s, none skipped.
- `uv run python scripts/check_retrieval.py` → **10/10** known-answer queries found the right source in the top 5 (8 at rank 1).
- Threshold calibration: relevant queries scored 0.53–0.75, off-topic queries 0.41–0.48. `FINBOT_MIN_DENSE_SCORE` raised from 0.45 to 0.50, so all 4 off-topic queries are rejected and all 10 known queries stay relevant.
- Known gap: the Income Tax VDA PDF could not be downloaded automatically (see blockers above).

### 2026-10-16 — M2 update: knowledge base rebuilt with 26 documents (user requirement: personal, corporate finance, guidelines)
- Corpus: 26 documents, each with one `kb/sources.csv` row and no mismatches. All 11 new files have extractable text (no scans, no glyph artifacts).
  - `personal/` has 5 documents.
  - `books/` has 2: *Principles of Finance 2e* and *Principles of Accounting Vol 1* (973 pages).
  - `guidelines/` has 19.
- `uv run python -m finbot.kb.ingest` → **3,934 chunks from 26 files in 520 s** (was 1,724 from 15), none skipped. The embedding shared Ollama with the live `qwen3:8b` demo.
  - New chunk counts: Accounting Vol 1 1,702; KYC Directions 172; NCFE 78; FAME 59; SEBI booklet 51; Finance Act 2025 highlights 38; Digital Lending 36; IRDAI 30; NPS FAQ 22; AY 2026-27 salaried page 20; VDA PDF 2.
  - The chunk counts of the old 15 files did not change.
- `uv run python scripts/check_retrieval.py`: before **10/10** (8 at rank 1, all in the top 3); after **9/10** (6 at rank 1).
  - FAIL: "Can a SEBI-registered broker work with an unregistered finfluencer?" The circular is now at rank 6 (it was 3). The new SEBI securities-market booklet's "unregistered investment advisers" pages take ranks 1, 2 and 5. The circular never uses the word "finfluencer", so the match depends on the wording: "What are SEBI's rules on finfluencers?" still finds it at rank 3.
  - The VDA query now ranks the official Income Tax VDA PDF first, which closes the M2 known gap. The check still counts only the ITR-2 FAQ (rank 2).
  - Off-topic queries still score 0.41–0.48, so all are rejected at 0.50.
  - The search was then changed in a parallel session: `MAX_PER_FILE` in `src/finbot/kb/search.py` went from 3 to 2. The check is now **10/10** (6 at rank 1). The finfluencer circular is at rank 5, so it is still **outside the top 3** that the app passes to the model. The other 9 queries are in the top 3.
- New-document queries (ad hoc): before 2/12 in the top 5, after **11/12**, all 11 in the top 3 (the same with cap 2 or 3). Accrual accounting, double entry and trial balance → Accounting Vol 1 at rank 1. NPS Tier II → NPS FAQ at 1. Term vs endowment → IRDAI at 1. New-regime slabs and 87A → Finance Act 2025 at 1. Demat → SEBI booklet at 1. Lending-app contacts → Digital Lending at 2. Bank KYC documents → KYC Directions at 2. The one miss was a long compound budget query. "How do I make a monthly budget?" finds FAME at 1 and NCFE at 2.
- Caveat: "How do I complain against my insurance company?" ranks the outdated IRDAI handbook first. Its complaint section is the outdated part.

### 2026-10-16 — Model decision (SPEC changelog)
`uv run python scripts/compare_models.py qwen3.5:9b-q4_K_M qwen3:8b qwen3.5:4b` (6 acceptance prompts, document search on):

| Model | Acceptance | Median answer | Slowest | Writing speed | Memory |
|---|---|---|---|---|---|
| qwen3.5:9b-q4_K_M | 6/6 | 46.0 s | 70.6 s | 7.9 tok/s | 5.4 GB |
| **qwen3:8b** | 6/6 | **28.2 s** | 66.0 s | 8.9 tok/s | 5.4 GB |
| qwen3.5:4b | 6/6 | 45.6 s | 73.0 s | 16.5 tok/s | 6.4 GB (4/6 drafts rewritten) |

Shipped: `qwen3:8b`. A manual audit then found invented tax figures, which led to review check 9; retrieval was cut to the top 3 chunks.

### 2026-10-16 — M3 closed
- `uv run pytest -q` → 128 passed.
- `uv run python -m finbot.cli --acceptance` on `qwen3:8b` → **6/6**, each answer read by hand. Times were 30.3, 20.8, 53.5 (one rewrite: missing risk note), 19.1, 8.7 and 36.5 s, for a median of about 25 s.
- Audit notes:
  - EMI exact (₹17,356.46).
  - Crypto: correct 30% under 115BBH, 4% cess, Schedule VDA, cited. It names the 1961 Act although the Income-tax Act, 2025 applies from 2026-04-01; this is a corpus gap, and the official VDA PDF needs a manual download.
  - Convexity: correct concept answer with no citation, which is right because the book index has no "convexity" text.

### 2026-10-16 — M4 closed
- Browser test (Chrome DevTools against `localhost:8501`):
  - The SIP example answered ₹50,45,760, which is correct.
  - The "Details" panel shows the pre-check, the calculator formula, three linked AMFI/SEBI sources, and the review result.
  - The question counter and example buttons refresh after each answer.
- Offline check: `lsof` on the app and Ollama processes during a live answer showed **no non-local connections**. The app is now bound to `127.0.0.1`. **The literal Wi-Fi-off test is left for the booth rehearsal (user action).**
- Fixed during the test: PDF glyph names (`/uni20B9` → ₹, 34 cases in the AMFI booklet) were cleaned and the index rebuilt (0 artifacts left, retrieval still 10/10).
- Temperature set to 0 after the crypto audit (SPEC changelog).

### 2026-10-16 — M4.5 closed (user request: tables, statements, cream UI)
- `uv run pytest -q` → **157 passed**.
- Browser results:
  - The EMI chip produced a 20-row year-wise schedule plus a total row (paid ₹41,65,552, interest ₹21,65,552, principal ₹20,00,000, closing 0).
  - The SIP chip produced a 15-row table ending at ₹50,45,760.
  - A CSV upload of a balance sheet and P&L produced a ratio table (current 1.77, D/E 0.53, coverage 5.78, ROE 18.75% = DuPont).
- Theme checked by screenshot:
  - cream `#FAF9F5`, terracotta accent, light tight headline, uppercase labels;
  - pill chips, a dark user pill on the right, white table cards with sticky header and total row, a quiet "Details" disclosure.
- Real-model failures found and fixed:
  1. The cash-flow tool's nested schema produced 3 invalid calls, a silently dropped working-capital argument, an invented profit-on-sale line and a wrong CFO (184 s). Fixed with flat inputs, `[opening, closing]` support, and unknown arguments rejected. Result: 1 call, correct CFO ₹94,000, reconciled, 31 s.
  2. The ratio tool received invented `total_liabilities=1150`, `operating_cash_flow=0` and `gross_profit=840`. Fixed with an input-grounding guard: statement-tool numbers must appear in the user's message, or be an exact difference for a working-capital change. The model retried with only the given figures, and the result was correct.
  3. "Leverage" in a ratio question triggered the high-risk flag, forcing a needless rewrite (120 s). Fixed.
- Regression: `finbot.cli --acceptance` → 6/6 (median about 35 s).

### 2026-10-16 — M5 baseline: calculation benchmark (in progress; question set not yet frozen)
`uv run python bench/run_finbot.py --config rag --set calc` on `qwen3:8b` → **27/30 correct** (0.5% tolerance), median 29.3 s, slowest 71.8 s, 17 min total, 0 errors, 5 rewrites. Raw file: `results/finbot-rag-calc.jsonl` (not committed).

| Problem | What went wrong | Root cause |
|---|---|---|
| c14 years to double at 9% | Used the Rule of 72 (8.0) instead of computing it (8.04) | "how many years … double" not recognised as a calculation, so no tool was required |
| c15 monthly saving for ₹1 crore | Called `sip_future_value` with ₹1 crore as the *monthly* amount and answered "₹10 lakh a month" | Wrong tool: a goal-based saving needs `tvm` solving for `pmt` |
| c28 15% of ₹2,45,000 plus 18% GST "on that amount" | Applied GST to ₹2,45,000 instead of to the 15% share | Misread the question |

c29 (real return) and c30 (effective annual rate) were correct but computed without a tool. That breaks rule 1, and the review did not catch it because those phrasings are not flagged as calculations.
Caution: any fix must be checked on *new* problems, not these 30, or the score measures tuning to the test rather than ability.

### 2026-10-17 — M4.6 work: fixes from the live McGraw run on Qwen3.6-27B (no boxes ticked)
- SPEC changelog 2026-10-17 (four entries): number-free fallback, context budget (compact schemas, compact tool results, output cap per backend, overflow retry, multi-turn history budget), new tool contracts (`tvm` annuities and withdrawals, `loan_emi` payments_per_year, payback, `perpetuity`, `project_cash_flows`, `other_expenses`), conventions, currency clean-up, `hra_50pct_cities`, and the question-bank addendum (Hindi/Tamil amounts, capital gains, decision questions, tool-name clean-up, missing inputs).
- `uv run --offline pytest -q` → **471 passed** (367 at d924788).
- Tool schemas on the vLLM path, Qwen tokenizer: 10,408 → 6,213 tokens (all 19 tools). History plus the current turn is held to about 6.1k estimated tokens on vLLM (1,500-token output cap).
- Exact course-set gold values were checked through the tools outside the repo (13/13 exact).
- Live smoke on the 27B through the tunnel, one question at a time: all 13 target problems answered with the right tool. The 3 failures in the first pass were fixed and re-run:
  - bid price: a cents figure was passed as dollars;
  - two-year balance sheet: grounding errors gave no balancing hint, so the final answer was empty;
  - a follow-up of a follow-up was not routed and invented a figure.
  - Before the fixes, 2 problems hit HTTP 400; now 0 do.
- First drafts still often skip the tool (6 of 18). The reviewer note fixes them, at the cost of one revision (10–40 s).
- 2026-10-17 M6 fine-tune v2: Qwen3-8B LoRA r32/a32, lr 1e-4, 2 epochs, 1,340 train / 73 eval examples, compact tool schemas, 168 steps on an A100.
  - Eval loss went from 0.682 to 0.455 (best at the last step). Q5_K_M GGUF sha256 ea340d67… verified on the Mac; the training A100 was terminated 08:19 IST.
  - Serving fix needed: `FINBOT_THINKING=auto` and `FINBOT_COMPACT_TOOLS=on`. With `think=false`, v2 never called a tool (see the SPEC changelog).
  - Side by side on the same A100 Ollama with the same code (commit f39c6d4), v2 against base qwen3:8b:
    - held-out: 20/20 against 18/20;
    - McGraw: 8/34 problems fully correct (10/184 parts) against 14/34 (79/184). v2 skipped the tool on 15/34 and fell back on 20/34;
    - team question bank: numeric 7/11 against 7/11; review passed 62/83 against 74/83.
  - **Winner: base qwen3:8b, which stays on finbot-local.** v2 is better on simple Indian calculations but much worse on textbook statement and NPV problems, which the training data barely covered after decontamination. Both results are recorded here; v2 stays registered on the Mac as `finbot-8b-v2`, but it is not served.
- 2026-10-17 overnight, 27B (finbot-blond) on the latest code against the night baseline:
  - McGraw: 32/34 problems (170/184 parts) against 26/34 (125/184);
  - held-out: 20/20 against 17/20;
  - question bank: 0 blank fallbacks against 4.
- 2026-10-17 v3 evaluation (A10, production code): test bank 113/159 (71%), CA 72/97 (74%), McGraw 17/34 (98/184 parts); concept 27/27, safety 15/16, Indian languages 3/10. The 27B on the same sets: 151/159, 92/97, 32/34. Failure causes are in docs/finetune_v4_plan.md. All Lambda GPUs were terminated at the user's request after the results and chat logs were copied to results/lambda_a10 and results/lambda_a100.
- 2026-10-19 v4 evaluation (A100, Ollama Q5_K_M, production code):
  - Training: 1,941 examples, eval_loss 0.397.
  - Test bank 133/159 (84%), up from 71%.
  - CA 72/97 → 76/97 (78%).
  - McGraw 17/34 → 19/34, with parts 98/184 → 92/184.
  - Indian languages 3/10 → 9/10, concept 26/27, safety 15/16, held-out 19/20, question bank 7/11.
  - **M8 acceptance not met:** the test bank misses 85% by 2 questions, CA by 6 and McGraw by 5; concept and held-out are each 1 lower.
  - Remaining causes: 17 blank fallbacks; 16 answers with no tool call; tool-argument errors (project_cash_flows "life", npv_irr "cashflows2", capital_gains prices, calculate syntax).
  - Report: results/lambda_v4_eval/v4_report.md. All Lambda GPUs terminated.
- 2026-10-19 v4.1 retest: the same finbot-8b-v4 GGUF on the v4.1 code and KB (commits d1c9990, 6bc4c7b, b756ffd); A100, Ollama, same sets.
  - Test bank 142/159 (89%; v4 84%). CA 78/97 (80%; v4 78%). McGraw 18/34 (v4 19). Question bank 8/11. Held-out 20/20.
  - Concept 27/27, safety 16/16, Indian languages 10/10, regulatory 21/22.
  - Against v4: 22 questions fixed, 9 broke. All 9 were routed the same as in v4, so they reflect run-to-run variation of the model, not the code.
  - The fraud question now answers 1930 [S1]; v4 invented a number. The "guaranteed 2% a month" question gets a clear warning; v4 gave a blank.
  - **M8 acceptance still not met:** CA is 5 questions short of 85%, McGraw 6 short of 24/34.
  - Caveat: the fixes were found from v4's failures on these sets, so this retest is optimistic.
  - Report: results/lambda_v41_eval/v41_report.md. All Lambda GPUs terminated.
- 2026-10-20 v5 evaluation (A100). Training: 2,189 examples (248 new), eval_loss 0.34.
  - Ollama Q5_K_M, v5 vs v4.1:
    - test bank 138/159 (87%) vs 142 (89%);
    - CA 69/97 (71%) vs 78 (80%), more skipped tools and 13 blank fallbacks;
    - McGraw 18/34 vs 18;
    - question bank 9/11 vs 8.
  - **Fresh blind set** (100 new questions, never used for tuning or training), v4.1 code on every row:
    - v4 on Ollama 63%;
    - v5 on Ollama 65%;
    - **v5 on vLLM (bf16 LoRA, forced tool calls) 72%**, median 14 s.
  - v5 on vLLM, older sets: test bank 144/159 (91%), CA 77/97 (79%, 2 blank fallbacks).
  - **Reading:** the extra training data barely changed the 8B on Ollama. The gain comes from serving it on vLLM with forced tool calls. v4 on vLLM was not tested, so part of the gain may come from the serving alone.
  - **M8 acceptance still not met:** CA is below 85% on every configuration.
  - Reports: results/lambda_v5_eval/{evals,evals_vllm}/v5_report.md. All Lambda GPUs terminated.
