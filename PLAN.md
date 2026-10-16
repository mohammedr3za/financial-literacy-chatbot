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
- [ ] Download the guideline corpus and the OpenStax book; fill in `kb/sources.csv`
- [ ] `ingest.py`: PDF and HTML to chunks and embeddings
- [ ] `search.py`: hybrid BM25 and dense search with reciprocal-rank fusion
- **Acceptance:** 10 known-answer queries find the correct source in the top 5 (`tests/test_search.py`).

## M3 — Graph and CLI (Sat)
- [ ] `precheck.py` and `review.py`, with unit tests
- [ ] `prompts.py` (trust policy) and `graph.py` (LangGraph)
- [ ] `cli.py`
- [ ] `tests/test_graph.py` with a fake chat model, so no model is needed to run it
- **Acceptance:** all 6 acceptance prompts in SPEC section 10 behave as specified through the CLI.

## M4 — UI and demo script (Sat PM)
- [ ] Streamlit `app.py` with the "How I got this" panel, banner, and session cap
- [ ] `run_demo.sh`
- **Acceptance:** works end to end with Wi-Fi off.

### ✅ Chatbot deliverable complete (target Sat Oct 17 night)

## M5 — Benchmark harness (Sun AM)
- [ ] `questions.yaml` (frozen by member 2 by Sat noon) and `calc_problems.yaml`
- [ ] `run_finbot.py`, `bank_responses.csv`, `make_score_sheet.py`, `report.py`
- **Acceptance:** the full grid runs for `base` and `rag`, and the blind score sheet is generated.

## M6 — Fine-tune (Sun, rented GPU, optional)
- [ ] GPU rented (user action) and SSH access shared
- [ ] Teacher served by vLLM; `make_dataset.py` produces filtered transcripts
- [ ] Unsloth LoRA, GGUF export, `ollama create finbot-9b`
- **Acceptance:** `rag` vs `rag_ft` compared on calculation accuracy and the rubric sample; the winner ships; both are recorded.

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
