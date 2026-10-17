# Fine-tune v4 plan (from the v3 evaluation, 2026-10-17)

## Where v3 stands

finbot-8b-v3 was tested on the Lambda A10 with the production code: Ollama with the Q5_K_M GGUF, compact tools and thinking off. Every set below is test-only and never trained on.

| Set | v3 | 27B (finbot-blond) | base qwen3:8b |
|---|---|---|---|
| New test bank (159) | 113 (71%) | 151 (95%) | — |
| CA Foundation, ICAI (97) | 72 (74%) | 92 (95%) | — |
| McGraw, problems fully correct (34) | 17–20 | 32 | 13 |
| Held-out (20) | 20 | 20 | 18 |
| Team question bank, numeric (11) | 9 | 10 | 7 |

v3 does best on concepts (27/27) and safety (15/16). It is weakest on Hindi, Tamil and Hinglish (3/10), Indian calculations (29/48), regulatory facts (14/22) and some CA topics:
- simple interest: 7/11
- sinking fund: 0/2
- bills of exchange: 0/2
- annuity due: 2/4

## Why it fails (88 failures)

1. **New tools are never used.** `tax_rules` and `capital_gains` were added after v3's data was made. Rule questions (standard deduction, LTCG rate, NPS 80CCD(1B), DICGC limit) were answered by hand, so they got the blank fallback.
2. **Tool arguments are silently dropped.** 44 `tvm` calls failed with "annual_rate_pct is required" in 17 questions, and every one of those questions gives the rate. The LangChain tool schema ignores unknown argument names, so a rate passed under another name, such as `rate`, simply disappears.
   - There were also 6 `goal_saving` calls missing `target` and 6 `bond_metrics` calls with neither or both of `ytm_pct` and `price`.
3. **Wrong tool for the job.** SIP questions went to `goal_saving` instead of `sip_future_value` (including step-up SIPs), and PPF yearly deposits got no tool.
4. **No tool, then the blank fallback.** 27 answers were worked out by hand. Ollama cannot force a tool call (no `tool_choice`), so the revision cannot make it use one. The 27B on vLLM can be forced.
5. **Missing facts in the knowledge base:** the Senior Citizens Savings Scheme limit, SEBI's study that 9 in 10 F&O traders lose money, and the 1930 cybercrime helpline.
6. **Indian languages.** In Hindi, Tamil and Hinglish calculation questions it rarely calls a tool.

## v4 plan

### A. FinBot code first

These fixes help every model and take about an hour, with tests.

1. **Tool arguments:** reject unknown argument names with a "did you mean …" error instead of dropping them. Accept common aliases (`rate`, `rate_pct`, `interest_rate` → `annual_rate_pct`; `target_amount`, `goal` → `target`).
2. **Routes and hints:**
   - simple interest → `calculate` with the SI formula;
   - sinking fund → `tvm` solving for `pmt` with `fv`;
   - PPF and other yearly deposits → `tvm` with `when="begin"`;
   - SIP future value → `sip_future_value`, never `goal_saving`.
3. **Knowledge base:** add the official sources for the SCSS limits, SEBI's F&O study and the 1930/cybercrime.gov.in reporting, then rebuild the index.

### B. v4 training data

The teacher is the 27B on vLLM running the fixed code. It answers new synthetic questions only, never test-set questions.

1. **New tools (about 180):**
   - `tax_rules`: about 60
   - `capital_gains`: about 60
   - freelance `income_tax`: about 30
   - `cash_flow_statement` with balance pairs: about 30
2. **Weak calculation types (about 360):**
   - SIP, including step-up and start-of-month: about 60
   - PPF and yearly deposits: about 30
   - FD/RD with quarterly compounding: about 40
   - annuity due and sinking fund: about 40
   - present value of uneven cash flows: about 30
   - credit-card and flat-rate loans: about 30
   - CA-style simple interest, compound interest, depreciation and bills, in our own wording: about 80
3. **Languages:** about 150 Hindi, Tamil, Hinglish and Tanglish calculation questions that must call a tool.
4. **Team chats:** the questions people asked on finbot-blond and finbot-local (redacted, kept in `results/lambda_a100`, `results/lambda_a10` and the Mac chat logs) become new prompts. The teacher writes fresh answers.
5. **Mix:**
   - lower the concept cap from 0.55 to 0.40 (v3 already gets every concept question right);
   - keep the fact check, gold checks and decontamination, with a stricter similarity cut-off of 0.6 against the test bank, CA, McGraw, question bank and held-out sets.
6. **Target:** about 2,300 examples (v3's 1,663 plus about 700 new).

### C. Training

- Same recipe as v3: LoRA r32/α32, lr 1e-4, 2 epochs, max_seq 12288, and the v3 text format (empty think block before every assistant turn, `/no_think` on the question).
- v3's eval loss was still falling at the end (0.371), so try 3 epochs if v4's eval loss keeps dropping.
- About 2–2.5 h on an H100.

### D. Serving

- Serve the 8B with vLLM instead of Ollama, so the revision of a hand-computed answer can be forced to call a tool (`tool_choice="required"`, the fix that took the 27B to 32/34 on McGraw).
- This needs a vLLM chat template that matches the training format.
- The A10 is enough (8B in bf16 is 16 GB).

### E. Acceptance (same test-only sets)

v4 replaces v3 if it reaches:
- at least 85% on the test bank and on CA;
- at least 24/34 McGraw problems;
- at least 7/10 Indian-language questions;
- no loss on concept, safety or held-out.

### F. Cost and time

| Step | GPU | Time | Cost |
|---|---|---|---|
| Data generation | A100 (27B) | about 2 h | about $4 |
| Training | H100 | about 2.5 h | about $8 |
| Evaluation | A10 | about 1 h | about $1.30 |
| **Total** | | **about 5–6 h** | **about $15** |
