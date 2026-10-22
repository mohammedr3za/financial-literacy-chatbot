# FinBot poster: content brief

*Measured 2026-10-16 to 2026-10-21. Prices are as of 2026-10-21 and exclude GST. USD/INR = 94.4 (Wise mid-market). The detailed cost model is in `docs/cost_and_value.md`.*

Exhibition theme: **"Can you trust a financial chatbot?"** The poster's answer: *only if it shows its working*. A small model that checks every number with a calculator, cites official sources and refuses scams can be trusted further than a big model that guesses. It also runs cheaply and privately.

## 0. Model names in plain words (use these labels on the poster)
- **"B" means billions of parameters,** the learned numbers inside a model. More usually means smarter, but it needs more memory and computing power.
- **FinBot-8B** ("our trained small model"):
  - It runs Qwen3-8B, fine-tuned by us.
  - The model file is 5.9 GB and runs offline on a 16 GB laptop.
- **FinBot-27B** ("big model, used as is"):
  - It runs Qwen3.6-27B unchanged.
  - It needs a rented GPU with about 28 GB of memory.
- **Teacher:** the 27B wrote worked answers to practice questions, a calculator kept only the correct ones, and the 8B was trained on them (teacher and student, or "distillation").
- **The same in both:** the calculator tools, official sources and answer checker. Only the model that plans and writes changes.

## 1. Title options
1. **"A finance chatbot that shows its working"**, with the subtitle "An 8-billion-parameter model, a calculator and official sources".
2. "Small model, checked answers: building a trustworthy finance chatbot for India".
3. "79% on CA Foundation maths, at ₹0.08 an answer".

## 2. The four big numbers (top strip)

| Number | Label | Small print |
|---|---|---|
| **79%** | FinBot-8B on ICAI CA Foundation Business Mathematics questions (77/97). The 27B scores **95%** (92/97). | ICAI pass mark 40% per paper; distinction 70%. Asked without the answer options. |
| **₹81 vs ₹2,300+** | Cost per 1,000 answers: FinBot-8B on a rented GPU vs the same work on Claude Opus 5.5 / GPT-5.6 Sol | ₹5 per 1,000 on a MacBook (electricity only) |
| **12 s** | Median time per answer on one GPU | 66 s on a MacBook, offline, with the same 80% score |
| **26/26** | Scam and risky-advice questions handled safely by the 8B | OTP fraud, "guaranteed" returns, Telegram tips, loan apps |

## 3. Panels (suggested: 3 columns × 3 rows on A1)

### Panel A: "Can it pass the CA exam?"
- **Test:** 97 questions from ICAI's own CA Foundation material (suggested answers and question banks, Business Mathematics). Each is scored against ICAI's official answer.
- **Harder than the real exam:** the real Paper 3 is multiple choice. FinBot gets **no answer options** and must work the number out.
- **Never trained on:** the v4 and v5 training data was checked for overlap with these questions, and near-copies were removed.
- **Chart:** a horizontal bar per model, with dashed lines at 40% (pass) and 70% (distinction).

| Model | Score | Median time |
|---|---|---|
| FinBot-8B v3 | 72/97 (74%) | 15 s |
| FinBot-8B v4 | 78/97 (80%) | 20 s |
| **FinBot-8B v5 on a GPU** | **77/97 (79%)** | **12 s** |
| FinBot-8B v4 on the MacBook (offline) | **78/97 (80%)**, the same as on the GPU | 66 s (timed while awake; see Panel D) |
| FinBot-27B (exhibition model) | **92/97 (95%)** | 20 s |

- **Story line:** "Above the distinction line, but still wrong about 1 time in 5. That's why it shows its working."
- **Example to print** (in `results/poster/examples.md`): *"In how many years will money double at 5% compound interest?"*
  - ICAI's key says 14.3 years, from log tables.
  - FinBot's calculator says **14.21 years** and notes that the Rule of 72 gives 14.4.
- **Context only, not a comparison:** in the January 2026 CA Foundation session, 21,099 of 1,09,694 candidates passed (19%). That is the whole four-paper exam, so don't put it next to our score.

### Panel B: "A test it had never seen"
- **What it is:** a blind set of 100 new questions, written by a separate helper. Nobody looked at them while fixing or training the bot, and they were kept out of the training data.
- **Why it matters:** this is the honest score.

| Kind of question | FinBot-8B v5 | FinBot-27B |
|---|---|---|
| CA-style maths (20) | 14 | 20 |
| Indian money maths: EMI, SIP, FD, tax (30) | 19 | 27 |
| Textbook finance (15) | 10 | 13 |
| RBI/SEBI/tax rules (12) | 7 | 8 |
| Safety and scams (10) | **10** | 9 |
| Concepts (6) | 5 | 6 |
| Hindi, Tamil, Hinglish (7) | 7 | 7 |
| **Total** | **72** | **90** |

- **Story line:** "The small model is as safe as the big one. It loses on harder maths, not on honesty."

### Panel C: "What does an answer cost?" (log-scale bar chart)
- **Basis:** the same FinBot work on each option, measured at 25k input tokens (most of them cached) and 500 output tokens per answer.
- **Fairness:** the cloud APIs get their prompt-caching discount, which makes the comparison generous to them.

| Option | ₹ per 1,000 answers |
|---|---|
| FinBot-8B on a MacBook (electricity, ₹7/kWh) | **₹5** |
| Claude Haiku 5.5 (cheapest Claude) | ₹70 |
| **FinBot-8B on a rented A100 GPU** | **₹81** |
| OpenAI GPT-5.6 Luna (cheapest GPT) | ₹140 |
| FinBot-27B on a rented A100 GPU | ₹245 |
| Claude Sonnet 5.5 | ₹1,174 |
| OpenAI GPT-5.6 Terra | ₹1,398 |
| Claude Opus 5.5 | ₹2,347 |
| OpenAI GPT-5.6 Sol (flagship) | ₹2,608 |
| OpenAI GPT-5.5 | ₹3,495 |

Recomputed from the vLLM logs:
- **8B:** 2,021 new input tokens per answer (92.2% of the 25,909 total are repeated) and 499 output tokens. It serves 2,330 answers an hour.
- **27B:** 766 answers an hour.
- An earlier draft used 2,289 answers an hour, which gave ₹82 for the 8B and ₹250 for the 27B.

- **Honest point:** the cheapest cloud models now cost about the same as renting a GPU. FinBot wins on two other things:
  1. **Privacy:** on a laptop or a bank's own GPU, no question leaves the building. RBI's data-storage rules and the DPDP Act favour this.
  2. **Control:** every number comes from our own calculator, which can be audited.
- **Subscriptions:**
  - ChatGPT Go costs ₹399 a month, ChatGPT Plus ₹1,999 and Claude Pro about ₹2,000.
  - One month of ChatGPT Plus pays for about 10.6 hours of the A100, which is about **25,000 FinBot-8B answers**.
  - A full exhibition day on the A100 (8 h) costs about ₹1,500.
- **At bank scale** (from `docs/cost_and_value.md`): about **₹0.08–0.10 per answer** on Indian GPUs. A call-centre call costs about ₹45–55 (a vendor estimate).

### Panel D: "How fast?"
- **Measured times** (the same 37 CA questions, timed while the laptop was awake and plugged in):
  - 8B on one GPU: **median 12 s**, 9 in 10 within 24 s, with about 10 answers at a time.
  - 27B: median 18 s, 9 in 10 within 47 s.
  - MacBook M5 (16 GB, offline): median 66 s, 9 in 10 within 143 s. It scores the same as on the GPU (78/97 on CA), only slower.
- ⚠️ **Correction:** the laptop's first 60 answers ran while the Mac was asleep with its lid closed. That inflated their times: the median over all 97 was 74 s and the p90 196 s. Their marks are unaffected.
- **Why it's slower than ChatGPT's first words:** FinBot does not stream a guess. Each answer goes through:
  1. a plan;
  2. calculator calls;
  3. a reviewer that checks every figure;
  4. one rewrite if the reviewer finds a problem.
- **One line for the poster:** "It waits until the maths is checked before it speaks."

### Panel E: "How it earns trust" (pipeline diagram)
**Question → Pre-check → Search official documents → Model plans → Calculator tools → Reviewer → Answer with sources**

Callouts:
- **Every number comes from a tool,** not the model's memory. A figure the reviewer can't trace to a tool is removed, or the bot says it isn't reliable.
- **Sources are cited** as [S1], [S2] from RBI, SEBI, PIB and National Savings documents.
- **Invented phone numbers are caught.** A helpline number must appear in a source. An early version once invented a fraud helpline; now it gives **1930** from the government notice.
- **It refuses:** stock tips, "guaranteed" returns, OTP sharing, and taking passwords.
- **It speaks Hindi, Tamil and Hinglish,** with the same calculator underneath. A Tamil EMI example is in `results/poster/examples.md`.

### Panel F: "How we built it" (timeline)
- **Base:** Qwen3-8B, open weights (Apache 2.0). Fine-tuned with LoRA, a small add-on layer, so the base model stays unchanged.
- **Teacher:** a bigger open model (Qwen3.6-27B) wrote worked examples. **A calculator checked each one, and wrong ones were thrown away.** Of the 455 newest examples, 359 were kept.
- **Training data:** 2,189 examples. Training took **4.0 hours on one A100 GPU (about ₹750)**.
- **Model file:** 5.9 GB. It runs on a 16 GB laptop.
- **Versions:**
  - v3 → v5 in 5 days (spec approved 16 Oct), 98 commits and 548 automated tests.
  - Blind-set score: v4 63 → v5 65 → v5 with tool calls forced: 72.
- **GPU spend:** the v4 and v5 rounds plus the blind-set test cost about $60 (₹5,700).
  - About half of that, roughly $30, paid for GPUs sitting idle after a stalled run or a sleeping laptop.
  - It was the project's most expensive lesson. Rented GPUs now shut themselves down.

### Panel G: "Where it still fails" (honest box)
- **RBI/SEBI/tax rules: 7–8 of 12** on the blind set. Rules change, and the bot only knows the documents we gave it.
- **About 1 in 5 CA questions wrong** for the 8B. Of its 20 misses:
  - 6 were calculator calls that failed;
  - 6 used the wrong method or numbers;
  - 5 were blocked by the reviewer because it couldn't verify the figures. The bot said "not sure" instead of guessing, which is wrong for the score but right for trust;
  - 3 never used the calculator.
- **Slow on a laptop** (over a minute per answer).
- **Our test, not the official exam:** 97 practice questions from one paper.
- **It cannot see your account.** It doesn't do balances, card blocking or payments.

### Panel H: "Try it"
- A QR code to the live bot. It is free, with no sign-up, and chats are not recorded.
- Suggested prompts:
  - "What is the EMI on a ₹25 lakh home loan at 9% for 15 years?"
  - "Someone asked for my OTP for KYC, should I share it?"
  - A Hindi question.

## 4. Claims to avoid
- **Don't** say "FinBot passed the CA exam". **Say** "79% on ICAI CA Foundation Business Mathematics practice questions".
- **Don't** say "better than ChatGPT" unless both are tested on the same questions (see section 5).
- **Don't** say "cheaper than any AI". Claude Haiku 5.5 and GPT-5.6 Luna are about as cheap. **Say** "as cheap as the cheapest cloud AI, and your data stays with you".

## 5. Open item: ChatGPT and Claude on the same 97 CA questions
- **Status:** not measured yet. It is the comparison visitors will ask about.
- **Recommended way:** run the 97 questions through the GPT-5.6 Sol, Claude Opus 5.5 and Claude Haiku 5.5 APIs as plain chat, graded by the same checker.
- **Needs:** an OpenAI key and an Anthropic key, and about $5–10 in total.
- **Alternative:** paste 20 questions into free ChatGPT and Claude by hand, and we grade them.

## 6. Sources
- Claude API prices: [platform.claude.com pricing](https://platform.claude.com/docs/en/about-claude/pricing)
- OpenAI API prices: [developers.openai.com pricing](https://developers.openai.com/api/docs/pricing)
- GPU: Lambda A100 40 GB at $1.99/h (Lambda API, 2026-10-21)
- ChatGPT India prices: [Bloomberg on ChatGPT Go](https://www.bloomberg.com/news/articles/2025-08-19/openai-rolls-out-5-chatgpt-go-plan-exclusively-for-india) and [Outlook Business](https://www.outlookbusiness.com/artificial-intelligence/openai-rolls-out-low-cost-chatgpt-go-for-indian-users-check-details-here)
- Claude India prices: [Outlook Business](https://www.outlookbusiness.com/deeptech/claude-pro-now-costs-2000-in-india-as-anthropic-localises-subscription-pricing) and [Free Press Journal](https://www.freepressjournal.in/tech/anthropic-introduces-indian-prices-for-claude-ai-subscriptions-heres-how-much-it-costs)
- CA Foundation pass rules and Jan 2026 result: [Careers360 passing marks](https://finance.careers360.com/articles/ca-foundation-and-intermediate-jan-2026-passing-marks) and [Careers360 Jan 2026 pass percentage](https://finance.careers360.com/articles/ca-foundation-pass-percentage-jan-2026-out)
- CA questions: ICAI BoS suggested answers and question banks (resource.cdn.icai.org, boslive.icai.org); the URL of each question is in `results/ca_foundation/questions.jsonl`
- Exchange rate: [Wise USD/INR](https://wise.com/us/currency-converter/usd-to-inr-rate)
- Indian household tariff of about ₹5–7.5/kWh: [GlobalPetrolPrices](https://www.globalpetrolprices.com/India/electricity_prices/)
- Laptop power: about 35 W system load during inference, measured with `ioreg` (AppleSmartBattery PowerTelemetryData)
