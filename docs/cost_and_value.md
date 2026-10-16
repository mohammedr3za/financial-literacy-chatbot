# Cost and value of running FinBot in an Indian bank app

*As of 2026-10-16. USD/INR = 94.96 (ECB reference rate). Prices exclude GST unless stated.*

**Bottom line.** At a bank's volume, a self-hosted open model on Indian GPUs costs **about ₹0.08–0.10 per answer**. That is 2–5× cheaper than small closed-model APIs, and it keeps customer data in India under the bank's control. A human-handled contact in an Indian banking contact centre costs roughly **₹45–55**, a vendor estimate. So every query the bot resolves instead of a call saves about 500× its own running cost.

## 1. What one FinBot answer uses (measured)

| Item | Tokens | How measured |
|---|---|---|
| Fixed system turn: rules plus 16 tool definitions | 8,385 | Qwen3 tokenizer, vLLM path. Tool definitions are 7,473 of these. |
| Per-question turn: pre-check notes, 3 retrieved passages, question | ~1,300 | Median over the benchmark questions |
| Model calls per answer | 2 typical, 4 heavy | One answer pass, plus a tool round or a review revision |
| **Input per typical answer** | **~19,900** | 2 × (8,385 + 1,300) + tool results |
| Output per typical answer | ~400 | Answer text plus tool calls |

The fixed 8.4k system turn is the same on every call, so vLLM's prefix cache processes it once. Self-hosted cost is therefore driven mainly by the ~2–4k new tokens per answer.

## 2. Cost per answer: hosted API vs self-hosted

| Option | Price used | Cost per typical answer | Per 10 lakh (1M) answers |
|---|---|---|---|
| Gemini 3.8 Flash (closed API) | $0.75 / $3.75 per 1M tokens in/out | ₹1.56 | ₹15.6 lakh |
| OpenAI gpt-5-mini (closed API) | $0.25 / $2.00 | ₹0.55 | ₹5.5 lakh |
| Claude Haiku 5.5 / OpenAI gpt-6-luna (closed API) | $0.10 / $0.50 | ₹0.21 | ₹2.1 lakh |
| Hosted open 8B (DeepInfra Llama-3.1-8B) | $0.02 / $0.04 | ₹0.04 | ₹0.4 lakh |
| **Self-hosted 8B on 1× L4, Mumbai (AWS g6.xlarge, 1-yr reserved $0.63/h)** | 1,800–3,600 answers/h (est.) | **₹0.017–0.033** | **₹0.17–0.33 lakh** |
| **Self-hosted 8B on 1× L4, E2E Networks (₹49/h)** | 1,800–3,600 answers/h (est.) | **₹0.014–0.027** | **₹0.14–0.27 lakh** |
| Self-hosted 27B on 1× A100 40 GB (Lambda $1.99/h) | 1,800–3,600 answers/h (est.) | ₹0.05–0.11 | ₹0.5–1.1 lakh |

The answers-per-hour figure is an estimate. It is based on published vLLM throughput for 8B models on A10-class GPUs (about 350–850 output tokens/s at 16–64 parallel users) and FinBot's short outputs. It assumes the GPU is busy. An idle GPU still costs money: at the exhibition (8 h/day, a few hundred questions) the A100 works out to about ₹2–3 per answer.

## 3. Bank-scale scenario

Assumption: **35 lakh (3.5M) queries a month**, the volume ICICI's iPal reported in 2021. Peak hour is 3× the average, about 14,600 answers per hour, which needs 8 L4 GPUs at about 2,000 answers/hour each.

| Option | Monthly cost | Per query |
|---|---|---|
| Self-hosted, 8× L4 at E2E (₹49/h) | **₹2.86 lakh** | ₹0.08 |
| Self-hosted, 8× L4 on AWS Mumbai (1-yr reserved) | ₹3.49 lakh | ₹0.10 |
| Hosted open 8B API (DeepInfra) | ₹1.37 lakh | ₹0.04, but data processed by a third party, likely abroad |
| Claude Haiku 5.5 / gpt-6-luna API | ₹7.27 lakh | ₹0.21 |
| OpenAI gpt-5-mini API | ₹19.2 lakh | ₹0.55 |
| Gemini 3.8 Flash API | ₹54.5 lakh | ₹1.56 |

**Against human agents.** If the bot fully resolves 10% of those queries that would otherwise have been calls, at ₹45–55 per banking call, it saves **₹1.6–1.9 crore a month**. At 30% the saving is **₹4.7–5.8 crore a month**, against under ₹3.5 lakh of GPU cost. The per-call figure is a vendor estimate; no official Indian figure exists. Gartner's global medians are $13.50 per assisted contact and $1.84 per self-service contact.

## 4. Why self-host instead of calling an API (beyond price)

| Requirement | What it says | How a self-hosted open model helps |
|---|---|---|
| RBI, Storage of Payment System Data (2018) | Payment data stored only in India; data processed abroad must be deleted there and returned within 24 h | Model and logs run on Indian GPUs (AWS Mumbai, E2E, Yotta); nothing leaves the country |
| DPDP Act 2023 and Rules 2025 | Transfers abroad can be restricted; penalties up to ₹250 crore for security failures | No third-party processor for customer chats |
| RBI FREE-AI report (2025) | Guardrails so that, "especially when using open source or external AI models", sensitive data stays "under the control of the institution"; indigenous models give "control over model behaviour, data pipelines, and fine-tuning cycles" | Weights, prompts, tools and fine-tunes are all owned and auditable by the bank |
| RBI draft Model Risk Management guidance (2026) | Independent validation of all models, including vendor models; access to documentation; explainability thresholds | Every number comes from a deterministic, testable tool, with a "How I got this" trace per answer |

## 5. What this project actually spent

| Item | Cost |
|---|---|
| Fine-tuning Qwen3-8B (1× A100, 3 h 49 min at $1.99/h) | $7.60 ≈ ₹722 |
| Serving the 27B for the exhibition (estimate: 2 days × 8 h × $1.99/h) | $31.84 ≈ ₹3,024 |
| Running the 8B offline on the MacBook (about 70 W, 8 h, ₹6.80/kWh) | ≈ ₹4 per day |
| Models and software | ₹0: Apache-2.0 open weights, open-source stack |

## 6. Quality: what the money buys (measured on 34 McGraw-Hill textbook problems)

| Model | Problems fully correct | Median answer time |
|---|---|---|
| Qwen3-8B on the MacBook | 10 / 29 so far (run in progress) | ~50 s |
| Qwen3.6-27B on 1× A100 | **24 / 34** | ~12 s |

Both runs used the same code from before tonight's tool fixes. The fixes target most of the remaining failures, and the scores will be re-measured.

## Sources
- GPU prices: [AWS EC2 g6/g5, Mumbai (ec2.shop mirror)](https://ec2.shop?region=ap-south-1&filter=g6.xlarge) · [E2E Networks pricing](https://www.e2enetworks.com/pricing.md) · [Lambda pricing](https://lambda.ai/pricing)
- API prices: [OpenAI](https://developers.openai.com/api/docs/pricing.md) · [Google Gemini](https://ai.google.dev/gemini-api/docs/pricing) · [Anthropic](https://platform.claude.com/docs/en/about-claude/pricing.md) · [DeepInfra](https://deepinfra.com/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo)
- Throughput: [TrueFoundry vLLM benchmark (Qwen3-8B on A10)](https://www.truefoundry.com/blog/vllm-benchmark) · [Red Hat MLPerf v5.1 (vLLM, Llama-3.1-8B)](https://www.redhat.com/de/blog/efficient-and-reproducible-llm-inference-red-hat-mlperf-inference-v51-results)
- Contact-centre costs: [YuVerse 2026 India benchmark (vendor estimate)](https://www.yuverse.ai/resources/posts/average-cost-per-call-in-indian-contact-centres-2026-benchmark-data) · Gartner cost per contact (2024, via secondary quotes)
- Bank chatbot volumes: [ICICI iPal (Asian Banker)](https://www.theasianbanker.com/updates-and-articles/best-retail-bank-in-india-icici-2022) · [HDFC Bank Annual Report FY2025-26](https://www.hdfc.bank.in/content/dam/hdfcbankpws/in/en/pdf/annual-reports/2025-26/reports/HDFC_Bank_IR26.pdf)
- Regulation: [RBI payment data storage circular](https://www.rbi.org.in/Scripts/NotificationUser.aspx?Id=11244) · [RBI FAQ](https://rbi.org.in/scripts/FAQView.aspx?Id=130) · [DPDP Rules 2025](https://static.pib.gov.in/WriteReadData/specificdocs/documents/2025/nov/doc20251117695301.pdf) · [RBI FREE-AI report](https://rbi.org.in/SCRIPTS/PublicationReportDetails.aspx?ID=1306) · [RBI draft model risk guidance](https://www.rbi.org.in/Scripts/bs_viewcontent.aspx?Id=5089)
- Exchange rate: [ECB reference rates, 2026-10-16](https://www.ecb.europa.eu/stats/eurofxref/eurofxref-daily.xml)
