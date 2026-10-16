"""M6 step 2: LoRA fine-tune with Unsloth on the rented GPU, then export GGUF for Ollama.

python train/finetune.py --base Qwen/Qwen3-8B --data train/data/sft_v2_final.jsonl --out train/out_v2
(bf16 LoRA. v1 used about 18 GB at 8k tokens on an A100 40 GB; 12k tokens fits in 40 GB.)

v2 settings (2026-10-17), from what v1 and the v2 data showed:
- max_seq 12288: every example carries the system prompt and all tool schemas (~9-10k tokens before the answer),
  so v2 examples run 10-11k tokens. Over-length examples are DROPPED, never truncated, because truncation cuts the
  assistant answer at the end and leaves nothing to learn.
- LoRA r=32, alpha=32 on all linear layers: twice v1's capacity for ~2x the data and more tools; alpha = r keeps scale 1.
- lr 1e-4 (v1: 2e-4), cosine, 5% warmup, 2 epochs: v1 ended with train loss 0.65 vs eval 0.76 (a little overfit).
- Evaluate and checkpoint every --eval-steps, keep the best by eval loss (v1 kept only the last step).
- Eval split stratified by source so every kind of example is measured.
- Export Q5_K_M (better fidelity for tool-call JSON and digits, 5.9 GB, fits the 16 GB Mac) and Q4_K_M as fallback.
"""
import unsloth  # noqa: F401  (must be imported before trl/transformers so its patches apply)

import argparse
import json
import random

from datasets import Dataset
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3-8B")
    ap.add_argument("--data", default="train/data/sft.jsonl")
    ap.add_argument("--out", default="train/out")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--max-seq", type=int, default=12288)  # system prompt + 16 tool schemas + sources is ~10k tokens before the answer
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--eval-steps", type=int, default=20)
    ap.add_argument("--quant", default="q5_k_m,q4_k_m")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--eval-frac", type=float, default=0.05)
    a = ap.parse_args()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=a.base, max_seq_length=a.max_seq, load_in_4bit=False, load_in_16bit=True, full_finetuning=False)
    model = FastLanguageModel.get_peft_model(
        model, r=a.rank, lora_alpha=a.alpha, lora_dropout=0, bias="none", random_state=3407, max_seq_length=a.max_seq,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth")

    rows = [json.loads(line) for line in open(a.data, encoding="utf-8")]
    random.Random(3407).shuffle(rows)
    # enable_thinking=False matches how FinBot serves the model (Ollama with thinking off)
    kept, dropped, lengths = [], 0, []
    for r in rows:
        text = tokenizer.apply_chat_template(r["messages"], tools=r.get("tools"), tokenize=False, enable_thinking=False)
        n = len(tokenizer(text, add_special_tokens=False)["input_ids"])
        if n > a.max_seq:
            dropped += 1          # never truncate: the answer is at the end
            continue
        lengths.append(n)
        kept.append((r.get("source", "other"), text))
    lengths.sort()
    print(f"examples kept {len(kept)}, dropped as too long {dropped}; tokens median {lengths[len(lengths)//2]}, max {lengths[-1]}")
    by_src = {}
    for src, text in kept:
        by_src.setdefault(src, []).append(text)
    train_txt, eval_txt = [], []
    for src, texts in sorted(by_src.items()):   # stratified: every source is represented in eval
        n_eval = max(1, round(len(texts) * a.eval_frac)) if len(texts) >= 10 else 0
        eval_txt += texts[:n_eval]
        train_txt += texts[n_eval:]
        print(f"  {src}: train {len(texts) - n_eval}, eval {n_eval}")
    random.Random(3407).shuffle(train_txt)
    train_ds, eval_ds = Dataset.from_dict({"text": train_txt}), Dataset.from_dict({"text": eval_txt})
    print(f"train={len(train_ds)} eval={len(eval_ds)}; example:\n{train_txt[0][:1500]}")

    trainer = SFTTrainer(  # TRL 1.x names: processing_class (not tokenizer), max_length (not max_seq_length)
        model=model, processing_class=tokenizer, train_dataset=train_ds, eval_dataset=eval_ds,
        args=SFTConfig(max_length=a.max_seq, dataset_text_field="text", per_device_train_batch_size=a.batch, gradient_accumulation_steps=a.grad_accum,
                       num_train_epochs=a.epochs, learning_rate=a.lr, warmup_ratio=0.05, lr_scheduler_type="cosine",
                       weight_decay=0.01, max_grad_norm=1.0, logging_steps=5,
                       eval_strategy="steps", eval_steps=a.eval_steps, save_strategy="steps", save_steps=a.eval_steps,
                       save_total_limit=3, load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
                       optim="adamw_8bit", seed=3407, output_dir=f"{a.out}/checkpoints", dataset_num_proc=1,
                       report_to="none"))
    # Learn only the assistant turns; system, context, user and tool-result text are masked (Qwen ChatML markers).
    trainer = train_on_responses_only(trainer, instruction_part="<|im_start|>user\n", response_part="<|im_start|>assistant\n")
    trainer.train()
    print(trainer.evaluate())

    model.save_pretrained(f"{a.out}/lora")
    tokenizer.save_pretrained(f"{a.out}/lora")
    model.save_pretrained_gguf(f"{a.out}/gguf", tokenizer, quantization_method=a.quant.split(","))
    print(f"Saved LoRA to {a.out}/lora and GGUF to {a.out}/gguf")


if __name__ == "__main__":
    main()
