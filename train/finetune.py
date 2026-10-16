"""M6 step 2: LoRA fine-tune with Unsloth on the rented GPU, then export GGUF for Ollama.

python train/finetune.py --base Qwen/Qwen3.5-9B --data train/data/sft.jsonl --out train/out
(bf16 LoRA; about 22 GB GPU memory for a 9B model per Unsloth's guide. Qwen3.5 needs transformers v5.)
"""
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
    ap.add_argument("--max-seq", type=int, default=4096)
    ap.add_argument("--eval-frac", type=float, default=0.05)
    a = ap.parse_args()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=a.base, max_seq_length=a.max_seq, load_in_4bit=False, load_in_16bit=True, full_finetuning=False)
    model = FastLanguageModel.get_peft_model(
        model, r=16, lora_alpha=16, lora_dropout=0, bias="none", random_state=3407, max_seq_length=a.max_seq,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth")

    rows = [json.loads(line) for line in open(a.data, encoding="utf-8")]
    random.Random(3407).shuffle(rows)
    texts = [tokenizer.apply_chat_template(r["messages"], tools=r.get("tools"), tokenize=False) for r in rows]
    n_eval = max(1, int(len(texts) * a.eval_frac))
    train_ds, eval_ds = Dataset.from_dict({"text": texts[n_eval:]}), Dataset.from_dict({"text": texts[:n_eval]})
    print(f"train={len(train_ds)} eval={len(eval_ds)}; example:\n{texts[0][:1500]}")

    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer, train_dataset=train_ds, eval_dataset=eval_ds,
        args=SFTConfig(max_seq_length=a.max_seq, per_device_train_batch_size=2, gradient_accumulation_steps=8,
                       num_train_epochs=a.epochs, learning_rate=2e-4, warmup_ratio=0.05, lr_scheduler_type="cosine",
                       logging_steps=5, eval_strategy="steps", eval_steps=50, save_strategy="no",
                       optim="adamw_8bit", seed=3407, output_dir=f"{a.out}/checkpoints", dataset_num_proc=1,
                       report_to="none"))
    # Learn only the assistant turns; system, context, user and tool-result text are masked (Qwen ChatML markers).
    trainer = train_on_responses_only(trainer, instruction_part="<|im_start|>user\n", response_part="<|im_start|>assistant\n")
    trainer.train()
    print(trainer.evaluate())

    model.save_pretrained(f"{a.out}/lora")
    tokenizer.save_pretrained(f"{a.out}/lora")
    model.save_pretrained_gguf(f"{a.out}/gguf", tokenizer, quantization_method="q4_k_m")
    print(f"Saved LoRA to {a.out}/lora and GGUF to {a.out}/gguf")


if __name__ == "__main__":
    main()
