"""Fine-tune v3 training text matches Ollama's think=false rendering (train/sft_format.py)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
from sft_format import THINK, mark_no_think, uniform_think  # noqa: E402

RENDERED = ("<|im_start|>system\nYou are FinBot.<|im_end|>\n"
            "<|im_start|>user\nWhat is an SIP?<|im_end|>\n"
            "<|im_start|>assistant\nA monthly investment plan.<|im_end|>\n"
            "<|im_start|>user\nEMI on 20 lakh at 8.5% for 20 years? /no_think<|im_end|>\n"
            "<|im_start|>assistant\n<tool_call>\n{\"name\": \"loan_emi\"}\n</tool_call><|im_end|>\n"
            "<|im_start|>user\n<tool_response>\n{\"emi\": 17356.46}\n</tool_response><|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\nYour EMI is 17,356.46.<|im_end|>\n")


def test_every_assistant_turn_after_the_question_starts_with_the_think_block():
    out = uniform_think(RENDERED)
    assert "<|im_start|>assistant\n" + THINK + "<tool_call>" in out                     # the tool-call turn gets it
    assert out.count(THINK) == 2                                                          # final turn not doubled
    assert "<|im_start|>assistant\nA monthly investment plan." in out                    # history turn untouched
    assert uniform_think(out) == out


def test_no_think_marks_only_the_last_question():
    msgs = [{"role": "user", "content": "What is an SIP?"}, {"role": "assistant", "content": "A plan."},
            {"role": "user", "content": "EMI?"}, {"role": "assistant", "content": "", "tool_calls": []},
            {"role": "tool", "content": "{}"}]
    out = mark_no_think(msgs)
    assert out[2]["content"] == "EMI? /no_think" and out[0]["content"] == "What is an SIP?"
    assert msgs[2]["content"] == "EMI?" and mark_no_think(out)[2]["content"] == "EMI? /no_think"
