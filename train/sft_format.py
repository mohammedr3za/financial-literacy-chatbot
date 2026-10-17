"""Training-text format that matches how Ollama serves FinBot with think=false (fine-tune v3).

v2 lesson: the Qwen3 chat template puts the empty think block only before the final answer, never before a tool-call turn,
so the model learned "empty block -> answer directly". Ollama with think=false pre-fills that block at every step (and adds
" /no_think" to the question), so v2 never called a tool. Here every assistant turn after the question starts with the
block and the question ends with " /no_think": the model then learns from the question whether to call a tool."""
THINK = "<think>\n\n</think>\n\n"
ASSISTANT = "<|im_start|>assistant\n"
USER = "<|im_start|>user\n"


def mark_no_think(messages: list[dict]) -> list[dict]:
    """A copy of messages with " /no_think" after the last user question (not a tool result), as Ollama sends it."""
    out = [dict(m) for m in messages]
    idx = max((i for i, m in enumerate(out) if m.get("role") == "user"), default=None)
    if idx is not None and not str(out[idx].get("content", "")).rstrip().endswith("/no_think"):
        out[idx]["content"] = str(out[idx].get("content", "")) + " /no_think"
    return out


def uniform_think(text: str) -> str:
    """Every assistant turn after the last user question starts with the empty think block."""
    start, pos = 0, 0
    while (i := text.find(USER, pos)) != -1:          # the last user turn that is not a tool result
        if not text.startswith("<tool_response>", i + len(USER)):
            start = i
        pos = i + 1
    head, tail = text[:start], text[start:]
    parts = tail.split(ASSISTANT)
    return head + parts[0] + "".join(ASSISTANT + (p if p.startswith(THINK) else THINK + p) for p in parts[1:])
