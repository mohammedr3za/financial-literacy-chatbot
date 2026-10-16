"""Conversation log for future training (SPEC 7.4): every answered turn is appended to a Markdown file, one per day.

Stored: the question as the model saw it (personal data already redacted by precheck), the answer, each tool call's
name and arguments (results are deterministic and can be recomputed), sources, flags, the review outcome, the model and
the latency. Not stored: IP addresses or the browser's thread id (a short hash links turns of one conversation).
Question and answer sit in fenced blocks so answers that contain Markdown headings still parse back unambiguously.
"""
import hashlib
import json
import os
import re
import threading
import time
from pathlib import Path

from .config import PROJECT_ROOT

_lock = threading.Lock()
DEFAULT_MAX_BYTES = 2 * 1024 ** 3   # 2 GiB across all log files


def log_dir() -> Path:
    return Path(os.getenv("FINBOT_CHATLOG_DIR", str(PROJECT_ROOT / "results" / "chatlog")))


def enabled() -> bool:
    return os.getenv("FINBOT_CHATLOG", "1") != "0"


def max_bytes() -> int:
    return int(os.getenv("FINBOT_CHATLOG_MAX_BYTES", str(DEFAULT_MAX_BYTES)))


def conversation_id(thread_id: str) -> str:
    return hashlib.sha256(f"finbot-chatlog:{thread_id}".encode()).hexdigest()[:10]


def _fenced(text: str, lang: str = "markdown") -> str:
    longest = max((len(m) for m in re.findall(r"`+", text or "")), default=0)
    fence = "`" * max(3, longest + 1)
    return f"{fence}{lang}\n{(text or '').rstrip()}\n{fence}"


def entry(thread_id: str, turn: int, trace: dict | None, *, question: str = "", model: str = "", error: str = "") -> str:
    trace = trace or {}
    review = trace.get("review") or {}
    meta = {"conversation": conversation_id(thread_id), "turn": turn, "model": trace.get("served_by") or model,
            "latency_s": trace.get("latency_s"), "review_passed": review.get("passed"), "review_codes": review.get("codes", []),
            "revisions": trace.get("revisions", 0), "flags": trace.get("flags", {}), "rag_status": trace.get("rag_status", ""),
            "error": error or None}
    head = (f"## {time.strftime('%Y-%m-%d %H:%M:%S')} · {meta['conversation']} · turn {turn} · {meta['model'] or 'unknown model'}"
            + (f" · {meta['latency_s']} s" if meta["latency_s"] is not None else "")
            + (" · review passed" if meta["review_passed"] else " · review flagged" if meta["review_passed"] is False else "")
            + (" · error" if error else ""))
    parts = [head, f"<!-- meta {json.dumps(meta, ensure_ascii=False, default=str)} -->",
             "### Question", _fenced(trace.get("question_sent_to_model") or question, "text")]
    parts += ["### Answer", _fenced(trace.get("answer", "") if not error else "", "markdown")]
    tools = trace.get("tools") or []
    if tools:
        parts.append("### Tools")
        parts += [f"- `{t.get('name')}` {json.dumps(t.get('args', {}), ensure_ascii=False, default=str)}" for t in tools]
    sources = trace.get("sources") or []
    if sources:
        parts.append("### Sources")
        parts += [f"- S{s.get('n')} {s.get('title', '')}" + (f", p.{s.get('page')}" if s.get("page") else "") for s in sources]
    return "\n\n".join(parts) + "\n\n---\n\n"


def append(thread_id: str, turn: int, trace: dict | None, **kw) -> bool:
    """Append one turn. Returns False when logging is off or the cap is reached; never raises."""
    if not enabled():
        return False
    try:
        text = entry(thread_id, turn, trace, **kw)
        d = log_dir()
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{time.strftime('%Y-%m-%d')}.md"
        with _lock:
            used = sum(p.stat().st_size for p in d.glob("*.md"))
            if used + len(text.encode("utf-8")) > max_bytes():
                return False
            new = not path.exists()
            with path.open("a", encoding="utf-8") as f:
                if new:
                    f.write(f"# FinBot conversations {path.stem}\n\nRedacted questions and answers, for evaluation and "
                            "future fine-tuning. Never commit this folder.\n\n")
                f.write(text)
        return True
    except Exception:   # logging must never break an answer
        return False
