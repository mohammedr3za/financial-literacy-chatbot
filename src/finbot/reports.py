"""'Report a problem' storage (SPEC 7.3). Saves the server's own copy of a conversation for fixing mistakes."""
import json
import os
import threading
import time
import uuid
from pathlib import Path

from .config import PROJECT_ROOT

_lock = threading.Lock()
MAX_NOTE = 1000
DEFAULT_MAX_BYTES = 5 * 1024 ** 3   # 5 GiB (SPEC 7.3)


class ReportsFull(Exception):
    """Report storage reached its cap; no more reports are accepted."""


DEFAULT_REPORT_MAX = 2 * 1024 ** 2  # 2 MiB per report (SPEC 7.3)


def max_bytes() -> int:
    return int(os.getenv("FINBOT_REPORTS_MAX_BYTES", str(DEFAULT_MAX_BYTES)))


def report_max_bytes() -> int:
    return int(os.getenv("FINBOT_REPORT_MAX_BYTES", str(DEFAULT_REPORT_MAX)))


def _fit(rec: dict, limit: int) -> str:
    """Serialise a report under `limit` bytes: drop oldest turns, then tool outputs, then shorten answers.
    The last turn (the reported answer) is always kept."""
    def dump():
        return json.dumps(rec, ensure_ascii=False) + "\n"
    line = dump()
    if len(line.encode("utf-8")) <= limit:
        return line
    rec["trimmed"] = True
    while len(rec["conversation"]) > 1 and len(dump().encode("utf-8")) > limit:
        rec["conversation"].pop(0)
    for turn in rec["conversation"]:
        if len(dump().encode("utf-8")) <= limit:
            break
        turn["tools"] = [{"name": t.get("name"), "args": t.get("args")} for t in turn.get("tools", [])]
    for turn in rec["conversation"]:
        while len(dump().encode("utf-8")) > limit and len(turn.get("answer", "")) > 200:
            turn["answer"] = turn["answer"][: len(turn["answer"]) // 2] + " …[shortened]"
    return dump()


def reports_path() -> Path:
    return Path(os.getenv("FINBOT_REPORTS_DIR", str(PROJECT_ROOT / "results" / "reports"))) / "reports.jsonl"


def turn_record(trace: dict) -> dict:
    """Everything needed to diagnose one answer. The question is the redacted text the model saw."""
    return {"question": trace.get("question_sent_to_model", ""), "answer": trace.get("answer", ""),
            "tools": trace.get("tools", []), "sources": [{k: s.get(k) for k in ("n", "title", "page", "url", "dense")} for s in trace.get("sources", [])],
            "flags": trace.get("flags", {}), "review": trace.get("review", {}), "revisions": trace.get("revisions", 0),
            "rag_status": trace.get("rag_status", ""), "served_by": trace.get("served_by", ""), "latency_s": trace.get("latency_s")}


MAX_NAME = 80


def save_report(conversation: list[dict], note: str = "", channel: str = "web", name: str = "") -> str:
    rid = uuid.uuid4().hex[:12]
    rec = {"report_id": rid, "created_at": time.strftime("%Y-%m-%d %H:%M:%S"), "channel": channel,
           "reporter": " ".join((name or "").split())[:MAX_NAME] or "anonymous",
           "note": (note or "").strip()[:MAX_NOTE], "turns": len(conversation), "conversation": conversation}
    line = _fit(rec, report_max_bytes())
    path = reports_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with _lock:
        used = path.stat().st_size if path.exists() else 0
        if used + len(line.encode("utf-8")) > max_bytes():
            raise ReportsFull("Report storage is full, so reports are paused.")
        with path.open("a", encoding="utf-8") as f:
            f.write(line)
    return rid
