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


def max_bytes() -> int:
    return int(os.getenv("FINBOT_REPORTS_MAX_BYTES", str(DEFAULT_MAX_BYTES)))


def reports_path() -> Path:
    return Path(os.getenv("FINBOT_REPORTS_DIR", str(PROJECT_ROOT / "results" / "reports"))) / "reports.jsonl"


def turn_record(trace: dict) -> dict:
    """Everything needed to diagnose one answer. The question is the redacted text the model saw."""
    return {"question": trace.get("question_sent_to_model", ""), "answer": trace.get("answer", ""),
            "tools": trace.get("tools", []), "sources": [{k: s.get(k) for k in ("n", "title", "page", "url", "dense")} for s in trace.get("sources", [])],
            "flags": trace.get("flags", {}), "review": trace.get("review", {}), "revisions": trace.get("revisions", 0),
            "rag_status": trace.get("rag_status", ""), "served_by": trace.get("served_by", ""), "latency_s": trace.get("latency_s")}


def save_report(conversation: list[dict], note: str = "", channel: str = "web") -> str:
    rid = uuid.uuid4().hex[:12]
    rec = {"report_id": rid, "created_at": time.strftime("%Y-%m-%d %H:%M:%S"), "channel": channel,
           "note": (note or "").strip()[:MAX_NOTE], "turns": len(conversation), "conversation": conversation}
    line = json.dumps(rec, ensure_ascii=False) + "\n"
    path = reports_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with _lock:
        used = path.stat().st_size if path.exists() else 0
        if used + len(line.encode("utf-8")) > max_bytes():
            raise ReportsFull("Report storage is full, so reports are paused.")
        with path.open("a", encoding="utf-8") as f:
            f.write(line)
    return rid
