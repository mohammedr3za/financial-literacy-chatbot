"""SPEC 7.4: the Markdown conversation log for future training."""
import re

from finbot import chatlog

TRACE = {"question_sent_to_model": "EMI on [PAN redacted] loan of ₹20 lakh at 8.5% for 20 years?",
         "answer": "### EMI\nYour EMI is ₹17,356.\n```\ncode with ``` inside\n```",
         "tools": [{"name": "loan_emi", "args": {"principal": 2000000, "annual_rate_pct": 8.5, "months": 240}, "result": {"emi": 17356}}],
         "sources": [{"n": 1, "title": "RBI FAME", "page": 12}], "review": {"passed": True, "codes": []},
         "flags": {"calc_types": ["emi"]}, "served_by": "qwen3.6-27b", "latency_s": 12.3, "revisions": 0}


def test_entry_keeps_what_training_needs_and_drops_identifiers(tmp_path, monkeypatch):
    monkeypatch.setenv("FINBOT_CHATLOG_DIR", str(tmp_path))
    assert chatlog.append("browser-thread-123", 1, TRACE)
    text = next(tmp_path.glob("*.md")).read_text()
    assert "[PAN redacted]" in text and "₹17,356" in text
    assert '`loan_emi` {"principal": 2000000, "annual_rate_pct": 8.5, "months": 240}' in text
    assert "S1 RBI FAME, p.12" in text and "qwen3.6-27b" in text and "review passed" in text
    assert "browser-thread-123" not in text and chatlog.conversation_id("browser-thread-123") in text
    assert '"emi": 17356' not in text                     # tool results are recomputable, not stored
    assert re.search(r"^````markdown$", text, re.M)       # fence longer than the answer's own ``` runs


def test_cap_stops_logging(tmp_path, monkeypatch):
    monkeypatch.setenv("FINBOT_CHATLOG_DIR", str(tmp_path))
    monkeypatch.setenv("FINBOT_CHATLOG_MAX_BYTES", "200")
    assert not chatlog.append("t", 1, TRACE)
    assert not list(tmp_path.glob("*.md"))


def test_can_be_switched_off(tmp_path, monkeypatch):
    monkeypatch.setenv("FINBOT_CHATLOG_DIR", str(tmp_path))
    monkeypatch.setenv("FINBOT_CHATLOG", "0")
    assert not chatlog.append("t", 1, TRACE)


def test_error_turn_is_logged_with_the_question(tmp_path, monkeypatch):
    monkeypatch.setenv("FINBOT_CHATLOG_DIR", str(tmp_path))
    assert chatlog.append("t", 2, None, question="What is NPV?", model="qwen3:8b", error="TimeoutError")
    text = next(tmp_path.glob("*.md")).read_text()
    assert "What is NPV?" in text and "turn 2" in text and "error" in text
