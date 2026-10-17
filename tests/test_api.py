import base64
import json
import uuid

import pytest
from fastapi.testclient import TestClient
from helpers import ScriptedModel, make_kb, tool_call

from finbot.api import create_app
from finbot.graph import FinBot

TOKEN = "test-token-123"


def client(*responses):
    model = ScriptedModel(responses=list(responses))
    return TestClient(create_app(bot_factory=lambda: FinBot(model=model, kb=make_kb()), token=TOKEN))


def post(c, msg, token=TOKEN, ip="1.1.1.1", **extra):
    return c.post("/api/chat", json={"message": msg, "thread_id": uuid.uuid4().hex, **extra},
                  headers={"Authorization": f"Bearer {token}", "X-Client-IP": ip})


def done_event(text):
    for block in text.split("\n\n"):
        if block.startswith("event: done"):
            return json.loads(block.split("data: ", 1)[1])
    raise AssertionError(text)


def test_requires_token():
    assert post(client(), "hi", token="wrong").status_code == 401


def test_streams_answer_with_table_and_followups():
    c = client(tool_call("loan_emi", {"principal": 2000000, "annual_rate_pct": 8.5, "months": 240}, "c1"),
               "Your EMI is ₹17,356.46; the schedule is below.")
    r = post(c, "What EMI will I pay on a ₹20 lakh loan at 8.5% for 20 years?")
    assert r.status_code == 200 and "event: step" in r.text
    d = done_event(r.text)
    assert d["tables"][0]["rows"][-1][0] == "Total" and d["followups"]
    assert "17,356.46" in d["answer_text"] and "excerpt" not in json.dumps(d["sources"])


def test_rate_limit_per_visitor():
    c = client(*["ok"] * 12)
    codes = [post(c, "What is a mutual fund?", ip="9.9.9.9").status_code for _ in range(11)]
    assert codes[:10] == [200] * 10 and codes[10] == 429
    assert post(c, "What is a mutual fund?", ip="8.8.8.8").status_code == 200  # other visitors unaffected


def test_size_limits():
    c = client()
    assert post(c, "x" * 4001).status_code == 422
    big = base64.b64encode(b"a," * 150_000).decode()
    assert post(c, "analyse", attachment={"name": "big.csv", "content_b64": big}).status_code == 413


def test_attachment_becomes_table_in_question():
    c = client(tool_call("financial_ratios", {"current_assets": 800, "current_liabilities": 400, "inventory": 300}, "c1"),
               "The current ratio is 2.00, so short-term liquidity looks comfortable.")
    csv = base64.b64encode(b"Item,FY26\nCurrent assets,800\nCurrent liabilities,400\nInventory,300\n").decode()
    r = post(c, "Ratios?", attachment={"name": "bs.csv", "content_b64": csv})
    d = done_event(r.text)
    assert r.status_code == 200 and d["flags"].get("statement_data") and d["tables"][0]["title"] == "Financial ratios"


def test_report_saves_server_side_conversation(tmp_path, monkeypatch):
    monkeypatch.setenv("FINBOT_REPORTS_DIR", str(tmp_path))
    c = client(tool_call("loan_emi", {"principal": 2000000, "annual_rate_pct": 8.5, "months": 240}, "c1"),
               "Your EMI is ₹17,356.46.")
    d = done_event(post(c, "My PAN is ABCDE1234F. EMI on ₹20 lakh at 8.5% for 20 years?").text)
    r = c.post("/api/report", json={"turn_id": d["turn_id"], "note": "the table is too long"},
               headers={"Authorization": f"Bearer {TOKEN}", "X-Client-IP": "2.2.2.2"})
    assert r.status_code == 200 and r.json()["ok"]
    rec = json.loads((tmp_path / "reports.jsonl").read_text().splitlines()[0])
    assert rec["note"] == "the table is too long" and rec["channel"] == "web" and rec["turns"] == 1
    turn = rec["conversation"][0]
    assert "ABCDE1234F" not in json.dumps(rec) and "17,356.46" in turn["answer"] and turn["tools"][0]["name"] == "loan_emi"


def test_report_rejects_unknown_turn_and_missing_token(tmp_path, monkeypatch):
    monkeypatch.setenv("FINBOT_REPORTS_DIR", str(tmp_path))
    c = client()
    h = {"Authorization": f"Bearer {TOKEN}"}
    assert c.post("/api/report", json={"turn_id": "nosuchthread:0"}, headers=h).status_code == 404
    assert c.post("/api/report", json={"turn_id": "nosuchthread:0"}).status_code == 401
    assert c.post("/api/report", json={"turn_id": "x" * 10, "note": "n" * 1001}, headers=h).status_code == 422


def test_reports_stop_at_storage_cap(tmp_path, monkeypatch):
    monkeypatch.setenv("FINBOT_REPORTS_DIR", str(tmp_path))
    monkeypatch.setenv("FINBOT_REPORTS_MAX_BYTES", "3000")   # tiny cap for the test
    c = client(*["A mutual fund pools money from many investors."] * 3)
    h = {"Authorization": f"Bearer {TOKEN}", "X-Client-IP": "3.3.3.3"}
    d = done_event(post(c, "What is a mutual fund?").text)
    assert c.post("/api/report", json={"turn_id": d["turn_id"], "note": "first"}, headers=h).status_code == 200
    size = (tmp_path / "reports.jsonl").stat().st_size
    monkeypatch.setenv("FINBOT_REPORTS_MAX_BYTES", str(size + 10))  # next report would exceed the cap
    r = c.post("/api/report", json={"turn_id": d["turn_id"], "note": "second"}, headers=h)
    assert r.status_code == 507 and "full" in r.json()["detail"]
    assert (tmp_path / "reports.jsonl").stat().st_size == size        # nothing written, nothing deleted


def test_report_name_optional(tmp_path, monkeypatch):
    monkeypatch.setenv("FINBOT_REPORTS_DIR", str(tmp_path))
    c = client(*["A mutual fund pools money from many investors."] * 2)
    h = {"Authorization": f"Bearer {TOKEN}", "X-Client-IP": "4.4.4.4"}
    d = done_event(post(c, "What is a mutual fund?").text)
    c.post("/api/report", json={"turn_id": d["turn_id"], "note": "x", "name": "  Riya   S  "}, headers=h)
    c.post("/api/report", json={"turn_id": d["turn_id"], "note": "y"}, headers=h)
    recs = [json.loads(line) for line in (tmp_path / "reports.jsonl").read_text().splitlines()]
    assert [r["reporter"] for r in recs] == ["Riya S", "anonymous"]
    assert c.post("/api/report", json={"turn_id": d["turn_id"], "name": "n" * 81}, headers=h).status_code == 422


def test_single_report_is_trimmed_to_its_cap_and_keeps_the_reported_answer(tmp_path, monkeypatch):
    from finbot.reports import save_report
    monkeypatch.setenv("FINBOT_REPORTS_DIR", str(tmp_path))
    monkeypatch.setenv("FINBOT_REPORT_MAX_BYTES", "4000")
    big = {"question": "q", "answer": "a" * 1500, "tools": [{"name": "loan_emi", "args": {}, "output": {"table": "x" * 3000}}]}
    convo = [dict(big, question=f"turn {i}") for i in range(5)]
    convo[-1]["answer"] = "THE REPORTED ANSWER " + "b" * 1200
    save_report(convo, "too big", channel="web")
    line = (tmp_path / "reports.jsonl").read_text()
    rec = json.loads(line)
    assert len(line.encode()) <= 4000 and rec["trimmed"] is True
    assert rec["conversation"][-1]["question"] == "turn 4" and rec["conversation"][-1]["answer"].startswith("THE REPORTED ANSWER")


def test_health_says_where_the_model_runs(monkeypatch):
    c = TestClient(create_app(bot_factory=lambda: None, token=TOKEN))
    monkeypatch.setenv("FINBOT_BACKEND", "openai"); monkeypatch.setenv("FINBOT_MODEL", "qwen3.6-27b")
    assert c.get("/api/health").json() == {"status": "ok", "host": "cloud", "model": "qwen3.6-27b"}
    monkeypatch.setenv("FINBOT_BACKEND", "ollama"); monkeypatch.setenv("FINBOT_MODEL", "qwen3:8b")
    assert c.get("/api/health").json() == {"status": "ok", "host": "local", "model": "qwen3:8b"}


def test_each_answer_is_appended_to_the_markdown_log(tmp_path, monkeypatch):
    monkeypatch.setenv("FINBOT_CHATLOG_DIR", str(tmp_path))
    c = client(tool_call("loan_emi", {"principal": 2000000, "annual_rate_pct": 8.5, "months": 240}, "c1"),
               "Your EMI is ₹17,356.46; the schedule is below.")
    assert post(c, "What EMI will I pay on a ₹20 lakh loan at 8.5% for 20 years? My PAN is ABCDE1234F").status_code == 200
    text = next(tmp_path.glob("*.md")).read_text()
    assert "₹17,356.46" in text and "`loan_emi`" in text and "turn 1" in text
    assert "ABCDE1234F" not in text            # the log keeps the redacted question the model saw


def test_health_host_can_be_set_for_ollama_on_the_gpu_box(monkeypatch):
    # finbot-8b-v3 runs on Ollama on the Lambda GPU: the badge must still say Lambda (cloud)
    c = TestClient(create_app(bot_factory=lambda: None, token=TOKEN))
    monkeypatch.setenv("FINBOT_BACKEND", "ollama"); monkeypatch.setenv("FINBOT_MODEL", "finbot-8b-v3"); monkeypatch.setenv("FINBOT_HOST", "cloud")
    assert c.get("/api/health").json() == {"status": "ok", "host": "cloud", "model": "finbot-8b-v3"}
