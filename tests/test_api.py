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
