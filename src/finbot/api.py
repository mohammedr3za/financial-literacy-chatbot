"""Public FinBot API (SPEC 7.2): the same graph behind a token, rate limits and size limits, streamed as Server-Sent Events.

Run on the GPU box:  FINBOT_API_TOKEN=... uv run uvicorn finbot.api:app --host 127.0.0.1 --port 8080
"""
import base64
import json
import os
import secrets
import threading
import time
from collections import defaultdict, deque

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from collections import OrderedDict

from .attachments import table_text
from .followups import suggest
from .reports import MAX_NAME, MAX_NOTE, ReportsFull, save_report, turn_record

MAX_CHARS, MAX_TURNS, MAX_ATTACH, PER_MINUTE = 4000, 25, 200_000, 10
STEP_LABELS = {"precheck": "Checking", "retrieve": "Searching sources", "tools": "Calculating", "review": "Reviewing",
               "present": "Preparing tables"}


class Attachment(BaseModel):
    name: str = Field(max_length=200)
    content_b64: str


class ReportRequest(BaseModel):
    turn_id: str = Field(min_length=8, max_length=100)
    note: str = Field(default="", max_length=MAX_NOTE)
    name: str = Field(default="", max_length=MAX_NAME)


class ChatRequest(BaseModel):
    message: str = Field(default="", max_length=MAX_CHARS)
    thread_id: str = Field(min_length=8, max_length=64)
    use_rag: bool = True
    attachment: Attachment | None = None


def public_trace(t: dict) -> dict:
    """What the browser needs; tool outputs are reduced to name, inputs, formula and error."""
    return {
        "answer_text": t["answer_text"], "tables": t["tables"], "redactions": t["redactions"],
        "flags": {k: v for k, v in t["flags"].items() if v and k != "calc_types"},
        "tools": [{"name": x["name"], "args": x["args"], "formula": x["output"].get("formula"), "error": x["output"].get("error")}
                  for x in t["tools"]],
        "sources": [{k: s[k] for k in ("n", "title", "page", "url")} for s in t["sources"]],
        "rag_status": t["rag_status"], "review": t["review"], "revisions": t["revisions"],
        "served_by": t.get("served_by", ""), "latency_s": t["latency_s"], "followups": suggest(t),
    }


def create_app(bot_factory=None, token: str | None = None) -> FastAPI:
    token = token or os.environ.get("FINBOT_API_TOKEN")
    if not token:
        raise RuntimeError("Set FINBOT_API_TOKEN.")
    app = FastAPI(title="FinBot API", docs_url=None, redoc_url=None, openapi_url=None)
    state = {"bot": None}
    lock = threading.Lock()
    hits: dict[str, deque] = defaultdict(deque)
    turns: dict[str, int] = defaultdict(int)
    history: "OrderedDict[str, list[dict]]" = OrderedDict()   # thread -> server-side turn records, for reports
    MAX_THREADS = 2000

    def remember(thread_id: str, trace: dict) -> str:
        with lock:
            log = history.setdefault(thread_id, [])
            history.move_to_end(thread_id)
            log.append(turn_record(trace))
            while len(history) > MAX_THREADS:
                history.popitem(last=False)
            return f"{thread_id}:{len(log) - 1}"

    def bot():
        with lock:
            if state["bot"] is None:
                if bot_factory is None:
                    from .graph import FinBot
                    state["bot"] = FinBot()
                else:
                    state["bot"] = bot_factory()
            return state["bot"]

    def check_auth(authorization: str | None):
        if not authorization or not secrets.compare_digest(authorization, f"Bearer {token}"):
            raise HTTPException(status_code=401, detail="Not authorised.")

    def check_rate(client: str):
        now, q = time.time(), hits[client]
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= PER_MINUTE:
            raise HTTPException(status_code=429, detail="Too many questions in a minute. Please wait a moment.")
        q.append(now)

    @app.get("/api/health")
    def health():
        return {"status": "ok"}

    @app.post("/api/chat")
    def chat(req: ChatRequest, request: Request, authorization: str | None = Header(default=None),
             x_client_ip: str | None = Header(default=None)):
        check_auth(authorization)
        check_rate(x_client_ip or (request.client.host if request.client else "unknown"))
        if turns[req.thread_id] >= MAX_TURNS:
            raise HTTPException(status_code=429, detail="This conversation reached its limit. Start a new one.")
        question = req.message.strip()
        if req.attachment:
            raw = base64.b64decode(req.attachment.content_b64, validate=True)
            if len(raw) > MAX_ATTACH:
                raise HTTPException(status_code=413, detail="Attachment too large (200 KB maximum).")
            try:
                question = (question or "Analyse this statement.") + \
                    f"\n\nAttached table ({req.attachment.name}):\n{table_text(req.attachment.name, raw)}"
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from None
        if not question:
            raise HTTPException(status_code=422, detail="Ask a question.")
        turns[req.thread_id] += 1

        def events():
            try:
                for ev in bot().ask_iter(question, thread_id=req.thread_id, use_rag=req.use_rag):
                    if ev["event"] == "step":
                        label = STEP_LABELS.get(ev["node"], "Writing")
                        if ev["node"] == "agent" and ev["update"].get("messages"):
                            calls = getattr(ev["update"]["messages"][-1], "tool_calls", None)
                            label = f"Calling {calls[0]['name']}" if calls else "Writing"
                        yield f"event: step\ndata: {json.dumps({'label': label})}\n\n"
                    else:
                        payload = public_trace(ev["trace"])
                        payload["turn_id"] = remember(req.thread_id, ev["trace"])
                        yield f"event: done\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
            except Exception:  # never leak internals to the public page
                yield f"event: error\ndata: {json.dumps({'detail': 'FinBot could not answer right now. Please try again.'})}\n\n"

        return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.post("/api/report")
    def report(req: ReportRequest, request: Request, authorization: str | None = Header(default=None),
               x_client_ip: str | None = Header(default=None)):
        check_auth(authorization)
        check_rate(x_client_ip or (request.client.host if request.client else "unknown"))
        thread_id, _, idx = req.turn_id.rpartition(":")
        with lock:
            log = history.get(thread_id)
            if log is None or not idx.isdigit() or int(idx) >= len(log):
                raise HTTPException(status_code=404, detail="That answer can no longer be reported.")
            conversation = log[: int(idx) + 1]
        try:
            return {"ok": True, "report_id": save_report(conversation, req.note, channel="web", name=req.name)}
        except ReportsFull as exc:
            raise HTTPException(status_code=507, detail=str(exc)) from None

    @app.exception_handler(HTTPException)
    def http_error(_, exc: HTTPException):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    return app


app = create_app() if os.environ.get("FINBOT_API_TOKEN") else None
