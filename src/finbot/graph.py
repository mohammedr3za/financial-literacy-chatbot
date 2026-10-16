"""LangGraph: START -> precheck -> retrieve -> agent <-> tools -> review -> END (SPEC section 3)."""
import json
import re
import time
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from .config import Settings, get_settings
from .precheck import precheck as run_precheck
from .prompts import SYSTEM_PROMPT, context_block
from .review import FOOTER, review_answer, reviewer_note
from .tables import to_markdown
from .tools.registry import CALC_TOOLS

_THINK = re.compile(r"<think>.*?</think>", re.S)


class State(TypedDict, total=False):
    user_input: str
    messages: Annotated[list[AnyMessage], add_messages]
    question_idx: int
    flags: dict
    redactions: list[str]
    amounts: list[dict]
    sources: list[dict]
    sources_relevant: bool
    best_dense: float
    rag_status: str
    tool_iterations: int
    revisions: int
    review: dict
    tables: list[dict]


def text_of(msg: AnyMessage) -> str:
    content = msg.content
    if isinstance(content, list):
        content = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return _THINK.sub("", content or "").strip()


def _load_kb(settings: Settings):
    index = settings.kb_dir / "index"
    if not (index / "chunks.jsonl").exists():
        return None
    from .kb.search import KnowledgeBase
    from .llm import get_embedder
    return KnowledgeBase.load(index, get_embedder(settings), settings.min_dense_score)


def build_graph(model=None, kb: Any = "auto", settings: Settings | None = None, checkpointer=None):
    s = settings or get_settings()
    if model is None:
        from .llm import get_chat_model
        model = get_chat_model(s)
    model_with_tools = model.bind_tools(CALC_TOOLS)
    if kb == "auto":
        kb = _load_kb(s)

    def precheck(state: State) -> dict:
        r = run_precheck(state.get("user_input", ""))
        return {"messages": [HumanMessage(content=r.text)], "question_idx": len(state.get("messages", [])),
                "flags": r.flags, "redactions": r.redactions, "amounts": r.amounts, "user_input": "",
                "sources": [], "sources_relevant": False, "best_dense": 0.0, "rag_status": "",
                "tool_iterations": 0, "revisions": 0, "review": {}, "tables": []}

    def retrieve(state: State, config) -> dict:
        if not config.get("configurable", {}).get("use_rag", True):
            return {"rag_status": "disabled"}
        if kb is None:
            return {"rag_status": "no_index"}
        msgs, qi = state["messages"], state["question_idx"]
        query = text_of(msgs[qi])
        if len(query.split()) < 6:  # short follow-up: add the previous user question for context
            prev = [m for m in msgs[:qi] if isinstance(m, HumanMessage) and m.name != "reviewer"]
            if prev:
                query = text_of(prev[-1]) + " " + query
        res = kb.search(query, k=s.top_k)
        return {"sources": res["hits"] if res["relevant"] else [], "sources_relevant": res["relevant"],
                "best_dense": res["best_dense"], "rag_status": "ok" if res["relevant"] else "no_match"}

    def agent(state: State) -> dict:
        msgs, qi = state["messages"], state["question_idx"]
        ctx = context_block(text_of(msgs[qi]), state["flags"], state.get("redactions", []),
                            state.get("sources", []), state.get("rag_status", ""))
        model_input = [SystemMessage(SYSTEM_PROMPT), *msgs[:qi], HumanMessage(ctx), *msgs[qi + 1:]]
        capped = state.get("tool_iterations", 0) >= s.max_tool_iterations
        response = (model if capped else model_with_tools).invoke(model_input)
        if capped and getattr(response, "tool_calls", None):  # defensive: never loop past the cap
            response = AIMessage(content=text_of(response) or "I could not finish the calculation.", id=response.id)
        return {"messages": [response],
                "tool_iterations": state.get("tool_iterations", 0) + (1 if getattr(response, "tool_calls", None) else 0)}

    def route_after_agent(state: State) -> str:
        return "tools" if getattr(state["messages"][-1], "tool_calls", None) else "review"

    def review(state: State) -> dict:
        msgs, qi = state["messages"], state["question_idx"]
        turn = msgs[qi + 1:]
        tools_used = [tc["name"] for m in turn if isinstance(m, AIMessage) for tc in (m.tool_calls or [])]
        last_ai = next(m for m in reversed(msgs) if isinstance(m, AIMessage))
        answer = text_of(last_ai)
        srcs = state.get("sources", [])
        evidence = "\n".join([text_of(msgs[qi])] + [x["text"] for x in srcs]
                             + [m.content for m in turn if isinstance(m, ToolMessage)])
        result = review_answer(answer, state["flags"], tools_used, len(srcs), tuple(x.get("url", "") for x in srcs), evidence)
        prior = state.get("review", {}).get("first_draft_codes", [])
        if result.passed:
            return {"review": {"passed": True, "codes": [], "first_draft_codes": prior, "footer": False}}
        if state.get("revisions", 0) < s.max_revisions:
            return {"messages": [HumanMessage(content=reviewer_note(result), name="reviewer")],
                    "revisions": state.get("revisions", 0) + 1,
                    "review": {"passed": False, "codes": result.codes, "first_draft_codes": result.codes}}
        return {"messages": [AIMessage(content=answer + FOOTER, id=last_ai.id)],
                "review": {"passed": False, "codes": result.codes, "first_draft_codes": prior, "footer": True}}

    def route_after_review(state: State) -> str:
        last = state["messages"][-1]
        return "agent" if isinstance(last, HumanMessage) and last.name == "reviewer" else "present"

    def present(state: State) -> dict:
        """Collect the exact tables the tools produced this turn (SPEC 4.2). Kept out of the model's memory."""
        tables = []
        for m in state["messages"][state["question_idx"] + 1:]:
            if isinstance(m, ToolMessage):
                try:
                    out = json.loads(m.content)
                except (TypeError, ValueError):
                    continue
                if isinstance(out, dict) and out.get("table"):
                    tables.append(out["table"])
        return {"tables": tables}

    g = StateGraph(State)
    g.add_node("precheck", precheck)
    g.add_node("retrieve", retrieve)
    g.add_node("agent", agent)
    g.add_node("tools", ToolNode(CALC_TOOLS))
    g.add_node("review", review)
    g.add_node("present", present)
    g.add_edge(START, "precheck")
    g.add_edge("precheck", "retrieve")
    g.add_edge("retrieve", "agent")
    g.add_conditional_edges("agent", route_after_agent, {"tools": "tools", "review": "review"})
    g.add_edge("tools", "agent")
    g.add_conditional_edges("review", route_after_review, {"agent": "agent", "present": "present"})
    g.add_edge("present", END)
    return g.compile(checkpointer=checkpointer if checkpointer is not None else InMemorySaver())


def turn_trace(state: dict) -> dict:
    """Everything the UI's "How I got this" panel and the benchmark log need for the latest turn."""
    msgs, qi = state["messages"], state["question_idx"]
    turn = msgs[qi + 1:]
    results = {m.tool_call_id: m.content for m in turn if isinstance(m, ToolMessage)}
    tools = []
    for m in turn:
        if isinstance(m, AIMessage):
            for tc in m.tool_calls or []:
                raw = results.get(tc["id"], "")
                try:
                    parsed = json.loads(raw)
                except (TypeError, ValueError):
                    parsed = {"raw": raw}
                tools.append({"name": tc["name"], "args": tc["args"], "output": parsed})
    final = next((m for m in reversed(msgs) if isinstance(m, AIMessage)), None)
    return {
        "question_sent_to_model": text_of(msgs[qi]),
        "flags": state.get("flags", {}), "redactions": state.get("redactions", []), "amounts": state.get("amounts", []),
        "tools": tools,
        "sources": [{"n": i + 1, "title": h["title"], "page": h.get("page"), "url": h.get("url", ""), "file": h["file"],
                     "dense": h.get("dense"), "excerpt": h["text"][:300]} for i, h in enumerate(state.get("sources", []))],
        "rag_status": state.get("rag_status", ""), "best_dense": state.get("best_dense", 0.0),
        "review": state.get("review", {}), "revisions": state.get("revisions", 0),
        "answer_text": text_of(final) if final else "",
        "tables": state.get("tables", []),
        "answer": "\n\n".join([text_of(final) if final else ""] + [to_markdown(t) for t in state.get("tables", [])]),
    }


class FinBot:
    """Thin wrapper used by the CLI, the Streamlit app and the benchmark runner."""

    def __init__(self, model=None, kb: Any = "auto", settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.graph = build_graph(model=model, kb=kb, settings=self.settings)

    def ask_iter(self, text: str, thread_id: str = "default", use_rag: bool = True):
        """Yields {"event": "step", "node": ..., "update": ...} per graph step, then {"event": "done", "trace": ...}."""
        cfg = {"configurable": {"thread_id": thread_id, "use_rag": use_rag}, "recursion_limit": 40}
        t0 = time.time()
        for chunk in self.graph.stream({"user_input": text}, cfg, stream_mode="updates"):
            for node, update in chunk.items():
                yield {"event": "step", "node": node, "update": update or {}}
        trace = turn_trace(self.graph.get_state(cfg).values)
        trace["latency_s"] = round(time.time() - t0, 2)
        yield {"event": "done", "trace": trace}

    def ask(self, text: str, thread_id: str = "default", use_rag: bool = True) -> dict:
        for event in self.ask_iter(text, thread_id, use_rag):
            if event["event"] == "done":
                return event["trace"]
        raise RuntimeError("graph finished without a result")
