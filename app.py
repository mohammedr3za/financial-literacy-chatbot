"""FinBot booth UI. Run: ./run_demo.sh  (or: uv run streamlit run app.py)"""
import re
import uuid

import streamlit as st

from finbot.config import get_settings
from finbot.graph import FinBot
from finbot.precheck import redact

st.set_page_config(page_title="FinBot", page_icon="₹", layout="centered")
SETTINGS = get_settings()

FLAG_LABELS = {
    "guarantee_seeking": "Asks for a guarantee of returns",
    "personal_advice_request": "Asks what to buy or sell",
    "high_risk_topic": "High-risk topic",
    "calculation_intent": "Needs a calculation",
    "regulatory_topic": "Indian rules or tax",
}
STEP_LABELS = {
    "precheck": "Checking the question and removing personal data…",
    "retrieve": "Searching the finance books and SEBI/RBI guidelines…",
    "agent": "Thinking…",
    "tools": "Calculating with a finance tool…",
    "review": "Checking the answer against the trust checklist…",
}
EXAMPLES = [
    "What EMI will I pay on a ₹20 lakh loan at 8.5% for 20 years?",
    "Confirm that this stock will give me 30% returns",
    "How are crypto gains taxed in India?",
    "If I invest ₹10,000 a month in a SIP at 12% for 15 years, how much will I have?",
]


@st.cache_resource(show_spinner="Loading the model and knowledge base…")
def get_bot() -> FinBot:
    return FinBot()


def to_markdown(text: str) -> str:
    text = re.sub(r"\\\[(.+?)\\\]", r"$$\1$$", text, flags=re.S)
    text = re.sub(r"\\\((.+?)\\\)", r"$\1$", text, flags=re.S)
    return text


def render_trace(t: dict) -> None:
    with st.expander("How I got this", expanded=False):
        if t["redactions"]:
            st.warning(f"Removed before reaching the model: {', '.join(t['redactions'])}. Please never share personal details with a chatbot.")
        flags = [FLAG_LABELS[k] for k, v in t["flags"].items() if v and k in FLAG_LABELS]
        st.markdown("**Pre-check:** " + (", ".join(flags) if flags else "nothing unusual"))
        converted = [a for a in t["amounts"] if re.search(r"(?i)lakh|lac|crore|cr|k\b", a["text"])]
        if converted:
            st.markdown("**Amounts understood as:** " + ", ".join(f"{a['text']} = {a['value']:,.0f}" for a in converted))
        for x in t["tools"]:
            out = x["output"]
            st.markdown(f"**Calculator `{x['name']}`**")
            if "error" in out:
                st.error(out["error"])
                continue
            st.caption(out.get("formula", ""))
            cols = st.columns(2)
            cols[0].json(out.get("inputs", x["args"]), expanded=False)
            cols[1].json(out.get("formatted") or out.get("result", {}), expanded=False)
        if t["sources"]:
            st.markdown("**Sources searched and passed to the model:**")
            for s in t["sources"]:
                page = f", page {s['page']}" if s["page"] else ""
                link = f" — [open]({s['url']})" if s["url"] else ""
                st.markdown(f"[S{s['n']}] *{s['title']}*{page}{link}")
                st.caption(s["excerpt"] + "…")
        else:
            st.markdown("**Sources:** " + {"disabled": "document search switched off",
                                             "no_match": "no relevant document found",
                                             "no_index": "knowledge base not built"}.get(t["rag_status"], "none"))
        rv = t["review"]
        if rv.get("passed") and t["revisions"] == 0:
            st.success("Answer passed every trust check on the first try.")
        elif rv.get("passed"):
            st.info(f"The first draft failed a check ({', '.join(rv.get('first_draft_codes', []))}) and was rewritten.")
        else:
            st.error(f"Still failing after a rewrite ({', '.join(rv.get('codes', []))}); a caution was added.")
        st.caption(f"Model: {SETTINGS.model} · {t['latency_s']} s")


# ---------------------------------------------------------------- page
st.title("₹ FinBot")
st.info("Educational information, not investment advice. Do not enter personal or account details.", icon="ℹ️")

if "thread" not in st.session_state:
    st.session_state.thread, st.session_state.history = uuid.uuid4().hex, []

with st.sidebar:
    st.markdown(f"**Model:** `{SETTINGS.model}` (runs on this laptop)")
    use_rag = st.toggle("Search books and guidelines", value=True,
                        help="Switch off to see how the same model answers without grounding.")
    used = sum(1 for m in st.session_state.history if m["role"] == "user")
    st.progress(min(used / SETTINGS.session_cap, 1.0), text=f"{used}/{SETTINGS.session_cap} questions")
    if st.button("New conversation", use_container_width=True):
        st.session_state.thread, st.session_state.history = uuid.uuid4().hex, []
        st.rerun()

for m in st.session_state.history:
    with st.chat_message(m["role"]):
        st.markdown(to_markdown(m["content"]))
        if m.get("trace"):
            render_trace(m["trace"])

prompt = None
examples = st.empty()
if not st.session_state.history:
    with examples.container():
        st.caption("Try one of these:")
        cols = st.columns(2)
        for i, ex in enumerate(EXAMPLES):
            if cols[i % 2].button(ex, key=f"ex{i}", use_container_width=True):
                prompt = ex

capped = used >= SETTINGS.session_cap
typed = st.chat_input("Ask a finance question…" if not capped else "Session limit reached. Start a new conversation.", disabled=capped)
prompt = typed or prompt

if prompt and not capped:
    examples.empty()
    shown = redact(prompt)[0]
    st.session_state.history.append({"role": "user", "content": shown})
    with st.chat_message("user"):
        st.markdown(shown)
    with st.chat_message("assistant"):
        trace = None
        with st.status("Working…", expanded=False) as status:
            try:
                for ev in get_bot().ask_iter(prompt, thread_id=st.session_state.thread, use_rag=use_rag):
                    if ev["event"] == "step":
                        label = STEP_LABELS.get(ev["node"], "Working…")
                        if ev["node"] == "agent" and ev["update"].get("messages"):
                            calls = getattr(ev["update"]["messages"][-1], "tool_calls", None)
                            label = f"Calling `{calls[0]['name']}`…" if calls else "Writing the answer…"
                        status.update(label=label)
                    else:
                        trace = ev["trace"]
                status.update(label=f"Done in {trace['latency_s']} s", state="complete")
            except Exception as exc:  # keep the booth demo alive
                status.update(label="Something went wrong", state="error")
                st.error(f"FinBot could not answer: {exc}. Is Ollama running? Try ./run_demo.sh")
        if trace:
            st.markdown(to_markdown(trace["answer"]))
            render_trace(trace)
            st.session_state.history.append({"role": "assistant", "content": trace["answer"], "trace": trace})
            st.rerun()  # redraw so the question counter and example buttons reflect the new turn
