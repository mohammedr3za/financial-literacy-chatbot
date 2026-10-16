"""FinBot booth UI (SPEC 7, 7.1). Run: ./run_demo.sh"""
import html
import re
import uuid

import streamlit as st

from finbot.attachments import table_text
from finbot.config import get_settings
from finbot.followups import suggest
from finbot.graph import FinBot
from finbot.precheck import redact
from finbot.reports import ReportsFull, save_report, turn_record
from finbot.tables import to_html

st.set_page_config(page_title="FinBot", page_icon="✳", layout="centered", initial_sidebar_state="collapsed")
SETTINGS = get_settings()

CSS = """
<style>
:root { --fb-ink:#111111; --fb-muted:#888888; --fb-line:#E4E4E7; --fb-cream:#FAF9F5; --fb-accent:#C85A32; }
header[data-testid="stHeader"] { background: transparent; }
[data-testid="stDecoration"] { display: none; }
.block-container { max-width: 780px; padding-top: 3rem; padding-bottom: 8rem; }
.fb-eyebrow { font-size: 11px; letter-spacing: .2em; text-transform: uppercase; color: var(--fb-muted); font-weight: 500;
              display: flex; align-items: center; gap: .55rem; }
.fb-star { color: var(--fb-accent); font-size: 15px; letter-spacing: 0; }
.fb-hero { font-size: 54px; line-height: 1.02; font-weight: 400; letter-spacing: -0.045em; color: var(--fb-ink);
           margin: 1.6rem 0 .7rem 0; padding: 0; }
.fb-sub { color: var(--fb-muted); font-size: 15px; margin: 0 0 2.2rem 0; }
.st-key-examples button { background: #fff; border: 1px solid var(--fb-line); color: var(--fb-ink); box-shadow: none;
                          padding: .55rem 1.1rem; justify-content: flex-start; text-align: left; min-height: 0; }
.st-key-examples button:hover { border-color: var(--fb-ink); color: var(--fb-ink); background: #fff; }
.st-key-examples button p { font-size: 13px; text-align: left; }
.st-key-followups { margin-top: .4rem; }
[data-testid="stPopover"] button { background: transparent; border: none; box-shadow: none; padding: 0; min-height: 0; }
[data-testid="stPopover"] button p { font-size: 11px; letter-spacing: .14em; text-transform: uppercase; color: var(--fb-muted); }
.st-key-followups button { background: transparent; border: 1px solid var(--fb-line); color: #444; box-shadow: none; min-height: 0;
                           padding: .3rem .85rem; }
.st-key-followups button:hover { border-color: var(--fb-accent); color: var(--fb-accent); background: transparent; }
.st-key-followups button p { font-size: 12.5px; }
.st-key-examples button > div { justify-content: flex-start; width: 100%; }
[data-testid="stChatMessageAvatarUser"], [data-testid="stChatMessageAvatarAssistant"] { display: none; }
[data-testid="stChatMessage"] { background: transparent; padding: .3rem 0; gap: 0; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) { flex-direction: row; justify-content: flex-end; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {
    background: var(--fb-ink); border-radius: 22px; padding: .7rem 1.15rem; max-width: 82%; margin: 0 0 0 auto; flex: 0 1 auto; width: auto; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] * { color: var(--fb-cream); }
[data-testid="stChatMessageContent"] p, [data-testid="stChatMessageContent"] li { font-size: 15px; line-height: 1.65; }
.fb-attach { font-size: 12px; opacity: .7; }
.fb-table { background: #fff; border: 1px solid var(--fb-line); border-radius: 22px; padding: 1.1rem 1.25rem .85rem; margin: 1.1rem 0; }
.fb-table .fb-eyebrow { margin-bottom: .75rem; }
.fb-scroll { overflow: auto; max-height: 440px; }
.fb-table thead th { position: sticky; top: 0; z-index: 1; }
.fb-table tr.last td { position: sticky; bottom: 0; background: #fff; box-shadow: 0 -1px 0 var(--fb-ink); border-top: none; }
.fb-table table { width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; border: none; }
.fb-table th { font-size: 10.5px; text-transform: uppercase; letter-spacing: .12em; color: var(--fb-muted); font-weight: 500;
               padding: .45rem .65rem; border: none; border-bottom: 1px solid var(--fb-line); white-space: nowrap; background: #fff; }
.fb-table td { font-size: 13.5px; padding: .42rem .65rem; border: none; border-bottom: 1px solid #F2F0EA; color: var(--fb-ink); white-space: nowrap; }
.fb-table tr:last-child td { border-bottom: none; }
.fb-table .right { text-align: right; }
.fb-table .left { text-align: left; }
.fb-table tr.total td { font-weight: 600; border-top: 1px solid var(--fb-ink); }
.fb-note { font-size: 12px; color: var(--fb-muted); margin-top: .65rem; }
[data-testid="stMarkdownContainer"] table { border-collapse: collapse; font-variant-numeric: tabular-nums; font-size: 13.5px; }
[data-testid="stMarkdownContainer"] th { font-size: 10.5px; text-transform: uppercase; letter-spacing: .12em; color: var(--fb-muted); font-weight: 500; }
[data-testid="stExpander"] details { border: none; background: transparent; }
[data-testid="stExpander"] summary { padding: .25rem 0; }
[data-testid="stExpander"] summary p { font-size: 11px; letter-spacing: .18em; text-transform: uppercase; color: var(--fb-muted); }
.fb-detail { font-size: 13px; color: #444; line-height: 1.65; margin: .15rem 0 .7rem; }
.fb-detail .k { font-size: 10.5px; letter-spacing: .14em; text-transform: uppercase; color: var(--fb-muted); display: block; margin-bottom: .15rem; }
.fb-detail a { color: var(--fb-ink); text-decoration: none; border-bottom: 1px solid var(--fb-line); }
.fb-detail .muted { color: var(--fb-muted); }
.fb-pill { display: inline-block; border: 1px solid var(--fb-line); border-radius: 999px; padding: .05rem .65rem; margin: 0 .3rem .3rem 0;
           font-size: 11.5px; color: #555; background: #fff; }
.fb-warn { color: var(--fb-accent); font-size: 12.5px; margin: .2rem 0 .4rem; }
[data-testid="stChatInput"] { border-radius: 26px; background: #fff; }
[data-testid="stBottomBlockContainer"] { background: var(--fb-cream); }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

FLAG_LABELS = {"guarantee_seeking": "asks for a guarantee", "personal_advice_request": "asks what to buy",
               "high_risk_topic": "high-risk topic", "calculation_intent": "calculation", "regulatory_topic": "Indian rules or tax",
               "statement_data": "financial statement"}
STEP_LABELS = {"precheck": "Checking", "retrieve": "Searching sources", "tools": "Calculating", "review": "Reviewing", "present": "Preparing tables"}
REVIEW_LABELS = {"guarantee_language": "guarantee language", "no_tool_for_numbers": "a number without a calculator",
                 "no_risk_disclosure": "a missing risk note", "no_citation": "a missing citation", "buy_sell_call": "a buy/sell call",
                 "no_adviser_referral": "no adviser referral", "fake_citation": "a citation to a missing source",
                 "unverified_regulatory": "an unverified rule", "unlisted_link": "an unofficial link",
                 "unsupported_figures": "a figure not in the sources", "empty": "an empty answer"}
EXAMPLES = [  # (chip label, question sent)
    ('My home loan is ₹20 lakh for 20 years at 8.5%. What will my EMI be?', 'My home loan is ₹20 lakh for 20 years at 8.5%. What will my EMI be?'),
    ('How much should I save monthly to get ₹1 crore in 15 years?', 'How much should I save monthly to get ₹1 crore in 15 years if my investments earn about 12% a year?'),
    ('If I put ₹10,000 a month in a SIP for 15 years, how much will I have?', 'If I put ₹10,000 a month in a SIP for 15 years at 12% a year, how much will I have?'),
    ('Can you prepare a cash flow statement from my numbers?', 'Prepare a cash flow statement using the indirect method. Net profit ₹80,000. Depreciation ₹15,000. Debtors increased from ₹40,000 to ₹55,000. Stock decreased from ₹30,000 to ₹22,000. Creditors increased by ₹6,000. Bought machinery for ₹60,000. Sold old furniture for ₹5,000. Took a bank loan of ₹25,000. Paid dividend ₹20,000. Opening cash ₹12,000 and closing cash ₹56,000.'),
    ('My friend says this stock will surely give 30% returns. Is that true?', 'My friend says this stock will surely give 30% returns. Is that true?'),
    ('I made a profit on crypto this year. How much tax will I pay?', 'I made a profit on crypto this year. How much tax will I pay in India?'),
]


@st.cache_resource(show_spinner=False)
def get_bot() -> FinBot:
    return FinBot()


def latex(text: str) -> str:
    """Map the model's math convention onto st.markdown's KaTeX syntax.

    A lone `$` is always currency, so it is escaped (st.markdown would otherwise read `$675,000, costs of $337,000`
    as inline math). `$$…$$` stays display math, `\\[…\\]` becomes `$$…$$`, and `\\(…\\)` becomes inline `$…$`.
    """
    text = re.sub(r"(?<![\\$])\$(?!\$)", r"\\$", text)
    text = re.sub(r"\\\[(.+?)\\\]", lambda m: f"$${m.group(1)}$$", text, flags=re.S)
    return re.sub(r"\\\((.+?)\\\)", lambda m: f"${m.group(1).strip()}$", text, flags=re.S)


def esc(x) -> str:
    return html.escape(str(x))


def render_details(t: dict) -> None:
    with st.expander("Details"):
        parts = []
        flags = [FLAG_LABELS[k] for k, v in t["flags"].items() if v and k in FLAG_LABELS]
        parts.append('<div class="fb-detail"><span class="k">Pre-check</span>'
                     + ("".join(f'<span class="fb-pill">{esc(f)}</span>' for f in flags) or '<span class="muted">nothing unusual</span>')
                     + "</div>")
        conv = [a for a in t["amounts"] if re.search(r"(?i)lakh|lac|crore|cr|k\b", a["text"])]
        if conv:
            parts.append('<div class="fb-detail"><span class="k">Amounts read as</span>'
                         + " · ".join(f"{esc(a['text'])} = {a['value']:,.0f}" for a in conv) + "</div>")
        for x in t["tools"]:
            out = x["output"]
            body = (f'<span class="fb-warn">{esc(out["error"])}</span>' if "error" in out
                    else f'<span class="muted">{esc(out.get("formula", ""))}</span>')
            parts.append(f'<div class="fb-detail"><span class="k">Calculator · {esc(x["name"])}</span>{body}</div>')
        if t["sources"]:
            items = []
            for s in t["sources"]:
                page = f" · p.{s['page']}" if s["page"] else ""
                title = f'<a href="{esc(s["url"])}" target="_blank">{esc(s["title"])}</a>' if s["url"] else esc(s["title"])
                items.append(f'S{s["n"]} &nbsp;{title}<span class="muted">{page}</span>')
            parts.append('<div class="fb-detail"><span class="k">Sources</span>' + "<br>".join(items) + "</div>")
        else:
            msg = {"disabled": "search off", "no_match": "no relevant document", "no_index": "index not built"}.get(t["rag_status"], "none")
            parts.append(f'<div class="fb-detail"><span class="k">Sources</span><span class="muted">{msg}</span></div>')
        rv = t["review"]
        fixed = ", ".join(REVIEW_LABELS.get(c, c) for c in rv.get("first_draft_codes", []))
        verdict = ("Passed every trust check." if rv.get("passed") and not t["revisions"]
                   else f"First draft had {fixed}; rewritten and passed." if rv.get("passed")
                   else "Still failing after a rewrite; a caution was added.")
        parts.append(f'<div class="fb-detail"><span class="k">Review</span>{esc(verdict)}</div>')
        served = t.get("served_by") or SETTINGS.model
        where = "on this laptop" if served == SETTINGS.fallback_model or SETTINGS.backend == "ollama" else "on the GPU server"
        parts.append(f'<div class="fb-detail muted">{esc(served)} · {where} · {t["latency_s"]} s</div>')
        st.markdown("".join(parts), unsafe_allow_html=True)


def render_answer(t: dict) -> None:
    if t["redactions"]:
        st.markdown(f'<div class="fb-warn">Removed {esc(", ".join(t["redactions"]))} before it reached the model. '
                    "Please don't share personal details.</div>", unsafe_allow_html=True)
    st.markdown(latex(t["answer_text"]))
    for table in t["tables"]:
        st.markdown(to_html(table), unsafe_allow_html=True)
    render_details(t)


# ---------------------------------------------------------------- state and sidebar
if "thread" not in st.session_state:
    st.session_state.thread, st.session_state.history = uuid.uuid4().hex, []
used = sum(1 for m in st.session_state.history if m["role"] == "user")

with st.sidebar:
    st.markdown('<div class="fb-eyebrow"><span class="fb-star">✳</span> Settings</div>', unsafe_allow_html=True)
    use_rag = st.toggle("Search books & guidelines", value=True, help="Switch off to compare an ungrounded answer.")
    where = "GPU server, laptop fallback" if SETTINGS.backend == "openai" else "on this laptop"
    st.caption(f"{SETTINGS.model} · {where} · {used}/{SETTINGS.session_cap} questions")
    if st.button("New conversation", use_container_width=True):
        st.session_state.thread, st.session_state.history = uuid.uuid4().hex, []
        st.rerun()

# ---------------------------------------------------------------- page
prompt = None
st.markdown('<div class="fb-eyebrow"><span class="fb-star">✳</span> FinBot &nbsp;·&nbsp; educational, not advice</div>',
            unsafe_allow_html=True)
attach = None
if not st.session_state.history:
    st.markdown('<h1 class="fb-hero">Ask about money.</h1>'
                '<p class="fb-sub">Exact calculators. Official sources. Runs on this laptop.</p>', unsafe_allow_html=True)
    with st.container(key="examples"):
        cols = st.columns(2)
        for i, (label, question) in enumerate(EXAMPLES):
            if cols[i % 2].button(label, key=f"ex{i}", use_container_width=True):
                prompt = question

for idx, m in enumerate(st.session_state.history):
    with st.chat_message(m["role"]):
        if m["role"] == "user":
            st.markdown(m["content"], unsafe_allow_html=True)  # already escaped when stored
        else:
            render_answer(m["trace"])
            with st.popover("Report a problem"):
                name = st.text_input("Your name (optional, leave blank to stay anonymous)", key=f"rname{idx}", max_chars=80)
                note = st.text_area("What was wrong? (optional)", key=f"rnote{idx}", max_chars=1000)
                if st.button("Send report", key=f"rsend{idx}"):
                    convo = [turn_record(h["trace"]) for h in st.session_state.history[: idx + 1] if h.get("trace")]
                    try:
                        st.success(f"Saved for review (report {save_report(convo, note, channel='mac-app', name=name)}).")
                    except ReportsFull as exc:
                        st.warning(str(exc))
            if idx == len(st.session_state.history) - 1:  # follow-up chips under the latest answer only
                chips = suggest(m["trace"])
                with st.container(key="followups"):
                    cols = st.columns(len(chips))
                    for j, chip in enumerate(chips):
                        if cols[j].button(chip, key=f"fu{idx}_{j}", use_container_width=True):
                            prompt = chip

capped = used >= SETTINGS.session_cap
submitted = st.chat_input("Session limit reached. Start a new conversation." if capped else "Ask, or attach a balance sheet…",
                          accept_file=True, file_type=["csv", "xlsx", "xls", "txt"], disabled=capped, key="ask")
if submitted:
    prompt = (submitted.text or "").strip()
    if submitted.files:
        f = submitted.files[0]
        try:
            prompt = (prompt or "Analyse this statement.") + f"\n\nAttached table ({f.name}):\n{table_text(f.name, f.getvalue())}"
            attach = f.name
        except Exception as exc:  # unreadable upload: tell the user, don't crash the booth
            st.toast(f"Could not read {f.name}: {exc}")
            prompt = prompt or None

if prompt and not capped:
    typed = prompt.split("\n\nAttached table (")[0]
    shown = esc(redact(typed)[0]).replace("\n", "<br>") + (f'<br><span class="fb-attach">↳ {esc(attach)}</span>' if attach else "")
    st.session_state.history.append({"role": "user", "content": shown})
    with st.chat_message("user"):
        st.markdown(shown, unsafe_allow_html=True)
    with st.chat_message("assistant"):
        trace = None
        with st.status("Checking", expanded=False) as status:
            try:
                for ev in get_bot().ask_iter(prompt, thread_id=st.session_state.thread, use_rag=use_rag):
                    if ev["event"] == "step":
                        label = STEP_LABELS.get(ev["node"], "Writing")
                        if ev["node"] == "agent" and ev["update"].get("messages"):
                            calls = getattr(ev["update"]["messages"][-1], "tool_calls", None)
                            label = f"Calling {calls[0]['name']}" if calls else "Writing"
                        status.update(label=label)
                    else:
                        trace = ev["trace"]
                status.update(label=f"{trace['latency_s']} s", state="complete")
            except Exception as exc:  # keep the booth demo alive
                status.update(label="Something went wrong", state="error")
                st.markdown(f'<div class="fb-warn">FinBot could not answer ({esc(exc)}). Is Ollama running? Try ./run_demo.sh</div>',
                            unsafe_allow_html=True)
        if trace:
            st.session_state.history.append({"role": "assistant", "content": trace["answer"], "trace": trace})
            st.rerun()
