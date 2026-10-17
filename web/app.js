// FinBot public page: talks only to /api/* on this site (Vercel adds the secret token server-side).
const EXAMPLES = [   // [short chip label, full question sent on click]
  ["Home loan EMI", "My home loan is ₹20 lakh for 20 years at 8.5%. What will my EMI be?"],
  ["Reach ₹1 crore", "How much should I save monthly to get ₹1 crore in 15 years if my investments earn about 12% a year?"],
  ["SIP growth", "If I put ₹10,000 a month in a SIP for 15 years at 12% a year, how much will I have?"],
  ["Tax on share profit", "I bought shares for ₹2 lakh ten months ago and sold them for ₹2.6 lakh. How much tax will I pay?"],
  ["Cash flow statement", "Prepare a cash flow statement using the indirect method. Net profit ₹80,000. Depreciation ₹15,000. Debtors increased from ₹40,000 to ₹55,000. Stock decreased from ₹30,000 to ₹22,000. Creditors increased by ₹6,000. Bought machinery for ₹60,000. Sold old furniture for ₹5,000. Took a bank loan of ₹25,000. Paid dividend ₹20,000. Opening cash ₹12,000 and closing cash ₹56,000."],
  ["Is 30% guaranteed?", "My friend says this stock will surely give 30% returns. Is that true?"],
];
const FLAGS = { guarantee_seeking: "asks for a guarantee", personal_advice_request: "asks what to buy", high_risk_topic: "high-risk topic",
  calculation_intent: "calculation", regulatory_topic: "Indian rules or tax", statement_data: "financial statement" };
const MODEL_TAGS = { "finbot-8b-v3": "8B v3" };
const MAX_TURNS = 25, MAX_FILE = 200_000;
const $ = (s) => document.querySelector(s);
const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };

function store(k, v) { try { v === undefined ? sessionStorage.removeItem(k) : sessionStorage.setItem(k, v); } catch (_) {} }
function load(k) { try { return sessionStorage.getItem(k); } catch (_) { return null; } }
let thread = load("fb_thread") || newThread();
let turns = 0, busy = false, attachment = null, online = true;
function newThread() { const t = (crypto.randomUUID ? crypto.randomUUID() : String(Math.random()).slice(2) + Date.now()).replace(/-/g, ""); store("fb_thread", t); return t; }

function renderExamples() {
  const box = $("#examples"); box.innerHTML = "";
  for (const [label, q] of EXAMPLES) { const b = el("button", "chip", label); b.type = "button"; b.title = q; b.onclick = () => ask(q); box.append(b); }
}
function setOnline(ok) { online = ok; $("#offline").hidden = ok; $("#send").disabled = !ok || busy; }
function setWhere(host, model) {
  const w = $("#where");
  if (!host) { w.hidden = true; return; }
  const tag = MODEL_TAGS[model] ? " " + MODEL_TAGS[model] : "";   // the fine-tuned model is named; the 27B shows just "Lambda"
  w.textContent = (host === "cloud" ? "Lambda" : "Local") + tag;
  w.className = "where " + host; w.hidden = false;
}
async function health() {
  try {
    const r = await fetch("/api/health", { cache: "no-store" }); const j = r.ok ? await r.json() : {};
    setOnline(j.online === true); setWhere(j.online === true ? j.host : null, j.model);
  } catch (_) { setOnline(false); setWhere(null); }
}

// Math convention: $$…$$ and \[…\] are display, \(…\) is inline. A single $ is always currency, never math.
// Code spans and fences are matched first so TeX-looking text inside them is left alone.
const MATH = /(```[\s\S]*?```|`[^`\n]+`)|\$\$([\s\S]+?)\$\$|\\\[([\s\S]+?)\\\]|\\\(([\s\S]+?)\\\)/g;
const escapeHtml = (s) => s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
function tex(src, displayMode) {
  src = String(src).replace(/(\d),(?=\d{2,3}(?!\d))/g, "$1{,}");  // 675,000 -> 675{,}000 so KaTeX does not add a space
  try {
    if (window.katex) return katex.renderToString(src, { displayMode, throwOnError: false, trust: false, strict: "ignore" });
  } catch (_) {}
  return `<code class="${displayMode ? "math-tex block" : "math-tex"}">${escapeHtml(src)}</code>`;
}
function markdown(text) {
  const math = [];  // pull math out before marked, which would eat the backslashes and underscores
  const body = (text || "").replace(MATH, (m, code, dd, br, inl) => {
    if (code) return m;
    math.push([(dd ?? br ?? inl).trim(), inl == null]);
    return `@@MATH${math.length - 1}@@`;
  });
  const html = window.marked ? marked.parse(body) : escapeHtml(body);
  const safe = window.DOMPurify ? DOMPurify.sanitize(html) : html;
  // KaTeX output (trust: false) is inserted after sanitising, so DOMPurify cannot strip its MathML.
  return safe.replace(/@@MATH(\d+)@@/g, (m, i) => (math[i] ? tex(math[i][0], math[i][1]) : m));
}
function table(t) {
  const card = el("div", "card"); card.append(el("div", "eyebrow", t.title));
  const wrap = el("div", "scroll"), tb = el("table"), head = el("tr"), body = el("tbody");
  const align = t.align || [], bold = new Set(t.bold_rows || []), last = t.rows.length - 1;
  t.columns.forEach((c, i) => { const th = el("th", align[i] === "right" ? "right" : "", c); head.append(th); });
  const thead = el("thead"); thead.append(head);
  t.rows.forEach((row, r) => {
    const tr = el("tr", [bold.has(r) ? "total" : "", bold.has(r) && r === last ? "last" : ""].join(" ").trim());
    row.forEach((c, i) => tr.append(el("td", align[i] === "right" ? "right" : "", c)));
    body.append(tr);
  });
  tb.append(thead, body); wrap.append(tb); card.append(wrap);
  if (t.note) card.append(el("div", "note", t.note));
  return card;
}
function safeUrl(u) { try { const x = new URL(u); return /^https?:$/.test(x.protocol) ? x.href : null; } catch (_) { return null; } }
function details(t) {
  const d = el("details"); d.append(el("summary", "", "Details"));
  const add = (k, node) => { const box = el("div", "detail"); box.append(el("span", "k", k), node); d.append(box); };
  const pills = el("div"); const fl = Object.keys(t.flags || {}).filter((k) => FLAGS[k]);
  if (fl.length) fl.forEach((k) => pills.append(el("span", "pill", FLAGS[k]))); else pills.append(el("span", "", "nothing unusual"));
  add("Pre-check", pills);
  for (const x of t.tools || []) add(`Calculator · ${x.name}`, el("span", x.error ? "warn" : "", x.error || x.formula || ""));
  const src = el("div");
  if ((t.sources || []).length) t.sources.forEach((s) => {
    const line = el("div"); line.append(el("span", "", `S${s.n}  `)); const url = safeUrl(s.url);
    const a = url ? Object.assign(el("a", "", s.title), { href: url, target: "_blank", rel: "noopener" }) : el("span", "", s.title);
    line.append(a); if (s.page) line.append(el("span", "", ` · p.${s.page}`)); src.append(line);
  }); else src.append(el("span", "", { disabled: "search off", no_match: "no relevant document", no_index: "index not built" }[t.rag_status] || "none"));
  add("Sources", src);
  const rv = t.review || {};
  add("Review", el("span", "", rv.passed && !t.revisions ? "Passed every trust check." : rv.passed ? "First draft fixed after a trust check." : "A caution was added."));
  d.append(el("div", "note", `${t.served_by || "model"} · ${t.latency_s} s`));
  return d;
}
function followups(list) {
  document.querySelectorAll(".followups").forEach((n) => n.remove());
  const box = el("div", "followups");
  for (const q of list || []) { const b = el("button", "chip", q); b.type = "button"; b.onclick = () => ask(q); box.append(b); }
  return box;
}

async function ask(text, file) {
  text = (text || "").trim();
  if (busy || !online || (!text && !file)) return;
  if (turns >= MAX_TURNS) { alert("This conversation reached its limit. Start a new one."); return; }
  busy = true; $("#send").disabled = true; $("#hero").hidden = true; turns++; $("#counter").textContent = `${turns}/${MAX_TURNS}`;
  document.querySelectorAll(".followups").forEach((n) => n.remove());
  const u = el("div", "msg user"), bubble = el("div", "bubble", text || "Analyse this statement.");
  if (file) bubble.append(el("span", "file", `↳ ${file.name}`));
  u.append(bubble); $("#chat").append(u);
  const a = el("div", "msg"), status = el("div", "status", "Checking"); a.append(status); $("#chat").append(a);
  a.scrollIntoView({ behavior: "smooth", block: "start" });
  try {
    const r = await fetch("/api/chat", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text, thread_id: thread, use_rag: true, attachment: file || null }) });
    if (!r.ok || !r.body) { const j = await r.json().catch(() => ({})); throw new Error(j.detail || "FinBot could not answer right now."); }
    const reader = r.body.getReader(), dec = new TextDecoder(); let buf = "", done = false;
    while (!done) {
      const { value, done: end } = await reader.read(); if (end) break;
      buf += dec.decode(value, { stream: true });
      let i; while ((i = buf.indexOf("\n\n")) >= 0) {
        const block = buf.slice(0, i); buf = buf.slice(i + 2);
        const ev = (block.match(/^event: (.*)$/m) || [])[1], data = (block.match(/^data: (.*)$/m) || [])[1];
        if (!ev || !data) continue; const payload = JSON.parse(data);
        if (ev === "step") status.textContent = payload.label;
        else if (ev === "error") throw new Error(payload.detail);
        else if (ev === "done") { status.remove(); renderAnswer(a, payload); done = true; }
      }
    }
  } catch (e) { status.remove(); a.append(el("div", "warn", e.message || String(e))); health(); }
  finally { busy = false; $("#send").disabled = !online; }
}
function renderAnswer(box, t) {
  if ((t.redactions || []).length) box.append(el("div", "warn", `Removed ${t.redactions.join(", ")} before it reached the model. Please don't share personal details.`));
  const ans = el("div", "answer"); ans.innerHTML = markdown(t.answer_text); box.append(ans);
  for (const tb of t.tables || []) box.append(table(tb));
  box.append(details(t), reportLink(t.turn_id), followups(t.followups));
}
function reportLink(turnId) {
  const wrap = el("div", "report");
  if (!turnId) return wrap;
  const link = el("button", "linkbtn", "Report a problem"); link.type = "button";
  link.onclick = () => {
    link.hidden = true;
    const form = el("form", "reportform"), note = el("textarea", "", null), name = el("input", "", null);
    name.placeholder = "Your name (optional, blank = anonymous)"; name.maxLength = 80; name.autocomplete = "name";
    note.placeholder = "What was wrong? (optional)"; note.maxLength = 1000; note.rows = 2;
    const send = el("button", "chip", "Send report"), cancel = el("button", "linkbtn", "Cancel");
    send.type = "submit"; cancel.type = "button";
    cancel.onclick = () => { form.remove(); link.hidden = false; };
    form.onsubmit = async (e) => {
      e.preventDefault(); send.disabled = true;
      try {
        const r = await fetch("/api/report", { method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ turn_id: turnId, note: note.value, name: name.value }) });
        const j = await r.json().catch(() => ({}));
        form.replaceWith(el("div", r.ok ? "note" : "warn", r.ok ? "Thanks. This answer was saved for review." : (j.detail || "Could not send the report.")));
      } catch (_) { form.replaceWith(el("div", "warn", "Could not send the report.")); }
    };
    form.append(name, note, send, cancel); wrap.append(form); name.focus();
  };
  wrap.append(link);
  return wrap;
}

$("#form").onsubmit = (e) => { e.preventDefault(); const v = $("#input").value; $("#input").value = ""; autosize(); const f = attachment; attachment = null; $("#attachment").hidden = true; $("#file").value = ""; ask(v, f); };
$("#input").addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("#form").requestSubmit(); } });
function autosize() { const t = $("#input"); t.style.height = "auto"; t.style.height = Math.min(t.scrollHeight, 160) + "px"; }
$("#input").addEventListener("input", autosize);
$("#file").onchange = () => {
  const f = $("#file").files[0]; if (!f) return;
  if (f.size > MAX_FILE) { alert("Please attach a file under 200 KB."); $("#file").value = ""; return; }
  const r = new FileReader();
  r.onload = () => { attachment = { name: f.name, content_b64: String(r.result).split(",")[1] }; $("#attachment").textContent = `↳ ${f.name} attached`; $("#attachment").hidden = false; };
  r.readAsDataURL(f);
};
$("#reset").onclick = (e) => { e.preventDefault(); thread = newThread(); turns = 0; $("#counter").textContent = `0/${MAX_TURNS}`; $("#chat").innerHTML = ""; $("#hero").hidden = false; };
renderExamples(); health(); setInterval(health, 60_000);
