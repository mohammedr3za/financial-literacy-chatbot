"""Render tool tables as markdown (CLI, benchmark) or HTML (Streamlit)."""
import html


def to_markdown(t: dict) -> str:
    align = t.get("align") or ["left"] * len(t["columns"])
    lines = [f"**{t['title']}**", "", "| " + " | ".join(t["columns"]) + " |",
             "|" + "|".join(":--" if a == "left" else "--:" for a in align) + "|"]
    bold = set(t.get("bold_rows", []))
    for i, row in enumerate(t["rows"]):
        cells = [str(c).replace("|", "\\|") for c in row]
        if i in bold:
            cells = [f"**{c}**" if c else c for c in cells]
        lines.append("| " + " | ".join(cells) + " |")
    if t.get("note"):
        lines += ["", f"*{t['note']}*"]
    return "\n".join(lines)


def to_html(t: dict) -> str:
    align = t.get("align") or ["left"] * len(t["columns"])
    bold = set(t.get("bold_rows", []))
    head = "".join(f'<th class="{a}">{html.escape(c)}</th>' for c, a in zip(t["columns"], align))
    body = "".join(
        f'<tr class="{"total" if i in bold else ""}">' + "".join(
            f'<td class="{a}">{html.escape(str(c))}</td>' for c, a in zip(row, align)) + "</tr>"
        for i, row in enumerate(t["rows"]))
    note = f'<div class="fb-note">{html.escape(t["note"])}</div>' if t.get("note") else ""
    return (f'<div class="fb-table"><div class="fb-eyebrow">{html.escape(t["title"])}</div>'
            f'<div class="fb-scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>{note}</div>')
