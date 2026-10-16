"""Build the knowledge-base index: kb/raw/** (PDF, HTML) -> kb/index/{chunks.jsonl, embeddings.npy, meta.json}.

Usage: uv run python -m finbot.kb.ingest
"""
import csv
import hashlib
import json
import re
import sys
import time
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
from pypdf import PdfReader

from ..config import get_settings
from ..llm import get_embedder

CHUNK_WORDS = 300
OVERLAP_WORDS = 50
MIN_PAGE_CHARS = 40


class _TextExtractor(HTMLParser):
    SKIP = {"script", "style", "nav", "footer", "header", "noscript", "svg"}

    def __init__(self):
        super().__init__()
        self.parts, self._skip = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skip += 1

    def handle_endtag(self, tag):
        if tag in self.SKIP and self._skip:
            self._skip -= 1

    def handle_data(self, data):
        if not self._skip and data.strip():
            self.parts.append(data.strip())


def html_to_text(html: str) -> str:
    p = _TextExtractor()
    p.feed(html)
    return clean("\n".join(p.parts))


_GLYPH = re.compile(r"/uni([0-9A-Fa-f]{4})(?:\.[A-Z][A-Za-z_]*)?")


def clean(text: str) -> str:
    text = _GLYPH.sub(lambda m: chr(int(m.group(1), 16)), text)  # PDF glyph names, e.g. /uni20B9 -> ₹
    text = text.replace(" ", " ")
    text = re.sub(r"-\n(\w)", r"\1", text)          # re-join hyphenated line breaks
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{2,}", "\n", text).strip()


def chunk_words(text: str, size: int = CHUNK_WORDS, overlap: int = OVERLAP_WORDS) -> list[str]:
    words = text.split()
    if not words:
        return []
    step = max(1, size - overlap)
    return [" ".join(words[i:i + size]) for i in range(0, max(1, len(words) - overlap), step)]


def load_manifest(kb_dir: Path) -> dict:
    path = kb_dir / "sources.csv"
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as f:
        return {row["file"]: row for row in csv.DictReader(f)}


def pages_of(path: Path) -> list[tuple[int | None, str]]:
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        return [(i + 1, clean(p.extract_text() or "")) for i, p in enumerate(reader.pages)]
    if path.suffix.lower() in (".html", ".htm"):
        return [(None, html_to_text(path.read_text(encoding="utf-8", errors="ignore")))]
    if path.suffix.lower() in (".txt", ".md"):
        return [(None, clean(path.read_text(encoding="utf-8", errors="ignore")))]
    return []


def build_chunks(kb_dir: Path) -> tuple[list[dict], list[str]]:
    raw = kb_dir / "raw"
    manifest = load_manifest(kb_dir)
    chunks, skipped = [], []
    for path in sorted(p for p in raw.rglob("*") if p.is_file() and not p.name.startswith(".")):
        rel = str(path.relative_to(raw))
        pages = pages_of(path)
        if not pages:
            continue
        if sum(len(t) for _, t in pages[:5]) < 200:
            skipped.append(f"{rel}: no extractable text (scanned?)")
            continue
        meta = manifest.get(rel, {})
        title = meta.get("title") or path.stem.replace("_", " ")
        kind = rel.split("/")[0]                      # books | guidelines
        for page, text in pages:
            if len(text) < MIN_PAGE_CHARS:
                continue
            for j, piece in enumerate(chunk_words(text)):
                cid = hashlib.sha1(f"{rel}|{page}|{j}".encode()).hexdigest()[:12]
                chunks.append({"id": cid, "file": rel, "kind": kind, "title": title, "url": meta.get("url", ""),
                               "publisher": meta.get("publisher", ""), "page": page, "text": piece})
    return chunks, skipped


def main() -> int:
    s = get_settings()
    out = s.kb_dir / "index"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    chunks, skipped = build_chunks(s.kb_dir)
    for msg in skipped:
        print("SKIPPED", msg)
    if not chunks:
        print("No documents found under kb/raw/. Add PDFs or HTML files first.")
        return 1
    emb = get_embedder(s)
    vectors = []
    batch = 32
    for i in range(0, len(chunks), batch):
        vectors.extend(emb.embed_documents([c["text"] for c in chunks[i:i + batch]]))
        print(f"\rembedded {min(i + batch, len(chunks))}/{len(chunks)}", end="", flush=True)
    print()
    mat = np.asarray(vectors, dtype=np.float32)
    mat /= np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    np.save(out / "embeddings.npy", mat)
    with (out / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    files = sorted({c["file"] for c in chunks})
    (out / "meta.json").write_text(json.dumps({
        "embed_model": s.embed_model, "chunks": len(chunks), "files": files, "skipped": skipped,
        "built_on": time.strftime("%Y-%m-%d %H:%M"), "chunk_words": CHUNK_WORDS, "overlap_words": OVERLAP_WORDS,
    }, indent=2))
    print(f"Indexed {len(chunks)} chunks from {len(files)} files in {time.time() - t0:.0f}s -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
