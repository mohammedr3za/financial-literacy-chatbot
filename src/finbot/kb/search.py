"""Hybrid retrieval: BM25 keywords + dense bge-m3 vectors, merged with reciprocal-rank fusion."""
import json
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

RRF_K = 60
CANDIDATES = 50
MAX_PER_FILE = 2  # 26 sources: keep the top 5 diverse so one long booklet cannot crowd out a circular
_STOP = set("a an and are as at be by can do does for from has have how i in is it its me my of on or should so "
            "that the this to was what when where which who why will with would you your".split())


# Short forms people type vs the words official documents use (e.g. SEBI's IA rules say "Artificial Intelligence").
# Two-letter forms match only in capitals so ordinary words ("ai" in Hinglish, "ia") are left alone.
ABBREVIATIONS = {
    "AI": "artificial intelligence", "ML": "machine learning", "MF": "mutual fund", "MFs": "mutual funds",
    "IA": "investment adviser", "RA": "research analyst", "FD": "fixed deposit", "RD": "recurring deposit",
    "nps": "national pension system", "ppf": "public provident fund", "epf": "employees provident fund",
    "kyc": "know your customer", "upi": "unified payments interface", "sip": "systematic investment plan",
    "tds": "tax deducted at source", "hra": "house rent allowance", "itr": "income tax return",
    "nbfc": "non-banking financial company", "elss": "equity linked savings scheme", "ltcg": "long-term capital gains",
    "stcg": "short-term capital gains", "dicgc": "deposit insurance", "ombudsman": "ombudsman complaint",
}


# Everyday words for a situation vs the official terms (v4: "I was tricked on UPI" never reached the 1930 helpline pages).
CONCEPTS = [
    (re.compile(r"(?i)\b(tricked|cheated|scam(med)?|fraud\w*|duped|hacked|money (is |was )?gone|unauthori[sz]ed)\b|"
                r"धोखाधड़ी|धोखा|ठगी|फ्रॉड|மோசடி|ஏமாற்ற"), "cyber fraud, report, helpline, cybercrime reporting portal"),
]


def expand_query(query: str) -> str:
    extra = []
    for short, full in ABBREVIATIONS.items():
        pattern = rf"\b{short}\b" if len(short) <= 3 and short[:2].isupper() else rf"(?i)\b{short}\b"
        if re.search(pattern, query) and full.lower() not in query.lower():
            extra.append(full)
    extra += [terms for pattern, terms in CONCEPTS if pattern.search(query)]
    return query + (" (" + ", ".join(extra) + ")" if extra else "")


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOP]


def rrf(rankings: list[list[int]], k: int = RRF_K) -> dict[int, float]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return scores


class KnowledgeBase:
    def __init__(self, chunks: list[dict], vectors: np.ndarray, embedder, min_dense_score: float = 0.45):
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors differ in length; rebuild the index.")
        self.chunks, self.vectors, self.embedder = chunks, vectors, embedder
        self.min_dense_score = min_dense_score
        self.bm25 = BM25Okapi([tokenize(c["text"] + " " + c["title"]) for c in chunks]) if chunks else None

    @classmethod
    def load(cls, index_dir: Path, embedder, min_dense_score: float = 0.45) -> "KnowledgeBase":
        chunks = [json.loads(line) for line in (index_dir / "chunks.jsonl").read_text(encoding="utf-8").splitlines() if line]
        vectors = np.load(index_dir / "embeddings.npy")
        return cls(chunks, vectors, embedder, min_dense_score)

    def search(self, query: str, k: int = 5) -> dict:
        """Returns {"hits": [...], "best_dense": float, "relevant": bool}."""
        if not self.chunks or not query.strip():
            return {"hits": [], "best_dense": 0.0, "relevant": False}
        query = expand_query(query)
        q = np.asarray(self.embedder.embed_query(query), dtype=np.float32)
        q /= np.linalg.norm(q) + 1e-12
        dense = self.vectors @ q
        dense_rank = list(np.argsort(-dense)[:CANDIDATES])
        bm = self.bm25.get_scores(tokenize(query))
        bm_rank = [int(i) for i in np.argsort(-bm)[:CANDIDATES] if bm[i] > 0]
        fused = sorted(rrf([[int(i) for i in dense_rank], bm_rank]).items(), key=lambda kv: -kv[1])
        hits, per_file = [], {}
        for idx, score in fused:
            c = self.chunks[idx]
            if per_file.get(c["file"], 0) >= MAX_PER_FILE:
                continue
            per_file[c["file"]] = per_file.get(c["file"], 0) + 1
            hits.append({**c, "rrf": round(score, 5), "dense": round(float(dense[idx]), 4), "bm25": round(float(bm[idx]), 3)})
            if len(hits) == k:
                break
        best = float(dense.max())
        return {"hits": hits, "best_dense": round(best, 4), "relevant": best >= self.min_dense_score}
