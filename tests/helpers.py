"""Test doubles: a scripted chat model and a hashed bag-of-words embedder (no Ollama needed)."""
import hashlib

import numpy as np
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from finbot.kb.search import KnowledgeBase, tokenize


class ScriptedModel(BaseChatModel):
    responses: list = Field(default_factory=list)
    calls: list = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls.append(messages)
        if not self.responses:
            raise AssertionError("ScriptedModel ran out of responses (graph looped too long)")
        msg = self.responses.pop(0)
        if isinstance(msg, Exception):     # e.g. a server error such as a context-length 400
            raise msg
        if isinstance(msg, str):
            msg = AIMessage(content=msg)
        return ChatResult(generations=[ChatGeneration(message=msg)])


def tool_call(name, args, cid):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": cid, "type": "tool_call"}])


class FakeEmbedder:
    dim = 256

    def _vec(self, text):
        v = np.zeros(self.dim, dtype=np.float32)
        for t in tokenize(text):
            v[int(hashlib.md5(t.encode()).hexdigest(), 16) % self.dim] += 1
        return v

    def embed_query(self, text):
        return self._vec(text).tolist()

    def embed_documents(self, texts):
        return [self._vec(t).tolist() for t in texts]


CHUNKS = [
    {"id": "1", "file": "guidelines/vda_tax.html", "kind": "guidelines", "title": "Tax on virtual digital assets", "url": "u1", "page": None,
     "text": "Income from transfer of virtual digital assets such as crypto is taxed at 30 percent under section 115BBH. TDS of 1 percent applies under section 194S."},
    {"id": "3", "file": "books/finance.pdf", "kind": "books", "title": "Principles of Finance", "url": "u3", "page": 120,
     "text": "Duration measures the sensitivity of a bond price to changes in interest rates. Convexity captures the curvature of the price yield relationship."},
    {"id": "5", "file": "guidelines/ia_regs.pdf", "kind": "guidelines", "title": "SEBI Investment Advisers Regulations", "url": "u5", "page": 2,
     "text": "No person shall act as an investment adviser unless registered with SEBI under these regulations."},
]


def make_kb(min_dense=0.3):
    emb = FakeEmbedder()
    vecs = np.asarray(emb.embed_documents([c["text"] for c in CHUNKS]), dtype=np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    return KnowledgeBase(CHUNKS, vecs, emb, min_dense_score=min_dense)
