import hashlib

import numpy as np

from finbot.kb.ingest import chunk_words, clean, html_to_text
from finbot.kb.search import KnowledgeBase, rrf, tokenize


class FakeEmbedder:
    """Hashed bag-of-words vectors: deterministic, no model needed."""
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
    {"id": "2", "file": "guidelines/ombudsman.pdf", "kind": "guidelines", "title": "RBI Integrated Ombudsman Scheme 2021", "url": "u2", "page": 3,
     "text": "A customer may file a complaint with the RBI Ombudsman if the bank does not reply within 30 days or the reply is unsatisfactory."},
    {"id": "3", "file": "books/finance.pdf", "kind": "books", "title": "Principles of Finance", "url": "u3", "page": 120,
     "text": "Duration measures the sensitivity of a bond price to changes in interest rates. Convexity captures the curvature of the price yield relationship."},
    {"id": "4", "file": "books/finance.pdf", "kind": "books", "title": "Principles of Finance", "url": "u3", "page": 45,
     "text": "An equated monthly instalment repays a loan with interest through fixed payments over the tenure."},
    {"id": "5", "file": "guidelines/ia_regs.pdf", "kind": "guidelines", "title": "SEBI Investment Advisers Regulations", "url": "u5", "page": 2,
     "text": "No person shall act as an investment adviser unless registered with SEBI under these regulations."},
]


def make_kb(min_dense=0.2):
    emb = FakeEmbedder()
    vecs = np.asarray(emb.embed_documents([c["text"] for c in CHUNKS]), dtype=np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    return KnowledgeBase(CHUNKS, vecs, emb, min_dense_score=min_dense)


def test_rrf_rewards_agreement():
    scores = rrf([[0, 1, 2], [2, 0, 1]])
    assert max(scores, key=scores.get) == 0


def test_exact_section_number_found():
    res = make_kb().search("How is crypto taxed? section 115BBH", k=3)
    assert res["hits"][0]["file"] == "guidelines/vda_tax.html"


def test_concept_query_finds_book():
    res = make_kb().search("bond duration and convexity interest rates", k=3)
    assert res["hits"][0]["id"] == "3"


def test_unrelated_query_marked_not_relevant():
    res = make_kb(min_dense=0.5).search("best biryani recipe in hyderabad", k=3)
    assert res["relevant"] is False


def test_hits_carry_citation_fields():
    hit = make_kb().search("complaint to RBI ombudsman", k=1)["hits"][0]
    assert {"title", "url", "page", "file", "text"} <= set(hit)


def test_chunking_overlaps_and_covers_text():
    words = [f"w{i}" for i in range(700)]
    chunks = chunk_words(" ".join(words), size=300, overlap=50)
    assert chunks[0].split()[0] == "w0" and chunks[-1].split()[-1] == "w699"
    assert chunks[1].split()[0] == "w250"


def test_html_to_text_drops_scripts():
    html = "<html><script>alert(1)</script><style>p{}</style><p>Tax is 30%</p></html>"
    assert html_to_text(html) == "Tax is 30%"


def test_clean_rejoins_hyphenation():
    assert clean("invest-\nment  adviser") == "investment adviser"
