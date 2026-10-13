import re

import pytest

from resume_screener.ingest import Resume
from resume_screener.retrieval import Chunk, ResumeIndex, cosine, format_context, rrf_merge, tokenize

VOCAB = ["kafka", "java", "python", "aws", "nursing", "patient", "marketing", "seo", "distributed", "systems"]


class FakeEmbedder:
    """Deterministic bag-of-words vectors, so the index can be tested without Ollama."""

    @staticmethod
    def _vec(text: str) -> list[float]:
        words = re.findall(r"[a-z]+", text.lower())
        return [float(words.count(w)) + 0.01 for w in VOCAB]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


def test_tokenize_keeps_language_names():
    assert tokenize("C++, C# and .NET; Node.js") == ["c++", "c#", "and", "net", "node.js"]


def test_cosine():
    assert cosine([1, 0], [1, 0]) == pytest.approx(1)
    assert cosine([1, 0], [0, 1]) == pytest.approx(0)
    assert cosine([0, 0], [1, 1]) == 0


def test_rrf_rewards_items_ranked_well_by_both_lists():
    assert rrf_merge([["a", "b", "c"], ["b", "c", "a"]])[0] == "b"
    assert set(rrf_merge([["a"], ["z"]])) == {"a", "z"}


def test_format_context_groups_by_resume_and_dedupes():
    chunks = [
        Chunk("1", "r1", "a.pdf", "Kafka"),
        Chunk("2", "r2", "b.pdf", "Go"),
        Chunk("3", "r1", "a.pdf", "Kafka"),
        Chunk("4", "r1", "a.pdf", "AWS"),
    ]
    ctx = format_context(chunks)
    assert ctx.count("### a.pdf") == 1 and ctx.count("Kafka") == 1
    assert "### b.pdf\nGo" in ctx


@pytest.fixture(scope="module")
def index():
    resumes = [
        Resume("r0", "dev.pdf", "Java engineer. Built distributed systems with Kafka on AWS. " * 30),
        Resume("r1", "nurse.pdf", "Nursing professional focused on patient care in the ICU. " * 30),
        Resume("r2", "seo.pdf", "Marketing lead running SEO campaigns and content. " * 30),
    ]
    idx = ResumeIndex(resumes, FakeEmbedder())
    yield idx
    idx.release()


def test_index_chunks_every_resume(index):
    assert {c.resume_id for c in index.chunks} == {"r0", "r1", "r2"}
    assert len(index.chunks) > 3  # long resumes are split


def test_search_prefers_matching_resume_and_caps_chunks_per_resume(index):
    hits = index.search("distributed systems kafka", k=4, per_resume=2)
    assert hits[0].resume_name == "dev.pdf"
    assert sum(h.resume_id == "r0" for h in hits) <= 2
    assert len({h.resume_id for h in hits}) >= 2


def test_relevance_ranks_resumes_against_query(index):
    sims = index.relevance(index.embed_query("patient nursing"))
    assert max(sims, key=sims.get) == "r1"


def test_tidy_bullets_makes_markdown_lists():
    from resume_screener.assistant import tidy_bullets

    assert tidy_bullets("• Ann [a.pdf] • Bob [b.pdf]") == "- Ann [a.pdf]\n- Bob [b.pdf]"
    assert tidy_bullets("No bullets here.") == "No bullets here."
