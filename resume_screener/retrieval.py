"""Vector + keyword index over resume chunks.

Search runs a dense (embedding) query and a BM25 keyword query, merges the two rankings with
Reciprocal Rank Fusion, and caps how many chunks any single resume may contribute so answers
draw on the whole candidate pool.
"""

import contextlib
import math
import re
import uuid
from collections.abc import Callable
from dataclasses import dataclass

import chromadb
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

from .config import CHUNK_CHARS, CHUNK_OVERLAP, MAX_CHUNKS_PER_RESUME, PROFILE_CHAR_BUDGET, SEARCH_K
from .ingest import Resume

RRF_K = 60  # standard damping constant for Reciprocal Rank Fusion
CANDIDATES_PER_RANKER = 24

# nomic-embed-text is trained with task prefixes; using them noticeably improves retrieval
DOC_PREFIX, QUERY_PREFIX = "search_document: ", "search_query: "


@dataclass(frozen=True)
class Chunk:
    id: str
    resume_id: str
    resume_name: str
    text: str


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9][a-z0-9+#.]*", text.lower())


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


def rrf_merge(rankings: list[list[str]], k: int = RRF_K) -> list[str]:
    """Reciprocal Rank Fusion: each list votes 1 / (k + rank) for its items."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda item: -scores[item])


def split_resume(resume: Resume) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_CHARS, chunk_overlap=CHUNK_OVERLAP)
    return [
        Chunk(id=f"{resume.id}-{i:03d}", resume_id=resume.id, resume_name=resume.name, text=t)
        for i, t in enumerate(splitter.split_text(resume.text))
    ]


class ResumeIndex:
    """Chunks, embeddings and keyword index for one uploaded batch of resumes."""

    def __init__(self, resumes: list[Resume], embedder: OllamaEmbeddings, log: Callable[[str], None] = lambda _: None):
        self.resumes = resumes
        self.embedder = embedder
        self.chunks = [c for r in resumes for c in split_resume(r)]
        self._by_id = {c.id: c for c in self.chunks}

        log(f"🧮 Embedding {len(self.chunks)} chunks and {len(resumes)} resumes…")
        chunk_vecs = embedder.embed_documents([DOC_PREFIX + c.text for c in self.chunks])
        # one vector per whole resume (truncated) for the relevance score
        self.resume_vecs = dict(
            zip(
                (r.id for r in resumes),
                embedder.embed_documents([DOC_PREFIX + r.text[:PROFILE_CHAR_BUDGET] for r in resumes]),
                strict=True,
            )
        )

        self._client = chromadb.EphemeralClient()
        self._collection = self._client.create_collection(
            name=f"resumes-{uuid.uuid4().hex[:12]}", metadata={"hnsw:space": "cosine"}
        )
        self._collection.add(
            ids=[c.id for c in self.chunks],
            embeddings=chunk_vecs,
            documents=[c.text for c in self.chunks],
            metadatas=[{"resume_id": c.resume_id} for c in self.chunks],
        )
        self._bm25 = BM25Okapi([tokenize(c.text) for c in self.chunks])

    def embed_query(self, text: str) -> list[float]:
        return self.embedder.embed_query(QUERY_PREFIX + text)

    def relevance(self, query_vec: list[float]) -> dict[str, float]:
        """Cosine similarity between a query vector and every whole resume."""
        return {rid: cosine(query_vec, vec) for rid, vec in self.resume_vecs.items()}

    def search(self, query: str, k: int = SEARCH_K, per_resume: int = MAX_CHUNKS_PER_RESUME) -> list[Chunk]:
        n = min(CANDIDATES_PER_RANKER, len(self.chunks))
        dense = self._collection.query(query_embeddings=[self.embed_query(query)], n_results=n)["ids"][0]
        bm25_scores = self._bm25.get_scores(tokenize(query))
        keyword = [self.chunks[i].id for i in sorted(range(len(self.chunks)), key=lambda i: -bm25_scores[i])[:n]]

        picked, per = [], {}
        for chunk_id in rrf_merge([dense, keyword]):
            chunk = self._by_id[chunk_id]
            if per.get(chunk.resume_id, 0) < per_resume:
                picked.append(chunk)
                per[chunk.resume_id] = per.get(chunk.resume_id, 0) + 1
            if len(picked) == k:
                break
        return picked

    def release(self) -> None:
        with contextlib.suppress(Exception):
            self._client.delete_collection(self._collection.name)


def format_context(chunks: list[Chunk]) -> str:
    """Group chunks under their resume so the model sees each candidate as one block."""
    grouped: dict[str, list[str]] = {}
    for c in chunks:
        grouped.setdefault(c.resume_name, []).append(c.text.strip())
    return "\n\n".join(f"### {name}\n" + "\n…\n".join(dict.fromkeys(parts)) for name, parts in grouped.items())
