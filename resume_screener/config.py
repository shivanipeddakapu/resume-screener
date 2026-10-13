"""Runtime settings. Anything model- or host-specific can be overridden in `.env`."""

import os

from dotenv import load_dotenv

load_dotenv()

# ── local models (Ollama) ─────────────────────────────────────────────────────
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2:3b")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
NUM_CTX = int(os.getenv("NUM_CTX", "8192"))
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "2"))  # resumes profiled in parallel

# ── indexing / retrieval ──────────────────────────────────────────────────────
CHUNK_CHARS = 800
CHUNK_OVERLAP = 120
PROFILE_CHAR_BUDGET = 6000  # resumes shorter than this are given to the LLM whole
SEARCH_K = 8  # chunks returned by hybrid search
MAX_CHUNKS_PER_RESUME = 2  # keeps chat context spread across candidates

# ── upload limits ─────────────────────────────────────────────────────────────
MAX_PDFS = 200
MAX_UNZIPPED_BYTES = 200 * 1024 * 1024

# ── fit score ─────────────────────────────────────────────────────────────────
DEFAULT_SHORTLIST = 65
DEFAULT_WEIGHTS = {
    "Must-have": 35,
    "Experience": 20,
    "Relevance": 20,
    "Education": 15,
    "Nice-to-have": 10,
}
