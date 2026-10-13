"""Local model access through Ollama."""

from dataclasses import dataclass, field

import requests
from langchain_ollama import ChatOllama, OllamaEmbeddings

from .config import EMBEDDING_MODEL, LLM_MODEL, NUM_CTX, OLLAMA_BASE_URL


def make_llm() -> ChatOllama:
    return ChatOllama(model=LLM_MODEL, base_url=OLLAMA_BASE_URL, temperature=0, num_ctx=NUM_CTX)


def make_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)


@dataclass
class OllamaStatus:
    reachable: bool
    missing: list[str] = field(default_factory=list)
    version: str = ""

    @property
    def ready(self) -> bool:
        return self.reachable and not self.missing

    @property
    def problem(self) -> str | None:
        if not self.reachable:
            return f"Ollama is not reachable at {OLLAMA_BASE_URL}. Start it with `ollama serve`."
        if self.missing:
            pulls = " && ".join(f"ollama pull {m}" for m in self.missing)
            return f"Missing Ollama model(s): {', '.join(self.missing)}. Run `{pulls}`."
        return None


def ollama_status(timeout: float = 3.0) -> OllamaStatus:
    """Check that the Ollama server answers and that both configured models are pulled."""
    try:
        tags = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=timeout).json()
        version = requests.get(f"{OLLAMA_BASE_URL}/api/version", timeout=timeout).json().get("version", "")
    except (requests.RequestException, ValueError):
        return OllamaStatus(reachable=False)
    names = {m.get("name", "") for m in tags.get("models", [])}
    missing = [m for m in (LLM_MODEL, EMBEDDING_MODEL) if m not in names and f"{m}:latest" not in names]
    return OllamaStatus(reachable=True, missing=missing, version=version)
