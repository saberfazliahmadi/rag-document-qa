"""Application configuration.

All settings are read from environment variables (optionally loaded from a
`.env` file) so that no secrets ever live in the source code.
"""

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    """Runtime settings for the RAG pipeline."""

    # LLM provider (OpenRouter exposes an OpenAI-compatible API)
    api_key: str = field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY", ""))
    base_url: str = field(
        default_factory=lambda: os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    )
    model: str = field(
        default_factory=lambda: os.getenv("LLM_MODEL", "meta-llama/llama-3.3-70b-instruct:free")
    )

    # Embeddings
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    )

    # Vector store
    persist_dir: str = field(default_factory=lambda: os.getenv("CHROMA_DIR", "chroma_db"))
    collection_name: str = field(
        default_factory=lambda: os.getenv("COLLECTION_NAME", "documents")
    )

    # Chunking
    chunk_size: int = field(default_factory=lambda: int(os.getenv("CHUNK_SIZE", "500")))
    chunk_overlap: int = field(default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "100")))

    # Retrieval
    # "dense" = vector search only; "hybrid" = vector + BM25 merged with RRF.
    search_mode: str = field(default_factory=lambda: os.getenv("SEARCH_MODE", "hybrid"))
    # How many candidates the first stage retrieves before the final top_k cut.
    candidates: int = field(default_factory=lambda: int(os.getenv("CANDIDATES", "20")))
    use_reranker: bool = field(
        default_factory=lambda: os.getenv("USE_RERANKER", "true").lower() in ("1", "true", "yes")
    )
    reranker_model: str = field(
        default_factory=lambda: os.getenv(
            "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
    )

    # Generation
    temperature: float = field(default_factory=lambda: float(os.getenv("TEMPERATURE", "0.2")))
    max_tokens: int = field(default_factory=lambda: int(os.getenv("MAX_TOKENS", "512")))
    top_k: int = field(default_factory=lambda: int(os.getenv("TOP_K", "4")))

    def require_api_key(self) -> None:
        """Fail fast with a clear message when the API key is missing."""
        if not self.api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set. "
                "Copy .env.example to .env and add your key."
            )
