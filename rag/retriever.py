"""Two-stage retrieval: candidate generation, then precision re-ranking.

Stage 1 (recall): fetch a generous candidate set — dense vector search alone,
or dense + BM25 fused with Reciprocal Rank Fusion in hybrid mode. The goal is
that the right chunk is *somewhere* in the candidates.

Stage 2 (precision): optionally re-score the candidates with a cross-encoder
and keep the best few. The goal is that the right chunk is in the final
context window the LLM actually sees.

This split is deliberate: recall problems and precision problems have
different fixes, and separating the stages makes each one measurable
(see eval/).
"""

from .config import Settings
from .ranking import CrossEncoderReranker, reciprocal_rank_fusion
from .store import RetrievedChunk, VectorStore


class Retriever:
    """Configurable retrieval pipeline over a VectorStore."""

    def __init__(self, settings: Settings, store: VectorStore):
        if settings.search_mode not in ("dense", "hybrid"):
            raise ValueError(
                f"SEARCH_MODE must be 'dense' or 'hybrid', got '{settings.search_mode}'"
            )
        self.settings = settings
        self.store = store
        self.reranker = (
            CrossEncoderReranker(settings.reranker_model) if settings.use_reranker else None
        )

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        """Return the top_k most relevant chunks for the query."""
        top_k = top_k or self.settings.top_k
        candidates = self._candidates(query)

        if self.reranker is not None:
            return self.reranker.rerank(query, candidates, top_k)
        return candidates[:top_k]

    def _candidates(self, query: str) -> list[RetrievedChunk]:
        n = self.settings.candidates
        dense = self.store.search_dense(query, n)

        if self.settings.search_mode == "dense":
            return dense

        keyword = self.store.search_keyword(query, n)
        return reciprocal_rank_fusion([dense, keyword])
