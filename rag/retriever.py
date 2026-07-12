"""Two-stage retrieval: candidate generation, then precision re-ranking.

Stage 1 (recall): fetch a generous candidate set — dense vector search alone,
or dense + BM25 fused with Reciprocal Rank Fusion in hybrid mode. The goal is
that the right chunk is *somewhere* in the candidates.

Stage 2 (precision): optionally re-score the candidates with a cross-encoder
and keep the best few. The goal is that the right chunk is in the final
context window the LLM actually sees.

This split is deliberate: recall problems and precision problems have
different fixes, and separating the stages makes each one measurable
(see eval/) and observable (see rag/trace.py).
"""

import time

from .config import Settings
from .ranking import CrossEncoderReranker, reciprocal_rank_fusion
from .store import RetrievedChunk, VectorStore
from .trace import RetrievalTrace, StageTimer


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
        chunks, _ = self.retrieve_traced(query, top_k)
        return chunks

    def retrieve_traced(
        self, query: str, top_k: int | None = None
    ) -> tuple[list[RetrievedChunk], RetrievalTrace]:
        """Retrieve chunks and record what every stage saw along the way."""
        top_k = top_k or self.settings.top_k
        trace = RetrievalTrace(
            query=query,
            search_mode=self.settings.search_mode,
            reranked=self.reranker is not None,
        )
        started = time.perf_counter()

        n = self.settings.candidates
        with StageTimer() as timer:
            dense = self.store.search_dense(query, n)
        trace.add_stage("dense", timer.elapsed_ms, dense)

        if self.settings.search_mode == "hybrid":
            with StageTimer() as timer:
                keyword = self.store.search_keyword(query, n)
            trace.add_stage("bm25", timer.elapsed_ms, keyword)

            with StageTimer() as timer:
                candidates = reciprocal_rank_fusion([dense, keyword])
            trace.add_stage("rrf_fusion", timer.elapsed_ms, candidates)
        else:
            candidates = dense

        if self.reranker is not None:
            with StageTimer() as timer:
                final = self.reranker.rerank(query, candidates, top_k)
            trace.add_stage("rerank", timer.elapsed_ms, final)
        else:
            final = candidates[:top_k]

        trace.total_ms = (time.perf_counter() - started) * 1000
        return final, trace
