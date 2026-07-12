"""Result fusion and re-ranking.

Two ideas from production retrieval systems:

1. Reciprocal Rank Fusion (RRF) merges result lists from different retrievers
   (here: vector search and BM25). It uses only the *rank* of each result,
   never the raw scores — BM25 scores and vector distances live on different
   scales and cannot be compared directly, but ranks always can.

2. Cross-encoder re-ranking is a second, more accurate scoring stage. A
   cross-encoder reads the query and a chunk *together*, so it captures
   interactions a vector similarity cannot. It is too slow to score a whole
   corpus, but fast enough to re-score a few dozen candidates.
"""

from .store import RetrievedChunk

RRF_K = 60  # standard constant from the original RRF paper; dampens the top ranks


def reciprocal_rank_fusion(
    result_lists: list[list[RetrievedChunk]], k: int = RRF_K
) -> list[RetrievedChunk]:
    """Merge ranked result lists into one list ordered by fused score.

    Each result earns 1 / (k + rank) from every list it appears in, so
    chunks ranked well by several retrievers rise to the top.
    """
    scores: dict[str, float] = {}
    by_id: dict[str, RetrievedChunk] = {}

    for results in result_lists:
        for rank, chunk in enumerate(results, start=1):
            scores[chunk.id] = scores.get(chunk.id, 0.0) + 1.0 / (k + rank)
            by_id.setdefault(chunk.id, chunk)

    ranked_ids = sorted(scores, key=lambda chunk_id: scores[chunk_id], reverse=True)
    return [
        RetrievedChunk(
            id=chunk_id,
            text=by_id[chunk_id].text,
            source=by_id[chunk_id].source,
            chunk_index=by_id[chunk_id].chunk_index,
            score=scores[chunk_id],
        )
        for chunk_id in ranked_ids
    ]


class CrossEncoderReranker:
    """Re-scores query/chunk pairs with a cross-encoder model.

    The model is loaded lazily on first use so that importing this module
    (e.g. in unit tests) never triggers a download.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(
        self, query: str, candidates: list[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        """Return the top_k candidates ordered by cross-encoder relevance."""
        if not candidates:
            return []

        model = self._load()
        pairs = [(query, chunk.text) for chunk in candidates]
        scores = model.predict(pairs)

        rescored = [
            RetrievedChunk(
                id=chunk.id,
                text=chunk.text,
                source=chunk.source,
                chunk_index=chunk.chunk_index,
                score=float(score),
            )
            for chunk, score in zip(candidates, scores)
        ]
        rescored.sort(key=lambda chunk: chunk.score, reverse=True)
        return rescored[:top_k]
