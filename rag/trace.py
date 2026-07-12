"""Retrieval traces: the answer to "why did it retrieve *that*?"

Every question that goes through the retriever produces a trace recording
what each stage saw — dense ranks, BM25 ranks, fused order, re-ranked order —
plus per-stage latency. With a trace, a bad answer becomes a lookup
("the evidence was rank 2 after fusion and the re-ranker demoted it to 9")
instead of an investigation. Without one, retrieval is a black box and every
debugging session starts from zero.

Traces are returned to API clients, rendered in the web app, and logged as
single-line JSON so any log pipeline can collect them.
"""

import time
from dataclasses import dataclass, field


@dataclass
class StageTrace:
    """What one retrieval stage produced: its top results and how long it took."""

    stage: str
    latency_ms: float
    results: list[dict]  # [{"id": ..., "score": ...}, ...] — top few only


@dataclass
class RetrievalTrace:
    """The full retrieval story for one query."""

    query: str
    search_mode: str
    reranked: bool
    stages: list[StageTrace] = field(default_factory=list)
    total_ms: float = 0.0

    def add_stage(self, stage: str, latency_ms: float, chunks, limit: int = 8) -> None:
        self.stages.append(
            StageTrace(
                stage=stage,
                latency_ms=round(latency_ms, 1),
                results=[
                    {"id": chunk.id, "score": round(chunk.score, 4)}
                    for chunk in chunks[:limit]
                ],
            )
        )

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "search_mode": self.search_mode,
            "reranked": self.reranked,
            "total_ms": round(self.total_ms, 1),
            "stages": [
                {"stage": s.stage, "latency_ms": s.latency_ms, "results": s.results}
                for s in self.stages
            ],
        }


class StageTimer:
    """Context manager measuring one stage's wall-clock latency in ms."""

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000
        return False
