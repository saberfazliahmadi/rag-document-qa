"""Deterministic retrieval metrics.

These need no LLM, cost nothing, and always give the same answer for the
same inputs — which is what makes them suitable for CI. Generation metrics
(faithfulness, answer relevance) are LLM-judged and live in judge.py.
"""


def _normalize(text: str) -> str:
    return " ".join(text.split()).lower()


def first_relevant_rank(retrieved_texts: list[str], evidence: str) -> int | None:
    """Return the 1-based rank of the first retrieved chunk containing the

    evidence string, or None if no retrieved chunk contains it.
    """
    needle = _normalize(evidence)
    for rank, text in enumerate(retrieved_texts, start=1):
        if needle in _normalize(text):
            return rank
    return None


def hit_rate(ranks: list[int | None]) -> float:
    """Fraction of questions where the evidence was retrieved at all."""
    if not ranks:
        return 0.0
    return sum(rank is not None for rank in ranks) / len(ranks)


def mean_reciprocal_rank(ranks: list[int | None]) -> float:
    """Average of 1/rank of the first relevant chunk (0 when never found).

    Rewards putting the evidence early in the context window, not just
    somewhere in it.
    """
    if not ranks:
        return 0.0
    return sum(1.0 / rank for rank in ranks if rank is not None) / len(ranks)
