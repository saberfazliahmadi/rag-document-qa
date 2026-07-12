from rag.ranking import reciprocal_rank_fusion
from rag.store import RetrievedChunk


def chunk(chunk_id: str, score: float = 0.0) -> RetrievedChunk:
    return RetrievedChunk(
        id=chunk_id, text=f"text {chunk_id}", source="doc.txt", chunk_index=0, score=score
    )


def test_single_list_keeps_order():
    results = reciprocal_rank_fusion([[chunk("a"), chunk("b"), chunk("c")]])
    assert [c.id for c in results] == ["a", "b", "c"]


def test_symmetric_positions_get_equal_scores():
    dense = [chunk("a"), chunk("b"), chunk("c")]
    keyword = [chunk("c"), chunk("b"), chunk("a")]
    results = reciprocal_rank_fusion([dense, keyword])
    scores = {c.id: c.score for c in results}
    assert scores["a"] == scores["c"]


def test_found_by_both_retrievers_beats_found_by_one():
    # "b" is mid-ranked by both retrievers; "d" is top-ranked by only one.
    dense = [chunk("a"), chunk("b"), chunk("c")]
    keyword = [chunk("d"), chunk("b")]
    results = reciprocal_rank_fusion([dense, keyword])
    scores = {c.id: c.score for c in results}
    assert scores["b"] > scores["d"]


def test_fused_scores_are_rrf_sums():
    results = reciprocal_rank_fusion([[chunk("a")], [chunk("a")]], k=60)
    assert abs(results[0].score - 2 / 61) < 1e-12


def test_result_present_in_only_one_list_is_kept():
    dense = [chunk("a"), chunk("b")]
    keyword = [chunk("c")]
    ids = {c.id for c in reciprocal_rank_fusion([dense, keyword])}
    assert ids == {"a", "b", "c"}


def test_empty_input():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []
