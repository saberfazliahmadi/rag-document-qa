from rag.store import RetrievedChunk
from rag.trace import RetrievalTrace, StageTimer


def chunk(chunk_id: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(id=chunk_id, text="t", source="doc.txt", chunk_index=0, score=score)


def test_trace_records_stages_in_order():
    trace = RetrievalTrace(query="q", search_mode="hybrid", reranked=True)
    trace.add_stage("dense", 12.34, [chunk("a", 0.9), chunk("b", 0.8)])
    trace.add_stage("bm25", 1.5, [chunk("b", 3.2)])

    payload = trace.to_dict()
    assert [s["stage"] for s in payload["stages"]] == ["dense", "bm25"]
    assert payload["stages"][0]["results"][0] == {"id": "a", "score": 0.9}
    assert payload["stages"][0]["latency_ms"] == 12.3


def test_trace_limits_recorded_results():
    trace = RetrievalTrace(query="q", search_mode="dense", reranked=False)
    trace.add_stage("dense", 1.0, [chunk(str(i), 1.0) for i in range(30)], limit=8)
    assert len(trace.to_dict()["stages"][0]["results"]) == 8


def test_stage_timer_measures_elapsed():
    with StageTimer() as timer:
        sum(range(1000))
    assert timer.elapsed_ms >= 0
