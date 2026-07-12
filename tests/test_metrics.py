from eval.metrics import first_relevant_rank, hit_rate, mean_reciprocal_rank


def test_rank_of_first_matching_chunk():
    chunks = ["nothing here", "the EVIDENCE  string", "evidence again"]
    assert first_relevant_rank(chunks, "evidence string") == 2


def test_rank_is_none_when_absent():
    assert first_relevant_rank(["a", "b"], "missing") is None


def test_matching_ignores_case_and_whitespace():
    assert first_relevant_rank(["Split   Across\nLines"], "split across lines") == 1


def test_hit_rate():
    assert hit_rate([1, None, 3, None]) == 0.5
    assert hit_rate([]) == 0.0


def test_mrr():
    # ranks 1, 2, and one miss -> (1 + 0.5 + 0) / 3
    assert abs(mean_reciprocal_rank([1, 2, None]) - 0.5) < 1e-12
    assert mean_reciprocal_rank([]) == 0.0
