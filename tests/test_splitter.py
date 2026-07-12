import pytest

from rag.splitter import split_text


def test_short_text_is_one_chunk():
    assert split_text("hello world", chunk_size=100, chunk_overlap=10) == ["hello world"]


def test_chunks_respect_size_limit():
    text = "a" * 1200
    chunks = split_text(text, chunk_size=500, chunk_overlap=100)
    assert all(len(chunk) <= 500 for chunk in chunks)


def test_adjacent_chunks_overlap():
    text = " ".join(f"word{i}" for i in range(300))
    chunks = split_text(text, chunk_size=200, chunk_overlap=50)
    assert len(chunks) > 2
    for previous, current in zip(chunks, chunks[1:], strict=False):
        if len(current) >= 60:
            # The tail of each chunk must reappear inside the next one.
            assert previous[-30:] in current


def test_whitespace_is_normalized():
    chunks = split_text("line one\n\nline   two\t line three", chunk_size=100, chunk_overlap=10)
    assert chunks == ["line one line two line three"]


def test_empty_text_gives_no_chunks():
    assert split_text("   \n  ", chunk_size=100, chunk_overlap=10) == []


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValueError):
        split_text("some text", chunk_size=100, chunk_overlap=100)


def test_full_text_is_covered():
    text = " ".join(f"token{i}" for i in range(500))
    chunks = split_text(text, chunk_size=300, chunk_overlap=60)
    # Every token must appear in at least one chunk.
    for i in range(500):
        assert any(f"token{i}" in chunk for chunk in chunks)
