"""Text chunking utilities."""


def split_text(text: str, chunk_size: int = 500, chunk_overlap: int = 100) -> list[str]:
    """Split text into overlapping, fixed-size chunks.

    Overlap keeps sentences that fall on a chunk boundary present in both
    neighboring chunks, so their meaning is not lost during retrieval.

    Args:
        text: The raw document text.
        chunk_size: Maximum number of characters per chunk.
        chunk_overlap: Number of characters shared between adjacent chunks.

    Returns:
        A list of non-empty text chunks.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    text = " ".join(text.split())  # normalize whitespace and line breaks
    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - chunk_overlap

    return chunks
