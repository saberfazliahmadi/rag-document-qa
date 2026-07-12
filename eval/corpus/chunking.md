# Chunking Strategies for RAG

Chunking is the step that splits documents into the retrieval units stored in
the vector database. It quietly determines the quality ceiling of the whole
system: if the answer to a question is split across two chunks, or buried in
a chunk full of unrelated text, no amount of clever retrieval or prompting
can fully recover.

The simplest strategy is fixed-size chunking with overlap: cut the text every
N characters and let adjacent chunks share a margin. This project uses 500
characters per chunk with a 100-character overlap as its default. The overlap
exists so that a sentence falling on a chunk boundary appears complete in at
least one chunk. Fixed-size chunking is easy to reason about, deterministic,
and a solid baseline — but it is blind to document structure, happily cutting
through the middle of a sentence, a table row, or a section heading.

Structure-aware strategies split on natural boundaries instead. Sentence
splitting keeps sentences whole. Recursive splitting tries paragraph breaks
first, then sentences, then words, only cutting mid-sentence as a last
resort. Layout-aware splitting for PDFs keeps headings attached to the
sections they introduce and keeps table cells together. These strategies cost
more implementation effort and are harder to make deterministic, but they
measurably improve retrieval on documents with strong structure, such as
manuals, contracts, and clinical guidelines.

Chunk size is a genuine trade-off, not a solved constant. Small chunks (200
to 400 characters) embed precisely — the vector represents one focused idea —
but each chunk carries little context, so the LLM may receive fragments.
Large chunks (1,000 characters and up) carry rich context but embed vaguely,
because one vector must average several ideas; retrieval precision drops. A
useful pattern called small-to-big retrieval indexes small chunks for precise
matching but hands the LLM the larger parent section each small chunk came
from, getting the best of both sizes.

Whatever strategy is chosen, two rules hold. First, store the source and
position of every chunk as metadata, because citations and debugging depend
on it. Second, never tune chunk size by intuition: change it, re-run the
evaluation set, and let hit rate and answer quality decide. Teams that skip
this step routinely ship whichever chunk size the first tutorial they read
happened to use.
