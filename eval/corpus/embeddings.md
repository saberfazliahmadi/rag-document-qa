# Text Embeddings for Retrieval

Text embeddings are dense numerical vectors that represent the meaning of a
piece of text. Two texts with similar meaning get vectors that lie close
together in the vector space, even when they share no words. This is what
makes semantic search possible: a query about "reducing made-up answers" can
find a passage about "mitigating hallucination" because their embeddings are
close, despite zero keyword overlap.

The embedding model used in this project is all-MiniLM-L6-v2, a small
sentence-transformer that produces 384-dimensional vectors. It runs
comfortably on a CPU, embeds roughly one thousand short chunks per minute on
a laptop, and is a strong default for English text. Larger models such as
bge-large or e5-large produce 1024-dimensional vectors and score higher on
retrieval benchmarks, but they are slower and need more memory. The right
choice depends on the corpus and the latency budget, and should be decided
with an evaluation set rather than by leaderboard position alone.

Similarity between embeddings is usually measured with cosine similarity,
which compares the angle between two vectors and ignores their length.
Sentence-transformer models typically normalize their output vectors to unit
length, which makes cosine similarity and Euclidean distance rank results
identically.

A critical and often overlooked rule: the query and the documents must be
embedded with the same model. Mixing models — for example, indexing documents
with one embedding model and embedding queries with a newer one — silently
breaks retrieval, because the two vector spaces are unrelated. For the same
reason, changing the embedding model requires re-indexing the entire corpus.

Embeddings also have blind spots. They compress meaning, so they can lose
exact identifiers: product codes, version numbers, personal names, and rare
acronyms often embed poorly because the model saw them rarely during
training. A query for a specific model number can retrieve passages about
similar products instead of the exact one. This weakness is the main reason
production systems pair vector search with lexical search instead of relying
on embeddings alone.
