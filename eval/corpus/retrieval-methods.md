# Retrieval Methods: Dense, Lexical, Hybrid, and Re-ranking

Dense retrieval ranks chunks by embedding similarity. It excels at
paraphrase: the query and the answer passage can use entirely different
words. Its weakness is exact terms — identifiers, acronyms, version strings,
and rare names often embed poorly, so the passage containing the exact token
the user typed may rank below passages that are merely on-topic.

Lexical retrieval ranks chunks by term overlap. The standard algorithm is
BM25, a scoring function refined over decades of information retrieval
research. BM25 rewards chunks that contain the query's terms, weights rare
terms more heavily than common ones, and normalizes for chunk length. It
finds the exact-token matches that embeddings miss, but it cannot see
synonyms: a query about "cars" will never match a chunk that only says
"automobiles".

Because their failure modes are complementary, production systems run both
and merge the results — this is hybrid retrieval. The merging step must solve
one problem: BM25 scores and vector similarities live on incomparable scales.
Reciprocal Rank Fusion (RRF) sidesteps the problem by ignoring scores
entirely and using only ranks. Each result contributes 1/(k + rank) to its
document's fused score, where k is a damping constant conventionally set to
60. Documents ranked well by both retrievers accumulate score from both lists
and rise to the top. RRF needs no tuning, no score normalization, and no
training data, which is why it is the default fusion method in most search
engines that offer hybrid search.

Hybrid retrieval improves recall — the right chunk is more likely to be
somewhere in the candidate list. Precision — the right chunk being in the
final few results handed to the LLM — is the job of re-ranking. A
cross-encoder re-ranker takes the query and one candidate chunk together as a
single input and outputs a relevance score. Because it reads both texts
jointly, it models word-level interactions that separate embeddings cannot,
making it substantially more accurate than the first-stage retrievers. It is
also far too slow to score a whole corpus, so it is applied only to the top
candidates — a typical setup retrieves 20 to 100 candidates cheaply, then
re-ranks them and keeps the best few. The model used in this project,
ms-marco-MiniLM-L-6-v2, scores a query against 20 candidates in roughly a
tenth of a second on a CPU.

This two-stage design — cheap high-recall candidate generation followed by
expensive high-precision re-ranking — is the same architecture used by web
search engines, and it is the single most reliable retrieval upgrade
available to a RAG system.
