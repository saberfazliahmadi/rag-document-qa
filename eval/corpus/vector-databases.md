# Vector Databases and Approximate Nearest Neighbor Search

A vector database stores embeddings and answers the question "which stored
vectors are closest to this query vector?" Doing this exactly requires
comparing the query against every stored vector, which is fine for a few
thousand chunks but becomes too slow at millions. Vector databases therefore
use approximate nearest neighbor (ANN) indexes that trade a tiny amount of
recall for a large amount of speed.

The most widely used ANN index is HNSW, short for Hierarchical Navigable
Small World. HNSW builds a layered graph over the vectors: upper layers hold
long-range links for coarse navigation, lower layers hold short-range links
for fine search. A query starts at the top layer and greedily walks toward
its nearest neighbors, descending layer by layer.

HNSW behavior is controlled by three main parameters. The parameter M sets
how many links each node keeps; higher M improves recall but grows the index.
The parameter efConstruction controls how thoroughly the graph is built;
a common default is efConstruction set to 200, and raising it improves index
quality at the cost of slower ingestion. The parameter efSearch controls how
wide the search fans out at query time; raising efSearch increases recall and
latency together, and tuning it is the standard way to trade speed for
quality after the index is built.

ChromaDB, the database used in this project, embeds documents automatically
via a pluggable embedding function, persists collections to local disk, and
uses an HNSW index internally. It runs inside the application process, which
removes all operational overhead — there is no separate server to deploy,
which makes it ideal for prototypes, educational projects, and small
production workloads. Dedicated engines such as Qdrant, Weaviate, Milvus, or
pgvector inside PostgreSQL become worthwhile when the corpus grows into
millions of vectors, when several services need to share one index, or when
features like replication and role-based access control are required.

One practical warning: vector similarity always returns *something*. The
nearest neighbor to an off-topic query is still returned, just with a poor
similarity score. Systems that ignore the score and blindly pass the top
results to an LLM will hand it irrelevant context. Checking retrieval scores,
or re-ranking candidates with a stronger model, is the standard defense.
