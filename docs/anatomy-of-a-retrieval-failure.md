# Anatomy of a Retrieval Failure

This is a debugging walkthrough of a real failure in this repository's own
benchmark — not a hypothetical. The evaluation harness reports that the
`hybrid + re-ranking` configuration misses one golden question that plain
`hybrid` gets right. This document traces that question through every
retrieval stage to show **where** it fails, **why** it fails, and **how an
engineer decides what to do about it**.

The point is not this one bug. The point is the method: with stage-by-stage
traces and a golden dataset, a retrieval failure takes minutes to localize.
Without them, it is guesswork.

## The failing question

> **Question:** "How can a malicious document attack a RAG system?"
>
> **Evidence** (the chunk the answer lives in, `production-ops.md_chunk_5`):
> *"…treat document content as untrusted input: a document can contain text
> that tries to override the system prompt ('ignore previous
> instructions…')…"*

Every number below comes from `retrieve_traced()` on the evaluation corpus
(41 chunks). You can reproduce it: ingest `eval/corpus/`, then run
`python main.py ask "How can a malicious document attack a RAG system?" --show-trace`.

## Stage 1 — dense vector search: evidence at rank 6

| Rank | Chunk | Score |
|---|---|---|
| 1 | production-ops.md_chunk_4 | -0.558 |
| 2 | chunking.md_chunk_0 | -0.593 |
| 3 | evaluation.md_chunk_0 | -0.602 |
| 4 | evaluation.md_chunk_6 | -0.613 |
| 5 | production-ops.md_chunk_3 | -0.663 |
| **6** | **production-ops.md_chunk_5 (evidence)** | **-0.688** |

The embedding of the question centers on *attack / malicious / RAG system*.
The evidence chunk talks about *guarding boundaries, capping uploads,
untrusted input* — related, but expressed defensively rather than
offensively. Semantically close, not closest. **Dense-only retrieval with
top_k = 4 misses this question** — which is exactly the miss the baseline
row of the benchmark shows.

## Stage 2 — BM25: evidence at rank 1

| Rank | Chunk | Score |
|---|---|---|
| **1** | **production-ops.md_chunk_5 (evidence)** | **8.64** |
| 2 | evaluation.md_chunk_0 | 8.57 |
| 3 | chunking.md_chunk_0 | 6.56 |

The evidence chunk literally contains the words *document*, *system*,
*prompt*, *RAG* — high term overlap, and BM25 rewards it. This is the
complementary-failure-mode story in one table: the retriever that cannot see
meaning finds what the retriever that can see meaning missed.

## Stage 3 — RRF fusion: evidence at rank 3

| Rank | Chunk | Fused score |
|---|---|---|
| 1 | chunking.md_chunk_0 | 0.0320 |
| 2 | evaluation.md_chunk_0 | 0.0320 |
| **3** | **production-ops.md_chunk_5 (evidence)** | **0.0315** |
| 4 | production-ops.md_chunk_4 | 0.0296 |

Rank 6 (dense) + rank 1 (BM25) fuse to rank 3. Inside the top-4 window —
**the `hybrid` configuration answers this question correctly.**

## Stage 4 — cross-encoder re-ranking: evidence demoted to rank 10

| Rank | Chunk | CE score |
|---|---|---|
| 1 | chunking.md_chunk_0 | -1.94 |
| 2 | production-ops.md_chunk_0 | -4.84 |
| 3 | evaluation.md_chunk_0 | -5.70 |
| 4 | production-ops.md_chunk_3 | -6.94 |
| … | | |
| **10** | **production-ops.md_chunk_5 (evidence)** | **-11.03** |

The re-ranker pushes the evidence from rank 3 to rank 10 — out of the
context window. The final answer degrades to "I don't know."

Why does a *more accurate* model make a *worse* decision here? Look at what
it preferred: `chunking.md_chunk_0` opens with a heading and a definition
("Chunking is the step that splits documents into the retrieval units…").
The cross-encoder was trained on MS MARCO — web search queries paired with
passages that *directly answer* them, which are very often definitional
openings. Our evidence chunk starts mid-thought ("Finally, guard the
boundaries. Cap uploaded file sizes…") and never uses the words *malicious*
or *attack*; the connection between "untrusted input that overrides the
system prompt" and "attack" is an inference the small re-ranker does not
make. It is not broken — it is out of domain for this phrasing.

## What are the options?

This is the actual engineering decision, with the reasoning:

1. **Raise `top_k`.** Evidence is at rank 10; `top_k=12` would include it. But
   that triples prompt size for every query to fix one, and dilutes context
   with weaker chunks. Rejected.
2. **Skip re-ranking.** Hybrid alone scores 1.00 hit rate. But MRR drops from
   0.92 to 0.87 — every other question gets worse ordering to fix this one.
   Rejected; the aggregate matters more than the single case.
3. **A larger or domain-tuned re-ranker.** The correct long-term fix — a
   bigger cross-encoder likely makes the malicious→untrusted-input inference.
   Costs latency and (for fine-tuning) labeled data. Deferred, and the eval
   harness is exactly the tool that will prove whether it earns its cost.
4. **Accept and document.** 0.96 hit rate with the best MRR, failure mode
   understood and written down. **Chosen** — with the failing question kept
   in the golden set so any future change that fixes it shows up as a
   measured improvement.

## The transferable lessons

- **A better model is not better everywhere.** Aggregate metrics can improve
  while individual queries regress. Only an evaluation set makes this
  visible; only traces make it explainable.
- **Failures localize to stages.** "Retrieval is bad" is not actionable.
  "The re-ranker demotes evidence phrased defensively when the query is
  phrased offensively" is: it names the component, the mechanism, and the
  fix.
- **Know your models' training data.** The re-ranker's preference for
  definitional passages is an MS MARCO fingerprint. Every off-the-shelf
  model imports its training distribution's biases into your system.
- **Keep your failures.** A golden-set question you cannot currently pass is
  not an embarrassment; it is a free regression test for the next model you
  try.
