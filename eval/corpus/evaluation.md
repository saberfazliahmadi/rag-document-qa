# Evaluating RAG Systems

A RAG system without evaluation is a demo. Every meaningful decision —
chunk size, embedding model, search mode, how many chunks to retrieve —
changes answer quality in ways that eyeballing a few queries cannot detect.
Evaluation turns those decisions from guesses into measurements.

The foundation is a golden dataset: a set of questions paired with the
evidence that answers them, written against a fixed corpus. Even twenty to
fifty carefully written questions expose most retrieval problems. Good golden
sets mix question styles: paraphrase questions that share no words with the
answer passage, exact-term questions built around identifiers or acronyms,
and questions whose answers span multiple documents.

Retrieval quality is measured first, because retrieval failures poison
everything downstream. Two metrics cover most needs. Hit rate at k asks: for
what fraction of questions does the evidence appear in the top k retrieved
chunks? Mean Reciprocal Rank (MRR) is more sensitive: it awards 1/rank for
the first relevant chunk — a hit at rank 1 scores 1.0, at rank 4 scores 0.25 —
and averages over all questions, rewarding systems that put the evidence
first rather than merely somewhere in the window. Both metrics are
deterministic, cost nothing to compute, and need no LLM, which makes them
perfect for continuous integration: a pull request that degrades retrieval
fails the pipeline before a human ever reviews it.

Generation quality sits on top. Faithfulness asks whether every claim in the
generated answer is supported by the retrieved context; an unfaithful answer
is a hallucination even when it happens to be factually true. Answer
relevance asks whether the answer actually addresses the question. These are
usually scored by a strong LLM acting as a judge, since human review does not
scale. LLM-as-judge scores are noisy and drift with the judge model, so teams
treat them as trend indicators rather than absolute truths, and re-anchor
them periodically against a small human-labeled sample.

The discipline that matters most is comparing configurations, not admiring a
single score. A table showing hit rate and MRR for dense retrieval alone,
hybrid retrieval, and hybrid plus re-ranking — measured on the same golden
set — tells you exactly what each component buys and what it costs. That
habit, measuring every stage instead of assuming it helps, is the clearest
marker separating production RAG engineering from tutorial code.
