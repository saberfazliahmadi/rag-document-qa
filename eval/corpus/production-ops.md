# Running RAG in Production

Moving a RAG system from laptop to production changes which problems matter.
Correctness stops being the only concern; latency, cost, and failure handling
join it.

Latency is dominated by the LLM call, which routinely takes several seconds.
Streaming is the standard mitigation: the server forwards tokens to the
client as they are generated, typically over Server-Sent Events (SSE), so the
user sees the answer forming within a second instead of staring at a spinner.
Retrieval itself is rarely the bottleneck — embedding one query and searching
a local index takes tens of milliseconds — but the re-ranking stage adds
around a hundred milliseconds and the first-stage candidate count should be
kept reasonable.

Cost control starts with caching. Embedding the same document twice is pure
waste, so ingestion should be idempotent — this project uses upserts keyed on
file name and chunk index for that reason. Query-side, a semantic cache can
return a stored answer when a new question is close enough to a previously
answered one, cutting LLM spend substantially on repetitive workloads.

External LLM APIs fail in mundane ways: HTTP 429 responses when the rate
limit is exceeded, HTTP 503 during provider incidents, and occasional
truncated or empty completions. Production clients wrap LLM calls with
timeouts, retries with exponential backoff for 429 and 5xx errors, and a
fallback model so that a provider incident degrades quality instead of
availability. Empty responses must be handled explicitly — returning a clear
error to the user beats returning a blank answer.

Observability for RAG means logging more than requests and status codes. The
useful trace for one question records the query, the retrieved chunk IDs and
their scores, the prompt token count, the model, the latency of each stage,
and the final answer. With that trace, "why did it answer this?" becomes a
lookup instead of an investigation. Retrieval score distributions are worth
monitoring over time: a slow drift downward usually means the corpus and the
questions users actually ask are moving apart — a signal to re-index, re-chunk,
or expand the document set.

Finally, guard the boundaries. Cap uploaded file sizes, validate file types
before parsing, and treat document content as untrusted input: a document can
contain text that tries to override the system prompt ("ignore previous
instructions..."), so prompts should clearly separate instructions from
retrieved context, and answers should be constrained to that context.
