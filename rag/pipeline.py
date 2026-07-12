"""The core RAG pipeline: retrieve relevant chunks, then generate a grounded answer."""

from collections.abc import Iterator
from dataclasses import dataclass

from openai import OpenAI

from .config import Settings
from .retriever import Retriever
from .store import RetrievedChunk, VectorStore

SYSTEM_PROMPT = (
    "You are a helpful assistant for retrieval-augmented generation (RAG).\n"
    "Answer ONLY using the provided context. "
    "If the answer is not found in the context, say: "
    "'I don't know based on the provided documents.'"
)


@dataclass
class RagResult:
    """The answer to a question together with its source citations."""

    answer: str
    sources: list[str]


class RagPipeline:
    """Retrieval-Augmented Generation over a local vector store."""

    def __init__(self, settings: Settings, store: VectorStore):
        settings.require_api_key()
        self.settings = settings
        self.retriever = Retriever(settings, store)
        self.client = OpenAI(base_url=settings.base_url, api_key=settings.api_key)

    def ask(self, question: str, top_k: int | None = None) -> RagResult:
        """Answer a question using only the ingested documents.

        Returns:
            A RagResult with the generated answer and its sources. If no
            relevant context is found, the answer explains that instead of
            letting the model guess.
        """
        chunks = self.retriever.retrieve(question, top_k)
        if not chunks:
            return RagResult(
                answer="No relevant context was found in the ingested documents.",
                sources=[],
            )

        response = self.client.chat.completions.create(
            model=self.settings.model,
            messages=self._build_messages(chunks, question),
            temperature=self.settings.temperature,
            max_tokens=self.settings.max_tokens,
        )

        content = response.choices[0].message.content
        answer = content.strip() if content else "The model returned an empty response."
        return RagResult(answer=answer, sources=self._citations(chunks))

    def ask_stream(self, question: str, top_k: int | None = None) -> tuple[Iterator[str], list[str]]:
        """Answer a question, streaming the answer token by token.

        Retrieval happens up front, so the sources are known before
        generation starts. Returns the token iterator and the source list.
        """
        chunks = self.retriever.retrieve(question, top_k)
        if not chunks:
            no_context = "No relevant context was found in the ingested documents."
            return iter([no_context]), []

        def token_stream() -> Iterator[str]:
            stream = self.client.chat.completions.create(
                model=self.settings.model,
                messages=self._build_messages(chunks, question),
                temperature=self.settings.temperature,
                max_tokens=self.settings.max_tokens,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    yield delta

        return token_stream(), self._citations(chunks)

    @staticmethod
    def _build_messages(chunks: list[RetrievedChunk], question: str) -> list[dict]:
        """Assemble the chat messages sent to the LLM."""
        context = "\n\n".join(chunk.text for chunk in chunks)
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}\nAnswer:",
            },
        ]

    @staticmethod
    def _citations(chunks: list[RetrievedChunk]) -> list[str]:
        """Unique citations, preserving retrieval order."""
        seen: set[str] = set()
        citations = []
        for chunk in chunks:
            if chunk.citation not in seen:
                seen.add(chunk.citation)
                citations.append(chunk.citation)
        return citations
