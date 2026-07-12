"""The core RAG pipeline: retrieve relevant chunks, then generate a grounded answer."""

from dataclasses import dataclass

from openai import OpenAI

from .config import Settings
from .store import VectorStore

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
        self.store = store
        self.client = OpenAI(base_url=settings.base_url, api_key=settings.api_key)

    def ask(self, question: str, top_k: int | None = None) -> RagResult:
        """Answer a question using only the ingested documents.

        Args:
            question: The user's natural-language question.
            top_k: How many chunks to retrieve (defaults to the configured value).

        Returns:
            A RagResult with the generated answer and its sources. If no
            relevant context is found, the answer explains that instead of
            letting the model guess.
        """
        results = self.store.search(question, top_k or self.settings.top_k)
        context, sources = self._format_context(results)

        if not context.strip():
            return RagResult(
                answer="No relevant context was found in the ingested documents.",
                sources=[],
            )

        response = self.client.chat.completions.create(
            model=self.settings.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {question}\nAnswer:",
                },
            ],
            temperature=self.settings.temperature,
            max_tokens=self.settings.max_tokens,
        )

        content = response.choices[0].message.content
        answer = content.strip() if content else "The model returned an empty response."
        return RagResult(answer=answer, sources=sources)

    @staticmethod
    def _format_context(results: dict) -> tuple[str, list[str]]:
        """Join retrieved chunks into one context block and collect unique sources."""
        documents = (results.get("documents") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]

        if not documents:
            return "", []

        context = "\n\n".join(documents)

        sources: list[str] = []
        seen: set[str] = set()
        for meta in metadatas:
            label = f"{meta.get('source', '?')} (chunk {meta.get('chunk', '?')})"
            if label not in seen:
                seen.add(label)
                sources.append(label)

        return context, sources
