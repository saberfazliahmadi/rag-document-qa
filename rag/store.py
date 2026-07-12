"""Vector store wrapper around ChromaDB, plus a BM25 keyword index.

Dense vector search finds text that *means* the same as the query; BM25 finds
text that *shares exact terms* with it (model names, error codes, acronyms).
Production systems almost always run both — each catches queries the other
misses. The two indexes are kept side by side here and merged by the caller
(see rag/ranking.py).
"""

import os
import re
from dataclasses import dataclass

import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi

from .config import Settings
from .loaders import read_document
from .splitter import split_text


@dataclass(frozen=True)
class RetrievedChunk:
    """One retrieved chunk with its provenance and retrieval score.

    The meaning of `score` depends on the stage that produced it (vector
    similarity, BM25, RRF, or cross-encoder) — scores are comparable within
    one result list, not across stages.
    """

    id: str
    text: str
    source: str
    chunk_index: int
    score: float

    @property
    def citation(self) -> str:
        return f"{self.source} (chunk {self.chunk_index})"


def _tokenize(text: str) -> list[str]:
    """Lowercase word tokenizer for BM25. Keeps alphanumerics and hyphens

    so model names like 'yolov7-e6' survive as single tokens.
    """
    return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower())


class VectorStore:
    """Persistent ChromaDB collection with a parallel in-memory BM25 index.

    The BM25 index is rebuilt from the collection at startup and after each
    ingestion. That is fine at this project's scale; a production system
    would keep the keyword index in a search engine (e.g. OpenSearch).
    """

    def __init__(self, settings: Settings, client: chromadb.ClientAPI | None = None):
        self.settings = settings
        client = client or chromadb.PersistentClient(path=settings.persist_dir)
        embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=settings.embedding_model
        )
        self.collection = client.get_or_create_collection(
            name=settings.collection_name,
            embedding_function=embedding_fn,
        )
        self._bm25 = None
        self._bm25_chunks: list[RetrievedChunk] = []
        self._rebuild_bm25()

    def add_document(self, file_path: str) -> int:
        """Read, chunk, embed, and store a document.

        Returns:
            The number of chunks added to the collection.
        """
        content = read_document(file_path)
        chunks = split_text(
            content,
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )
        if not chunks:
            return 0

        file_name = os.path.basename(file_path)
        ids = [f"{file_name}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [{"source": file_name, "chunk": i} for i in range(len(chunks))]

        # Upsert so re-ingesting the same file updates chunks instead of duplicating them.
        self.collection.upsert(documents=chunks, metadatas=metadatas, ids=ids)
        self._rebuild_bm25()
        return len(chunks)

    def search_dense(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """Semantic search: rank chunks by embedding similarity to the query."""
        results = self.collection.query(
            query_texts=[query],
            n_results=min(top_k, max(self.count(), 1)),
            include=["documents", "metadatas", "distances"],
        )
        documents = (results.get("documents") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]
        ids = (results.get("ids") or [[]])[0]

        return [
            RetrievedChunk(
                id=chunk_id,
                text=text,
                source=meta.get("source", "?"),
                chunk_index=meta.get("chunk", -1),
                # Chroma returns a distance (lower = closer); negate so that
                # a higher score always means more relevant, like every
                # other stage in the pipeline.
                score=-distance,
            )
            for chunk_id, text, meta, distance in zip(
                ids, documents, metadatas, distances, strict=True
            )
        ]

    def search_keyword(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """Lexical search: rank chunks by BM25 term overlap with the query."""
        if self._bm25 is None:
            return []

        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        results = []
        for index in ranked[:top_k]:
            if scores[index] <= 0:
                break  # remaining chunks share no terms with the query
            chunk = self._bm25_chunks[index]
            results.append(
                RetrievedChunk(
                    id=chunk.id,
                    text=chunk.text,
                    source=chunk.source,
                    chunk_index=chunk.chunk_index,
                    score=float(scores[index]),
                )
            )
        return results

    def count(self) -> int:
        """Return the number of chunks currently stored."""
        return self.collection.count()

    def _rebuild_bm25(self) -> None:
        records = self.collection.get(include=["documents", "metadatas"])
        documents = records.get("documents") or []
        metadatas = records.get("metadatas") or []
        ids = records.get("ids") or []

        self._bm25_chunks = [
            RetrievedChunk(
                id=chunk_id,
                text=text,
                source=meta.get("source", "?"),
                chunk_index=meta.get("chunk", -1),
                score=0.0,
            )
            for chunk_id, text, meta in zip(ids, documents, metadatas, strict=True)
        ]
        corpus = [_tokenize(chunk.text) for chunk in self._bm25_chunks]
        self._bm25 = BM25Okapi(corpus) if corpus else None
