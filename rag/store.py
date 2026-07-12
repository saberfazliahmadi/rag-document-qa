"""Vector store wrapper around ChromaDB."""

import os

import chromadb
from chromadb.utils import embedding_functions

from .config import Settings
from .loaders import read_document
from .splitter import split_text


class VectorStore:
    """Persistent ChromaDB collection with sentence-transformer embeddings."""

    def __init__(self, settings: Settings):
        self.settings = settings
        client = chromadb.PersistentClient(path=settings.persist_dir)
        embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=settings.embedding_model
        )
        self.collection = client.get_or_create_collection(
            name=settings.collection_name,
            embedding_function=embedding_fn,
        )

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
        return len(chunks)

    def search(self, query: str, top_k: int = 4) -> dict:
        """Return the most semantically similar chunks for a query."""
        return self.collection.query(
            query_texts=[query],
            n_results=top_k,
            include=["documents", "metadatas"],
        )

    def count(self) -> int:
        """Return the number of chunks currently stored."""
        return self.collection.count()
