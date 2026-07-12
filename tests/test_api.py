"""API contract tests. The store and pipeline are replaced with fakes so the
tests exercise routing, validation, and serialization without downloading
models or calling an LLM."""

import pytest
from fastapi.testclient import TestClient

from api import app
from rag.pipeline import RagResult


class FakeStore:
    settings = None

    def add_document(self, path: str) -> int:
        return 3

    def count(self) -> int:
        return 42


class FakeSettings:
    collection_name = "documents"
    embedding_model = "fake-embedder"
    model = "fake-llm"


class FakePipeline:
    def ask(self, question: str, top_k=None) -> RagResult:
        return RagResult(answer=f"answer to: {question}", sources=["doc.txt (chunk 0)"])

    def ask_stream(self, question: str, top_k=None):
        return iter(["streamed ", "answer"]), ["doc.txt (chunk 0)"]


@pytest.fixture()
def client():
    app.state.settings = FakeSettings()
    app.state.store = FakeStore()
    app.state.pipeline = FakePipeline()
    return TestClient(app)


def test_status(client):
    response = client.get("/status")
    assert response.status_code == 200
    body = response.json()
    assert body["chunks"] == 42
    assert body["collection"] == "documents"


def test_ask_returns_answer_with_sources(client):
    response = client.post("/ask", json={"question": "What is RAG?"})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "answer to: What is RAG?"
    assert body["sources"] == ["doc.txt (chunk 0)"]


def test_ask_rejects_empty_question(client):
    assert client.post("/ask", json={"question": ""}).status_code == 422


def test_ask_rejects_out_of_range_top_k(client):
    assert client.post("/ask", json={"question": "q", "top_k": 99}).status_code == 422


def test_ingest_accepts_supported_format(client):
    response = client.post("/ingest", files={"file": ("notes.txt", b"some text")})
    assert response.status_code == 200
    assert response.json() == {"file": "notes.txt", "chunks_added": 3}


def test_ingest_rejects_unsupported_format(client):
    response = client.post("/ingest", files={"file": ("data.csv", b"a,b")})
    assert response.status_code == 415


def test_stream_emits_sources_then_tokens_then_done(client):
    with client.stream("POST", "/ask/stream", json={"question": "q"}) as response:
        assert response.status_code == 200
        events = [line for line in response.iter_lines() if line.startswith("data: ")]
    assert '"type": "sources"' in events[0]
    assert '"type": "token"' in events[1]
    assert '"type": "done"' in events[-1]
