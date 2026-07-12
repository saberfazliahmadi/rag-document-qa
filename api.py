"""REST API for the RAG Document Q&A system, built with FastAPI.

Run with:
    uvicorn api:app --reload

Interactive documentation is served at http://127.0.0.1:8000/docs
"""

import json
import os
import tempfile
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from rag.config import Settings
from rag.loaders import SUPPORTED_EXTENSIONS
from rag.pipeline import RagPipeline
from rag.store import VectorStore


# --- Request / response schemas -------------------------------------------------


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The question to answer")
    top_k: int | None = Field(None, ge=1, le=20, description="How many chunks to retrieve")


class AskResponse(BaseModel):
    answer: str
    sources: list[str]


class IngestResponse(BaseModel):
    file: str
    chunks_added: int


class StatusResponse(BaseModel):
    collection: str
    chunks: int
    embedding_model: str
    llm_model: str


# --- Application setup ----------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the embedding model and vector store once, at startup."""
    settings = Settings()
    store = VectorStore(settings)
    app.state.settings = settings
    app.state.store = store
    app.state.pipeline = RagPipeline(settings, store)
    yield


app = FastAPI(
    title="RAG Document Q&A API",
    description=(
        "Ask questions about your own documents. Answers are grounded in "
        "retrieved context and always include source citations."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# --- Endpoints ------------------------------------------------------------------


@app.get("/status", response_model=StatusResponse, tags=["monitoring"])
def status() -> StatusResponse:
    """Report what is currently stored and which models are configured."""
    settings: Settings = app.state.settings
    return StatusResponse(
        collection=settings.collection_name,
        chunks=app.state.store.count(),
        embedding_model=settings.embedding_model,
        llm_model=settings.model,
    )


@app.post("/ingest", response_model=IngestResponse, tags=["ingestion"])
async def ingest(file: UploadFile) -> IngestResponse:
    """Upload a document (.pdf, .docx, .txt, .md) and add it to the knowledge base."""
    filename = file.filename or "upload"
    extension = os.path.splitext(filename)[1].lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file format '{extension}'. "
            f"Supported: {', '.join(SUPPORTED_EXTENSIONS)}",
        )

    # Persist the upload to a temporary file so the loaders can read it by path.
    # The original filename is kept so citations point to a recognizable source.
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = os.path.join(tmp_dir, os.path.basename(filename))
        with open(tmp_path, "wb") as tmp_file:
            tmp_file.write(await file.read())
        chunks_added = app.state.store.add_document(tmp_path)

    return IngestResponse(file=filename, chunks_added=chunks_added)


@app.post("/ask", response_model=AskResponse, tags=["question-answering"])
def ask(request: AskRequest) -> AskResponse:
    """Answer a question using only the ingested documents."""
    result = app.state.pipeline.ask(request.question, request.top_k)
    return AskResponse(answer=result.answer, sources=result.sources)


@app.post("/ask/stream", tags=["question-answering"])
def ask_stream(request: AskRequest) -> StreamingResponse:
    """Answer a question, streaming the answer token by token (Server-Sent Events).

    Event order: one `sources` event first, then `token` events, then `done`.
    """
    tokens, sources = app.state.pipeline.ask_stream(request.question, request.top_k)

    def event_stream():
        yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"
        for token in tokens:
            yield f"data: {json.dumps({'type': 'token', 'text': token})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
