# RAG Document Q&A

Ask questions about your own documents and get answers with source citations.

This project is a complete Retrieval-Augmented Generation (RAG) pipeline built in Python. You give it documents (PDF, Word, text, or Markdown). It stores them in a local vector database. Then you ask questions in plain language, and a large language model answers **using only your documents** — and tells you exactly which parts of which files the answer came from.

## Why RAG?

Large language models are powerful, but they have two problems: they sometimes invent facts ("hallucination"), and they know nothing about your private files. RAG solves both. The model is only allowed to answer from text retrieved out of your documents, and every answer cites its sources. If the answer is not in your documents, the system says so instead of guessing.

## Key Features

- **Multi-format ingestion** — reads `.pdf`, `.docx`, `.txt`, and `.md` files
- **Semantic search** — finds relevant text by meaning, not just keywords, using sentence-transformer embeddings
- **Grounded answers** — the LLM is instructed to answer only from retrieved context
- **Source citations** — every answer lists the file and chunk it was built from
- **Persistent local vector store** — ChromaDB keeps your index on disk between runs; re-ingesting a file updates it instead of duplicating it
- **No secrets in code** — all configuration comes from environment variables
- **Simple CLI** — ingest, ask a single question, or chat interactively

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| Vector database | ChromaDB (persistent, local) |
| Embeddings | Sentence-Transformers (`all-MiniLM-L6-v2`) |
| LLM access | OpenRouter (OpenAI-compatible API) |
| Document parsing | PyPDF2, python-docx |
| Configuration | python-dotenv |

## Architecture

The pipeline has three stages:

```
 INGESTION                      RETRIEVAL                    GENERATION
┌─────────────────┐          ┌─────────────────┐          ┌─────────────────┐
│ Read document    │          │ Embed the       │          │ Build prompt:    │
│ (PDF/DOCX/TXT)   │          │ user question   │          │ context +        │
│       │          │          │       │         │          │ question         │
│ Split into       │          │ Query ChromaDB  │          │       │          │
│ overlapping      │  ─────▶  │ for the most    │  ─────▶  │ LLM answers      │
│ chunks           │          │ similar chunks  │          │ from context     │
│       │          │          │                 │          │ only             │
│ Embed + store    │          │ Return chunks   │          │       │          │
│ in ChromaDB      │          │ with metadata   │          │ Cite sources     │
└─────────────────┘          └─────────────────┘          └─────────────────┘
```

Each module owns one stage:

- `rag/loaders.py` — reads files into plain text
- `rag/splitter.py` — cuts text into overlapping chunks
- `rag/store.py` — embeds chunks and stores/queries them in ChromaDB
- `rag/pipeline.py` — retrieves context and generates the cited answer
- `rag/config.py` — loads all settings from the environment
- `main.py` — the command-line interface

## Installation

**1. Clone the repository**

```bash
git clone https://github.com/saberfazliahmadi/rag-document-qa.git
cd rag-document-qa
```

**2. Create a virtual environment and install dependencies**

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

**3. Configure your API key**

```bash
# Windows:
copy .env.example .env
# macOS / Linux:
cp .env.example .env
```

Open `.env` and set `OPENROUTER_API_KEY` to your key from [openrouter.ai/keys](https://openrouter.ai/keys). The default model is free to use.

## Usage

**Ingest documents** (any mix of PDF, DOCX, TXT, MD):

```bash
python main.py ingest data/sample.txt
python main.py ingest path/to/paper.pdf path/to/notes.docx
```

**Ask a single question:**

```bash
python main.py ask "What are the main benefits of RAG?"
```

**Chat interactively:**

```bash
python main.py chat
```

**Check what is stored:**

```bash
python main.py status
```

## Example Workflow

```text
$ python main.py ingest data/sample.txt
Ingested 'data/sample.txt' -> 4 chunks.

$ python main.py ask "What are the main benefits of RAG?"

=== ANSWER ===
The main benefits of RAG are accuracy and traceability. Because the model
answers from real documents, it is far less likely to invent facts
(hallucination), and every answer can cite the exact chunks it was built
from, so users can verify the sources themselves.

=== SOURCES ===
1. sample.txt (chunk 2)
2. sample.txt (chunk 1)
```

The first run downloads the embedding model (about 90 MB), so it takes a little longer. Later runs start quickly.

## Folder Structure

```
rag-document-qa/
├── main.py              # Command-line interface
├── rag/
│   ├── __init__.py
│   ├── config.py        # Settings loaded from environment variables
│   ├── loaders.py       # PDF / DOCX / TXT / MD readers
│   ├── splitter.py      # Overlapping text chunking
│   ├── store.py         # ChromaDB vector store wrapper
│   └── pipeline.py      # Retrieval + generation with citations
├── data/
│   └── sample.txt       # Small demo document
├── .env.example         # Configuration template (copy to .env)
├── requirements.txt
├── LICENSE
└── README.md
```

The vector database is written to `chroma_db/` at runtime and is not committed to the repository.

## Configuration

All settings have sensible defaults and can be overridden in `.env`:

| Variable | Default | Meaning |
|---|---|---|
| `OPENROUTER_API_KEY` | — (required) | Your OpenRouter API key |
| `LLM_MODEL` | `meta-llama/llama-3.3-70b-instruct:free` | Chat model used for answers |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformer embedding model |
| `CHUNK_SIZE` | `500` | Characters per chunk |
| `CHUNK_OVERLAP` | `100` | Characters shared between adjacent chunks |
| `TOP_K` | `4` | Chunks retrieved per question |
| `TEMPERATURE` | `0.2` | Lower = more factual answers |
| `MAX_TOKENS` | `512` | Maximum answer length |

## Future Improvements

- Hybrid search (combine keyword and semantic retrieval)
- Answer evaluation with Ragas (faithfulness and relevance scores)
- A web interface with Streamlit or FastAPI
- Support for more formats (HTML, CSV) and OCR for scanned PDFs
- Sentence-aware chunking instead of fixed character windows

## License

This project is licensed under the [MIT License](LICENSE).
