"""Streamlit chat client for the RAG Document Q&A API.

Start the API first, then this app:
    uvicorn api:app --reload
    streamlit run app.py

The app talks to the API over HTTP only — it contains no RAG logic itself.
"""

import json
import os

import requests
import streamlit as st

API_URL = os.getenv("RAG_API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="RAG Document Q&A", page_icon="📄", layout="centered")


def api_status() -> dict | None:
    """Return API status, or None when the API is unreachable."""
    try:
        response = requests.get(f"{API_URL}/status", timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None


def upload_document(file) -> dict:
    """Send an uploaded file to the /ingest endpoint."""
    response = requests.post(
        f"{API_URL}/ingest",
        files={"file": (file.name, file.getvalue())},
        timeout=300,
    )
    response.raise_for_status()
    return response.json()


def stream_answer(question: str):
    """Yield answer tokens from the /ask/stream endpoint; store sources in session state."""
    with requests.post(
        f"{API_URL}/ask/stream",
        json={"question": question},
        stream=True,
        timeout=300,
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            event = json.loads(line[len("data: "):])
            if event["type"] == "sources":
                st.session_state.last_sources = event["sources"]
            elif event["type"] == "token":
                yield event["text"]


# --- Sidebar: connection status and document upload ------------------------------

with st.sidebar:
    st.title("📄 RAG Document Q&A")

    status = api_status()
    if status is None:
        st.error(f"API not reachable at {API_URL}. Start it with:\n\n`uvicorn api:app`")
        st.stop()

    st.success(f"Connected — {status['chunks']} chunks indexed")
    st.caption(f"LLM: `{status['llm_model']}`")
    st.caption(f"Embeddings: `{status['embedding_model']}`")

    st.divider()
    st.subheader("Add a document")
    uploaded = st.file_uploader("PDF, DOCX, TXT, or MD", type=["pdf", "docx", "txt", "md"])
    if uploaded and st.button("Ingest", use_container_width=True):
        with st.spinner("Reading, chunking, and embedding..."):
            try:
                result = upload_document(uploaded)
                st.success(f"Added {result['chunks_added']} chunks from {result['file']}")
            except requests.RequestException as error:
                st.error(f"Ingestion failed: {error}")


# --- Chat -----------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Sources"):
                for i, source in enumerate(message["sources"], 1):
                    st.markdown(f"{i}. `{source}`")

if question := st.chat_input("Ask a question about your documents"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        st.session_state.last_sources = []
        try:
            answer = st.write_stream(stream_answer(question))
        except requests.RequestException as error:
            answer = f"Request failed: {error}"
            st.error(answer)

        sources = st.session_state.get("last_sources", [])
        if sources:
            with st.expander("Sources"):
                for i, source in enumerate(sources, 1):
                    st.markdown(f"{i}. `{source}`")

    st.session_state.messages.append(
        {"role": "assistant", "content": str(answer), "sources": sources}
    )
