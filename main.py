"""Command-line interface for the RAG Document Q&A system.

Usage:
    python main.py ingest <file> [<file> ...]   Ingest documents into the vector store
    python main.py ask "<question>"             Ask a single question
    python main.py chat                         Start an interactive Q&A session
    python main.py status                       Show how many chunks are stored
"""

import argparse
import sys

from rag.config import Settings
from rag.pipeline import RagPipeline, RagResult
from rag.store import VectorStore


def print_result(result: RagResult) -> None:
    """Print an answer and its source citations."""
    print("\n=== ANSWER ===")
    print(result.answer)
    print("\n=== SOURCES ===")
    if result.sources:
        for i, source in enumerate(result.sources, 1):
            print(f"{i}. {source}")
    else:
        print("(none)")


def cmd_ingest(store: VectorStore, files: list[str]) -> None:
    for file_path in files:
        try:
            added = store.add_document(file_path)
            print(f"Ingested '{file_path}' -> {added} chunks.")
        except (FileNotFoundError, ValueError) as error:
            print(f"Skipped '{file_path}': {error}")


def cmd_ask(settings: Settings, store: VectorStore, question: str) -> None:
    pipeline = RagPipeline(settings, store)
    print_result(pipeline.ask(question))


def cmd_chat(settings: Settings, store: VectorStore) -> None:
    pipeline = RagPipeline(settings, store)
    print("Interactive mode. Type a question, or 'exit' to quit.")
    while True:
        try:
            question = input("\nQuestion> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question or question.lower() in ("exit", "quit"):
            break
        print_result(pipeline.ask(question))
    print("Goodbye.")


def cmd_status(store: VectorStore) -> None:
    print(f"Collection '{store.settings.collection_name}' holds {store.count()} chunks.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag",
        description="Ask questions about your own documents using Retrieval-Augmented Generation.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest", help="Ingest one or more documents")
    ingest.add_argument("files", nargs="+", help="Paths to .txt, .md, .pdf, or .docx files")

    ask = subparsers.add_parser("ask", help="Ask a single question")
    ask.add_argument("question", help="The question to answer")

    subparsers.add_parser("chat", help="Interactive Q&A session")
    subparsers.add_parser("status", help="Show vector store statistics")

    return parser


def main() -> int:
    args = build_parser().parse_args()
    settings = Settings()
    store = VectorStore(settings)

    if args.command == "ingest":
        cmd_ingest(store, args.files)
    elif args.command == "ask":
        cmd_ask(settings, store, args.question)
    elif args.command == "chat":
        cmd_chat(settings, store)
    elif args.command == "status":
        cmd_status(store)
    return 0


if __name__ == "__main__":
    sys.exit(main())
