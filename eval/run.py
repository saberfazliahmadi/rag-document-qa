"""Retrieval evaluation harness.

Builds a fresh in-memory index over eval/corpus, then measures every
retrieval configuration against the golden dataset:

    python -m eval.run              # hit rate + MRR for all configurations
    python -m eval.run --top-k 8    # evaluate a wider context window

The corpus and questions are fixed, the metrics are deterministic, and no
LLM or API key is involved — so two runs on two machines produce identical
numbers, and a regression fails loudly in CI.
"""

import argparse
import dataclasses
import glob
import json
import os

import chromadb

from rag.config import Settings
from rag.retriever import Retriever
from rag.store import VectorStore

from .metrics import first_relevant_rank, hit_rate, mean_reciprocal_rank

EVAL_DIR = os.path.dirname(__file__)
CORPUS_DIR = os.path.join(EVAL_DIR, "corpus")
GOLDEN_PATH = os.path.join(EVAL_DIR, "golden.jsonl")

# Each configuration is (label, search_mode, use_reranker).
CONFIGURATIONS = [
    ("dense (baseline)", "dense", False),
    ("hybrid (BM25 + RRF)", "hybrid", False),
    ("hybrid + re-ranking", "hybrid", True),
]


def load_golden() -> list[dict]:
    with open(GOLDEN_PATH, encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def build_store(settings: Settings) -> VectorStore:
    """Index the evaluation corpus into an in-memory ChromaDB instance."""
    store = VectorStore(settings, client=chromadb.EphemeralClient())
    for path in sorted(glob.glob(os.path.join(CORPUS_DIR, "*.md"))):
        store.add_document(path)
    return store


def check_answerable(store: VectorStore, golden: list[dict]) -> None:
    """Fail fast if any golden evidence string got split across chunks —

    a broken golden set would silently corrupt every metric below.
    """
    records = store.collection.get(include=["documents"])
    chunks = records.get("documents") or []
    for item in golden:
        if first_relevant_rank(chunks, item["evidence"]) is None:
            raise SystemExit(
                f"Golden set error: evidence not found in any chunk: {item['evidence']!r}"
            )


def evaluate(retriever: Retriever, golden: list[dict], top_k: int) -> dict:
    ranks: list[int | None] = []
    misses: list[str] = []
    for item in golden:
        chunks = retriever.retrieve(item["question"], top_k=top_k)
        rank = first_relevant_rank([chunk.text for chunk in chunks], item["evidence"])
        ranks.append(rank)
        if rank is None:
            misses.append(item["question"])
    return {
        "hit_rate": hit_rate(ranks),
        "mrr": mean_reciprocal_rank(ranks),
        "misses": misses,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval configurations.")
    parser.add_argument("--top-k", type=int, default=4, help="Context window size (default 4)")
    parser.add_argument("--verbose", action="store_true", help="List missed questions")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if the default configuration falls below the "
        "regression thresholds (used as a CI gate)",
    )
    args = parser.parse_args()

    golden = load_golden()
    base = Settings()
    store = build_store(base)
    check_answerable(store, golden)

    print(f"\nCorpus: {store.count()} chunks | Golden questions: {len(golden)} | top_k={args.top_k}\n")
    print(f"{'configuration':<24} {'hit_rate@' + str(args.top_k):>12} {'MRR':>8}")
    print("-" * 46)

    results = {}
    for label, mode, rerank in CONFIGURATIONS:
        settings = dataclasses.replace(base, search_mode=mode, use_reranker=rerank)
        retriever = Retriever(settings, store)
        result = evaluate(retriever, golden, args.top_k)
        results[label] = result
        print(f"{label:<24} {result['hit_rate']:>12.2f} {result['mrr']:>8.2f}")
        if args.verbose and result["misses"]:
            for question in result["misses"]:
                print(f"    missed: {question}")

    print()

    if args.check:
        # Regression gate for the shipped default (hybrid + re-ranking).
        # Thresholds sit just below the current measured scores, so any
        # change that hurts retrieval fails CI while normal noise passes.
        final = results["hybrid + re-ranking"]
        min_hit_rate, min_mrr = 0.90, 0.85
        if final["hit_rate"] < min_hit_rate or final["mrr"] < min_mrr:
            raise SystemExit(
                f"Retrieval regression: hit_rate={final['hit_rate']:.2f} "
                f"(min {min_hit_rate}), mrr={final['mrr']:.2f} (min {min_mrr})"
            )
        print(f"Regression gate passed (hit_rate >= {min_hit_rate}, MRR >= {min_mrr}).")


if __name__ == "__main__":
    main()
