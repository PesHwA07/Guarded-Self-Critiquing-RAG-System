"""LangGraph pipeline — linear retrieve → generate chain (v1).

Wires the retriever and generator into a LangGraph StateGraph so we have
a solid foundation to add the critic/reformulate loop in Week 2.  For now
it's a simple two-node linear chain, but the graph structure makes it trivial
to insert new nodes later.

Usage:
    from rag.graph import build_graph, run_query

    # Quick one-shot
    result = run_query("How do I send a POST request?")
    print(result["answer"])

    # Or build the graph and invoke directly
    graph = build_graph()
    state = graph.invoke({"question": "..."})
"""

from __future__ import annotations

import logging
import time
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from rag.generator import GeneratorResponse, generate_answer
from rag.retriever import (
    RetrievalResult,
    VectorStoreRetriever,
    format_context,
)

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# State definition
# ------------------------------------------------------------------


class RAGState(TypedDict, total=False):
    """State that flows through the RAG graph.

    Designed to be extended in Week 2 with critic/reformulate fields.
    Using ``total=False`` so nodes only need to set the fields they produce.
    """

    # Input
    question: str

    # Retriever output
    retrieved_chunks: list[RetrievalResult]
    context: str

    # Generator output
    generator_response: GeneratorResponse
    answer: str
    sources_used: list[int]
    confidence: str

    # Metadata
    latency_ms: float
    error: str | None

    # --- Week 2 additions (reserved) ---
    # critic_verdict: str
    # critic_reasoning: str
    # retry_count: int
    # reformulated_query: str


# ------------------------------------------------------------------
# Graph nodes
# ------------------------------------------------------------------

# Module-level retriever singleton (lazy-initialized)
_retriever: VectorStoreRetriever | None = None


def _get_retriever() -> VectorStoreRetriever:
    """Lazy-init the retriever singleton."""
    global _retriever
    if _retriever is None:
        _retriever = VectorStoreRetriever()
    return _retriever


def retrieve_node(state: RAGState) -> dict[str, Any]:
    """Retrieve relevant document chunks for the question.

    Reads: ``state["question"]``
    Writes: ``retrieved_chunks``, ``context``
    """
    question = state["question"]
    logger.info("🔍 Retrieving chunks for: '%s'", question)

    retriever = _get_retriever()
    results = retriever.retrieve(question)
    context = format_context(results)

    logger.info("📄 Retrieved %d chunks.", len(results))
    return {
        "retrieved_chunks": results,
        "context": context,
    }


def generate_node(state: RAGState) -> dict[str, Any]:
    """Generate an answer from the retrieved context.

    Reads: ``state["question"]``, ``state["context"]``
    Writes: ``generator_response``, ``answer``, ``sources_used``,
            ``confidence``, ``latency_ms``, ``error``
    """
    question = state["question"]
    context = state.get("context", "(No context available.)")

    logger.info("🤖 Generating answer...")

    response = generate_answer(question, context)

    logger.info(
        "✅ Answer generated (confidence=%s, latency=%.0f ms).",
        response.confidence,
        response.latency_ms,
    )

    return {
        "generator_response": response,
        "answer": response.answer,
        "sources_used": response.sources_used,
        "confidence": response.confidence,
        "latency_ms": response.latency_ms,
        "error": response.error,
    }


# ------------------------------------------------------------------
# Graph builder
# ------------------------------------------------------------------


def build_graph() -> StateGraph:
    """Build the v1 linear RAG graph: retrieve → generate.

    Returns a compiled LangGraph StateGraph ready for ``.invoke()``.
    """
    graph = StateGraph(RAGState)

    # Add nodes
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)

    # Linear flow: retrieve → generate → END
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    return graph.compile()


# ------------------------------------------------------------------
# Convenience runner
# ------------------------------------------------------------------


def run_query(
    question: str,
    *,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run a single question through the full RAG pipeline.

    This is the main entry point for CLI usage and testing.

    Args:
        question: The user's question.
        verbose: If True, prints intermediate steps.

    Returns:
        The final RAGState dict with answer, sources, confidence, etc.
    """
    if verbose:
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    graph = build_graph()

    start = time.perf_counter()
    result = graph.invoke({"question": question})
    total_ms = (time.perf_counter() - start) * 1000

    if verbose:
        print(f"\n⏱️  Total pipeline latency: {total_ms:.0f} ms")

    return result


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------


def main() -> None:
    """Interactive CLI for asking questions against the RAG pipeline."""
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )

    print("=" * 60)
    print("🛡️  Guarded RAG System — v1 (linear pipeline)")
    print("=" * 60)
    print("Type your question and press Enter.")
    print("Type 'quit' or 'exit' to stop.\n")

    # Quick sanity check: is the vector store populated?
    retriever = _get_retriever()
    doc_count = retriever.count
    if doc_count == 0:
        print("⚠️  Vector store is empty! Run ingestion first:")
        print("   python scripts/ingest.py")
        print()
        sys.exit(1)
    else:
        print(f"📚 Vector store contains {doc_count} document chunks.\n")

    while True:
        try:
            question = input("❓ Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question or question.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        result = run_query(question, verbose=True)

        print(f"\n💬 Answer (confidence: {result.get('confidence', '?')}):")
        print("-" * 40)
        print(result.get("answer", "(no answer)"))
        print("-" * 40)

        sources = result.get("sources_used", [])
        if sources:
            print(f"📎 Sources used: {sources}")

        if result.get("error"):
            print(f"⚠️  Error: {result['error']}")

        print()


if __name__ == "__main__":
    main()
