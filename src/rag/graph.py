"""LangGraph pipeline — agentic self-critiquing RAG (v2).

V2 upgrades the linear retrieve → generate chain into a full agentic loop:

    retrieve → generate → critic → (reformulate → retrieve) | fallback | done

The critic uses a *different, stronger* model (llama-3.3-70b-versatile) than the
generator (llama-3.1-8b-instant) to avoid correlated blind spots.

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
from typing import Any, Literal, TypedDict

from langgraph.graph import END, StateGraph

from rag.critic import evaluate_answer
from rag.generator import GeneratorResponse, generate_answer
from rag.retriever import (
    RetrievalResult,
    VectorStoreRetriever,
    format_context,
)

logger = logging.getLogger(__name__)

# Default maximum retries before hitting the fallback path
DEFAULT_MAX_RETRIES = 2


# ------------------------------------------------------------------
# State definition (v2 — extended for agentic loop)
# ------------------------------------------------------------------


class RAGState(TypedDict, total=False):
    """State that flows through the agentic RAG graph.

    Using ``total=False`` so nodes only need to set the fields they produce.

    Fields are grouped by the node that writes them:
    - **Input**: set by the caller
    - **Retriever**: written by ``retrieve_node``
    - **Generator**: written by ``generate_node``
    - **Critic**: written by ``critic_node``
    - **Reformulator**: written by ``reformulate_node``
    - **Fallback**: written by ``fallback_node``
    - **Control flow**: used by routing logic
    """

    # --- Input ---
    question: str               # The user's original question
    original_question: str      # Preserved copy (never mutated by reformulator)

    # --- Retriever output ---
    retrieved_chunks: list[RetrievalResult]
    context: str

    # --- Generator output ---
    generator_response: GeneratorResponse
    answer: str
    sources_used: list[int]
    confidence: str

    # --- Critic output ---
    critic_verdict: Literal[
        "grounded",
        "not_grounded",
        "partially_grounded",
    ]
    critic_reasoning: str       # Free-text explanation of the verdict

    # --- Reformulator output ---
    reformulated_query: str     # Improved query targeting the gap the critic found

    # --- Control flow ---
    retry_count: int            # How many reformulate→retrieve loops so far
    max_retries: int            # Ceiling (default DEFAULT_MAX_RETRIES)

    # --- Metadata ---
    latency_ms: float
    error: str | None


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

    On the first call, uses ``state["question"]``.
    On retries after reformulation, uses ``state["reformulated_query"]``
    if available, falling back to the original question.

    Reads: ``question``, ``reformulated_query`` (optional)
    Writes: ``retrieved_chunks``, ``context``
    """
    # Use reformulated query on retries, original question on first pass
    query = state.get("reformulated_query") or state["question"]
    logger.info("Retrieving chunks for: '%s'", query)

    retriever = _get_retriever()
    results = retriever.retrieve(query)
    context = format_context(results)

    logger.info("Retrieved %d chunks.", len(results))
    return {
        "retrieved_chunks": results,
        "context": context,
    }


def generate_node(state: RAGState) -> dict[str, Any]:
    """Generate an answer from the retrieved context.

    Always uses the *original* question (not the reformulated one) so the
    answer addresses what the user actually asked.

    Reads: ``original_question`` (or ``question``), ``context``
    Writes: ``generator_response``, ``answer``, ``sources_used``,
            ``confidence``, ``latency_ms``, ``error``
    """
    # Always answer the original question, not the reformulated search query
    question = state.get("original_question") or state["question"]
    context = state.get("context", "(No context available.)")

    logger.info("Generating answer...")

    response = generate_answer(question, context)

    logger.info(
        "Answer generated (confidence=%s, latency=%.0f ms).",
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


def critic_node(state: RAGState) -> dict[str, Any]:
    """Evaluate whether the generated answer is grounded in the retrieved chunks.

    Uses the critic model to check if claims trace back to the context.

    Reads: ``original_question`` (or ``question``), ``answer``, ``context``
    Writes: ``critic_verdict``, ``critic_reasoning``, ``latency_ms``, ``error``
    """
    question = state.get("original_question") or state["question"]
    answer = state.get("answer", "")
    context = state.get("context", "")

    logger.info("Evaluating answer for groundedness...")

    response = evaluate_answer(question, answer, context)

    logger.info(
        "Critic verdict: %s (latency=%.0f ms).",
        response.verdict,
        response.latency_ms,
    )

    result_dict = {
        "critic_verdict": response.verdict,
        "critic_reasoning": response.reasoning,
    }
    
    # Optional: We could sum latencies or keep them separate.
    # We will just write any errors if they occurred.
    if response.error:
        result_dict["error"] = response.error

    return result_dict


def reformulate_node(state: RAGState) -> dict[str, Any]:
    """Reformulate the search query based on the critic's feedback.

    PLACEHOLDER — will be fully implemented in Week 2 Day 3 (reformulator.py).
    For now, just increments retry_count and echoes the original question.

    Reads: ``question``, ``critic_reasoning``, ``retry_count``
    Writes: ``reformulated_query``, ``retry_count``
    """
    retry = state.get("retry_count", 0) + 1
    question = state.get("original_question") or state["question"]

    logger.info(
        "Reformulating query (attempt %d, placeholder — echoing original)...",
        retry,
    )

    return {
        "reformulated_query": question,
        "retry_count": retry,
    }


def fallback_node(state: RAGState) -> dict[str, Any]:
    """Return a graceful degradation response when retries are exhausted.

    PLACEHOLDER — will be fully implemented in Week 2 Day 4 (fallback.py).
    For now, returns a simple "insufficient information" message.

    Reads: ``retry_count``, ``critic_reasoning``
    Writes: ``answer``, ``confidence``, ``error``
    """
    retries = state.get("retry_count", 0)
    logger.info(
        "Fallback triggered after %d retries (placeholder).",
        retries,
    )

    return {
        "answer": (
            "I don't have enough information in the available documentation "
            "to answer this question."
        ),
        "confidence": "low",
        "error": f"Exhausted {retries} retries without a grounded answer.",
    }


# ------------------------------------------------------------------
# Routing logic
# ------------------------------------------------------------------


def _route_after_critic(state: RAGState) -> str:
    """Decide what happens after the critic evaluates the answer.

    - ``grounded`` → done (go to END)
    - ``not_grounded`` or ``partially_grounded``:
      - retries left → ``reformulate``
      - retries exhausted → ``fallback``
    """
    verdict = state.get("critic_verdict", "not_grounded")
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", DEFAULT_MAX_RETRIES)

    if verdict == "grounded":
        logger.info("Critic approved — answer is grounded.")
        return END

    if retry_count < max_retries:
        logger.info(
            "Critic rejected (verdict=%s, retry %d/%d) — reformulating.",
            verdict,
            retry_count + 1,
            max_retries,
        )
        return "reformulate"

    logger.info(
        "Critic rejected and retries exhausted (%d/%d) — falling back.",
        retry_count,
        max_retries,
    )
    return "fallback"


# ------------------------------------------------------------------
# Graph builders
# ------------------------------------------------------------------


def build_graph_v1() -> StateGraph:
    """Build the v1 linear RAG graph: retrieve → generate → END.

    Preserved for backwards compatibility and simple testing.
    """
    graph = StateGraph(RAGState)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    return graph.compile()


def build_graph() -> StateGraph:
    """Build the v2 agentic RAG graph with self-critiquing loop.

    Flow::

        retrieve → generate → critic ─┬─ (grounded)       → END
                                       ├─ (retries left)   → reformulate → retrieve
                                       └─ (retries exhausted) → fallback → END

    Returns a compiled LangGraph StateGraph ready for ``.invoke()``.
    """
    graph = StateGraph(RAGState)

    # --- Nodes ---
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("critic", critic_node)
    graph.add_node("reformulate", reformulate_node)
    graph.add_node("fallback", fallback_node)

    # --- Edges ---
    # Entry: retrieve
    graph.set_entry_point("retrieve")

    # retrieve → generate → critic
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "critic")

    # critic → conditional routing (grounded/reformulate/fallback)
    graph.add_conditional_edges("critic", _route_after_critic)

    # reformulate loops back to retrieve
    graph.add_edge("reformulate", "retrieve")

    # fallback goes to END
    graph.add_edge("fallback", END)

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
    result = graph.invoke({
        "question": question,
        "original_question": question,
        "retry_count": 0,
    })
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
    print("Guarded RAG System - v2 (agentic self-critiquing pipeline)")
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

        # Show critic verdict
        verdict = result.get("critic_verdict", "?")
        retries = result.get("retry_count", 0)
        print(f"Critic verdict: {verdict} (retries: {retries})")

        sources = result.get("sources_used", [])
        if sources:
            print(f"Sources used: {sources}")

        if result.get("error"):
            print(f"Error: {result['error']}")

        print()


if __name__ == "__main__":
    main()
