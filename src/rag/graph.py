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

from guardrails.input_guard import InjectionGuard, PIIGuard
from guardrails.output_guard import TopicGuard, ToxicityGuard
from guardrails.policy_engine import load_policies
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
# Guard Singletons
# ------------------------------------------------------------------
_pii_guard = None
_injection_guard = None
_toxicity_guard = None
_topic_guard = None
_guards_loaded = False

def _get_guards():
    global _pii_guard, _injection_guard, _toxicity_guard, _topic_guard, _guards_loaded
    if not _guards_loaded:
        logger.info("Loading guardrail policies and initializing guards...")
        policies = load_policies()

        _pii_guard = PIIGuard(action=policies.pii.action) if policies.pii.enabled else None
        _injection_guard = InjectionGuard(enabled=policies.injection.enabled)

        _toxicity_guard = ToxicityGuard(threshold=policies.toxicity.threshold) if policies.toxicity.enabled else None
        _topic_guard = TopicGuard(allowed_topics=policies.topics.allowed) if policies.topics.enabled else None

        _guards_loaded = True



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

def input_guard_node(state: RAGState) -> dict[str, Any]:
    """Run input guardrails before retrieval."""
    question = state["question"]
    logger.info("Running input guards...")
    _get_guards()

    if _injection_guard and _injection_guard.enabled:
        inj_res = _injection_guard.evaluate(question)
        if not inj_res.passed:
            logger.warning(f"Injection blocked: {inj_res.reason}")
            return {"error": inj_res.reason}

    if _pii_guard:
        pii_res = _pii_guard.evaluate(question)
        if not pii_res.passed:
            logger.warning(f"PII blocked: {pii_res.reason}")
            return {"error": pii_res.reason}
        elif pii_res.modified_text and pii_res.modified_text != question:
            logger.info("PII redacted from input.")
            return {"question": pii_res.modified_text, "original_question": pii_res.modified_text}

    return {}



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

    Reads: ``question``, ``critic_reasoning``, ``retry_count``
    Writes: ``reformulated_query``, ``retry_count``
    """
    from rag.reformulator import reformulate_query

    retry = state.get("retry_count", 0) + 1
    question = state.get("original_question") or state["question"]
    reasoning = state.get("critic_reasoning", "")

    logger.info("Reformulating query (attempt %d)...", retry)

    new_query = reformulate_query(question, reasoning)

    logger.info("New query: '%s'", new_query)

    return {
        "reformulated_query": new_query,
        "retry_count": retry,
    }


def fallback_node(state: RAGState) -> dict[str, Any]:
    """Return a graceful degradation response when retries are exhausted.

    Reads: ``retry_count``, ``retrieved_chunks``
    Writes: ``answer``, ``confidence``, ``error``
    """
    from rag.fallback import generate_fallback_response

    retries = state.get("retry_count", 0)
    chunks = state.get("retrieved_chunks", [])
    logger.info(
        "Fallback triggered after %d retries.",
        retries,
    )

    return generate_fallback_response(chunks, retries)


def output_guard_node(state: RAGState) -> dict[str, Any]:
    """Run output guardrails on the final answer."""
    answer = state.get("answer", "")
    question = state.get("original_question") or state.get("question", "")
    logger.info("Running output guards...")
    _get_guards()

    if _toxicity_guard:
        tox_res = _toxicity_guard.evaluate(answer)
        if not tox_res.passed:
            logger.warning(f"Toxicity blocked: {tox_res.reason}")
            return {"error": tox_res.reason, "answer": "Answer blocked due to toxicity policy."}

    if _topic_guard:
        top_res = _topic_guard.evaluate(answer, question)
        if not top_res.passed:
            logger.warning(f"Topic blocked: {top_res.reason}")
            return {"error": top_res.reason, "answer": "Answer blocked due to topic policy."}

    return {}


# ------------------------------------------------------------------
# Routing logic
# ------------------------------------------------------------------


def _route_after_input_guard(state: RAGState) -> str:
    """Decide what happens after the input guard.

    If error is set (blocked), go to END. Otherwise proceed to retrieve.
    """
    if state.get("error"):
        return END
    return "retrieve"


def _route_after_critic(state: RAGState) -> str:
    """Decide what happens after the critic evaluates the answer.

    - ``grounded`` → output_guard
    - ``not_grounded`` or ``partially_grounded``:
      - retries left → ``reformulate``
      - retries exhausted → ``fallback``
    """
    verdict = state.get("critic_verdict", "not_grounded")
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", DEFAULT_MAX_RETRIES)

    if verdict == "grounded":
        logger.info("Critic approved — answer is grounded.")
        return "output_guard"

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
    graph.add_node("input_guard", input_guard_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("critic", critic_node)
    graph.add_node("reformulate", reformulate_node)
    graph.add_node("fallback", fallback_node)
    graph.add_node("output_guard", output_guard_node)

    # --- Edges ---
    # Entry: input_guard
    graph.set_entry_point("input_guard")

    # input_guard → conditional routing (pass/block)
    graph.add_conditional_edges(
        "input_guard",
        _route_after_input_guard,
        {
            END: END,
            "retrieve": "retrieve",
        }
    )

    # retrieve → generate → critic
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "critic")

    # critic → conditional routing (grounded/reformulate/fallback)
    graph.add_conditional_edges(
        "critic",
        _route_after_critic,
        {
            "output_guard": "output_guard",
            "reformulate": "reformulate",
            "fallback": "fallback",
        }
    )

    # reformulate loops back to retrieve
    graph.add_edge("reformulate", "retrieve")

    # fallback goes to output_guard
    graph.add_edge("fallback", "output_guard")

    # output_guard goes to END
    graph.add_edge("output_guard", END)

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
        from telemetry import setup_telemetry
        setup_telemetry(json_format=False, level=logging.INFO)
    else:
        # Default for production: JSON telemetry
        from telemetry import setup_telemetry
        setup_telemetry(json_format=True, level=logging.INFO)
        
    logger.info("Starting run_query pipeline", extra={"metadata": {"question": question}})


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

    from telemetry import setup_telemetry
    # Interactive CLI should remain readable text, not JSON
    setup_telemetry(json_format=False, level=logging.INFO)

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
