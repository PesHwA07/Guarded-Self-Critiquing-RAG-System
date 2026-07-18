"""Fallback response generation for the agentic RAG pipeline.

Used when the critic rejects the generated answer and all retry attempts
have been exhausted. It returns a graceful refusal that includes the
context snippets we attempted to use, so the user has some breadcrumbs.
"""

from typing import Any

from rag.retriever import RetrievalResult


def generate_fallback_response(
    retrieved_chunks: list[RetrievalResult],
    retries: int,
) -> dict[str, Any]:
    """Generate a graceful degradation response.

    Args:
        retrieved_chunks: The chunks retrieved in the final attempt.
        retries: The number of reformulate/retrieve loops executed.

    Returns:
        A dictionary with answer, confidence, and error fields to update the state.
    """
    base_msg = (
        "I don't have enough information in the available documentation "
        "to answer this question with certainty."
    )

    if not retrieved_chunks:
        answer = base_msg + " I could not find any relevant documents."
    else:
        answer = base_msg + " However, here are the excerpts I found that might be relevant:\n\n"

        parts = []
        for i, r in enumerate(retrieved_chunks, 1):
            source = r.source
            parts.append(f"**Source {i}: {source}**\n```text\n{r.content.strip()}\n```")

        answer += "\n\n".join(parts)

    return {
        "answer": answer,
        "confidence": "none",
        "error": f"Exhausted {retries} retries without a grounded answer.",
    }
