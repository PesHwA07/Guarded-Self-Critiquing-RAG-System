"""Core evaluation metrics for the Guarded RAG System.

Instead of relying on heavy frameworks like Ragas which can have dependency
conflicts, we implement custom LLM-as-a-judge metrics using our strong
critic model (llama-3.3-70b-versatile).

Metrics:
1. Faithfulness: Is the answer derived *only* from the retrieved context?
2. Answer Relevancy: Does the answer directly address the user's query?
"""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field

from rag.llm_provider import get_critic_llm


class FaithfulnessScore(BaseModel):
    """Structured output for Faithfulness evaluation."""

    reasoning: str = Field(
        description="Step-by-step reasoning checking if the answer is derived ONLY from the context."
    )
    is_faithful: bool = Field(
        description="True if the answer contains NO hallucinated information, False otherwise."
    )


class RelevancyScore(BaseModel):
    """Structured output for Answer Relevancy evaluation."""

    reasoning: str = Field(
        description="Step-by-step reasoning about whether the answer addresses the user's query."
    )
    is_relevant: bool = Field(
        description="True if the answer directly addresses the query, False if it is evasive or off-topic."
    )


def evaluate_faithfulness(query: str, contexts: List[str], answer: str) -> dict:
    """Evaluate if the answer is faithful to the retrieved context.

    Args:
        query: The user's original query.
        contexts: List of retrieved document chunks.
        answer: The generated answer to evaluate.

    Returns:
        dict: {"score": float, "reasoning": str}
    """
    llm = get_critic_llm(temperature=0.0)
    structured_llm = llm.with_structured_output(FaithfulnessScore)

    context_str = "\n\n".join(contexts) if contexts else "NO CONTEXT PROVIDED."

    prompt = f"""You are an expert evaluator. Your task is to determine if the given Answer is perfectly faithful to the provided Context.

An answer is FAITHFUL if all claims made in the answer can be directly inferred from the context.
An answer is NOT FAITHFUL if it includes hallucinated information, external knowledge not present in the context, or contradicts the context.

Query: {query}

Context:
{context_str}

Answer:
{answer}

Evaluate the faithfulness of the answer.
"""
    try:
        result = structured_llm.invoke(prompt)
        return {
            "score": 1.0 if result.is_faithful else 0.0,
            "reasoning": result.reasoning,
        }
    except Exception as e:
        return {"score": 0.0, "reasoning": f"Evaluation failed: {str(e)}"}


def evaluate_relevancy(query: str, answer: str) -> dict:
    """Evaluate if the answer is relevant to the original query.

    Args:
        query: The user's original query.
        answer: The generated answer to evaluate.

    Returns:
        dict: {"score": float, "reasoning": str}
    """
    llm = get_critic_llm(temperature=0.0)
    structured_llm = llm.with_structured_output(RelevancyScore)

    prompt = f"""You are an expert evaluator. Your task is to determine if the given Answer directly addresses the Query.

An answer is RELEVANT if it provides a direct, helpful, and accurate response to the user's query.
An answer is NOT RELEVANT if it is evasive, talks about unrelated topics, or fails to answer the core question.

Query: {query}

Answer:
{answer}

Evaluate the relevancy of the answer.
"""
    try:
        result = structured_llm.invoke(prompt)
        return {
            "score": 1.0 if result.is_relevant else 0.0,
            "reasoning": result.reasoning,
        }
    except Exception as e:
        return {"score": 0.0, "reasoning": f"Evaluation failed: {str(e)}"}
