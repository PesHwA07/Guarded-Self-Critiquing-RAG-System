"""Core evaluation metrics for the Guarded RAG System.

Instead of relying on heavy frameworks like Ragas which can have dependency
conflicts, we implement custom LLM-as-a-judge metrics using our strong
critic model (llama-3.3-70b-versatile).

Metrics:
1. Faithfulness: Is the answer derived *only* from the retrieved context?
2. Answer Relevancy: Does the answer directly address the user's query?
"""

from __future__ import annotations

import math
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

    prompt = f"""You are an expert evaluator. Your task is to determine if the given Answer
is perfectly faithful to the provided Context.

An answer is FAITHFUL if all claims made in the answer can be directly inferred from the context.
An answer is NOT FAITHFUL if it includes hallucinated information, external knowledge
not present in the context, or contradicts the context.

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

    prompt = f"""You are an expert evaluator. Your task is to determine if the given Answer
directly addresses the Query.

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


def calculate_hallucination_rate(faithfulness_scores: List[float]) -> float:
    """Calculate the hallucination rate from a list of faithfulness scores.

    A faithfulness score of 0.0 means the answer contained hallucinated information.
    Hallucination rate = (count of 0.0 scores) / (total scores).
    """
    if not faithfulness_scores:
        return 0.0
    hallucinated = sum(1 for score in faithfulness_scores if score < 0.5)
    return hallucinated / len(faithfulness_scores)


def calculate_latency_percentiles(latencies: List[float]) -> dict:
    """Calculate p50 (median) and p95 latencies.

    Args:
        latencies: List of latency measurements in seconds.

    Returns:
        dict: {"p50": float, "p95": float}
    """
    if not latencies:
        return {"p50": 0.0, "p95": 0.0}

    sorted_lats = sorted(latencies)
    n = len(sorted_lats)

    # Calculate p50
    p50_idx = int(math.ceil(0.50 * n)) - 1
    p50 = sorted_lats[max(0, p50_idx)]

    # Calculate p95
    p95_idx = int(math.ceil(0.95 * n)) - 1
    p95 = sorted_lats[max(0, p95_idx)]

    return {"p50": p50, "p95": p95}


def calculate_cost(input_tokens: int, output_tokens: int, model: str = "llama-3.1-8b-instant") -> float:
    """Calculate the estimated API cost based on token usage.

    Args:
        input_tokens: Number of input/prompt tokens used.
        output_tokens: Number of output/completion tokens used.
        model: The model name to look up pricing for.

    Returns:
        float: The estimated cost in USD.
    """
    # Pricing per 1M tokens (Groq free tier has a nominal value, listed here for evaluation realism)
    pricing = {
        "llama-3.1-8b-instant": {"input": 0.05, "output": 0.08},
        "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
    }

    rates = pricing.get(model, {"input": 0.0, "output": 0.0})

    input_cost = (input_tokens / 1_000_000) * rates["input"]
    output_cost = (output_tokens / 1_000_000) * rates["output"]

    return input_cost + output_cost
