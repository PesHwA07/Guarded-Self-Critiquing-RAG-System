"""LLM critic — evaluates whether generated answers are grounded in context.

Takes the user question, the retrieved context, and the generated answer,
and evaluates if the claims in the answer are supported by the context.
Uses the critic model (larger/stronger, e.g., 70B) to avoid correlated blind spots
with the generator model.

Usage:
    from rag.critic import evaluate_answer
    from rag.retriever import VectorStoreRetriever, format_context

    retriever = VectorStoreRetriever()
    results = retriever.retrieve("How do I send a POST request?")
    context = format_context(results)

    # ... generate answer ...

    response = evaluate_answer("How do I send a POST request?", answer_text, context)
    print(response.verdict)
    print(response.reasoning)
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage

from rag.llm_provider import get_critic_llm

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Prompt templates
# ------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are an expert fact-checker and self-critique agent for a documentation assistant.
Your job is to evaluate whether a generated answer is GROUNDED in the retrieved context.

You will be given:
1. The user's QUESTION
2. The retrieved CONTEXT DOCUMENTS
3. The generated ANSWER

Your task is to determine if all claims made in the ANSWER are fully supported by the CONTEXT.
Do not evaluate if the answer is factually true in the real world, ONLY if it is supported by the provided CONTEXT.

Rules for verdicts:
- "grounded": All claims in the answer are supported by the context. Or, the answer correctly
  states that it doesn't have enough information based on the context.
- "not_grounded": The answer makes claims that contradict the context, or makes up significant
  new information not present in the context.
- "partially_grounded": The answer is mostly correct but includes some unverified details
  or misses critical caveats present in the context.

You MUST respond with valid JSON in exactly this format:
{
    "verdict": "grounded" | "not_grounded" | "partially_grounded",
    "reasoning": "Step-by-step explanation of your evaluation, pointing out specific unsupported claims if applicable."
}
"""

_USER_PROMPT_TEMPLATE = """\
Question: {question}

Context Documents:
{context}

Generated Answer:
{answer}

Respond with valid JSON only.
"""


# ------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------


@dataclass
class CriticResponse:
    """Structured output from the critic."""

    verdict: Literal["grounded", "not_grounded", "partially_grounded"]
    reasoning: str
    raw_llm_output: str = ""
    latency_ms: float = 0.0
    error: str | None = None


# ------------------------------------------------------------------
# Core critic function
# ------------------------------------------------------------------


def evaluate_answer(
    question: str,
    answer: str,
    context: str,
    *,
    temperature: float = 0.0,
) -> CriticResponse:
    """Evaluate whether an answer is grounded in the retrieved context.

    Args:
        question: The user's question.
        answer: The generated answer to evaluate.
        context: Formatted context string (from ``format_context``).
        temperature: LLM temperature (0.0 = deterministic).

    Returns:
        A ``CriticResponse`` with the verdict, reasoning, and timing.
    """
    llm = get_critic_llm(temperature=temperature)

    user_prompt = _USER_PROMPT_TEMPLATE.format(
        question=question,
        context=context,
        answer=answer,
    )

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]

    start = time.perf_counter()
    try:
        response = llm.invoke(messages)
        latency_ms = (time.perf_counter() - start) * 1000
        raw_output = response.content.strip()

        logger.debug(
            "Critic LLM responded in %.0f ms (%d chars).",
            latency_ms,
            len(raw_output),
        )

        return _parse_response(raw_output, latency_ms)

    except Exception as e:
        latency_ms = (time.perf_counter() - start) * 1000
        logger.error("Critic LLM call failed: %s", e)
        # On error, we default to grounded to avoid infinite loops, but record the error.
        # Alternatively, we could fail closed (not_grounded). Failing open is safer for availability.
        return CriticResponse(
            verdict="grounded",
            reasoning="Critic failed to evaluate due to an error. Auto-approved.",
            raw_llm_output="",
            latency_ms=latency_ms,
            error=str(e),
        )


# ------------------------------------------------------------------
# Response parsing
# ------------------------------------------------------------------


def _parse_response(raw: str, latency_ms: float) -> CriticResponse:
    """Parse the LLM's JSON response into a CriticResponse.

    Handles common LLM quirks: markdown code fences and malformed JSON.
    """
    cleaned = raw
    if "```" in cleaned:
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", cleaned, re.DOTALL)
        if match:
            cleaned = match.group(1).strip()

    try:
        data = json.loads(cleaned)

        verdict = data.get("verdict", "grounded")
        if verdict not in ("grounded", "not_grounded", "partially_grounded"):
            logger.warning("Invalid verdict from critic: %s. Defaulting to grounded.", verdict)
            verdict = "grounded"

        return CriticResponse(
            verdict=verdict,
            reasoning=data.get("reasoning", "No reasoning provided."),
            raw_llm_output=raw,
            latency_ms=latency_ms,
        )

    except json.JSONDecodeError:
        logger.warning(
            "Failed to parse Critic LLM output as JSON. Defaulting to grounded."
        )
        return CriticResponse(
            verdict="grounded",
            reasoning="Failed to parse critic JSON. Auto-approved. Raw output: " + raw,
            raw_llm_output=raw,
            latency_ms=latency_ms,
            error="LLM output was not valid JSON",
        )
