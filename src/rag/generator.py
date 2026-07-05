"""LLM answer generator — turns retrieved context into grounded answers.

Takes a user question + retrieved document chunks and produces a structured
response with the answer text and which sources were actually used.  Uses
the generator model (8B, fast/cheap) by default.

Usage:
    from rag.generator import generate_answer
    from rag.retriever import VectorStoreRetriever, format_context

    retriever = VectorStoreRetriever()
    results = retriever.retrieve("How do I send a POST request?")
    context = format_context(results)
    response = generate_answer("How do I send a POST request?", context)
    print(response.answer)
    print(response.sources_used)
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage

from rag.llm_provider import get_generator_llm

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Prompt templates
# ------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a technical documentation assistant. Your job is to answer questions
accurately based ONLY on the provided context documents.

Rules:
1. ONLY use information from the provided context to answer the question.
2. If the context does not contain enough information to answer, say exactly:
   "I don't have enough information in the available documentation to answer this question."
3. When you use information from the context, cite the source using [Source N] notation.
4. Be concise and direct. Avoid filler phrases.
5. If the question is ambiguous, acknowledge the ambiguity and answer the most
   likely interpretation based on the context.

You MUST respond with valid JSON in exactly this format:
{
    "answer": "Your answer text here, with [Source N] citations.",
    "sources_used": [1, 2],
    "confidence": "high" | "medium" | "low"
}

- "sources_used" is a list of Source numbers you actually referenced.
- "confidence" reflects how well the context supports your answer:
  - "high": The answer is directly and clearly stated in the context.
  - "medium": The answer requires some inference from the context.
  - "low": The context only partially addresses the question.
"""

_USER_PROMPT_TEMPLATE = """\
Context Documents:
{context}

Question: {question}

Respond with valid JSON only.
"""


# ------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------


@dataclass
class GeneratorResponse:
    """Structured output from the answer generator."""

    answer: str
    sources_used: list[int] = field(default_factory=list)
    confidence: str = "low"
    raw_llm_output: str = ""
    latency_ms: float = 0.0
    error: Optional[str] = None

    @property
    def is_refusal(self) -> bool:
        """Check if the LLM refused to answer (insufficient context)."""
        refusal_phrases = [
            "don't have enough information",
            "do not have enough information",
            "cannot answer",
            "no relevant information",
            "not enough context",
        ]
        return any(phrase in self.answer.lower() for phrase in refusal_phrases)


# ------------------------------------------------------------------
# Core generator function
# ------------------------------------------------------------------


def generate_answer(
    question: str,
    context: str,
    *,
    model: str | None = None,
    temperature: float = 0.0,
) -> GeneratorResponse:
    """Generate a grounded answer from retrieved context.

    Args:
        question: The user's question.
        context: Formatted context string (from ``format_context``).
        model: Override the default generator model.
        temperature: LLM temperature (0.0 = deterministic).

    Returns:
        A ``GeneratorResponse`` with the answer, sources, confidence, and timing.
    """
    llm = get_generator_llm(model=model, temperature=temperature)

    user_prompt = _USER_PROMPT_TEMPLATE.format(
        context=context,
        question=question,
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
            "LLM responded in %.0f ms (%d chars).",
            latency_ms,
            len(raw_output),
        )

        return _parse_response(raw_output, latency_ms)

    except Exception as e:
        latency_ms = (time.perf_counter() - start) * 1000
        logger.error("Generator LLM call failed: %s", e)
        return GeneratorResponse(
            answer="An error occurred while generating the answer.",
            raw_llm_output="",
            latency_ms=latency_ms,
            error=str(e),
        )


# ------------------------------------------------------------------
# Response parsing
# ------------------------------------------------------------------


def _parse_response(raw: str, latency_ms: float) -> GeneratorResponse:
    """Parse the LLM's JSON response into a GeneratorResponse.

    Handles common LLM quirks: markdown code fences, trailing commas,
    and malformed JSON (falls back to treating the whole thing as the answer).
    """
    # Strip markdown code fences if the LLM wrapped its response
    cleaned = raw
    if "```" in cleaned:
        # Extract content between code fences
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", cleaned, re.DOTALL)
        if match:
            cleaned = match.group(1).strip()

    try:
        data = json.loads(cleaned)

        return GeneratorResponse(
            answer=data.get("answer", raw),
            sources_used=data.get("sources_used", []),
            confidence=data.get("confidence", "low"),
            raw_llm_output=raw,
            latency_ms=latency_ms,
        )

    except json.JSONDecodeError:
        logger.warning(
            "Failed to parse LLM output as JSON. Using raw text as answer."
        )
        return GeneratorResponse(
            answer=raw,
            sources_used=[],
            confidence="low",
            raw_llm_output=raw,
            latency_ms=latency_ms,
            error="LLM output was not valid JSON",
        )
