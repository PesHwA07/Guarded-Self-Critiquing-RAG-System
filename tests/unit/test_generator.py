"""Unit tests for the answer generator (src/rag/generator.py).

Tests cover:
- JSON response parsing (valid, malformed, code-fenced)
- GeneratorResponse properties (is_refusal detection)
- Prompt template formatting

These tests mock the LLM call to avoid network requests and API keys.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from rag.generator import (
    GeneratorResponse,
    _parse_response,
    generate_answer,
)

# ── TestParseResponse ─────────────────────────────────────────────────


class TestParseResponse:
    """Tests for the JSON parsing logic."""

    def test_valid_json(self):
        """Well-formed JSON should be parsed correctly."""
        raw = '{"answer": "Use requests.post()", "sources_used": [1, 2], "confidence": "high"}'
        resp = _parse_response(raw, latency_ms=100.0)

        assert resp.answer == "Use requests.post()"
        assert resp.sources_used == [1, 2]
        assert resp.confidence == "high"
        assert resp.latency_ms == 100.0
        assert resp.error is None

    def test_json_with_code_fences(self):
        """JSON wrapped in markdown code fences should be unwrapped."""
        raw = '```json\n{"answer": "The answer.", "sources_used": [1], "confidence": "medium"}\n```'
        resp = _parse_response(raw, latency_ms=50.0)

        assert resp.answer == "The answer."
        assert resp.sources_used == [1]
        assert resp.confidence == "medium"

    def test_json_with_bare_code_fences(self):
        """JSON wrapped in bare ``` fences (no language tag) should work."""
        raw = '```\n{"answer": "Another answer.", "sources_used": [], "confidence": "low"}\n```'
        resp = _parse_response(raw, latency_ms=75.0)

        assert resp.answer == "Another answer."

    def test_malformed_json_fallback(self):
        """Non-JSON output should fall back to using raw text as answer."""
        raw = "I think the answer is to use requests.post()."
        resp = _parse_response(raw, latency_ms=200.0)

        assert resp.answer == raw
        assert resp.sources_used == []
        assert resp.confidence == "low"
        assert resp.error is not None

    def test_partial_json_missing_fields(self):
        """JSON with missing optional fields should use defaults."""
        raw = '{"answer": "Just the answer."}'
        resp = _parse_response(raw, latency_ms=80.0)

        assert resp.answer == "Just the answer."
        assert resp.sources_used == []
        assert resp.confidence == "low"


# ── TestGeneratorResponse ─────────────────────────────────────────────


class TestGeneratorResponse:
    """Tests for the GeneratorResponse dataclass."""

    def test_is_refusal_true(self):
        """Should detect refusal phrases."""
        resp = GeneratorResponse(
            answer="I don't have enough information in the available documentation to answer this question.",
        )
        assert resp.is_refusal is True

    def test_is_refusal_false(self):
        """Normal answers should not be flagged as refusals."""
        resp = GeneratorResponse(
            answer="Use requests.post() to send a POST request [Source 1].",
        )
        assert resp.is_refusal is False

    def test_is_refusal_case_insensitive(self):
        """Refusal detection should be case-insensitive."""
        resp = GeneratorResponse(
            answer="I Don't Have Enough Information to answer.",
        )
        assert resp.is_refusal is True

    def test_is_refusal_cannot_answer(self):
        """'cannot answer' should also be detected."""
        resp = GeneratorResponse(
            answer="Based on the context, I cannot answer this question.",
        )
        assert resp.is_refusal is True


# ── TestGenerateAnswer ────────────────────────────────────────────────


class TestGenerateAnswer:
    """Tests for the main generate_answer function (mocked LLM)."""

    @patch("rag.generator.get_generator_llm")
    def test_successful_generation(self, mock_get_llm):
        """Should return a parsed GeneratorResponse on success."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content='{"answer": "Use requests.post(url, data=payload).", "sources_used": [1], "confidence": "high"}'
        )
        mock_get_llm.return_value = mock_llm

        resp = generate_answer(
            question="How do I send a POST request?",
            context="[Source 1: quickstart.rst]\nUse requests.post() with url and data.",
        )

        assert resp.answer == "Use requests.post(url, data=payload)."
        assert resp.sources_used == [1]
        assert resp.confidence == "high"
        assert resp.error is None
        assert resp.latency_ms > 0

    @patch("rag.generator.get_generator_llm")
    def test_llm_error_handling(self, mock_get_llm):
        """Should return an error response if the LLM call fails."""
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = RuntimeError("API rate limit exceeded")
        mock_get_llm.return_value = mock_llm

        resp = generate_answer(
            question="Anything",
            context="Some context.",
        )

        assert "error occurred" in resp.answer.lower()
        assert resp.error is not None
        assert "rate limit" in resp.error

    @patch("rag.generator.get_generator_llm")
    def test_llm_returns_non_json(self, mock_get_llm):
        """Non-JSON LLM output should fall back gracefully."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content="Just a plain text answer without JSON."
        )
        mock_get_llm.return_value = mock_llm

        resp = generate_answer(
            question="Question?",
            context="Context.",
        )

        assert resp.answer == "Just a plain text answer without JSON."
        assert resp.error is not None  # indicates parse failure
