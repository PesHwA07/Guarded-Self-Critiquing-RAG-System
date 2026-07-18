"""Unit tests for the critic module (src/rag/critic.py).

Tests cover:
- JSON response parsing (valid, malformed, code-fenced)
- CriticResponse data structure defaults
- Error handling in LLM invocation
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from rag.critic import (
    _parse_response,
    evaluate_answer,
)

# ── TestParseResponse ─────────────────────────────────────────────────


class TestParseResponse:
    """Tests for the JSON parsing logic."""

    def test_valid_json_grounded(self):
        """Well-formed JSON with 'grounded' verdict should be parsed correctly."""
        raw = '{"verdict": "grounded", "reasoning": "All claims are supported."}'
        resp = _parse_response(raw, latency_ms=100.0)

        assert resp.verdict == "grounded"
        assert resp.reasoning == "All claims are supported."
        assert resp.latency_ms == 100.0
        assert resp.error is None

    def test_valid_json_not_grounded(self):
        """Well-formed JSON with 'not_grounded' verdict."""
        raw = '{"verdict": "not_grounded", "reasoning": "Claim XYZ is false."}'
        resp = _parse_response(raw, latency_ms=100.0)

        assert resp.verdict == "not_grounded"
        assert resp.reasoning == "Claim XYZ is false."

    def test_invalid_verdict_defaults_to_grounded(self):
        """If the LLM returns an unknown verdict string, default to grounded."""
        raw = '{"verdict": "perfectly_fine", "reasoning": "Looks good."}'
        resp = _parse_response(raw, latency_ms=50.0)

        # We default to grounded to avoid infinite retry loops on weird LLM behavior
        assert resp.verdict == "grounded"
        assert resp.reasoning == "Looks good."

    def test_json_with_code_fences(self):
        """JSON wrapped in markdown code fences should be unwrapped."""
        raw = '```json\n{"verdict": "partially_grounded", "reasoning": "Missing caveat."}\n```'
        resp = _parse_response(raw, latency_ms=50.0)

        assert resp.verdict == "partially_grounded"
        assert resp.reasoning == "Missing caveat."

    def test_malformed_json_fallback(self):
        """Non-JSON output should fall back to grounded verdict with an error."""
        raw = "I evaluate this as grounded because..."
        resp = _parse_response(raw, latency_ms=200.0)

        assert resp.verdict == "grounded"
        assert "Failed to parse" in resp.reasoning
        assert resp.error is not None


# ── TestEvaluateAnswer ────────────────────────────────────────────────


class TestEvaluateAnswer:
    """Tests for the main evaluate_answer function (mocked LLM)."""

    @patch("rag.critic.get_critic_llm")
    def test_successful_evaluation(self, mock_get_llm):
        """Should return a parsed CriticResponse on success."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content='{"verdict": "grounded", "reasoning": "Matches context perfectly."}'
        )
        mock_get_llm.return_value = mock_llm

        resp = evaluate_answer(
            question="How do I send a POST request?",
            answer="Use requests.post()",
            context="[Source 1: quickstart.rst]\nUse requests.post()",
        )

        assert resp.verdict == "grounded"
        assert resp.reasoning == "Matches context perfectly."
        assert resp.error is None
        assert resp.latency_ms > 0

    @patch("rag.critic.get_critic_llm")
    def test_llm_error_handling(self, mock_get_llm):
        """Should return a default 'grounded' response if the LLM call fails."""
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = RuntimeError("API rate limit exceeded")
        mock_get_llm.return_value = mock_llm

        resp = evaluate_answer(
            question="Anything",
            answer="Something",
            context="Context.",
        )

        assert resp.verdict == "grounded"
        assert "Critic failed to evaluate" in resp.reasoning
        assert resp.error is not None
        assert "rate limit" in resp.error
