import pytest

from rag.graph import (
    DEFAULT_MAX_RETRIES,
    RAGState,
    _route_after_critic,
    build_graph,
    build_graph_v1,
    fallback_node,
    run_query,
)

# ------------------------------------------------------------------
# Graph structure tests
# ------------------------------------------------------------------


class TestGraphStructure:
    """Test that both v1 and v2 graphs compile with the expected nodes."""

    def test_build_graph_v1_has_expected_nodes(self):
        """V1 graph should have retrieve and generate nodes."""
        graph = build_graph_v1()
        nodes = list(graph.nodes.keys())
        assert "retrieve" in nodes
        assert "generate" in nodes

    def test_build_graph_v2_has_all_nodes(self):
        """V2 graph should have retrieve, generate, critic, reformulate, and fallback."""
        graph = build_graph()
        nodes = list(graph.nodes.keys())
        assert "retrieve" in nodes
        assert "generate" in nodes
        assert "critic" in nodes
        assert "reformulate" in nodes
        assert "fallback" in nodes


# ------------------------------------------------------------------
# Routing logic tests (no LLM calls needed)
# ------------------------------------------------------------------


class TestRouteAfterCritic:
    """Test the conditional routing logic after the critic node."""

    def test_grounded_routes_to_output_guard(self):
        """A grounded verdict should route to output_guard."""
        state: RAGState = {
            "critic_verdict": "grounded",
            "retry_count": 0,
            "max_retries": DEFAULT_MAX_RETRIES,
        }
        result = _route_after_critic(state)
        assert result == "output_guard"

    def test_not_grounded_with_retries_routes_to_reformulate(self):
        """A not_grounded verdict with retries left should route to reformulate."""
        state: RAGState = {
            "critic_verdict": "not_grounded",
            "retry_count": 0,
            "max_retries": 2,
        }
        result = _route_after_critic(state)
        assert result == "reformulate"

    def test_partially_grounded_with_retries_routes_to_reformulate(self):
        """A partially_grounded verdict should also trigger reformulation."""
        state: RAGState = {
            "critic_verdict": "partially_grounded",
            "retry_count": 1,
            "max_retries": 2,
        }
        result = _route_after_critic(state)
        assert result == "reformulate"

    def test_not_grounded_retries_exhausted_routes_to_fallback(self):
        """When retries are exhausted, should route to fallback."""
        state: RAGState = {
            "critic_verdict": "not_grounded",
            "retry_count": 2,
            "max_retries": 2,
        }
        result = _route_after_critic(state)
        assert result == "fallback"

    def test_default_max_retries_used_when_not_set(self):
        """If max_retries is not in state, DEFAULT_MAX_RETRIES should be used."""
        state: RAGState = {
            "critic_verdict": "not_grounded",
            "retry_count": 0,
        }
        result = _route_after_critic(state)
        assert result == "reformulate"

    def test_zero_max_retries_goes_straight_to_fallback(self):
        """With max_retries=0, any rejection should go straight to fallback."""
        state: RAGState = {
            "critic_verdict": "not_grounded",
            "retry_count": 0,
            "max_retries": 0,
        }
        result = _route_after_critic(state)
        assert result == "fallback"


# ------------------------------------------------------------------
# Placeholder node tests (no LLM calls needed)
# ------------------------------------------------------------------


class TestPlaceholderNodes:
    """Test that placeholder nodes produce the expected state updates."""





    def test_fallback_returns_refusal(self):
        """Fallback should return a refusal-like answer and format chunks."""
        state: RAGState = {
            "retry_count": 2,
        }
        result = fallback_node(state)
        assert "don't have enough information" in result["answer"]
        assert result["confidence"] == "none"
        assert result["error"] is not None


# ------------------------------------------------------------------
# Integration test (requires vector store + API key)
# ------------------------------------------------------------------


from unittest.mock import MagicMock, patch


@pytest.mark.integration
class TestV2PipelineExecution:
    """End-to-end test of the v2 agentic pipeline.

    Requires a populated vector store and valid GROQ_API_KEY.
    """

    @patch("rag.graph.evaluate_answer")
    def test_v2_pipeline_path_grounded(self, mock_evaluate):
        """Test the path where the generated answer is immediately grounded."""
        mock_response = MagicMock()
        mock_response.verdict = "grounded"
        mock_response.reasoning = "Looks good."
        mock_response.latency_ms = 100.0
        mock_evaluate.return_value = mock_response

        result = run_query("How do I send a GET request?", verbose=False)

        # Verify it went straight to END
        assert result["critic_verdict"] == "grounded"
        assert result["retry_count"] == 0
        assert "answer" in result
        assert mock_evaluate.call_count == 1

    @patch("rag.graph.evaluate_answer")
    def test_v2_pipeline_path_reformulate_then_grounded(self, mock_evaluate):
        """Test the path where critic rejects once, reformulates, then accepts."""
        # First call: not grounded. Second call: grounded.
        mock_fail = MagicMock()
        mock_fail.verdict = "not_grounded"
        mock_fail.reasoning = "Missing info."
        mock_fail.latency_ms = 100.0

        mock_pass = MagicMock()
        mock_pass.verdict = "grounded"
        mock_pass.reasoning = "Now it's good."
        mock_pass.latency_ms = 100.0

        mock_evaluate.side_effect = [mock_fail, mock_pass]

        result = run_query("How do I send a GET request?", verbose=False)

        # Verify it went through the reformulate loop once
        assert result["critic_verdict"] == "grounded"
        assert result["retry_count"] == 1
        assert mock_evaluate.call_count == 2
        assert "answer" in result

    @patch("rag.graph.evaluate_answer")
    def test_v2_pipeline_path_fallback(self, mock_evaluate):
        """Test the path where retries are exhausted and it triggers the fallback."""
        # Always return not grounded to force exhaustion of retries
        mock_fail = MagicMock()
        mock_fail.verdict = "not_grounded"
        mock_fail.reasoning = "Still bad."
        mock_fail.latency_ms = 100.0
        mock_evaluate.return_value = mock_fail

        result = run_query("How do I send a GET request?", verbose=False)

        # Verify it exhausted retries and hit fallback
        assert result["critic_verdict"] == "not_grounded"
        assert result["retry_count"] == 2
        assert mock_evaluate.call_count == 3 # Initial generation + 2 retries = 3 generations
        assert "don't have enough information" in result["answer"]
