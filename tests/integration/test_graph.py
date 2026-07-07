import pytest
from rag.graph import (
    build_graph,
    build_graph_v1,
    run_query,
    RAGState,
    retrieve_node,
    generate_node,
    critic_node,
    reformulate_node,
    fallback_node,
    _route_after_critic,
    DEFAULT_MAX_RETRIES,
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

    def test_grounded_routes_to_end(self):
        """A grounded verdict should route to __end__."""
        state: RAGState = {
            "critic_verdict": "grounded",
            "retry_count": 0,
            "max_retries": DEFAULT_MAX_RETRIES,
        }
        result = _route_after_critic(state)
        assert result == "__end__"

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



    def test_reformulate_placeholder_increments_retry(self):
        """Placeholder reformulator should increment retry_count."""
        state: RAGState = {
            "question": "How do I send a GET request?",
            "original_question": "How do I send a GET request?",
            "retry_count": 0,
        }
        result = reformulate_node(state)
        assert result["retry_count"] == 1
        assert "reformulated_query" in result

    def test_reformulate_preserves_original_question(self):
        """Placeholder should echo the original question as the reformulated query."""
        state: RAGState = {
            "question": "reformulated version",
            "original_question": "original version",
            "retry_count": 0,
        }
        result = reformulate_node(state)
        assert result["reformulated_query"] == "original version"

    def test_fallback_returns_refusal(self):
        """Placeholder fallback should return a refusal-like answer."""
        state: RAGState = {
            "retry_count": 2,
        }
        result = fallback_node(state)
        assert "don't have enough information" in result["answer"]
        assert result["confidence"] == "low"
        assert result["error"] is not None


# ------------------------------------------------------------------
# Integration test (requires vector store + API key)
# ------------------------------------------------------------------


@pytest.mark.integration
class TestV2PipelineExecution:
    """End-to-end test of the v2 agentic pipeline.

    Requires a populated vector store and valid GROQ_API_KEY.
    """

    def test_v2_pipeline_returns_expected_state_keys(self):
        """The v2 pipeline should return all expected state fields."""
        result = run_query("How do I send a GET request?", verbose=False)

        # Input fields
        assert "question" in result
        assert "original_question" in result

        # Retriever fields
        assert "retrieved_chunks" in result
        assert "context" in result

        # Generator fields
        assert "answer" in result
        assert "sources_used" in result

        # Critic fields (from placeholder)
        assert "critic_verdict" in result
        assert "critic_reasoning" in result

        # Control flow
        assert "retry_count" in result

        # Basic sanity
        assert len(result["retrieved_chunks"]) > 0
        assert isinstance(result["answer"], str)
        assert len(result["answer"]) > 0
        assert result["critic_verdict"] in ("grounded", "not_grounded", "partially_grounded")
