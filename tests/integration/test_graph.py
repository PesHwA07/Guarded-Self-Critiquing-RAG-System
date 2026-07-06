import pytest
from rag.graph import build_graph, run_query

def test_build_graph():
    """Test that the graph compiles and has the correct nodes."""
    graph = build_graph()
    
    # Verify the structure (just checking basic properties)
    nodes = list(graph.nodes.keys())
    # Depending on langgraph version, '__start__' and '__end__' might be present.
    # We just want to ensure 'retrieve' and 'generate' are in there.
    assert "retrieve" in nodes
    assert "generate" in nodes

@pytest.mark.integration
def test_linear_chain_execution():
    """Test running a real query through the pipeline.
    
    This requires that the vector store is populated and LLM provider is configured.
    Since we don't want to burn Groq credits in every CI run for integration tests,
    this is just a basic smoke test. In the future, this can be mocked.
    """
    question = "How do I send a GET request?"
    
    # We use verbose=False to not pollute test output unnecessarily
    result = run_query(question, verbose=False)
    
    # Assert expected keys in the output state
    assert "question" in result
    assert "retrieved_chunks" in result
    assert "context" in result
    assert "answer" in result
    assert "sources_used" in result
    assert "latency_ms" in result
    
    # Verify we got some chunks
    assert len(result["retrieved_chunks"]) > 0
    assert isinstance(result["answer"], str)
    assert len(result["answer"]) > 0
