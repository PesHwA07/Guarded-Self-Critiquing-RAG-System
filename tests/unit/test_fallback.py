"""Unit tests for the fallback module."""

from langchain_core.documents import Document
from rag.retriever import RetrievalResult
from rag.fallback import generate_fallback_response

def test_generate_fallback_response_no_chunks():
    """Test fallback when no chunks were retrieved."""
    result = generate_fallback_response([], retries=2)
    
    assert "don't have enough information" in result["answer"]
    assert "I could not find any relevant documents" in result["answer"]
    assert result["confidence"] == "none"
    assert "Exhausted 2 retries" in result["error"]

def test_generate_fallback_response_with_chunks():
    """Test fallback formats retrieved chunks correctly."""
    chunks = [
        RetrievalResult(
            document=Document(page_content="Snippet A", metadata={"source": "file1.txt"}),
            score=0.1
        ),
        RetrievalResult(
            document=Document(page_content="Snippet B", metadata={"source": "file2.txt"}),
            score=0.2
        )
    ]
    
    result = generate_fallback_response(chunks, retries=1)
    
    assert "don't have enough information" in result["answer"]
    assert "here are the excerpts I found" in result["answer"]
    assert "**Source 1: file1.txt**" in result["answer"]
    assert "Snippet A" in result["answer"]
    assert "**Source 2: file2.txt**" in result["answer"]
    assert "Snippet B" in result["answer"]
    assert result["confidence"] == "none"
    assert "Exhausted 1 retries" in result["error"]
