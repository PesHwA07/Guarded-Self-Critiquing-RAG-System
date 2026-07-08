"""Unit tests for the query reformulator module."""

from unittest.mock import MagicMock, patch

import pytest

from rag.reformulator import reformulate_query


class TestReformulator:
    """Test the reformulate_query function."""

    @patch("rag.reformulator.get_generator_llm")
    def test_successful_reformulation(self, mock_get_llm):
        """Test that a valid new query is returned and cleaned."""
        from langchain_core.language_models import FakeListChatModel
        
        # Setup fake LLM that returns a predictable response
        fake_llm = FakeListChatModel(responses=['  "better search query"  \n'])
        mock_get_llm.return_value = fake_llm
        
        result = reformulate_query("original", "critic reasoning")
        
        # Quotes and whitespace should be stripped
        assert result == "better search query"

    @patch("rag.reformulator.get_generator_llm")
    def test_empty_response_fallback(self, mock_get_llm):
        """Test that it falls back to original question if LLM returns empty."""
        from langchain_core.language_models import FakeListChatModel
        
        # Setup fake LLM that returns an empty string (whitespace)
        fake_llm = FakeListChatModel(responses=["   \n  "])
        mock_get_llm.return_value = fake_llm
        
        result = reformulate_query("original question", "critic reasoning")
        
        # Should return the original question
        assert result == "original question"

    @patch("rag.reformulator.get_generator_llm")
    def test_error_fallback(self, mock_get_llm):
        """Test that it falls back to original question on LLM error."""
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
        # Make the LLM raise an exception
        mock_llm.invoke.side_effect = Exception("API rate limit")
        
        result = reformulate_query("original question", "critic reasoning")
        
        # Should return the original question
        assert result == "original question"
