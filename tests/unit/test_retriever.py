"""Unit tests for the vector store retriever (src/rag/retriever.py).

Tests cover:
- Ingestion (adding docs, deduplication, batch handling)
- Retrieval (similarity search, empty collection guard, top_k)
- Context formatting
- Collection management (count, clear)

All tests use a temporary ChromaDB directory so they don't touch real data.
"""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

from rag.retriever import (
    RetrievalResult,
    VectorStoreRetriever,
    format_context,
)

# ── Fixtures ──────────────────────────────────────────────────────────


class FakeEmbeddingFunction:
    """Deterministic embedding function for testing.

    Maps text to a simple vector based on length, so we get repeatable
    results without loading a real model.  Implements the ChromaDB
    EmbeddingFunction protocol (name, is_legacy).
    """

    is_legacy = False

    def name(self) -> str:
        return "fake_test_embedding"

    def __call__(self, input: list[str]) -> list[list[float]]:
        # 10-dim vector: [len, len/10, len/100, ...]
        return [
            [len(text) / (10**i) for i in range(10)]
            for text in input
        ]

    def embed_documents(self, documents: list[str]) -> list[list[float]]:
        return self(documents)

    def embed_query(self, input: list[str]) -> list[list[float]]:
        return self(input)


@pytest.fixture
def retriever(tmp_path):
    """Create a retriever backed by a temp directory."""
    return VectorStoreRetriever(
        collection_name="test_collection",
        persist_directory=str(tmp_path / "chroma_test"),
        top_k=3,
        embedding_fn=FakeEmbeddingFunction(),
    )


@pytest.fixture
def sample_docs():
    """A few sample documents to ingest."""
    return [
        Document(
            page_content="To send a POST request, use requests.post() with the url and data parameters.",
            metadata={"source": "quickstart.rst", "chunk_index": 0, "total_chunks": 3},
        ),
        Document(
            page_content="Authentication can be added using the auth parameter in any request method.",
            metadata={"source": "authentication.rst", "chunk_index": 0, "total_chunks": 2},
        ),
        Document(
            page_content="For sending JSON data, use the json parameter instead of data.",
            metadata={"source": "quickstart.rst", "chunk_index": 1, "total_chunks": 3},
        ),
        Document(
            page_content="The requests.Session object allows you to persist parameters across requests.",
            metadata={"source": "advanced.rst", "chunk_index": 0, "total_chunks": 5},
        ),
        Document(
            page_content="Timeouts prevent your program from hanging indefinitely on network issues.",
            metadata={"source": "quickstart.rst", "chunk_index": 2, "total_chunks": 3},
        ),
    ]


# ── TestIngestion ─────────────────────────────────────────────────────


class TestIngestion:
    """Tests for adding documents to the vector store."""

    def test_ingest_adds_documents(self, retriever, sample_docs):
        """Documents should be added and countable."""
        added = retriever.ingest(sample_docs)
        assert added == 5
        assert retriever.count == 5

    def test_ingest_deduplication(self, retriever, sample_docs):
        """Ingesting the same docs twice should not duplicate them."""
        retriever.ingest(sample_docs)
        retriever.ingest(sample_docs)  # upsert — same IDs
        assert retriever.count == 5

    def test_ingest_empty_list(self, retriever):
        """Ingesting an empty list should return 0."""
        added = retriever.ingest([])
        assert added == 0
        assert retriever.count == 0

    def test_ingest_batch_size(self, retriever):
        """Ingesting more docs than batch_size should still work."""
        docs = [
            Document(
                page_content=f"Document number {i} content.",
                metadata={"source": "bulk.rst", "chunk_index": i, "total_chunks": 150},
            )
            for i in range(150)
        ]
        added = retriever.ingest(docs, batch_size=50)
        assert added == 150
        assert retriever.count == 150


# ── TestRetrieval ─────────────────────────────────────────────────────


class TestRetrieval:
    """Tests for querying the vector store."""

    def test_retrieve_returns_results(self, retriever, sample_docs):
        """Basic retrieve should return a list of RetrievalResult."""
        retriever.ingest(sample_docs)
        results = retriever.retrieve("POST request")

        assert len(results) > 0
        assert len(results) <= 3  # top_k=3
        assert all(isinstance(r, RetrievalResult) for r in results)

    def test_retrieve_empty_collection(self, retriever):
        """Retrieving from an empty collection should return an empty list."""
        results = retriever.retrieve("anything")
        assert results == []

    def test_retrieve_custom_top_k(self, retriever, sample_docs):
        """top_k override should limit results."""
        retriever.ingest(sample_docs)
        results = retriever.retrieve("request", top_k=2)
        assert len(results) <= 2

    def test_retrieval_result_properties(self, retriever, sample_docs):
        """RetrievalResult should expose content and source."""
        retriever.ingest(sample_docs)
        results = retriever.retrieve("authentication")

        r = results[0]
        assert isinstance(r.content, str)
        assert len(r.content) > 0
        assert isinstance(r.source, str)
        assert isinstance(r.score, float)


# ── TestCollectionManagement ──────────────────────────────────────────


class TestCollectionManagement:
    """Tests for collection-level operations."""

    def test_clear_resets_collection(self, retriever, sample_docs):
        """clear() should remove all documents."""
        retriever.ingest(sample_docs)
        assert retriever.count == 5

        retriever.clear()
        assert retriever.count == 0

    def test_count_reflects_state(self, retriever, sample_docs):
        """count should update after ingest and clear."""
        assert retriever.count == 0
        retriever.ingest(sample_docs[:2])
        assert retriever.count == 2


# ── TestFormatContext ─────────────────────────────────────────────────


class TestFormatContext:
    """Tests for the context formatting helper."""

    def test_format_context_with_results(self):
        """Should format each result with source labels."""
        results = [
            RetrievalResult(
                document=Document(
                    page_content="Use requests.post() for POST.",
                    metadata={"source": "quickstart.rst", "chunk_index": 0},
                ),
                score=0.1,
            ),
            RetrievalResult(
                document=Document(
                    page_content="JSON data uses the json= parameter.",
                    metadata={"source": "quickstart.rst", "chunk_index": 1},
                ),
                score=0.3,
            ),
        ]

        context = format_context(results)

        assert "[Source 1: quickstart.rst (chunk 0)]" in context
        assert "[Source 2: quickstart.rst (chunk 1)]" in context
        assert "requests.post()" in context
        assert "---" in context

    def test_format_context_empty(self):
        """Empty results should produce a fallback message."""
        context = format_context([])
        assert "No relevant documents found" in context
