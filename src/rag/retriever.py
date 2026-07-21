"""Vector store retriever — similarity search over ChromaDB and Qdrant.

Wraps vector databases behind a clean interface so the rest of the pipeline only deals
with LangChain Documents. Uses the same local embedding function from
``embedder.py`` — no API key, no cost for embeddings.

Usage:
    from rag.retriever import get_retriever

    retriever = get_retriever()
    retriever.ingest(chunks)             # one-time
    results = retriever.retrieve("How do I send a POST request?")
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import chromadb
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

from config import settings
from rag.embedder import LocalEmbeddingFunction, get_embedding_function

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """A single retrieved chunk with its similarity score."""

    document: Document
    score: float  # lower distance = better match (L2 / Cosine distance)

    @property
    def content(self) -> str:
        return self.document.page_content

    @property
    def source(self) -> str:
        return self.document.metadata.get("source", "unknown")


class BaseRetriever(ABC):
    """Abstract base class for vector store retrievers."""

    @abstractmethod
    def ingest(self, documents: Sequence[Document], *, batch_size: int = 100) -> int:
        pass

    @abstractmethod
    def retrieve(self, query: str, *, top_k: int | None = None) -> list[RetrievalResult]:
        pass

    @property
    @abstractmethod
    def count(self) -> int:
        pass

    @abstractmethod
    def clear(self) -> None:
        pass


@dataclass
class ChromaRetriever(BaseRetriever):
    """ChromaDB-backed retriever with local embeddings."""

    collection_name: str = field(
        default_factory=lambda: settings.retriever.collection_name
    )
    persist_directory: str = field(
        default_factory=lambda: settings.retriever.persist_directory
    )
    top_k: int = field(default_factory=lambda: settings.retriever.top_k)
    embedding_fn: LocalEmbeddingFunction | None = None

    def __post_init__(self) -> None:
        if self.embedding_fn is None:
            self.embedding_fn = get_embedding_function()

        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=self.persist_directory)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "l2"},
        )

    def ingest(
        self,
        documents: Sequence[Document],
        *,
        batch_size: int = 100,
    ) -> int:
        if not documents:
            return 0

        added = 0
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            ids, texts, metadatas = [], [], []

            for doc in batch:
                source = doc.metadata.get("source", "unknown")
                chunk_idx = doc.metadata.get("chunk_index", 0)
                doc_id = f"{source}::chunk_{chunk_idx}"

                ids.append(doc_id)
                texts.append(doc.page_content)
                metadatas.append(doc.metadata)

            self._collection.upsert(ids=ids, documents=texts, metadatas=metadatas)
            added += len(batch)

        logger.info("Ingested %d documents into Chroma collection '%s'.", added, self.collection_name)
        return added

    def retrieve(self, query: str, *, top_k: int | None = None) -> list[RetrievalResult]:
        k = top_k or self.top_k
        if self._collection.count() == 0:
            return []

        results = self._collection.query(
            query_texts=[query],
            n_results=min(k, self._collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        retrieval_results = []
        for text, meta, dist in zip(documents, metadatas, distances):
            doc = Document(page_content=text, metadata=meta or {})
            retrieval_results.append(RetrievalResult(document=doc, score=dist))

        return retrieval_results

    @property
    def count(self) -> int:
        return self._collection.count()

    def clear(self) -> None:
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "l2"},
        )


@dataclass
class QdrantRetriever(BaseRetriever):
    """Qdrant-backed retriever with local embeddings."""

    collection_name: str = field(
        default_factory=lambda: settings.retriever.collection_name
    )
    top_k: int = field(default_factory=lambda: settings.retriever.top_k)
    url: str = field(default_factory=lambda: settings.retriever.qdrant_url)
    api_key: str = field(default_factory=lambda: settings.retriever.qdrant_api_key)
    embedding_fn: LocalEmbeddingFunction | None = None

    def __post_init__(self) -> None:
        if self.embedding_fn is None:
            self.embedding_fn = get_embedding_function()

        # If no URL is provided, fall back to an in-memory/local Qdrant setup for testing
        if not self.url:
            logger.warning("No QDRANT_URL found. Using in-memory Qdrant client.")
            self._client = QdrantClient(location=":memory:")
        else:
            self._client = QdrantClient(url=self.url, api_key=self.api_key)

        # Ensure collection exists
        try:
            self._client.get_collection(self.collection_name)
        except Exception:
            # Get embedding dimension
            test_embedding = self.embedding_fn(["test"])[0]
            vector_size = len(test_embedding)

            self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config=rest.VectorParams(
                    size=vector_size,
                    distance=rest.Distance.COSINE,
                ),
            )

    def ingest(self, documents: Sequence[Document], *, batch_size: int = 100) -> int:
        if not documents:
            return 0

        added = 0
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            points = []

            texts = [doc.page_content for doc in batch]
            embeddings = self.embedding_fn(texts)

            for doc, emb in zip(batch, embeddings):
                source = doc.metadata.get("source", "unknown")
                chunk_idx = doc.metadata.get("chunk_index", 0)
                # Qdrant IDs must be integers or UUIDs, we can use a hash
                import hashlib
                doc_id_str = f"{source}::chunk_{chunk_idx}"
                doc_id = hashlib.md5(doc_id_str.encode()).hexdigest()

                payload = doc.metadata.copy()
                payload["page_content"] = doc.page_content

                points.append(
                    rest.PointStruct(
                        id=doc_id,
                        vector=emb,
                        payload=payload
                    )
                )

            self._client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            added += len(batch)

        logger.info("Ingested %d documents into Qdrant collection '%s'.", added, self.collection_name)
        return added

    def retrieve(self, query: str, *, top_k: int | None = None) -> list[RetrievalResult]:
        k = top_k or self.top_k
        if self.count == 0:
            return []

        query_vector = self.embedding_fn([query])[0]

        results = self._client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=k,
            with_payload=True
        )

        retrieval_results = []
        for scored_point in results:
            payload = scored_point.payload or {}
            content = payload.pop("page_content", "")
            doc = Document(page_content=content, metadata=payload)
            # Qdrant uses cosine similarity typically; score is higher for better matches if cosine.
            # But we map it to our score format.
            retrieval_results.append(RetrievalResult(document=doc, score=scored_point.score))

        return retrieval_results

    @property
    def count(self) -> int:
        try:
            return self._client.get_collection(self.collection_name).points_count
        except Exception:
            return 0

    def clear(self) -> None:
        try:
            self._client.delete_collection(self.collection_name)
        except Exception:
            pass

        test_embedding = self.embedding_fn(["test"])[0]
        vector_size = len(test_embedding)

        self._client.create_collection(
            collection_name=self.collection_name,
            vectors_config=rest.VectorParams(
                size=vector_size,
                distance=rest.Distance.COSINE,
            ),
        )


def get_retriever() -> BaseRetriever:
    """Factory function to get the configured vector store."""
    if settings.retriever.vector_store == "qdrant":
        return QdrantRetriever()
    else:
        return ChromaRetriever()


def format_context(results: list[RetrievalResult]) -> str:
    """Format retrieval results into a context string for the LLM prompt.

    Each chunk is wrapped with its source and index for traceability.

    Args:
        results: Output from ``VectorStoreRetriever.retrieve()``.

    Returns:
        A formatted string ready to inject into the generator prompt.
    """
    if not results:
        return "(No relevant documents found.)"

    parts: list[str] = []
    for i, r in enumerate(results, 1):
        source = r.source
        chunk_idx = r.document.metadata.get("chunk_index", "?")
        parts.append(
            f"[Source {i}: {source} (chunk {chunk_idx})]\n{r.content}"
        )

    return "\n\n---\n\n".join(parts)
