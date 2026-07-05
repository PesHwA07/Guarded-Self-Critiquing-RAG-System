"""Vector store retriever — similarity search over ChromaDB.

Wraps ChromaDB behind a clean interface so the rest of the pipeline only deals
with LangChain Documents.  Uses the same local embedding function from
``embedder.py`` — no API key, no cost.

Usage:
    from rag.retriever import VectorStoreRetriever

    retriever = VectorStoreRetriever()
    retriever.ingest(chunks)             # one-time
    results = retriever.retrieve("How do I send a POST request?")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import chromadb
from langchain_core.documents import Document

from config import settings
from rag.embedder import get_embedding_function, LocalEmbeddingFunction

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """A single retrieved chunk with its similarity score."""

    document: Document
    score: float  # lower distance = better match (L2)

    @property
    def content(self) -> str:
        return self.document.page_content

    @property
    def source(self) -> str:
        return self.document.metadata.get("source", "unknown")


@dataclass
class VectorStoreRetriever:
    """ChromaDB-backed retriever with local embeddings.

    Manages a single ChromaDB collection.  Documents are embedded using
    ``sentence-transformers`` (free, local) and stored on disk so you only
    need to run ingestion once.

    Parameters:
        collection_name: ChromaDB collection name.
        persist_directory: Where ChromaDB stores data on disk.
        top_k: Number of results to return per query.
        embedding_fn: Custom embedding function (defaults to project settings).
    """

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

        # Ensure persist directory exists
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(path=self.persist_directory)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "l2"},  # L2 distance — lower is better
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest(
        self,
        documents: Sequence[Document],
        *,
        batch_size: int = 100,
    ) -> int:
        """Add documents to the vector store.

        Skips documents whose content is already present (deduplication by
        content hash via ChromaDB's ID mechanism).

        Args:
            documents: LangChain Documents (typically from ``DocumentProcessor``).
            batch_size: How many to upsert at once (ChromaDB recommends ≤5000).

        Returns:
            Number of documents actually added (after dedup).
        """
        if not documents:
            logger.warning("ingest() called with no documents — nothing to do.")
            return 0

        added = 0
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]

            ids = []
            texts = []
            metadatas = []

            for doc in batch:
                # Deterministic ID from source + chunk_index for deduplication
                source = doc.metadata.get("source", "unknown")
                chunk_idx = doc.metadata.get("chunk_index", 0)
                doc_id = f"{source}::chunk_{chunk_idx}"

                ids.append(doc_id)
                texts.append(doc.page_content)
                metadatas.append(doc.metadata)

            self._collection.upsert(
                ids=ids,
                documents=texts,
                metadatas=metadatas,
            )
            added += len(batch)

        logger.info(
            "Ingested %d documents into collection '%s'.",
            added,
            self.collection_name,
        )
        return added

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve the most similar chunks for a query.

        Args:
            query: The user's question.
            top_k: Override the default number of results.

        Returns:
            A list of ``RetrievalResult`` objects, sorted by relevance
            (best match first).
        """
        k = top_k or self.top_k

        # Guard against empty collection
        if self._collection.count() == 0:
            logger.warning(
                "Collection '%s' is empty — run ingest first.",
                self.collection_name,
            )
            return []

        results = self._collection.query(
            query_texts=[query],
            n_results=min(k, self._collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        # Unpack ChromaDB's nested list format
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        retrieval_results: list[RetrievalResult] = []
        for text, meta, dist in zip(documents, metadatas, distances):
            doc = Document(page_content=text, metadata=meta or {})
            retrieval_results.append(RetrievalResult(document=doc, score=dist))

        logger.debug(
            "Retrieved %d chunks for query: '%s...'",
            len(retrieval_results),
            query[:60],
        )
        return retrieval_results

    @property
    def count(self) -> int:
        """Number of documents currently in the collection."""
        return self._collection.count()

    def clear(self) -> None:
        """Delete the collection entirely. Useful for re-ingestion."""
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "l2"},
        )
        logger.info("Cleared collection '%s'.", self.collection_name)


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
