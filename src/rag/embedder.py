"""Document loading, chunking, and embedding for the RAG pipeline.

Handles three responsibilities:
1. **Loading** — reads .rst, .md, and .txt files from a directory
2. **Chunking** — splits documents into overlapping chunks using a recursive
   character text splitter (configurable size + overlap)
3. **Embedding** — wraps sentence-transformers for local, free embeddings

Usage:
    from rag.embedder import DocumentProcessor, get_embedding_function

    processor = DocumentProcessor()
    chunks = processor.load_and_chunk("data/documents/")

    embed_fn = get_embedding_function()
    vectors = embed_fn(["some text"])  # list[list[float]]
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings

# File extensions we know how to load
_SUPPORTED_EXTENSIONS = {".rst", ".md", ".txt"}


@dataclass
class ChunkMetadata:
    """Metadata attached to each chunk for traceability."""

    source_file: str
    chunk_index: int
    total_chunks: int


@dataclass
class DocumentProcessor:
    """Loads documents from disk and splits them into embeddable chunks.

    Parameters match the project's .env defaults (500 token chunk size, 50 overlap)
    but can be overridden at construction time for experimentation.
    """

    chunk_size: int = field(default_factory=lambda: settings.chunking.chunk_size)
    chunk_overlap: int = field(default_factory=lambda: settings.chunking.chunk_overlap)

    def __post_init__(self) -> None:
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_and_chunk(
        self,
        docs_path: str | Path | None = None,
    ) -> list[Document]:
        """Load all supported files from *docs_path* and return chunked Documents.

        Each returned Document carries metadata:
          - ``source``: relative path to the original file
          - ``chunk_index``: position within the source file
          - ``total_chunks``: how many chunks the source file produced

        Args:
            docs_path: Directory to scan. Defaults to ``settings.documents_dir``.

        Returns:
            A flat list of LangChain Documents, ready for embedding.
        """
        docs_path = Path(docs_path) if docs_path else settings.documents_dir

        if not docs_path.exists():
            raise FileNotFoundError(f"Documents directory not found: {docs_path}")

        raw_docs = self._load_files(docs_path)

        if not raw_docs:
            raise ValueError(
                f"No supported files ({', '.join(_SUPPORTED_EXTENSIONS)}) found "
                f"in {docs_path}. Add some documents first."
            )

        all_chunks: list[Document] = []
        for doc in raw_docs:
            chunks = self._split_document(doc)
            all_chunks.extend(chunks)

        return all_chunks

    def chunk_text(self, text: str, source: str = "inline") -> list[Document]:
        """Chunk a raw text string (useful for testing or one-off ingestion).

        Args:
            text: The text to split.
            source: A label for the source metadata.

        Returns:
            A list of chunked Documents.
        """
        if source.lower().endswith(".rst"):
            text = self._clean_rst(text)

        doc = Document(page_content=text, metadata={"source": source})
        return self._split_document(doc)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_files(self, directory: Path) -> list[Document]:
        """Recursively scan a directory and load supported files."""
        documents: list[Document] = []

        for file_path in sorted(directory.rglob("*")):
            if file_path.suffix.lower() not in _SUPPORTED_EXTENSIONS:
                continue
            if file_path.name.startswith("."):
                continue

            text = self._read_file(file_path)
            if not text.strip():
                continue

            # Clean up the text based on format
            if file_path.suffix.lower() == ".rst":
                text = self._clean_rst(text)

            rel_path = file_path.relative_to(directory)
            documents.append(
                Document(
                    page_content=text,
                    metadata={"source": str(rel_path)},
                )
            )

        return documents

    @staticmethod
    def _read_file(path: Path) -> str:
        """Read a file, trying UTF-8 first then falling back to latin-1."""
        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return path.read_text(encoding="latin-1")

    @staticmethod
    def _clean_rst(text: str) -> str:
        """Light cleanup for reStructuredText files.

        Removes RST-specific markup (directives, cross-references) that don't
        add semantic value to the chunks. Keeps the actual prose and code blocks.
        """
        # Remove RST directive headers (.. note::, .. code-block::, etc.)
        # but keep the indented content that follows
        text = re.sub(r"^\.\.\s+\w+::\s*.*$", "", text, flags=re.MULTILINE)

        # Remove RST cross-reference markup like :func:`foo` → foo
        text = re.sub(r":[\w]+:`([^`]+)`", r"\1", text)

        # Remove RST section underlines (==== ---- ~~~~ ^^^^ etc.)
        text = re.sub(r"^[=\-~^\"\'`]+\s*$", "", text, flags=re.MULTILINE)

        # Collapse excessive blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()

    def _split_document(self, doc: Document) -> list[Document]:
        """Split a single document into chunks, adding positional metadata."""
        chunks = self._splitter.split_documents([doc])

        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["total_chunks"] = len(chunks)

        return chunks


# ------------------------------------------------------------------
# Embedding function
# ------------------------------------------------------------------


class LocalEmbeddingFunction:
    """Wraps sentence-transformers as a ChromaDB-compatible embedding function.

    Loads the model lazily on first call to avoid slow import at startup.
    The model runs locally (CPU or GPU) — no API key, no cost.
    """

    is_legacy = False

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or settings.embedding.model_name
        self._model = None

    def name(self) -> str:
        """ChromaDB protocol: unique name for this embedding function."""
        return f"local_sentence_transformers_{self.model_name}"

    def _ensure_model(self):
        """Lazy-load the sentence-transformers model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            # Force CPU. ZeroGPU makes torch.cuda.is_available() return True,
            # but crashes if we use it without the @spaces.GPU decorator.
            self._model = SentenceTransformer(self.model_name, device="cpu")

    def __call__(self, input: Sequence[str]) -> list[list[float]]:
        """Embed a list of texts. Compatible with ChromaDB's EmbeddingFunction protocol."""
        self._ensure_model()
        embeddings = self._model.encode(list(input), show_progress_bar=False)
        return embeddings.tolist()

    def embed_documents(self, documents: list[str]) -> list[list[float]]:
        """Embed a list of documents (ChromaDB 1.5+ protocol)."""
        return self(documents)

    def embed_query(self, input: list[str]) -> list[list[float]]:
        """Embed a query string (ChromaDB 1.5+ protocol)."""
        return self(input)

    @property
    def dimension(self) -> int:
        """Return the embedding dimensionality (e.g., 384 for all-MiniLM-L6-v2)."""
        self._ensure_model()
        return self._model.get_sentence_embedding_dimension()


def get_embedding_function(model_name: str | None = None) -> LocalEmbeddingFunction:
    """Factory for the embedding function. Uses the model from settings by default."""
    return LocalEmbeddingFunction(model_name)
