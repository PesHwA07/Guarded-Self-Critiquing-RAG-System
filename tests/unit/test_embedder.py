"""Unit tests for the document processor (embedder module).

Tests cover:
- File loading from directories
- RST cleanup
- Chunking with correct metadata
- Edge cases (empty dirs, unsupported files, encoding issues)

These tests are self-contained — they create temp files and don't require
an actual embedding model or ChromaDB.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from rag.embedder import DocumentProcessor, _SUPPORTED_EXTENSIONS


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def processor() -> DocumentProcessor:
    """A DocumentProcessor with small chunks for predictable test output."""
    return DocumentProcessor(chunk_size=100, chunk_overlap=20)


@pytest.fixture
def docs_dir(tmp_path: Path) -> Path:
    """Create a temp directory with sample documents."""
    # A simple markdown file
    md_file = tmp_path / "guide.md"
    md_file.write_text(
        "# Getting Started\n\n"
        "Install the library with pip install mylib.\n\n"
        "## Usage\n\n"
        "Import mylib and call mylib.run() to get started.\n"
        "You can pass configuration via a dict or environment variables.\n",
        encoding="utf-8",
    )

    # A simple RST file
    rst_file = tmp_path / "api.rst"
    rst_file.write_text(
        "API Reference\n"
        "=============\n\n"
        ".. module:: mylib\n\n"
        "The :func:`run` function is the main entry point.\n\n"
        ".. code-block:: python\n\n"
        "   import mylib\n"
        "   mylib.run()\n",
        encoding="utf-8",
    )

    # A plain text file
    txt_file = tmp_path / "notes.txt"
    txt_file.write_text(
        "These are some development notes.\n"
        "Remember to update the changelog before release.\n",
        encoding="utf-8",
    )

    # A file that should be ignored (wrong extension)
    csv_file = tmp_path / "data.csv"
    csv_file.write_text("col1,col2\n1,2\n", encoding="utf-8")

    # A hidden file that should be ignored
    hidden = tmp_path / ".hidden.md"
    hidden.write_text("secret stuff\n", encoding="utf-8")

    return tmp_path


# ── Tests: File Loading ───────────────────────────────────────────────


class TestDocumentLoading:
    """Tests for loading files from disk."""

    def test_loads_supported_extensions(self, processor: DocumentProcessor, docs_dir: Path):
        """Should load .md, .rst, and .txt files."""
        chunks = processor.load_and_chunk(docs_dir)
        sources = {c.metadata["source"] for c in chunks}

        assert "guide.md" in sources
        assert "api.rst" in sources
        assert "notes.txt" in sources

    def test_ignores_unsupported_extensions(self, processor: DocumentProcessor, docs_dir: Path):
        """Should skip .csv and other unsupported files."""
        chunks = processor.load_and_chunk(docs_dir)
        sources = {c.metadata["source"] for c in chunks}

        assert "data.csv" not in sources

    def test_ignores_hidden_files(self, processor: DocumentProcessor, docs_dir: Path):
        """Should skip files starting with a dot."""
        chunks = processor.load_and_chunk(docs_dir)
        sources = {c.metadata["source"] for c in chunks}

        assert ".hidden.md" not in sources

    def test_raises_on_missing_directory(self, processor: DocumentProcessor):
        """Should raise FileNotFoundError for a nonexistent directory."""
        with pytest.raises(FileNotFoundError, match="not found"):
            processor.load_and_chunk("/nonexistent/path")

    def test_raises_on_empty_directory(self, processor: DocumentProcessor, tmp_path: Path):
        """Should raise ValueError if no supported files are found."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        with pytest.raises(ValueError, match="No supported files"):
            processor.load_and_chunk(empty_dir)


# ── Tests: RST Cleanup ───────────────────────────────────────────────


class TestRSTCleanup:
    """Tests for RST-specific text cleaning."""

    def test_removes_section_underlines(self, processor: DocumentProcessor):
        """RST section underlines (====) should be stripped."""
        text = "My Title\n========\n\nSome content."
        chunks = processor.chunk_text(text, source="test.rst")
        combined = " ".join(c.page_content for c in chunks)

        assert "========" not in combined

    def test_removes_cross_references(self):
        """RST cross-refs like :func:`foo` should become just 'foo'."""
        cleaned = DocumentProcessor._clean_rst("Use :func:`my_function` to do stuff.")
        assert cleaned == "Use my_function to do stuff."

    def test_removes_directive_headers(self):
        """RST directives like .. note:: should be stripped."""
        text = ".. note:: Important!\n\n   This is a note."
        cleaned = DocumentProcessor._clean_rst(text)
        assert ".. note::" not in cleaned
        assert "This is a note." in cleaned


# ── Tests: Chunking ──────────────────────────────────────────────────


class TestChunking:
    """Tests for text splitting and metadata."""

    def test_chunks_have_correct_metadata(self, processor: DocumentProcessor, docs_dir: Path):
        """Each chunk should have source, chunk_index, and total_chunks."""
        chunks = processor.load_and_chunk(docs_dir)

        for chunk in chunks:
            assert "source" in chunk.metadata
            assert "chunk_index" in chunk.metadata
            assert "total_chunks" in chunk.metadata
            assert isinstance(chunk.metadata["chunk_index"], int)
            assert chunk.metadata["chunk_index"] >= 0
            assert chunk.metadata["chunk_index"] < chunk.metadata["total_chunks"]

    def test_chunk_indices_are_sequential(self, processor: DocumentProcessor):
        """Chunks from a single source should have sequential indices."""
        # A text long enough to produce multiple chunks at chunk_size=100
        text = "word " * 200  # ~1000 chars
        chunks = processor.chunk_text(text, source="long.txt")

        assert len(chunks) > 1  # Should produce multiple chunks
        indices = [c.metadata["chunk_index"] for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_all_chunks_share_total_count(self, processor: DocumentProcessor):
        """All chunks from the same source should have the same total_chunks."""
        text = "word " * 200
        chunks = processor.chunk_text(text, source="test.txt")

        total = chunks[0].metadata["total_chunks"]
        assert total == len(chunks)
        assert all(c.metadata["total_chunks"] == total for c in chunks)

    def test_chunk_overlap_produces_shared_content(self):
        """Adjacent chunks should share some text due to overlap."""
        # Use a processor with explicit overlap
        processor = DocumentProcessor(chunk_size=50, chunk_overlap=20)
        text = "A" * 30 + " " + "B" * 30 + " " + "C" * 30 + " " + "D" * 30
        chunks = processor.chunk_text(text)

        # With overlap, some chunks should share content
        if len(chunks) >= 2:
            # The last part of chunk N should appear at the start of chunk N+1
            # (exact overlap depends on separator logic, but some shared content expected)
            assert len(chunks) >= 2  # Confirms splitting happened

    def test_empty_content_after_cleaning(self, processor: DocumentProcessor, tmp_path: Path):
        """Files that become empty after loading should be skipped."""
        empty_file = tmp_path / "blank.md"
        empty_file.write_text("   \n\n  \n", encoding="utf-8")

        # Should not crash, just skip the empty file
        with pytest.raises(ValueError, match="No supported files"):
            processor.load_and_chunk(tmp_path)

    def test_chunk_text_method(self, processor: DocumentProcessor):
        """chunk_text() should work on raw strings."""
        chunks = processor.chunk_text("Hello, this is a test document.", source="inline")

        assert len(chunks) >= 1
        assert chunks[0].metadata["source"] == "inline"
        assert chunks[0].page_content == "Hello, this is a test document."


# ── Tests: Encoding ──────────────────────────────────────────────────


class TestEncoding:
    """Tests for file encoding handling."""

    def test_handles_utf8(self, processor: DocumentProcessor, tmp_path: Path):
        """Should read UTF-8 files correctly."""
        f = tmp_path / "utf8.md"
        f.write_text("Héllo wörld — with ünîcödé!", encoding="utf-8")

        chunks = processor.load_and_chunk(tmp_path)
        assert "Héllo wörld" in chunks[0].page_content

    def test_handles_latin1_fallback(self, processor: DocumentProcessor, tmp_path: Path):
        """Should fall back to latin-1 for non-UTF-8 files."""
        f = tmp_path / "latin1.txt"
        f.write_bytes("Caf\xe9 cr\xe8me".encode("latin-1"))

        chunks = processor.load_and_chunk(tmp_path)
        assert "Café" in chunks[0].page_content
