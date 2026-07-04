#!/usr/bin/env python3
"""Ingest documents into ChromaDB for the RAG pipeline.

Usage:
    # Ingest all files from the default documents directory
    python scripts/ingest.py

    # Ingest from a specific directory
    python scripts/ingest.py --docs-path data/documents/

    # Wipe the collection before ingesting (start fresh)
    python scripts/ingest.py --reset

    # Show what would be ingested without actually doing it
    python scripts/ingest.py --dry-run
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Add project root to path so we can import from the src package
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from rich.console import Console
from rich.table import Table

from config import settings
from rag.embedder import DocumentProcessor, get_embedding_function

console = Console()


def _get_chroma_client():
    """Create a persistent ChromaDB client."""
    import chromadb

    persist_dir = settings.retriever.persist_directory
    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=persist_dir)


def ingest(
    docs_path: str | Path | None = None,
    reset: bool = False,
    dry_run: bool = False,
) -> dict:
    """Load, chunk, embed, and store documents in ChromaDB.

    Args:
        docs_path: Directory containing documents. Defaults to settings.documents_dir.
        reset: If True, delete the existing collection before ingesting.
        dry_run: If True, show what would be ingested without writing to ChromaDB.

    Returns:
        A dict with ingestion stats: file_count, chunk_count, duration_seconds.
    """
    docs_path = Path(docs_path) if docs_path else settings.documents_dir

    # ── Step 1: Load and chunk documents ──────────────────────────
    console.print(f"\n[bold cyan]📂 Loading documents from:[/] {docs_path}")

    processor = DocumentProcessor()
    chunks = processor.load_and_chunk(docs_path)

    # Gather per-file stats
    sources = {}
    for chunk in chunks:
        src = chunk.metadata.get("source", "unknown")
        sources[src] = sources.get(src, 0) + 1

    # Display a summary table
    table = Table(title="Documents Found", show_lines=True)
    table.add_column("File", style="green")
    table.add_column("Chunks", justify="right", style="cyan")
    for src, count in sorted(sources.items()):
        table.add_row(src, str(count))
    table.add_row("[bold]TOTAL[/]", f"[bold]{len(chunks)}[/]")
    console.print(table)

    if dry_run:
        console.print("\n[yellow]🔍 Dry run — no data written to ChromaDB.[/]")
        return {
            "file_count": len(sources),
            "chunk_count": len(chunks),
            "duration_seconds": 0.0,
        }

    # ── Step 2: Set up ChromaDB + embedding function ──────────────
    console.print(f"\n[bold cyan]🗄️  ChromaDB persist dir:[/] {settings.retriever.persist_directory}")
    console.print(f"[bold cyan]📦 Collection:[/] {settings.retriever.collection_name}")

    client = _get_chroma_client()

    if reset:
        console.print("[yellow]⚠️  Resetting collection (--reset flag)...[/]")
        try:
            client.delete_collection(settings.retriever.collection_name)
        except Exception:
            pass  # Collection didn't exist — that's fine

    embed_fn = get_embedding_function()
    console.print(f"[bold cyan]🧠 Embedding model:[/] {embed_fn.model_name}")

    collection = client.get_or_create_collection(
        name=settings.retriever.collection_name,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )

    # ── Step 3: Embed and store ───────────────────────────────────
    console.print(f"\n[bold cyan]⏳ Embedding {len(chunks)} chunks...[/]")
    start_time = time.time()

    # ChromaDB has a batch size limit; process in batches of 100
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]

        ids = [f"{chunk.metadata['source']}::chunk_{chunk.metadata['chunk_index']}" for chunk in batch]
        documents = [chunk.page_content for chunk in batch]
        metadatas = [
            {
                "source": chunk.metadata["source"],
                "chunk_index": chunk.metadata["chunk_index"],
                "total_chunks": chunk.metadata["total_chunks"],
            }
            for chunk in batch
        ]

        collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

        progress = min(i + batch_size, len(chunks))
        console.print(f"  ✅ {progress}/{len(chunks)} chunks stored")

    duration = time.time() - start_time

    # ── Step 4: Verify ────────────────────────────────────────────
    stored_count = collection.count()
    console.print(f"\n[bold green]✅ Ingestion complete![/]")
    console.print(f"   Documents: {len(sources)}")
    console.print(f"   Chunks stored: {stored_count}")
    console.print(f"   Duration: {duration:.1f}s")

    return {
        "file_count": len(sources),
        "chunk_count": stored_count,
        "duration_seconds": round(duration, 2),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Ingest documents into ChromaDB for the RAG pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--docs-path",
        type=str,
        default=None,
        help=f"Path to documents directory (default: {settings.documents_dir})",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete the existing ChromaDB collection before ingesting",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be ingested without writing to ChromaDB",
    )

    args = parser.parse_args()

    try:
        ingest(
            docs_path=args.docs_path,
            reset=args.reset,
            dry_run=args.dry_run,
        )
    except (FileNotFoundError, ValueError) as e:
        console.print(f"\n[bold red]❌ Error:[/] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
