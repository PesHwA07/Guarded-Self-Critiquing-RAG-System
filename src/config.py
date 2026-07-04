"""Centralized configuration for the Guarded RAG System.

Reads settings from environment variables (loaded from .env by python-dotenv)
with sensible defaults. Everything configurable lives here — no magic strings
scattered across modules.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

# Load .env from the project root (two levels up from src/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


def _env(key: str, default: str = "") -> str:
    """Read an env var, stripping whitespace."""
    return os.getenv(key, default).strip()


def _env_int(key: str, default: int) -> int:
    """Read an env var as int."""
    raw = os.getenv(key, "")
    return int(raw) if raw.strip() else default


@dataclass(frozen=True)
class LLMConfig:
    """LLM provider settings."""

    provider: Literal["groq", "ollama"] = field(
        default_factory=lambda: _env("LLM_PROVIDER", "groq")  # type: ignore[arg-type]
    )

    # Groq (default — free tier)
    groq_api_key: str = field(default_factory=lambda: _env("GROQ_API_KEY"))
    groq_generator_model: str = field(
        default_factory=lambda: _env("GROQ_GENERATOR_MODEL", "llama-3.1-8b-instant")
    )
    groq_critic_model: str = field(
        default_factory=lambda: _env("GROQ_CRITIC_MODEL", "llama-3.3-70b-versatile")
    )

    # Ollama (optional — local GPU)
    ollama_model: str = field(
        default_factory=lambda: _env("OLLAMA_MODEL", "llama3.1:8b-instruct-q4_K_M")
    )
    ollama_base_url: str = field(
        default_factory=lambda: _env("OLLAMA_BASE_URL", "http://localhost:11434")
    )


@dataclass(frozen=True)
class EmbeddingConfig:
    """Local embedding model settings."""

    model_name: str = field(
        default_factory=lambda: _env("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    )


@dataclass(frozen=True)
class ChunkingConfig:
    """Document chunking parameters."""

    chunk_size: int = field(default_factory=lambda: _env_int("CHUNK_SIZE", 500))
    chunk_overlap: int = field(default_factory=lambda: _env_int("CHUNK_OVERLAP", 50))


@dataclass(frozen=True)
class RetrieverConfig:
    """Vector store retrieval settings."""

    top_k: int = field(default_factory=lambda: _env_int("TOP_K", 5))
    collection_name: str = field(
        default_factory=lambda: _env("CHROMA_COLLECTION", "rag_documents")
    )
    persist_directory: str = field(
        default_factory=lambda: _env(
            "CHROMA_PERSIST_DIR",
            str(_PROJECT_ROOT / "data" / "chroma_db"),
        )
    )


@dataclass(frozen=True)
class RAGConfig:
    """Top-level config for the RAG pipeline."""

    max_retries: int = field(default_factory=lambda: _env_int("MAX_RETRIES", 2))


@dataclass(frozen=True)
class Settings:
    """Root settings object — the single source of truth for all configuration."""

    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    project_root: Path = _PROJECT_ROOT
    documents_dir: Path = field(
        default_factory=lambda: _PROJECT_ROOT / "data" / "documents"
    )


# Module-level singleton — import this everywhere
settings = Settings()
