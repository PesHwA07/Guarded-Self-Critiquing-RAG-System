# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- Project scaffold: `pyproject.toml`, `.gitignore`, `.env.example`
- Folder structure: `src/rag/`, `src/guardrails/`, `src/eval/`, `src/dashboard/`
- Test infrastructure: `tests/unit/`, `tests/integration/`, `conftest.py`
- README with architecture diagram
- `src/config.py` — centralized settings loader with Groq/Ollama LLM_PROVIDER switch
- `src/rag/llm_provider.py` — Groq + Ollama behind one LangChain interface
- `src/rag/embedder.py` — document loading (.rst/.md/.txt), RST cleanup, chunking, local embeddings
- `scripts/ingest.py` — CLI for ingesting documents into ChromaDB (supports --reset, --dry-run)
- Python `requests` library docs ingested as the Week 1–3 knowledge base (14 RST files)
- Unit tests for the document processor (embedder)
