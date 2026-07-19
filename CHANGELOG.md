# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [v1.0.0] - 2026-07-19

### Added
- **LangGraph Agentic RAG Pipeline (Week 2):** Complete retrieve-generate-critic cycle forcing self-evaluation and retry logic on hallucinations.
- **Guardrails Gateway (Week 3):** Integrated Presidio for PII scrubbing, regex + LLM classifier for prompt injection, and `toxic-bert` filtering. Driven by `data/policies.yaml`.
- **Automated CI/CD Quality Gate (Week 4 & 5):** GitHub Actions executing a 100+ Q&A golden dataset on PRs. Blocks merges if Hallucination Rate > 15%, Relevancy drops, or Latency spikes.
- **SQLite Storage & Telemetry (Week 5 & 6):** Persistent local tracking of eval metrics per commit, alongside structured JSON pipeline logging (`src/telemetry.py`).
- **Streamlit Dashboard (Week 6):** Interactive UI (`src/dashboard/app.py`) for querying historical runs, visualizing trends, and an interactive "Query Debugger" trace view.
- Project scaffold, folder structure, `.env`, `pyproject.toml` (Week 1).
- Document ingestion for `requests` and `fastapi` using local ChromaDB and `sentence-transformers` embeddings (Week 1 & 4).
- Comprehensive unit and integration test suites covering the entire pipeline.

### Changed
- Refactored `retriever.py` to support fallback responses gracefully when documents don't match.
- Switched default Generator to `llama-3.1-8b-instant` and Critic to `llama-3.3-70b-versatile` on Groq.

### Fixed
- Addressed Streamlit deprecation warnings for `use_container_width`.
