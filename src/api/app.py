"""FastAPI server — REST API wrapper for the Guarded RAG pipeline.

Provides two endpoints:
  - ``GET /health``  — Liveness + readiness check (vector store status).
  - ``POST /query``  — Submit a question and receive the RAG pipeline answer.

Usage::

    # Dev server (auto-reload)
    uvicorn src.api.app:app --reload --port 8000

    # Production (from project root)
    python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Ensure ``src/`` is on sys.path so all internal imports resolve correctly
# when running via ``uvicorn src.api.app:app``.
# ---------------------------------------------------------------------------
_SRC_DIR = Path(__file__).resolve().parent.parent  # src/
_PROJECT_ROOT = _SRC_DIR.parent                    # project root
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from config import settings
from rag.graph import run_query, _get_retriever
from telemetry import setup_telemetry

# ---------------------------------------------------------------------------
# Telemetry — JSON structured logs for production
# ---------------------------------------------------------------------------
setup_telemetry(json_format=True, level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pydantic models for request / response
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    """Request body for the ``/query`` endpoint."""

    question: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="The question to ask the RAG pipeline.",
        examples=["How do I send a POST request with JSON data?"],
    )


class QueryResponse(BaseModel):
    """Response body for the ``/query`` endpoint."""

    answer: str = Field(description="The generated answer.")
    confidence: str = Field(description="Generator confidence level (high/medium/low).")
    critic_verdict: str = Field(description="Critic groundedness verdict.")
    retry_count: int = Field(description="Number of reformulation retries.")
    sources_used: list[int] = Field(default_factory=list, description="Source chunk indices cited.")
    latency_ms: float = Field(description="End-to-end pipeline latency in milliseconds.")
    error: str | None = Field(default=None, description="Error message, if any guardrail blocked the query.")


class HealthResponse(BaseModel):
    """Response body for the ``/health`` endpoint."""

    status: str = Field(description="Service health status: 'ok' or 'degraded'.")
    vector_store: str = Field(description="Active vector store backend (chroma or qdrant).")
    documents_indexed: int = Field(description="Number of document chunks in the vector store.")
    llm_provider: str = Field(description="Active LLM provider (groq or ollama).")
    version: str = Field(description="Application version string.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Guarded RAG System",
    description=(
        "A guarded, self-critiquing Retrieval-Augmented Generation API. "
        "Input guardrails block PII / injection / jailbreaks. "
        "A critic loop verifies answer groundedness before returning."
    ),
    version="1.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — allow all origins for demo purposes; tighten for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check() -> HealthResponse:
    """Liveness and readiness probe.

    Returns service status, vector store info, and active configuration.
    Use this endpoint for container orchestration health checks (Docker,
    Kubernetes) and uptime monitors.
    """
    try:
        retriever = _get_retriever()
        doc_count = retriever.count
        status = "ok" if doc_count > 0 else "degraded"
        logger.debug(f"Health check: doc_count={doc_count}, status={status}")
    except Exception as exc:
        logger.error("Health check failed: %s", exc)
        doc_count = 0
        status = "degraded"

    return HealthResponse(
        status=status,
        vector_store=settings.retriever.vector_store,
        documents_indexed=doc_count,
        llm_provider=settings.llm.provider,
        version="1.1.0",
    )


@app.post("/query", response_model=QueryResponse, tags=["RAG"])
async def query_endpoint(request: QueryRequest) -> QueryResponse:
    """Submit a question to the Guarded RAG pipeline.

    The pipeline runs through:
    1. **Input Guardrails** — PII scrubbing, injection detection
    2. **Retrieval** — Similarity search over the document index
    3. **Generation** — LLM answer synthesis with context
    4. **Critic** — Groundedness evaluation (may trigger reformulation)
    5. **Output Guardrails** — Toxicity and topic policy checks

    Returns the final answer along with metadata about the pipeline run.
    """
    logger.info("API /query received", extra={"metadata": {"question": request.question}})

    start = time.perf_counter()
    try:
        result = run_query(request.question, verbose=False)
    except Exception as exc:
        logger.error("Pipeline error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(exc)}")
    total_ms = (time.perf_counter() - start) * 1000

    return QueryResponse(
        answer=result.get("answer", "(no answer)"),
        confidence=result.get("confidence", "unknown"),
        critic_verdict=result.get("critic_verdict", "unknown"),
        retry_count=result.get("retry_count", 0),
        sources_used=result.get("sources_used", []),
        latency_ms=round(result.get("latency_ms", total_ms), 1),
        error=result.get("error"),
    )
