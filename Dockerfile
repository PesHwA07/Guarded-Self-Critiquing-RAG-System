# ============================================================
# Guarded RAG System — Production Dockerfile
# ============================================================
# Multi-stage build:
#   Stage 1 (builder): installs Python deps into a venv
#   Stage 2 (runtime): slim image with only the venv + app code
#
# Build:
#   docker build -t guarded-rag-api .
#
# Run:
#   docker run --env-file .env -p 8000:8000 guarded-rag-api
# ============================================================

# ---------- Stage 1: builder ----------
FROM python:3.11-slim AS builder

WORKDIR /build

# System deps for building native wheels (presidio, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
    && rm -rf /var/lib/apt/lists/*

# Create a virtual environment inside the build stage at the exact path
# it will be copied to in the runtime stage, because venvs are not relocatable.
RUN python -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

# Install Python dependencies first (cache-friendly layer)
COPY pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir .

# Pre-download the sentence-transformers model so it's baked
# into the image and doesn't need network access at runtime.
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('all-MiniLM-L6-v2')"


# ---------- Stage 2: runtime ----------
FROM python:3.11-slim AS runtime

# Set working directory for runtime
WORKDIR /app

# Copy the pre-built venv from builder
COPY --from=builder /app/venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

# Copy the pre-downloaded model cache from builder
COPY --from=builder /root/.cache/huggingface /root/.cache/huggingface

# Copy application source code
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY data/policies.yaml ./data/policies.yaml
COPY data/eval_config.yaml ./data/eval_config.yaml
COPY data/golden_dataset.json ./data/golden_dataset.json
COPY data/documents/ ./data/documents/
# Copy test scripts
COPY scripts/api_smoke_test.py ./scripts/api_smoke_test.py

# Create directories for runtime data
RUN mkdir -p data/chroma_db

# Environment defaults (can be overridden at runtime via --env-file)
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV VECTOR_STORE=chroma
ENV LLM_PROVIDER=groq

# Expose the API port
EXPOSE 8000

# Health check — Docker / orchestrators use this to verify the
# container is ready to serve traffic.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request, os; port = os.environ.get('PORT', '8000'); urllib.request.urlopen(f'http://localhost:{port}/health')" || exit 1

# Run the FastAPI server using the PORT environment variable (default 8000)
# Hugging Face Spaces uses port 7860 by default.
CMD uvicorn src.api.app:app --host 0.0.0.0 --port ${PORT:-8000}
