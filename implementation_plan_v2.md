# Guarded Self-Critiquing RAG System — Implementation Plan (v2, zero-cost)

Revised from the original plan to run at **$0 cost** and to replace the daily-commit-streak
structure with a **scope-based weekly plan**. Same architecture, same milestones — just a
saner build cadence and a free-tier-friendly tech stack.

## What changed from v1, and why

| Issue in v1 | Fix in v2 |
|---|---|
| Assumed OpenAI for everything → real cost | Groq (free tier) for LLM calls, local `sentence-transformers` for embeddings — $0 |
| "42 consecutive daily commits" streak framing | Scope-based weekly milestones; commit whenever a subtask is actually done |
| Critic used the same model family as the generator (correlated blind spots) | Critic + eval judge use a *different, stronger* model than the generator |
| CI eval-gate cost/rate-limit risk unclear | Tiered eval: smoke-test subset on every push, full golden dataset only on PR to `main` |
| No local/offline dev option | Ollama on your RTX 3050 as a swappable local provider for fast offline iteration |

---

## System Architecture

*(unchanged from v1 — same LangGraph flow: input guardrails → retrieve → generate → critic →
reformulate/fallback/done → output guardrails → answer, with the eval CI/CD loop running
against the golden dataset on every push)*

---

## Tech Stack (zero-cost)

| Component | Tool | Why |
|---|---|---|
| Language | Python 3.11+ | LangChain/LangGraph ecosystem |
| Orchestration | LangGraph | Stateful, cyclical agent graphs |
| **LLM — default (dev + CI)** | **Groq API** (`llama-3.1-8b-instant`) | Free tier, fast, no GPU needed — works identically on your laptop and in GitHub Actions |
| **LLM — critic / eval judge** | **Groq API** (`llama-3.3-70b-versatile`) | Different, stronger model than the generator — reduces the risk of the critic missing the generator's own characteristic hallucination patterns |
| **LLM — optional local provider** | **Ollama** (`llama3.1:8b-instruct-q4_K_M`, ~4.9GB) | Runs on your RTX 3050 (6GB VRAM). Use for offline prompt iteration and demoing without internet/rate limits — swappable via a config flag, not a second parallel system |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) | Free, runs on CPU or your GPU, no API key, no rate limit, no cost ever |
| Vector store | ChromaDB (local dev) + Qdrant (free cloud tier, Week 7) | Chroma for offline dev, Qdrant for anything needing to scale beyond one machine — swappable via `VECTOR_STORE` env var, same pattern as the LLM provider switch |
| Guardrails — PII | Presidio | Free, open-source, local |
| Guardrails — toxicity / on-topic | Local classifier (`unitary/toxic-bert`) as default; Groq LLM-as-judge as a fallback | Keeps your Groq quota free for the core pipeline instead of spending it on every guardrail check |
| Policy engine | Custom YAML loader | Free, no dependency |
| Eval | RAGAS + custom LLM-as-judge (via Groq critic model) | Free |
| Experiment tracking | Weights & Biases (free tier, Week 7) | Logs every eval run — hallucination rate, relevancy, latency, cost — comparable across commits; the MLOps signal a plain SQLite table can't give you |
| CI/CD | GitHub Actions | Free tier (2,000 min/month) — plenty for tiered eval strategy below |
| Serving | FastAPI + Docker (Week 8) | Turns "trained/built" into "shipped and running" — the deployment story most portfolios are missing |
| Deployment host | Hugging Face Spaces or Render (free tier, Week 8) | Real, clickable, live demo URL at $0 |
| Dashboard | Streamlit, optionally deployed free on Streamlit Community Cloud | Free hosting for a live demo link |
| Tracing | Structured JSON logs by default; LangSmith free tier (5k traces/month) optional | Avoids a second paid dependency |
| Package Mgmt | `uv` or `pip` + `pyproject.toml` | Modern Python packaging |

### Provider abstraction (important)

`src/rag/llm_provider.py` wraps both Groq and Ollama behind the same interface
(`LangChain`'s `ChatGroq` / `ChatOllama` are drop-in compatible), switched via an
`LLM_PROVIDER=groq|ollama` env var. This means:
- **Groq is the "real" system** — what actually gets tested in CI and what your eval
  numbers are based on.
- **Ollama is a local sandbox** — flip the env var to iterate on prompts offline, fast,
  without touching your Groq rate limit. Not used for golden-dataset eval runs, since CI
  can't use your GPU and you want dev/CI numbers to be comparable.

### Rate-limit strategy for CI

Groq's free tier has request-per-minute/day caps. To stay under them and keep feedback fast:
- **Every push** → smoke test: ~10 representative queries against `llama-3.1-8b-instant`.
- **Every PR to `main`** → full golden dataset (100+ queries), batched with small delays if needed.

This is also just better CI practice than running a full 100-query eval on every single commit.

---

## Repository Structure (Target)

```
guarded-rag-system/
├── .github/
│   └── workflows/
│       ├── ci.yml                  # Lint + unit tests on every push
│       └── eval-gate.yml           # Smoke test on push, full golden dataset on PR to main
├── src/
│   ├── __init__.py
│   ├── config.py                   # Env + settings loader (incl. LLM_PROVIDER switch)
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── llm_provider.py         # Groq / Ollama abstraction
│   │   ├── embedder.py             # Chunking + local embedding logic
│   │   ├── retriever.py            # Vector store query
│   │   ├── generator.py            # LLM answer generation
│   │   ├── critic.py               # Self-critique node (stronger model)
│   │   ├── reformulator.py         # Query reformulation node
│   │   ├── fallback.py             # Graceful degradation
│   │   └── graph.py                # LangGraph wiring
│   ├── guardrails/
│   │   ├── __init__.py
│   │   ├── input_guard.py          # PII, injection, jailbreak
│   │   ├── output_guard.py         # Schema, toxicity, on-topic
│   │   └── policy_engine.py        # YAML policy loader
│   ├── eval/
│   │   ├── __init__.py
│   │   ├── dataset.py              # Golden dataset loader
│   │   ├── metrics.py              # Faithfulness, relevancy, hallucination, latency, cost
│   │   ├── runner.py               # End-to-end eval runner (supports smoke/full modes)
│   │   └── reporter.py             # Metrics reporting
│   └── dashboard/
│       ├── app.py                  # Streamlit dashboard
│       └── components/             # Dashboard widgets
├── data/
│   ├── documents/                  # Source docs for RAG
│   ├── golden_dataset.json         # 100+ Q&A eval pairs
│   └── policies.yaml               # Guardrail policy config
├── tests/
│   ├── unit/
│   ├── integration/
│   └── conftest.py
├── scripts/
│   ├── ingest.py                   # Document ingestion script
│   └── run_eval.py                 # Standalone eval runner (--mode smoke|full)
├── .env.example
├── .gitignore
├── pyproject.toml
├── README.md
└── CHANGELOG.md
```

---

## User Review Required

> [!IMPORTANT]
> **API Key**: Groq account already set up — just drop your key into `.env` as
> `GROQ_API_KEY` (never commit it). Ollama needs no key — just
> `ollama pull llama3.1:8b-instruct-q4_K_M` locally.

> [!TIP]
> **Cost**: $0, as long as you stay within Groq's free-tier rate limits — the tiered
> eval strategy above is designed specifically to keep you there.

## Document Domain (resolved)

Public technical docs, cloned locally — no scraping, no API, no cost:

- **Weeks 1–3**: [`psf/requests`](https://github.com/psf/requests) docs (`docs/*.rst`).
  Small (~20–25 files), well-structured into quickstart/advanced/API-reference sections,
  fast to ingest and iterate on while the graph itself is still being debugged.
- **Weeks 4–6**: swap/add [`tiangolo/fastapi`](https://github.com/tiangolo/fastapi) tutorial
  docs (`docs/en/docs/tutorial/*.md`, ~40 files) for a bigger, more demo-worthy final
  knowledge base once the pipeline is proven out on the smaller corpus.

```bash
# Week 1
git clone https://github.com/psf/requests.git data/sources/requests
cp data/sources/requests/docs/*.rst data/documents/

# Week 4 (additive — keep requests docs, add fastapi tutorial subset)
git clone https://github.com/tiangolo/fastapi.git data/sources/fastapi
cp -r data/sources/fastapi/docs/en/docs/tutorial data/documents/fastapi-tutorial
```

Both are plain-text/markdown-ish and permissively licensed for this kind of local,
non-redistributed use — no cost, no rate limit, no account needed to obtain them.

## Open Questions

~~1. Repo name/visibility~~ — **Resolved: public repo, `guarded-rag-system`**
~~2. Groq account~~ — **Resolved: already set up**

All setup blockers are now cleared — Week 1 can start immediately.

---

## Weekly Build Plan (7-Day Daily Cadence)

This plan breaks down each week into 7 daily tasks so you can operate systematically and push cohesive daily updates to GitHub. Each day represents a logical chunk of work that you can commit and push.

```
main ← feature/week1-rag-foundation
     ← feature/week2-agentic-graph
     ← feature/week3-guardrails
     ← feature/week4-eval-harness
     ← feature/week5-cicd
     ← feature/week6-dashboard
     ← feature/week7-vector-mlops
     ← feature/week8-deploy
```

### Week 1 — Foundation: basic RAG pipeline

- [x] **Day 1: Project Setup** — Project scaffold (`pyproject.toml`, `.gitignore`, `.env.example`, folder structure) and `config.py` with `LLM_PROVIDER` switch.
- [x] **Day 2: Document Ingestion** — Clone `psf/requests`, copy `docs/*.rst` into `data/documents/`, and set up `llm_provider.py`.
- [x] **Day 3: Embeddings** — `embedder.py` (chunking + local `sentence-transformers` embeddings) and `ingest.py` (CLI to ingest into ChromaDB).
- [x] **Day 4: Retriever & Generator** — `retriever.py` (similarity search) and `generator.py` (prompt template with injected context).
- [x] **Day 5: Pipeline Graph** — `graph.py` v1 (linear `retrieve → generate` chain) and CLI entry point.
- [x] **Day 6: Sanity Check & Metrics** — Manual sanity check (15 test questions) and measure/document baseline latency in `docs/baseline_metrics.md`.
- [x] **Day 7: Tests & Review** — Unit tests for embedder, retriever, generator, and an integration test for the linear chain. Open PR for Week 1.

**Milestone:** linear RAG returns grounded answers on 15 manual test queries.

---

### Week 2 — Agentic self-critiquing RAG (LangGraph)

- [x] **Day 1: State Setup** — `RAGState` TypedDict and `StateGraph` skeleton.
- [x] **Day 2: Critic Node** — `critic.py` (checks whether claims trace back to chunks; uses `llama-3.3-70b-versatile`).
- [x] **Day 3: Reformulator Node** — `reformulator.py` (takes original query + critic reasoning, produces better search query).
- [x] **Day 4: Fallback Node** — `fallback.py` (graceful degradation response including attempted chunks).
- [x] **Day 5: Routing Logic** — Conditional routing in `graph.py` (critic → reformulate → retrieve) and retry counter.
- [x] **Day 6: Logging & Visualization** — Trace logging and `.get_graph().draw_mermaid()` visualization added to README.
- [ ] **Day 7: Integration Tests** — Tests for grounded, ungrounded, unanswerable queries, and edge cases. Open PR for Week 2.

**Milestone:** trace logs clearly show a path through critic → reformulate → retrieve → accept, and a separate path hitting fallback.

---

### Week 3 — Guardrails gateway

- [ ] **Day 1: PII Guard** — `input_guard.py` (Presidio PII detection, configurable action: block/redact/warn).
- [ ] **Day 2: Injection Guard** — `input_guard.py` (injection/jailbreak detection with regex heuristics + Groq classifier).
- [ ] **Day 3: Output Schema Guard** — `output_guard.py` (Pydantic schema validation and auto-retry on violation).
- [ ] **Day 4: Toxicity & Topic Guard** — `output_guard.py` (toxicity check via `toxic-bert` + on-topic check).
- [ ] **Day 5: Policy Engine** — `policy_engine.py` + `policies.yaml` (YAML-driven rules, editable without touching code).
- [ ] **Day 6: Graph Integration** — Wire `input_guard` before `retrieve`, and `output_guard` after critic accepts.
- [ ] **Day 7: Testing** — Unit tests per guardrail and integration test (PII, injection, clean query). Open PR for Week 3.

**Milestone:** PII, injection, and toxicity test cases are all caught; a clean query passes through untouched.

---

### Week 4 — Golden dataset + eval harness

- [ ] **Day 1: Add New Corpus** — Clone `tiangolo/fastapi`, copy tutorial docs, and re-run ingestion.
- [ ] **Day 2: Base Dataset** — `golden_dataset.json` (30 straightforward factual Q&A pairs across both corpora).
- [ ] **Day 3: Expand Dataset** — Add ambiguous, unanswerable, and adversarial queries to reach 100+ total.
- [ ] **Day 4: Core Metrics** — `metrics.py` (faithfulness/groundedness and answer relevancy).
- [ ] **Day 5: Operational Metrics** — `metrics.py` (hallucination rate, latency p50/p95, and cost per query).
- [ ] **Day 6: Eval Runner** — `dataset.py` + `runner.py` (load dataset, support `--mode smoke` and `--mode full`).
- [ ] **Day 7: Reporting** — `reporter.py`, run first full eval to calibrate thresholds, output JSON/console table. Open PR for Week 4.

**Milestone:** eval harness produces a metrics report across 100+ Q&A pairs with calibrated, realistic thresholds.

---

### Week 5 — CI/CD automation

- [ ] **Day 1: Basic CI** — `.github/workflows/ci.yml` (lint + unit tests on every push).
- [ ] **Day 2: Eval Gate CI** — `.github/workflows/eval-gate.yml` (smoke test on push, full golden dataset on PR).
- [ ] **Day 3: Eval Config** — `eval_config.yaml` (configurable thresholds that CI reads to gate pass/fail).
- [ ] **Day 4: Result Storage** — SQLite storage for eval results per commit.
- [ ] **Day 5: Trend CLI** — CLI tool to query historical results and show improving/degrading trends.
- [ ] **Day 6: Optimization & Docs** — Dependency caching in CI and branch protection documentation in README.
- [ ] **Day 7: End-to-End Test** — Test the full pipeline with an intentionally bad PR to confirm the gate blocks it. Open PR for Week 5.

**Milestone:** an intentionally bad change gets a red check and is blocked from merging; a good change passes cleanly.

---

### Week 6 — Dashboard + polish

- [ ] **Day 1: Dashboard Foundation** — `dashboard/app.py` Streamlit layout, loading eval results from SQLite.
- [ ] **Day 2: Analytics Charts** — Add charts for hallucination rate, latency trend, cost, and pass/fail history.
- [ ] **Day 3: Query Debugger** — View to inspect per-query results (fired guardrails, retry count, critic reasoning).
- [ ] **Day 4: Telemetry** — Structured JSON logging across the pipeline (optional LangSmith integration).
- [ ] **Day 5: Documentation** — README overhaul (architecture, setup, local run, extending policies).
- [ ] **Day 6: Demo Prep** — Prep scenarios: normal query, retry, guardrail trigger, CI block. Cleanup and type hints.
- [ ] **Day 7: Launch** — Tag `v1.0.0`, record demo video, final `CHANGELOG.md` and optional Streamlit deploy. Merge final PR.

**Milestone:** live demo walkthrough of all four scenarios works end-to-end, dashboard shows accurate trends.

---

### Week 7 — Vector database depth + MLOps tracking

*Goal: go from "used one vector store" and "no experiment tracking" to being able to speak
to real trade-offs in an interview — extends the existing system, no new architecture.*

- [ ] **Day 1: Qdrant Integration** — Add **Qdrant** (free cloud tier) as a second retriever backend via `VECTOR_STORE=chroma|qdrant`.
- [ ] **Day 2: W&B Setup** — Sign up for **Weights & Biases** free tier and configure the project.
- [ ] **Day 3: W&B Logging** — Wire W&B logging into `eval/reporter.py` as an additional sink alongside SQLite.
- [ ] **Day 4: Metrics Tracking** — Log hallucination rate, relevancy, latency (p50/p95), and cost per query to W&B.
- [ ] **Day 5: Verification** — Run `run_eval.py --mode full` and confirm a new W&B run appears with correct metrics.
- [ ] **Day 6: Documentation** — Document the trade-off in the README: local ChromaDB for dev/offline, hosted Qdrant for scaling.
- [ ] **Day 7: Review & Merge** — Ensure metrics are comparable across 3+ historical runs and open PR for Week 7.

**Milestone:** eval runs are tracked in W&B with a comparison view across commits; the
retriever can be swapped between Chroma and Qdrant via one env var with no code changes
elsewhere.

---

### Week 8 — Dockerize + deploy (closes the deployment gap)

*Goal: go from "trained/built" to "shipped and running" — the single biggest resume gap
this project was meant to close.*

- [ ] **Day 1: FastAPI Wrapper** — Wrap `graph.py` in a **FastAPI** app with `/query` and `/health` endpoints.
- [ ] **Day 2: Dockerization** — Create `Dockerfile` for the API service.
- [ ] **Day 3: Docker Compose** — Create `docker-compose.yml` running the API + ChromaDB (or Qdrant) together.
- [ ] **Day 4: Environment Config** — Secure environment variables passed via `.env`, never baked into the image.
- [ ] **Day 5: Deployment** — Deploy free on **Hugging Face Spaces** (Docker SDK) or **Render** free tier.
- [ ] **Day 6: Smoke Test** — Test the deployed endpoint with `curl`/Postman to confirm guardrails and fallback work.
- [ ] **Day 7: Docs & Launch** — Update README with architecture diagram, live demo link, and W&B link. Open PR for Week 8.

**Milestone:** a live, publicly reachable API endpoint answers questions correctly,
respects guardrails, and its eval history is visible in W&B — the full loop, resume-ready.

---

## Verification Plan

### Automated Tests
```bash
# Unit tests (run as you go)
pytest tests/unit/ -v

# Integration tests (run at least weekly)
pytest tests/integration/ -v

# Eval — smoke test (fast, cheap, run on every push)
python scripts/run_eval.py --config data/eval_config.yaml --mode smoke

# Eval — full golden dataset (run on PR to main)
python scripts/run_eval.py --config data/eval_config.yaml --mode full

# Lint
ruff check src/ tests/
```

### Manual Verification
- Week 1: CLI demo — ask 15 questions, verify answers + source chunks
- Week 2: Trace a query through retrieve → generate → critic → reformulate → accept
- Week 3: Test PII query (blocked), injection (blocked), clean query (passes)
- Week 4: Inspect eval report — are thresholds reasonable given real numbers?
- Week 5: Push a bad change, verify CI blocks the merge; confirm you're within Groq rate limits
- Week 6: Walk through dashboard, verify charts are accurate

### Milestones

| Week | Milestone | Verification |
|------|-----------|-------------|
| 1 | Linear RAG returns grounded answers | 15 manual test queries pass |
| 2 | Agentic graph with visible retry loop | Trace logs show critic → reformulate → accept path |
| 3 | Guardrails block bad input/output | PII, injection, toxicity test cases all caught |
| 4 | Eval harness produces metrics report | 100+ Q&A pairs scored, thresholds calibrated |
| 5 | CI blocks bad changes automatically, stays free | Intentionally bad PR gets red check; rate limits respected |
| 6 | Dashboard shows trends, system is demo-ready | Live demo with all 4 scenarios works |
| 7 | Vector store swappable, eval runs tracked in W&B | Chroma↔Qdrant swap works; 3+ comparable W&B runs logged |
| 8 | System is deployed and publicly reachable | Live API URL answers queries correctly, guardrails hold |
