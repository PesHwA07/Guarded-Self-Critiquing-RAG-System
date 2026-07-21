# 🛡️ Guarded RAG System

A **guarded, self-critiquing Retrieval-Augmented Generation (RAG) system** with automated evaluations — built with LangGraph, Groq, and ChromaDB.

> **Zero cost.** Runs entirely on free-tier APIs and local models.

---

## What This Project Does

This system answers questions about technical documentation with **three layers of intelligence**:

1. **Agentic RAG Core** — Retrieves relevant document chunks, generates an answer, then a *critic* checks if the answer is actually supported by the evidence. If not, it reformulates the query and tries again.
2. **Guardrails Gateway** — Input filters block PII leaks, prompt injection, and jailbreak attempts. Output filters catch toxic, off-topic, or malformed responses.
3. **Automated Eval Pipeline** — A golden dataset of 100+ Q&A pairs runs on every PR, computing hallucination rate, faithfulness, latency, and cost. Bad changes are blocked from merging.

---

## Architecture

```
User Query
   → Input Guardrails      (block PII / injection / jailbreaks)
   → Agentic RAG Core      (LangGraph: retrieve → generate → critic → retry/reformulate)
   → Output Guardrails     (schema, toxicity, on-topic checks)
   → Final Grounded Answer

Every Git Push
   → Run golden dataset (100+ Q&A pairs) through the pipeline
   → Compute hallucination rate, relevancy, faithfulness, latency
   → Gate the merge if thresholds are breached
   → Log results to a metrics dashboard
```

### RAG Flow (LangGraph)

```mermaid
---
config:
  flowchart:
    curve: linear
---
graph TD;
	__start__([<p>__start__</p>]):::first
	retrieve(retrieve)
	generate(generate)
	critic(critic)
	reformulate(reformulate)
	fallback(fallback)
	__end__([<p>__end__</p>]):::last
	__start__ --> retrieve;
	critic -.-> __end__;
	critic -.-> fallback;
	critic -.-> reformulate;
	generate --> critic;
	reformulate --> retrieve;
	retrieve --> generate;
	fallback --> __end__;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc
```

---

## Tech Stack

| Component | Tool | Cost |
|---|---|---|
| LLM (generator) | Groq `llama-3.1-8b-instant` | Free |
| LLM (critic/judge) | Groq `llama-3.3-70b-versatile` | Free |
| LLM (offline dev) | Ollama (optional, local GPU) | Free |
| Embeddings | `sentence-transformers` `all-MiniLM-L6-v2` | Free (local) |
| Vector Store | ChromaDB | Free (local) |
| PII Detection | Presidio | Free |
| Toxicity | `unitary/toxic-bert` | Free (local) |
| CI/CD | GitHub Actions | Free |
| Dashboard | Streamlit | Free |

---

## Quick Start

### Prerequisites
- Python 3.10+
- A free [Groq API key](https://console.groq.com/keys)

### Setup

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/guarded-rag-system.git
cd guarded-rag-system

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# 3. Install dependencies
pip install -e ".[dev]"

# 4. Set up environment
cp .env.example .env
# Edit .env and add your GROQ_API_KEY

# 5. Ingest documents
python scripts/ingest.py

# 6. Ask a question via CLI
python -m src.rag.graph "How do I send a POST request with JSON data?"

# 7. Run the interactive Dashboard
python -m streamlit run src/dashboard/app.py
```

### Optional: Local LLM with Ollama

```bash
# Install Ollama: https://ollama.com/download
ollama pull llama3.1:8b-instruct-q4_K_M

# Switch to local provider
# In .env: LLM_PROVIDER=ollama
```

---

## Project Structure

```
guarded-rag-system/
├── src/
│   ├── config.py              # Settings loader
│   ├── rag/
│   │   ├── llm_provider.py    # Groq/Ollama abstraction
│   │   ├── embedder.py        # Chunking + embeddings
│   │   ├── retriever.py       # Vector search
│   │   ├── generator.py       # Answer generation
│   │   ├── critic.py          # Self-critique (70B model)
│   │   ├── reformulator.py    # Query reformulation
│   │   ├── fallback.py        # Graceful "I don't know"
│   │   └── graph.py           # LangGraph wiring
│   ├── guardrails/
│   │   ├── input_guard.py     # PII, injection, jailbreak
│   │   ├── output_guard.py    # Schema, toxicity, on-topic
│   │   └── policy_engine.py   # YAML policy loader
│   ├── eval/
│   │   ├── metrics.py         # Faithfulness, hallucination, latency
│   │   ├── runner.py          # Eval runner (smoke/full modes)
│   │   └── reporter.py        # Metrics reporting
│   └── dashboard/
│       └── app.py             # Streamlit dashboard
├── data/
│   ├── documents/             # Source docs (requests + FastAPI)
│   ├── golden_dataset.json    # 100+ eval Q&A pairs
│   └── policies.yaml          # Guardrail config
├── tests/                     # Unit + integration tests
├── scripts/                   # CLI tools
├── docs/                      # Documentation + learning journal
└── .github/workflows/         # CI/CD pipelines (Lint, Unit Tests, Eval Gate)
```

---

## CI/CD and Branch Protection

This project uses a rigorous CI/CD pipeline implemented via GitHub Actions:

- **Continuous Integration (`ci.yml`)**: Runs on every push and PR. Executes `ruff check` for linting and `pytest` for all unit tests. Uses dependency caching to run in <1 minute.
- **Eval Gate (`eval-gate.yml`)**: A data-driven quality gate.
  - On every push: runs a fast "smoke test" evaluation (10 queries).
  - On PRs to `main`: runs the full golden dataset (100+ queries).
  - Validates metrics against `data/eval_config.yaml` thresholds (Faithfulness, Relevancy, Hallucination Rate, Latency p95).
  - Stores historical results in a local SQLite DB, viewable via `python scripts/trend.py`.

### Branch Protection Strategy
To ensure quality, the `main` branch should be protected on GitHub:
1. Go to **Settings > Branches > Add branch ruleset**.
2. Target the `main` branch.
3. Check **Require status checks to pass before merging**.
4. Add the following required checks:
   - `lint-and-test` (from `ci.yml`)
   - `eval-gate` (from `eval-gate.yml`)
5. This guarantees no untested or regressing code (e.g. prompt changes that increase hallucination rate above 15%) can be merged.

---

## 📈 Dashboard & Telemetry

You can track the performance of the system visually using the **Streamlit Dashboard**:
```bash
python -m streamlit run src/dashboard/app.py
```
**Features:**
- **Analytics Trends:** Visual line and area charts tracking hallucination rates, latencies (p95), cost per query, and success rates across your historical commits.
- **Raw Data:** Access to the underlying SQLite records for every eval run.
- **Query Debugger:** An interactive testing ground where you can input queries, see the LangGraph pipeline execute, and inspect the internal tracing (Critic verdicts, retries, source context, and latency).

The pipeline also uses built-in **structured JSON logging** via `src/telemetry.py`, making it production-ready for log aggregation systems like Datadog or ELK.

### Weights & Biases (W&B) MLOps Tracking
In addition to the local SQLite dashboard, the automated evaluation pipeline tracks historical metrics in **Weights & Biases**. 
When you run `run_eval.py`, it logs:
- **Hallucination Rate** and **Relevancy**
- **Latency (p50/p95)** and **Cost per Query**
- Comparisons across multiple runs, giving you the full MLOps view for comparing offline prompt engineering iterations.

Ensure you set `WANDB_API_KEY` and `WANDB_PROJECT` in your `.env` file to enable this.

---

## 🗄️ Vector Database Trade-offs

This system supports both **ChromaDB** and **Qdrant** as vector stores, easily switchable via the `VECTOR_STORE` environment variable in your `.env` file (`VECTOR_STORE=chroma` or `VECTOR_STORE=qdrant`).

**Why support both?**
- **ChromaDB (Local):** Perfect for offline development, rapid prototyping, and CI testing. It runs entirely locally without any network latency or API keys.
- **Qdrant (Cloud/Hosted):** Better suited for scaling to production. By using Qdrant's free cloud tier, the system can scale beyond a single machine, persisting a large document index centrally rather than duplicating the database locally for every developer or container.

---

## 🛡️ Extending Guardrail Policies

The RAG system uses a YAML-driven policy engine (`data/policies.yaml`) so you can tweak rules without modifying Python code.

To extend the system, simply edit `data/policies.yaml`:
```yaml
# Add a new allowed topic for the TopicGuard
topics:
  enabled: true
  allowed:
    - "Python programming"
    - "New Topic Here"

# Adjust Toxicity threshold (lower = stricter)
toxicity:
  enabled: true
  threshold: 0.4
```
The graph will automatically pick up these changes on the next query.

---

## Development Status

- [x] 📁 Week 1: Project scaffold & Linear RAG pipeline
- [x] 🤖 Week 2: Agentic self-critiquing graph
- [x] 🛡️ Week 3: Guardrails gateway
- [x] 📊 Week 4: Golden dataset + eval harness
- [x] ⚙️ Week 5: CI/CD automation & SQLite Result Storage
- [x] 📈 Week 6: Dashboard + polish
- [x] 🧠 Week 7: Vector database depth + MLOps tracking
- [ ] 🚀 Week 8: Dockerize + deploy

---

## License

MIT
