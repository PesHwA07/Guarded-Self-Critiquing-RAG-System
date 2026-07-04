# Build roadmap: guarded, self-critiquing RAG with automated evals
*A week-by-week plan to build all three projects as one integrated system*

## System overview

```
User query
   → Input Guardrails      (block PII / injection / jailbreaks)
   → Agentic RAG core      (LangGraph: retrieve → generate → critic → retry/reformulate)
   → Output Guardrails     (schema, toxicity, on-topic checks)
   → Final grounded answer

Every git push
   → Run golden dataset (100+ Q&A pairs) through the pipeline above
   → Compute hallucination rate, relevancy, faithfulness, latency (p50/p95), cost
   → Gate the merge if thresholds are breached
   → Log results to a metrics dashboard
```

The three projects aren't separate — the RAG core is the *thing being protected and tested*; guardrails are the *middleware around it*; the eval pipeline is the *safety net that watches it over time*.

---

## Week 1 — Foundation: basic RAG pipeline (no agents yet)

**Goal:** a working linear RAG chain before you make it stateful or agentic.

- Set up a vector store (Chroma or Pinecone/Qdrant for something more production-like).
- Chunk and embed a small document set (pick a domain you know — docs, a book, internal wiki export).
- Build a simple `retrieve → generate` chain with LangChain or raw API calls.
- Get baseline latency and manually sanity-check 10–15 answers.

**Deliverable:** a CLI or notebook that takes a question and returns an answer + the chunks it used.

---

## Week 2 — Make it agentic and self-critiquing (Project 1 core)

**Goal:** turn the linear chain into a stateful, cyclical LangGraph.

- Define the LangGraph state: `query, reformulated_query, retrieved_chunks, answer, critic_verdict, retry_count`.
- Build nodes: `retrieve`, `generate`, `critic`.
- Critic node prompt: ask the LLM to check if every claim in the answer is traceable to the retrieved chunks — return a structured verdict (grounded / not grounded / partially grounded) plus reasoning.
- Add conditional edges: if `critic_verdict == not_grounded` and `retry_count < max_retries` → loop back to a `reformulate_query` node → `retrieve` again. If retries exhausted → route to a `fallback` node that returns "I don't have enough information."
- Cap retries (e.g., 2) to avoid infinite loops and runaway cost.

**Deliverable:** a LangGraph app where you can visibly trace: retrieve → generate → critic rejects → reformulate → retrieve → generate → critic accepts → done.

---

## Week 3 — Guardrails gateway (Project 2)

**Goal:** wrap the RAG graph with input/output middleware.

- **Input guardrails:**
  - Prompt injection / jailbreak detection (use a classifier model, regex heuristics, or a library like `llm-guard` / `guardrails-ai` as a starting point, then customize).
  - PII detection (Presidio is a solid open-source option) — flag or redact credit card numbers, emails, SSNs, etc.
- **Output guardrails:**
  - Schema validation — if you want structured JSON output, validate against a Pydantic/JSON schema and retry generation on failure.
  - Toxicity / off-topic checks — a lightweight classifier or a second LLM call as judge.
- **Policy engine:**
  - Write a YAML config (`policies.yaml`) with rules like `never_discuss_competitors: true`, `require_citations: true`, `blocked_topics: [medical_advice]`.
  - Build a small policy-loader that turns these into checks the input/output guardrails run against — so a non-engineer can edit the YAML without touching code.

**Deliverable:** the same RAG graph from Week 2, now wrapped by an input filter and output filter, both driven by a config file.

---

## Week 4 — Golden dataset + eval harness (Project 3, part 1)

**Goal:** build the test set and the metrics before automating anything.

- Curate 100+ Q&A pairs from your document set: mix of easy factual questions, ambiguous questions, questions with no good answer in the docs (to test the fallback path), and adversarial/injection attempts (to test guardrails).
- Pick your eval metrics and how to compute each:
  - **Faithfulness / groundedness** — reuse or extend your critic logic, or use a framework like RAGAS.
  - **Answer relevancy** — semantic similarity between answer and expected answer, or LLM-as-judge scoring.
  - **Hallucination rate** — % of answers flagged not-grounded by the critic across the dataset.
  - **Latency p50/p95** — wall-clock time per query, including retries.
  - **Cost per query** — token counts × model pricing.
- Run the full dataset through your pipeline once, manually inspect failures, and calibrate thresholds (don't guess — look at real numbers first).

**Deliverable:** a script that runs the golden dataset end-to-end and prints a metrics report.

---

## Week 5 — CI/CD automation (Project 3, part 2)

**Goal:** wire the eval harness into your git workflow.

- Add a GitHub Actions (or GitLab CI) workflow that triggers on push/PR.
- Run the Week 4 eval script against the current code.
- Set thresholds as config (e.g., `max_hallucination_rate: 0.05`, `max_p95_latency_ms: 4000`) and fail the CI job if breached — this is your merge gate.
- Store historical results (a simple SQLite file, or push to a time-series-friendly store) so you can track trend over time, not just pass/fail.

**Deliverable:** a red/green CI check on every PR, plus a results log you can query historically.

---

## Week 6 — Dashboard + polish

**Goal:** make the system observable and demo-ready.

- Build a small dashboard (Streamlit or a simple React page) showing: hallucination rate over time, latency trend, cost trend, pass/fail history per commit.
- Add basic tracing/logging across the whole pipeline (LangSmith, or your own structured logs) so you can debug a specific failed query end-to-end — which guardrail fired, how many retries, what the critic said.
- Write the README: architecture diagram, how to run locally, how to add new policies, how to extend the golden dataset.
- Record a 2–3 minute demo video walking through: a normal query, a query that triggers a retry, a query that triggers guardrails, and a CI run that blocks a bad change.

**Deliverable:** the finished, integrated system — a portfolio piece that shows agentic reasoning, safety engineering, and MLOps discipline in one place.

---

## Stretch goals (if you have more time)

- Swap in multiple LLM providers and add cost/quality comparison to the dashboard.
- Add a second critic pass that scores answer *helpfulness*, not just groundedness, to catch "technically grounded but useless" answers.
- Add canary deployment logic — route 5% of traffic to a new prompt/model version and compare live metrics before full rollout.
- Add human-in-the-loop review for borderline critic verdicts, with a simple approve/reject UI that also feeds back into the golden dataset.

---

## Tech stack suggestions

| Component | Suggested tools |
|---|---|
| Orchestration | LangGraph |
| Vector store | Chroma (local) or Qdrant/Pinecone (hosted) |
| Guardrails | Presidio (PII), guardrails-ai or custom classifiers, YAML policy config |
| Eval framework | RAGAS, or custom LLM-as-judge scoring |
| CI/CD | GitHub Actions |
| Dashboard | Streamlit (fastest) or a small React + charting lib |
| Tracing | LangSmith, or structured JSON logs + a simple viewer |

