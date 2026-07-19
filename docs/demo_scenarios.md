# Guarded RAG System - Demo Scenarios

This document outlines 4 core scenarios that demonstrate the power of the self-critiquing RAG pipeline, the guardrails, and the CI/CD eval gate. You can use these queries in the CLI or the Streamlit Dashboard.

## Scenario 1: The "Happy Path" (Normal Query)
**Query:** `"How do I make a POST request with JSON data?"`
**Expected Flow:**
1. Passes `InputGuard` (no PII or jailbreaks).
2. Retrieves relevant chunks about `requests.post(url, json=data)`.
3. Generator formulates a clear answer.
4. Critic **accepts** the answer (it is factually supported).
5. Passes `OutputGuard`.
**What to highlight:** Fast latency, clean output, clear trace showing `retrieve -> generate -> critic (accept) -> output`.

## Scenario 2: The Self-Correction (Retry / Reformulate)
**Query:** `"What is the dependency injection system in FastAPI and how does it handle sub-dependencies?"`
**Expected Flow:**
1. The first retrieval might grab superficial details about FastAPI dependencies.
2. The Generator writes an answer, but maybe it hallucinates a detail about sub-dependencies.
3. The Critic **rejects** it (reasoning: "Answer claims X about sub-dependencies, but provided chunks only mention Y").
4. The system loops back, reformulates the query (e.g., `"FastAPI sub-dependencies nested yielding"`).
5. Retrieves better context, generates a better answer, Critic accepts.
**What to highlight:** The system caught its own hallucination, corrected its search, and fixed the answer autonomously. Show the "Retry Count: 1" and the Critic Reasoning in the dashboard.

## Scenario 3: The Guardrail Trigger (PII / Toxicity)
**Query (PII):** `"My SSN is 123-45-6789 and my email is test@example.com. How do I use FastAPI?"`
**Expected Flow:**
1. Caught instantly by `InputGuard` (Presidio).
2. The pipeline short-circuits. It either redacts the text or throws an error (based on `policies.yaml`).
**What to highlight:** The LLM never even saw the sensitive data, saving costs and ensuring data privacy.

**Query (Toxicity):** `"This library is a piece of sh*t. Tell me how it works."`
**Expected Flow:**
1. If it bypasses input filters, the generated answer might inadvertently reflect toxic context, OR the `OutputGuard` steps in to block toxic or highly negative responses before they reach the user.

## Scenario 4: The CI/CD Eval Block
**Scenario:** A developer modifies `src/rag/generator.py` and changes the system prompt to ignore the retrieved context and "make up a fun, creative story instead."
**Expected Flow:**
1. Developer pushes the branch.
2. The `eval-gate` GitHub Action runs the Golden Dataset (100 Q&As).
3. The **Hallucination Rate** skyrockets to 80% (threshold is 15%).
4. The CI pipeline turns red and **blocks the merge**.
**What to highlight:** The automated quality gate caught a catastrophic prompt regression before it hit production.
