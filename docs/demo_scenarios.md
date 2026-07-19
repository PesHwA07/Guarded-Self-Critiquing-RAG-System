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