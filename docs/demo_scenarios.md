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