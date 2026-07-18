import logging
import time
from typing import List

from eval.metrics import (
    calculate_cost,
    calculate_hallucination_rate,
    calculate_latency_percentiles,
    evaluate_faithfulness,
    evaluate_relevancy,
)
from rag.graph import run_query

logger = logging.getLogger(__name__)

def run_evaluation(dataset: List[dict], delay: float = 0.0) -> dict:
    """Run an end-to-end evaluation on a given dataset subset.

    Args:
        dataset: List of query dictionaries.
        delay: Seconds to wait between queries (use >0 in CI to respect rate limits).

    Returns:
        dict: containing summary metrics and detailed per-query results.
    """
    results = []
    latencies = []
    total_input_tokens = 0
    total_output_tokens = 0

    logger.info(f"Starting evaluation on {len(dataset)} items...")

    for i, item in enumerate(dataset):
        query = item["query"]
        logger.info(f"Evaluating {i+1}/{len(dataset)}: {query}")

        if i > 0 and delay > 0:
            time.sleep(delay)

        # 1. Run pipeline
        start_time = time.time()
        state = run_query(question=query, verbose=False)
        actual_latency_ms = state.get("latency_ms")

        if actual_latency_ms is None:
            actual_latency_ms = (time.time() - start_time) * 1000

        latency_sec = actual_latency_ms / 1000.0
        latencies.append(latency_sec)

        answer = state.get("answer", "")
        error = state.get("error", None)

        # We pass the full context block as a single string item in the list
        context = state.get("context", "")
        contexts = [context] if context else []

        # Approximate tokens (1 token ~ 4 chars) as a fallback for the Groq tracker
        input_tokens = len(query) // 4 + sum(len(c) for c in contexts) // 4
        output_tokens = len(answer) // 4
        total_input_tokens += input_tokens
        total_output_tokens += output_tokens

        # 2. Evaluate
        faithfulness = {"score": 0.0, "reasoning": "Skipped due to error"}
        relevancy = {"score": 0.0, "reasoning": "Skipped due to error"}

        if not error and answer:
            faithfulness = evaluate_faithfulness(query, contexts, answer)
            relevancy = evaluate_relevancy(query, answer)

        results.append({
            "query": query,
            "expected_category": item.get("category", "unknown"),
            "answer": answer,
            "faithfulness_score": faithfulness["score"],
            "faithfulness_reasoning": faithfulness["reasoning"],
            "relevancy_score": relevancy["score"],
            "relevancy_reasoning": relevancy["reasoning"],
            "latency_ms": actual_latency_ms,
            "error": error
        })

    # 3. Aggregate metrics
    faithfulness_scores = [r["faithfulness_score"] for r in results if not r["error"]]
    relevancy_scores = [r["relevancy_score"] for r in results if not r["error"]]

    avg_relevancy = sum(relevancy_scores) / len(relevancy_scores) if relevancy_scores else 0.0
    hallucination_rate = calculate_hallucination_rate(faithfulness_scores)
    lat_percentiles = calculate_latency_percentiles(latencies)
    cost = calculate_cost(total_input_tokens, total_output_tokens)

    return {
        "summary": {
            "total_queries": len(dataset),
            "successful_queries": len(faithfulness_scores),
            "hallucination_rate": hallucination_rate,
            "average_relevancy": avg_relevancy,
            "latency_p50_sec": lat_percentiles["p50"],
            "latency_p95_sec": lat_percentiles["p95"],
            "total_estimated_cost": cost,
            "cost_per_query": cost / len(dataset) if dataset else 0.0
        },
        "details": results
    }
