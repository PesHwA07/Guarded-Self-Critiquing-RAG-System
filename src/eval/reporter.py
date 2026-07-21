import json

from rich.console import Console
from rich.table import Table


def print_report(results: dict):
    """Print a formatted console table summarizing the evaluation results."""
    console = Console()
    summary = results["summary"]

    table = Table(title=f"Eval Summary (Total Queries: {summary['total_queries']})")

    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")

    table.add_row("Successful Queries", str(summary['successful_queries']))
    table.add_row("Hallucination Rate", f"{summary['hallucination_rate']*100:.2f}%")
    table.add_row("Average Relevancy", f"{summary['average_relevancy']:.2f}")
    table.add_row("Latency p50", f"{summary['latency_p50_sec']:.2f}s")
    table.add_row("Latency p95", f"{summary['latency_p95_sec']:.2f}s")
    table.add_row("Cost per Query", f"${summary['cost_per_query']:.6f}")
    table.add_row("Total Cost", f"${summary['total_estimated_cost']:.6f}")

    console.print(table)


def save_report(results: dict, filepath: str):
    """Save the full evaluation results (summary + details) to a JSON file.
    
    Args:
        results: The complete dictionary of evaluation results including summary metrics.
        filepath: Destination path for the output JSON file.
    """
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def log_to_wandb(results: dict):
    """Log the evaluation summary metrics to Weights & Biases."""
    from config import settings

    if not settings.mlops.wandb_api_key:
        print("[INFO] WANDB_API_KEY not set. Skipping W&B logging.")
        return

    import wandb

    print(f"\n[INFO] Logging metrics to W&B Project: {settings.mlops.wandb_project}...")

    # Initialize wandb run
    wandb.init(
        project=settings.mlops.wandb_project,
        config={
            "total_queries": results["summary"]["total_queries"],
            "successful_queries": results["summary"]["successful_queries"],
            "llm_provider": settings.llm.provider,
            "vector_store": settings.retriever.vector_store
        }
    )

    # Log the summary metrics
    wandb.log({
        "hallucination_rate": results["summary"]["hallucination_rate"],
        "average_relevancy": results["summary"]["average_relevancy"],
        "latency_p50_sec": results["summary"]["latency_p50_sec"],
        "latency_p95_sec": results["summary"]["latency_p95_sec"],
        "cost_per_query": results["summary"]["cost_per_query"],
        "total_estimated_cost": results["summary"]["total_estimated_cost"],
        "average_faithfulness": results["summary"].get("average_faithfulness", 0.0)
    })

    # Finish the run
    wandb.finish()
    print("[INFO] W&B logging complete.")
