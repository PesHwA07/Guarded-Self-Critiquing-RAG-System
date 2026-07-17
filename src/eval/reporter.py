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
    """Save the full evaluation results (summary + details) to a JSON file."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
