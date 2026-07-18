import argparse
import os
import sqlite3
import sys

from rich.console import Console
from rich.table import Table

# Ensure src is in the python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))


def main():
    parser = argparse.ArgumentParser(description="Guarded RAG System - Evaluation Trends")
    parser.add_argument("--db", default="data/eval_history.db", help="Path to SQLite DB")
    parser.add_argument("--limit", type=int, default=10, help="Number of historical runs to show")
    parser.add_argument("--mode", type=str, default="full", help="Filter by run mode (smoke or full)")
    args = parser.parse_args()

    if not os.path.exists(args.db):
        print(f"Database not found at {args.db}. Run an evaluation first to generate history.")
        return

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute(
        "SELECT * FROM eval_runs WHERE mode = ? ORDER BY timestamp DESC LIMIT ?",
        (args.mode, args.limit)
    )
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        print(f"No {args.mode} mode evaluation runs found in the database.")
        return

    console = Console()
    table = Table(title=f"Historical Evaluation Trends ({args.mode.upper()} Mode)")

    table.add_column("ID", style="cyan")
    table.add_column("Date (UTC)", style="dim")
    table.add_column("Commit", style="blue")
    table.add_column("Branch", style="dim")
    table.add_column("Faithfulness", style="magenta")
    table.add_column("Relevancy", style="magenta")
    table.add_column("Hallucination", style="red")
    table.add_column("Latency p95", style="yellow")
    table.add_column("Cost", style="green")

    for r in rows:
        timestamp_str = r["timestamp"]
        if "T" in timestamp_str:
            date_short = timestamp_str.split("T")[0] + " " + timestamp_str.split("T")[1][:5]
        else:
            date_short = timestamp_str[:16]

        commit_short = r["commit_hash"][:7] if r["commit_hash"] != "unknown" else "unknown"

        table.add_row(
            str(r["id"]),
            date_short,
            commit_short,
            r["branch"][:15],
            f"{r['average_faithfulness']:.2f}",
            f"{r['average_relevancy']:.2f}",
            f"{r['hallucination_rate']*100:.1f}%",
            f"{r['latency_p95_sec']:.2f}s",
            f"${r['total_estimated_cost']:.4f}"
        )

    console.print(table)


if __name__ == "__main__":
    main()
