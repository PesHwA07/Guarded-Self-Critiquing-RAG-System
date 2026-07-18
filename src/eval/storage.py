import os
import sqlite3
import subprocess
from datetime import datetime, timezone


def _get_git_info() -> tuple[str, str]:
    """Try to get the current git commit hash and branch name."""
    try:
        commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).decode("utf-8").strip()
        return commit_hash, branch
    except Exception:
        return "unknown", "unknown"


def init_db(db_path: str):
    """Initialize the SQLite database schema for eval runs."""
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS eval_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            commit_hash TEXT,
            branch TEXT,
            mode TEXT,
            total_queries INTEGER,
            successful_queries INTEGER,
            hallucination_rate REAL,
            average_relevancy REAL,
            average_faithfulness REAL,
            latency_p50_sec REAL,
            latency_p95_sec REAL,
            total_estimated_cost REAL,
            cost_per_query REAL
        )
    """)
    conn.commit()
    conn.close()


def save_to_sqlite(results: dict, mode: str, db_path: str = "data/eval_history.db"):
    """Save the evaluation summary to the SQLite database."""
    init_db(db_path)
    summary = results.get("summary", {})
    commit_hash, branch = _get_git_info()

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Use UTC timezone-aware datetime for isoformat
    now_iso = datetime.now(timezone.utc).isoformat()

    cursor.execute("""
        INSERT INTO eval_runs (
            timestamp, commit_hash, branch, mode, total_queries, successful_queries,
            hallucination_rate, average_relevancy, average_faithfulness,
            latency_p50_sec, latency_p95_sec, total_estimated_cost, cost_per_query
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        now_iso,
        commit_hash,
        branch,
        mode,
        summary.get("total_queries", 0),
        summary.get("successful_queries", 0),
        summary.get("hallucination_rate", 0.0),
        summary.get("average_relevancy", 0.0),
        summary.get("average_faithfulness", 0.0),
        summary.get("latency_p50_sec", 0.0),
        summary.get("latency_p95_sec", 0.0),
        summary.get("total_estimated_cost", 0.0),
        summary.get("cost_per_query", 0.0)
    ))
    conn.commit()
    conn.close()
