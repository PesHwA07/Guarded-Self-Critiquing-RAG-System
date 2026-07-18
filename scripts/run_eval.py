import argparse
import os
import sys

import yaml

# Ensure src is in the python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from eval.dataset import load_dataset
from eval.reporter import print_report, save_report
from eval.runner import run_evaluation
from eval.storage import save_to_sqlite


def load_thresholds(config_path: str) -> dict | None:
    """Load quality gate thresholds from a YAML config file.

    Returns None if the file doesn't exist (thresholds are optional).
    """
    if not os.path.exists(config_path):
        return None
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config.get("thresholds", None)


def check_thresholds(summary: dict, thresholds: dict) -> list[str]:
    """Check eval summary against thresholds. Returns a list of failure messages."""
    failures = []

    avg_faith = summary.get("average_faithfulness", 0.0)
    if avg_faith < thresholds.get("min_faithfulness", 0.0):
        failures.append(
            f"Faithfulness {avg_faith:.2f} < min {thresholds['min_faithfulness']}"
        )

    hall_rate = summary.get("hallucination_rate", 0.0)
    if hall_rate > thresholds.get("max_hallucination_rate", 1.0):
        failures.append(
            f"Hallucination rate {hall_rate:.2%} > max {thresholds['max_hallucination_rate']:.2%}"
        )

    avg_rel = summary.get("average_relevancy", 0.0)
    if avg_rel < thresholds.get("min_relevancy", 0.0):
        failures.append(
            f"Relevancy {avg_rel:.2f} < min {thresholds['min_relevancy']}"
        )

    lat_p95 = summary.get("latency_p95_sec", 0.0)
    if lat_p95 > thresholds.get("max_latency_p95_sec", float("inf")):
        failures.append(
            f"Latency p95 {lat_p95:.2f}s > max {thresholds['max_latency_p95_sec']}s"
        )

    return failures


def main():
    parser = argparse.ArgumentParser(description="Guarded RAG System - Evaluation Runner")
    parser.add_argument(
        "--mode", choices=["smoke", "full"], default="smoke",
        help="Run mode: 'smoke' for a quick test, 'full' for the entire dataset.",
    )
    parser.add_argument(
        "--dataset", default="data/golden_dataset.json",
        help="Path to the dataset JSON file.",
    )
    parser.add_argument(
        "--output", default="eval_report.json",
        help="Output path for the evaluation report JSON.",
    )
    parser.add_argument(
        "--config", default="data/eval_config.yaml",
        help="YAML config file with quality gate thresholds.",
    )
    parser.add_argument(
        "--delay", type=float, default=0.0,
        help="Seconds to wait between queries (use 2.0 in CI full mode for rate limits).",
    )

    args = parser.parse_args()

    print(f"Running evaluation in {args.mode.upper()} mode...")

    # 1. Load dataset
    dataset = load_dataset(args.dataset, mode=args.mode)

    # 2. Run evaluation
    results = run_evaluation(dataset, delay=args.delay)

    # 3. Compute average faithfulness for threshold checking
    details = results.get("details", [])
    faith_scores = [r["faithfulness_score"] for r in details if not r.get("error")]
    avg_faithfulness = sum(faith_scores) / len(faith_scores) if faith_scores else 0.0
    results["summary"]["average_faithfulness"] = avg_faithfulness

    # 4. Report results
    print_report(results)
    save_report(results, args.output)
    save_to_sqlite(results, mode=args.mode)
    print(f"\n[DONE] Full report saved to {args.output} and SQLite DB")

    # 5. Quality gate check
    thresholds = load_thresholds(args.config)
    if thresholds:
        failures = check_thresholds(results["summary"], thresholds)
        if failures:
            print("\n[FAIL] Quality gate failed:")
            for f in failures:
                print(f"  - {f}")
            sys.exit(1)
        else:
            print("\n[PASS] All quality gate thresholds met.")
            sys.exit(0)
    else:
        print("\n[INFO] No threshold config found, skipping quality gate check.")


if __name__ == "__main__":
    main()
