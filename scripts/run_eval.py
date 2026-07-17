import argparse
import sys
import os

# Ensure src is in the python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from eval.dataset import load_dataset
from eval.runner import run_evaluation
from eval.reporter import print_report, save_report

def main():
    parser = argparse.ArgumentParser(description="Guarded RAG System - Evaluation Runner")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke",
                        help="Run mode: 'smoke' for a quick test, 'full' for the entire dataset.")
    parser.add_argument("--dataset", default="data/golden_dataset.json",
                        help="Path to the dataset JSON file.")
    parser.add_argument("--output", default="eval_report.json",
                        help="Output path for the evaluation report JSON.")
    parser.add_argument("--config", default="data/eval_config.yaml", 
                        help="Configuration file for thresholds (to be implemented in Week 5).")
    
    args = parser.parse_args()

    print(f"Running evaluation in {args.mode.upper()} mode...")
    
    # 1. Load dataset
    dataset = load_dataset(args.dataset, mode=args.mode)
    
    # 2. Run evaluation
    results = run_evaluation(dataset)
    
    # 3. Report results
    print_report(results)
    save_report(results, args.output)
    
    print(f"\n[DONE] Full report saved to {args.output}")

if __name__ == "__main__":
    main()
