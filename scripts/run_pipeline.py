from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from investment_intelligence.pipeline import run_pipeline, save_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the NIFTY-50 investment intelligence pipeline.")
    parser.add_argument("--data-dir", default="data/raw", help="Folder containing Kaggle NIFTY-50 CSV files.")
    parser.add_argument("--profile", default="balanced", choices=["conservative", "balanced", "aggressive"])
    parser.add_argument("--output-dir", default="reports/generated")
    parser.add_argument("--model-dir", default="models")
    args = parser.parse_args()

    results = run_pipeline(data_dir=args.data_dir, profile=args.profile)
    save_outputs(results, args.output_dir, args.model_dir)

    print("Model metrics")
    for key, value in results["model_metrics"].items():
        print(f"  {key}: {value:.4f}")

    print("\nPortfolio metrics")
    for key, value in results["portfolio_metrics"].items():
        print(f"  {key}: {value:.4f}")

    print("\nTop allocation")
    print(results["allocation"][["symbol", "weight", "predicted_next_return", "sharpe_ratio"]].to_string(index=False))


if __name__ == "__main__":
    main()
