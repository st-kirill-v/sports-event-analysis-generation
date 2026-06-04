from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run_step(args: list[str]) -> None:
    subprocess.run([sys.executable, *args], cwd=PROJECT_ROOT, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the active NBA clutch regression pipeline.")
    parser.add_argument(
        "--skip-merge",
        action="store_true",
        help="Use the existing merged NBA CSV and start from preprocessing.",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Build data/features but skip LSTM training.",
    )
    parser.add_argument(
        "--merge-games",
        type=int,
        default=400,
        help="Number passed to the merge step. The active pipeline uses 400 by default.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.skip_merge:
        run_step(
            [
                "scripts/nba/merge_nba_400.py",
                "--skip-download",
                "--target-games",
                str(args.merge_games),
            ]
        )
    run_step(["scripts/nba/run_nba_preprocessing.py"])
    run_step(["scripts/nba/run_nba_feature_engineering.py"])
    run_step(["scripts/nba/validate_historical_timeline.py"])
    run_step(["scripts/nba/add_historical_team_features.py"])
    if not args.skip_training:
        run_step(["scripts/nba/run_nba_lstm_clutch_regression.py"])
    print("\nNBA PIPELINE COMPLETE")


if __name__ == "__main__":
    main()
