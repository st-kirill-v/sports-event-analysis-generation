from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run active Football and/or NBA pipelines.")
    parser.add_argument("--football", action="store_true", help="Run the Football pipeline.")
    parser.add_argument("--nba", action="store_true", help="Run the NBA pipeline.")
    parser.add_argument(
        "--skip-nba-training",
        action="store_true",
        help="Run NBA data steps but skip NBA LSTM training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_football = args.football or not args.nba
    run_nba = args.nba or not args.football

    if run_football:
        subprocess.run(
            [sys.executable, "scripts/football/run_football.py"],
            cwd=PROJECT_ROOT,
            check=True,
        )
    if run_nba:
        nba_args = ["scripts/nba/run_nba.py", "--skip-merge"]
        if args.skip_nba_training:
            nba_args.append("--skip-training")
        subprocess.run([sys.executable, *nba_args], cwd=PROJECT_ROOT, check=True)

    print("\nMATCH DYNAMICS PIPELINES COMPLETE")


if __name__ == "__main__":
    main()
