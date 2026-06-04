from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from match_dynamics.common.config import ProjectConfig
from match_dynamics.nba_pipeline.merge import NbaMergePaths, build_nba_valid_games_merge


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build an NBA merge with movement + events + shots_fixed. "
            "Broken/incomplete games are skipped and replaced by later games."
        )
    )
    parser.add_argument("--target-games", type=int, default=200)
    parser.add_argument("--max-archives", type=int, default=700)
    parser.add_argument("--max-events", type=int, default=700)
    parser.add_argument("--moment-stride", type=int, default=50)
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Use cached files in data/nba and only rebuild the merged dataset.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Defaults to data/nba/nba_merged_<target-games>.csv.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable step-by-step progress output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ProjectConfig()
    nba_dir = cfg.data_dir / "nba"
    output_csv = args.output_csv or nba_dir / f"nba_merged_{args.target_games}.csv"
    paths = NbaMergePaths(
        data_dir=nba_dir,
        movement_dir=nba_dir / "movement",
        events_dir=nba_dir / "events",
        shots_dir=nba_dir / "shots",
        merged_csv=output_csv,
        audit_dir=cfg.output_dir / "audits" / "data_quality",
        report_dir=cfg.output_dir / "reports" / f"nba_merge_{args.target_games}",
    )

    reports = build_nba_valid_games_merge(
        paths=paths,
        target_games=args.target_games,
        max_archives=args.max_archives,
        max_events=args.max_events,
        moment_stride=args.moment_stride,
        download=not args.skip_download,
        verbose=not args.quiet,
    )

    print(f"\nNBA {args.target_games}-game merge summary:")
    print(reports["merged_summary"].to_string(index=False))
    print("\nValid game selection:")
    print(
        reports["valid_game_selection"]
        .tail(20)[["game_id", "has_events", "has_shots_fixed", "selected", "warning"]]
        .to_string(index=False)
    )
    print("\nTop missing columns:")
    print(
        reports["merged_top_missing_columns"][
            ["column", "dtype", "null_count", "null_percent", "unique_count"]
        ]
        .head(20)
        .to_string(index=False)
    )
    print("\nSaved merged dataset:")
    print(output_csv)
    print(f"\nNBA {args.target_games}-GAME MERGE READY")


if __name__ == "__main__":
    main()
