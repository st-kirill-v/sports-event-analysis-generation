from __future__ import annotations

import argparse

from match_dynamics.common.config import ProjectConfig
from match_dynamics.nba_pipeline.lstm_feature_sets import (
    FULL_TRAINING_COMMANDS,
    run_nba_lstm_feature_set_pipeline,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train NBA LSTM regression feature-set models.")
    parser.add_argument(
        "--feature-set",
        choices=["top30", "top50", "top75", "all_features", "all"],
        default="top50",
        help="Feature set to train. Use 'all' to train all feature-set models.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run only top50 smoke training for one epoch and save smoke-test outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ProjectConfig()
    feature_set = "all" if args.smoke_test else args.feature_set
    reports = run_nba_lstm_feature_set_pipeline(
        input_path=cfg.data_dir / "nba" / "nba_feature_engineering_400_enhanced.csv",
        reports_dir=cfg.output_dir / "reports",
        sequences_dir=cfg.data_dir / "nba" / "sequences",
        scalers_dir=cfg.models_dir / "nba" / "scalers",
        metrics_dir=cfg.metrics_dir / "nba",
        models_dir=cfg.models_dir / "nba",
        feature_set=feature_set,
        smoke_test=args.smoke_test,
    )

    print("NBA LSTM feature-set pipeline summary")
    print("\nSplit summary:")
    print(
        reports["split_summary"]
        .groupby("split")
        .size()
        .reset_index(name="games")
        .to_string(index=False)
    )
    print("\nFeature sets:")
    print(reports["feature_sets"].to_string(index=False))
    print("\nLeakage validation:")
    print(reports["validation"].to_string(index=False))
    print("\nSequence shapes:")
    print(reports["shapes"].to_string(index=False))
    print("\nBaselines:")
    print(reports["baselines"].to_string(index=False))
    if args.smoke_test:
        print("\nSmoke-test metrics:")
        print(reports["metrics"].to_string(index=False))
        print("\nNBA LSTM SMOKE TEST PASSED")
        print("\nFULL TRAINING COMMANDS:")
        for command in FULL_TRAINING_COMMANDS:
            print(command)
    print("\nNBA LSTM FEATURE-SET TRAINING PIPELINE READY")


if __name__ == "__main__":
    main()
