from __future__ import annotations

from match_dynamics.common.config import ProjectConfig
from match_dynamics.nba_pipeline.historical_team_features import run_nba_historical_team_features


def main() -> None:
    cfg = ProjectConfig()
    input_path = cfg.data_dir / "nba" / "nba_feature_engineering_final_400.csv"
    output_path = cfg.data_dir / "nba" / "nba_feature_engineering_400_enhanced.csv"
    report_path = cfg.output_dir / "reports" / "nba_historical_team_features_report.txt"

    reports = run_nba_historical_team_features(
        input_path=input_path,
        output_path=output_path,
        report_path=report_path,
    )

    print("NBA historical team features summary:")
    print(f"Output: {output_path}")
    print("\nLeakage validation:")
    print(reports["validation"].to_string(index=False))
    print("\nTop 30 features:")
    print(reports["top30"].to_string(index=False))
    print("\nRecommended drop features:")
    print(reports["recommended_drop_features"].head(30).to_string(index=False))
    print("\nReport:")
    print(report_path)
    print("\nNBA HISTORICAL TEAM FEATURES COMPLETE")


if __name__ == "__main__":
    main()
