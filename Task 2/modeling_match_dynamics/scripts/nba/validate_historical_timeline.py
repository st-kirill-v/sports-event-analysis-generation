from __future__ import annotations

from match_dynamics.common.config import ProjectConfig
from match_dynamics.nba_pipeline.historical_timeline import validate_nba_historical_timeline


def main() -> None:
    cfg = ProjectConfig()
    input_path = cfg.data_dir / "nba" / "nba_feature_engineering_400.csv"
    final_output_path = cfg.data_dir / "nba" / "nba_feature_engineering_final_400.csv"
    report_path = cfg.output_dir / "reports" / "nba_historical_timeline_validation.txt"

    result = validate_nba_historical_timeline(
        input_path=input_path,
        final_output_path=final_output_path,
        report_path=report_path,
    )

    match_level = result["match_level"]
    validation = result["validation"]
    print("NBA historical timeline validation")
    print(f"Historical timeline source: {result['timeline_source']}")
    print(f"Matches: {len(match_level)}")
    print("\nFirst 20 matches:")
    print(result["preview"].to_string(index=False))
    print("\nLeakage validation sample:")
    print(validation.to_string(index=False))
    print("\nSaved report:")
    print(report_path)
    print("\nSaved final feature file:")
    print(final_output_path)
    print("\nNBA HISTORICAL TIMELINE VALIDATED")


if __name__ == "__main__":
    main()
