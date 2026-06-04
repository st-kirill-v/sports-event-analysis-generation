from __future__ import annotations

from match_dynamics.common.config import ProjectConfig
from match_dynamics.nba_pipeline.preprocessing import run_nba_preprocessing


def main() -> None:
    cfg = ProjectConfig()
    input_path = cfg.data_dir / "nba" / "nba_merged_400.csv"
    fallback_input_path = input_path
    output_path = cfg.data_dir / "nba" / "nba_merged_preprocessed_400.csv"
    report_dir = cfg.output_dir / "reports" / "nba_preprocessing_400"

    reports = run_nba_preprocessing(
        input_path=input_path,
        fallback_input_path=fallback_input_path,
        output_path=output_path,
        report_dir=report_dir,
    )

    print("\nNBA preprocessing summary:")
    print(reports["summary"].to_string(index=False))
    print("\nDropped columns:")
    print(reports["dropped_columns"].to_string(index=False))
    print("\nCreated features:")
    print(reports["created_features"].to_string(index=False))
    print("\nQuality checks:")
    print(reports["quality_checks"].to_string(index=False))
    print("\nTop missing before:")
    print(
        reports["top_missing_before"][["column", "null_count", "null_percent"]]
        .head(20)
        .to_string(index=False)
    )
    print("\nTop missing after:")
    print(
        reports["top_missing_after"][["column", "null_count", "null_percent"]]
        .head(20)
        .to_string(index=False)
    )
    print("\nSaved:")
    print(output_path)
    print("\nNBA 400 PREPROCESSING COMPLETE")


if __name__ == "__main__":
    main()
