from __future__ import annotations

from match_dynamics.common.config import ProjectConfig
from match_dynamics.nba_pipeline.feature_engineering import run_nba_feature_engineering


def main() -> None:
    cfg = ProjectConfig()
    input_path = cfg.data_dir / "nba" / "nba_merged_preprocessed_400.csv"
    fallback_input_path = input_path
    final_preprocessing_path = cfg.data_dir / "nba" / "nba_merged_preprocessed_400_final.csv"
    output_path = cfg.data_dir / "nba" / "nba_feature_engineering_400.csv"
    fix_report_path = cfg.output_dir / "reports" / "nba_preprocessing_400_final_report.txt"
    feature_report_dir = cfg.output_dir / "reports" / "nba_feature_engineering_400"

    reports = run_nba_feature_engineering(
        input_path=input_path,
        fallback_input_path=fallback_input_path,
        final_preprocessing_path=final_preprocessing_path,
        output_path=output_path,
        fix_report_path=fix_report_path,
        feature_report_dir=feature_report_dir,
    )

    print("NBA feature engineering summary:")
    print("\nTarget distribution:")
    print(reports["target_distribution"].to_string(index=False))
    print("\nTop correlations with target:")
    print(reports["correlations"].head(20).to_string(index=False))
    print("\nTop RF importance:")
    print(reports["rf_importance"].head(20).to_string(index=False))
    print("\nRecommended drop features:")
    print(reports["recommended_drop_features"].head(30).to_string(index=False))
    print("\nTop missing columns:")
    print(
        reports["top_missing_columns"][["column", "null_count", "null_percent"]]
        .head(20)
        .to_string(index=False)
    )
    print("\nSaved:")
    print(final_preprocessing_path)
    print(output_path)
    print("\nNBA FEATURE ENGINEERING 400 COMPLETE")


if __name__ == "__main__":
    main()
