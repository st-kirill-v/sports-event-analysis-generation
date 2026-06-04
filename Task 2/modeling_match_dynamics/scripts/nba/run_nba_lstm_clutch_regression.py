from __future__ import annotations

from match_dynamics.common.config import ProjectConfig
from match_dynamics.nba_pipeline.lstm_training import run_nba_lstm_clutch_regression


def main() -> None:
    cfg = ProjectConfig()
    reports = run_nba_lstm_clutch_regression(
        input_path=cfg.data_dir / "nba" / "nba_feature_engineering_400_enhanced.csv",
        data_dir=cfg.data_dir / "nba",
        models_dir=cfg.models_dir / "nba",
        metrics_dir=cfg.metrics_dir / "nba",
        figures_dir=cfg.figures_dir / "nba",
    )

    print("\nNBA LSTM clutch regression shapes:")
    print(reports["shapes"].to_string(index=False))
    print("\nNBA LSTM clutch regression metrics:")
    print(reports["metrics"].to_string(index=False))
    print("\nTraining summary:")
    print(reports["training_summary"].to_string(index=False))
    print("\nTest predictions:")
    test_predictions = reports["predictions"][reports["predictions"]["split"].eq("test")]
    print(test_predictions.to_string(index=False))
    print("\nFeature count:", len(reports["features"]))
    print("\nNBA LSTM CLUTCH REGRESSION COMPLETE")


if __name__ == "__main__":
    main()
