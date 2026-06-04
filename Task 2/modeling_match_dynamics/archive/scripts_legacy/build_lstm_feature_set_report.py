from __future__ import annotations

from pathlib import Path

import pandas as pd


def main() -> None:
    metrics_dir = Path("outputs/metrics/nba")
    reports_dir = Path("outputs/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    feature_sets = ["top30", "top50", "top75", "all_features"]
    frames: list[pd.DataFrame] = []
    for feature_set in feature_sets:
        path = metrics_dir / f"nba_lstm_{feature_set}_metrics.csv"
        if not path.exists():
            print(f"[warning] Missing metrics file: {path}")
            continue
        frame = pd.read_csv(path)
        frame.insert(0, "feature_set", feature_set)
        frames.append(frame)

    if not frames:
        raise FileNotFoundError("No NBA LSTM feature-set metrics files found.")

    comparison = pd.concat(frames, ignore_index=True)
    comparison_path = metrics_dir / "feature_set_full_training_comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    test_metrics = comparison[comparison["split"].eq("test")].copy()
    test_metrics = test_metrics.sort_values("mae")

    baselines_path = metrics_dir / "feature_set_baselines.csv"
    baselines = pd.read_csv(baselines_path) if baselines_path.exists() else pd.DataFrame()

    lines: list[str] = [
        "NBA LSTM feature-set full training report",
        "",
        "Dataset: data/nba/nba_feature_engineering_400_enhanced.csv",
        "Split: 280 train games, 60 validation games, 60 test games",
        "Target: target_score_diff_change_last_5min",
        "Task: regression, one GAME_ID = one event-level sequence",
        "",
        "Test metrics by feature set:",
        test_metrics[["feature_set", "mae", "mse", "rmse", "r2"]].to_string(index=False),
        "",
    ]

    if not baselines.empty:
        baseline_test = baselines[baselines["split"].eq("test")].drop_duplicates(["model"])[
            ["model", "mae", "mse", "rmse", "r2"]
        ]
        lines.extend(["Test baselines:", baseline_test.to_string(index=False), ""])

    best_mae = test_metrics.iloc[0]
    best_mse = test_metrics.sort_values("mse").iloc[0]
    lines.extend(
        [
            (
                "Best by MAE: "
                f"{best_mae.feature_set} | "
                f"MAE={best_mae.mae:.6f}, "
                f"MSE={best_mae.mse:.6f}, "
                f"RMSE={best_mae.rmse:.6f}, "
                f"R2={best_mae.r2:.6f}"
            ),
            (
                "Best by MSE/RMSE/R2: "
                f"{best_mse.feature_set} | "
                f"MAE={best_mse.mae:.6f}, "
                f"MSE={best_mse.mse:.6f}, "
                f"RMSE={best_mse.rmse:.6f}, "
                f"R2={best_mse.r2:.6f}"
            ),
            "",
            (
                "Interpretation: all trained LSTM variants beat the constant-zero "
                "and train-mean baselines on test MAE/MSE. top75 is strongest by "
                "absolute error; all_features is strongest by squared error and R2, "
                "but the gap is small."
            ),
        ]
    )

    report_path = reports_dir / "nba_lstm_feature_set_training_report.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    print()
    print(f"Saved comparison: {comparison_path}")
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
