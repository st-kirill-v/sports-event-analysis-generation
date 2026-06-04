from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


TARGET = "target_score_diff_change_last_5min"
DATA_PATH = Path("data/nba/nba_feature_engineering_400_enhanced.csv")
METRICS_DIR = Path("outputs/metrics/nba")
FIGURES_DIR = Path("outputs/figures/nba")
REPORTS_DIR = Path("outputs/reports")


def main() -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    if TARGET not in df.columns:
        raise KeyError(f"Missing target column: {TARGET}")
    if "GAME_ID" not in df.columns:
        raise KeyError("Missing GAME_ID column.")

    match_target = (
        df[["GAME_ID", TARGET]]
        .drop_duplicates("GAME_ID")
        .sort_values("GAME_ID")
        .reset_index(drop=True)
    )
    target = match_target[TARGET].astype(float)
    abs_target = target.abs()

    describe = target.describe().rename("value").reset_index()
    describe.columns = ["statistic", "value"]
    describe.to_csv(METRICS_DIR / "nba_target_describe.csv", index=False)

    abs_stats = pd.DataFrame(
        [
            {
                "mean_abs_target": abs_target.mean(),
                "median_abs_target": abs_target.median(),
                "std_abs_target": abs_target.std(),
                "p75_abs_target": abs_target.quantile(0.75),
                "p90_abs_target": abs_target.quantile(0.90),
                "p95_abs_target": abs_target.quantile(0.95),
                "max_abs_target": abs_target.max(),
            }
        ]
    )
    abs_stats.to_csv(METRICS_DIR / "nba_target_abs_statistics.csv", index=False)

    thresholds = [
        ("abs_target_le_1", abs_target <= 1),
        ("abs_target_le_3", abs_target <= 3),
        ("abs_target_le_5", abs_target <= 5),
        ("abs_target_le_10", abs_target <= 10),
        ("abs_target_gt_10", abs_target > 10),
    ]
    percentages = pd.DataFrame(
        [
            {
                "segment": name,
                "matches": int(mask.sum()),
                "percent": float(mask.mean()),
            }
            for name, mask in thresholds
        ]
    )
    percentages.to_csv(METRICS_DIR / "nba_target_threshold_percentages.csv", index=False)

    comparison_path = METRICS_DIR / "feature_set_full_training_comparison.csv"
    if comparison_path.exists():
        comparison = pd.read_csv(comparison_path)
        best_row = comparison[comparison["split"].eq("test")].sort_values("mae").iloc[0]
        best_model = str(best_row["model"])
        best_feature_set = str(best_row["feature_set"])
        best_mae = float(best_row["mae"])
    else:
        best_model = "NBA_LSTM_top75"
        best_feature_set = "top75"
        best_mae = 4.5989

    mean_abs = float(abs_target.mean())
    median_abs = float(abs_target.median())
    mae_context = pd.DataFrame(
        [
            {
                "best_model": best_model,
                "best_feature_set": best_feature_set,
                "mae": best_mae,
                "mean_abs_target": mean_abs,
                "median_abs_target": median_abs,
                "mae_percent_of_mean_abs_target": best_mae / mean_abs if mean_abs else pd.NA,
                "mae_percent_of_median_abs_target": best_mae / median_abs if median_abs else pd.NA,
            }
        ]
    )
    mae_context.to_csv(METRICS_DIR / "nba_target_mae_context.csv", index=False)

    plt.figure(figsize=(10, 6))
    plt.hist(target, bins=31, color="#2d6cdf", edgecolor="white")
    plt.axvline(target.mean(), color="black", linestyle="--", label=f"mean={target.mean():.2f}")
    plt.title("NBA target distribution: score diff change in last 5 minutes")
    plt.xlabel(TARGET)
    plt.ylabel("Games")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "nba_target_histogram.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.hist(abs_target, bins=25, color="#1f9d70", edgecolor="white")
    plt.axvline(mean_abs, color="black", linestyle="--", label=f"mean={mean_abs:.2f}")
    plt.axvline(median_abs, color="#a83232", linestyle="--", label=f"median={median_abs:.2f}")
    plt.title("NBA absolute target distribution")
    plt.xlabel(f"abs({TARGET})")
    plt.ylabel("Games")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "nba_abs_target_histogram.png", dpi=160)
    plt.close()

    lines = [
        "NBA target analysis",
        "",
        f"Dataset: {DATA_PATH}",
        f"Rows: {len(df)}",
        f"Unique games: {match_target['GAME_ID'].nunique()}",
        f"Target: {TARGET}",
        "",
        "target.describe():",
        describe.to_string(index=False),
        "",
        "abs(target) statistics:",
        abs_stats.to_string(index=False),
        "",
        "Match percentages:",
        percentages.to_string(index=False),
        "",
        "Best LSTM MAE context:",
        mae_context.to_string(index=False),
    ]
    report_path = REPORTS_DIR / "nba_target_analysis_report.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    print()
    print(f"Saved target describe: {METRICS_DIR / 'nba_target_describe.csv'}")
    print(f"Saved abs target stats: {METRICS_DIR / 'nba_target_abs_statistics.csv'}")
    print(f"Saved percentages: {METRICS_DIR / 'nba_target_threshold_percentages.csv'}")
    print(f"Saved MAE context: {METRICS_DIR / 'nba_target_mae_context.csv'}")
    print(f"Saved report: {report_path}")
    print("NBA TARGET ANALYSIS COMPLETE")


if __name__ == "__main__":
    main()
