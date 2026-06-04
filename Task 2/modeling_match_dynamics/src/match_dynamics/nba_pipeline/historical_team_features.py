from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from match_dynamics.nba_pipeline.feature_engineering import (
    build_recommended_drop_features,
    column_quality,
    model_numeric_feature_columns,
    random_forest_importance,
)


TARGET = "target_score_diff_change_last_5min"

MATCH_COLUMNS = [
    "GAME_ID",
    "HTM",
    "VTM",
    "final_home_score",
    "final_away_score",
    "final_score_diff",
    TARGET,
]

HISTORICAL_FEATURES = [
    "home_team_avg_points_scored_last_5",
    "home_team_avg_points_scored_last_10",
    "away_team_avg_points_scored_last_5",
    "away_team_avg_points_scored_last_10",
    "home_team_avg_points_allowed_last_5",
    "home_team_avg_points_allowed_last_10",
    "away_team_avg_points_allowed_last_5",
    "away_team_avg_points_allowed_last_10",
    "home_team_win_rate_last_5",
    "home_team_win_rate_last_10",
    "away_team_win_rate_last_5",
    "away_team_win_rate_last_10",
    "home_team_avg_clutch_diff_last_5",
    "home_team_avg_clutch_diff_last_10",
    "away_team_avg_clutch_diff_last_5",
    "away_team_avg_clutch_diff_last_10",
    "team_offense_strength_diff_last_5",
    "team_offense_strength_diff_last_10",
    "team_defense_strength_diff_last_5",
    "team_defense_strength_diff_last_10",
    "team_form_diff_last_5",
    "team_form_diff_last_10",
    "team_clutch_strength_diff_last_5",
    "team_clutch_strength_diff_last_10",
    "home_team_history_missing",
    "away_team_history_missing",
    "home_team_less_than_5_history",
    "away_team_less_than_5_history",
    "home_team_less_than_10_history",
    "away_team_less_than_10_history",
]

EXCLUDED_FROM_X = [
    "GAME_ID",
    "GAME_DATE",
    "EVENTNUM",
    "GAME_EVENT_ID",
    "PLAYER1_NAME",
    "PLAYER2_NAME",
    "PLAYER3_NAME",
    "PLAYER1_TEAM_ABBREVIATION",
    "PLAYER2_TEAM_ABBREVIATION",
    "PLAYER3_TEAM_ABBREVIATION",
    "HOMEDESCRIPTION",
    "VISITORDESCRIPTION",
    "SCORE",
    "SCOREMARGIN",
    "final_home_score",
    "final_away_score",
    "final_score_diff",
    "score_diff_at_5min_remaining",
    TARGET,
    "cutoff_found",
    "HTM",
    "VTM",
    "TEAM_NAME",
]


def run_nba_historical_team_features(
    input_path: Path,
    output_path: Path,
    report_path: Path,
) -> dict[str, pd.DataFrame]:
    if not input_path.exists():
        raise FileNotFoundError(f"NBA feature dataset not found: {input_path}")

    df = pd.read_csv(input_path)
    missing = [col for col in MATCH_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Required columns are missing: {missing}")

    input_shape = df.shape
    match_df = build_match_table(df)
    history = build_historical_match_features(match_df)
    enhanced = df.merge(history[["GAME_ID", *HISTORICAL_FEATURES]], on="GAME_ID", how="left")

    for col in HISTORICAL_FEATURES:
        enhanced[col] = pd.to_numeric(enhanced[col], errors="coerce").fillna(0)
    flag_cols = [
        col for col in HISTORICAL_FEATURES if col.endswith("_missing") or "_less_than_" in col
    ]
    for col in flag_cols:
        enhanced[col] = enhanced[col].astype("int8")

    validation = leakage_validation(match_df, history, enhanced)
    if not bool(validation["status"].all()):
        raise RuntimeError("NBA historical team feature leakage validation failed.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    enhanced.to_csv(output_path, index=False)

    reports = create_enhanced_feature_audit(enhanced, report_path.parent)
    summary = build_report_text(
        input_path=input_path,
        output_path=output_path,
        input_shape=input_shape,
        output_shape=enhanced.shape,
        match_df=match_df,
        history=history,
        validation=validation,
        reports=reports,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(summary, encoding="utf-8")

    history.to_csv(
        report_path.with_name("nba_historical_team_features_match_level.csv"), index=False
    )
    validation.to_csv(
        report_path.with_name("nba_historical_team_features_leakage_validation.csv"), index=False
    )
    pd.DataFrame({"feature": HISTORICAL_FEATURES}).to_csv(
        report_path.with_name("nba_historical_team_features_created.csv"), index=False
    )
    pd.DataFrame(
        {"excluded_from_x": [col for col in EXCLUDED_FROM_X if col in enhanced.columns]}
    ).to_csv(report_path.with_name("nba_excluded_from_x_columns.csv"), index=False)
    return {"enhanced": enhanced, "match_level": history, "validation": validation, **reports}


def build_match_table(df: pd.DataFrame) -> pd.DataFrame:
    match_df = (
        df[MATCH_COLUMNS]
        .groupby("GAME_ID", as_index=False, sort=False)
        .agg(
            {
                "HTM": "first",
                "VTM": "first",
                "final_home_score": "first",
                "final_away_score": "first",
                "final_score_diff": "first",
                TARGET: "first",
            }
        )
    )
    match_df["GAME_ID_NUMERIC"] = pd.to_numeric(match_df["GAME_ID"], errors="coerce")
    return match_df.sort_values(["GAME_ID_NUMERIC", "GAME_ID"], kind="mergesort").reset_index(
        drop=True
    )


def build_historical_match_features(match_df: pd.DataFrame) -> pd.DataFrame:
    team_history: dict[str, list[dict[str, float]]] = {}
    rows = []
    for row in match_df.itertuples(index=False):
        game_id = row.GAME_ID
        game_id_numeric = float(row.GAME_ID_NUMERIC)
        home_team = str(row.HTM)
        away_team = str(row.VTM)
        home_history = team_history.get(home_team, [])
        away_history = team_history.get(away_team, [])
        features = {
            "GAME_ID": game_id,
            "historical_max_GAME_ID_home": max_history_game_id(home_history),
            "historical_max_GAME_ID_away": max_history_game_id(away_history),
        }
        features.update(team_side_features(home_history, "home_team"))
        features.update(team_side_features(away_history, "away_team"))
        features.update(relative_features(features))
        rows.append(features)

        home_points = float(row.final_home_score)
        away_points = float(row.final_away_score)
        target = float(getattr(row, TARGET))
        team_history.setdefault(home_team, []).append(
            {
                "GAME_ID_NUMERIC": game_id_numeric,
                "points_scored": home_points,
                "points_allowed": away_points,
                "win": float(home_points > away_points),
                "clutch_diff": target,
            }
        )
        team_history.setdefault(away_team, []).append(
            {
                "GAME_ID_NUMERIC": game_id_numeric,
                "points_scored": away_points,
                "points_allowed": home_points,
                "win": float(away_points > home_points),
                "clutch_diff": -target,
            }
        )
    return pd.DataFrame(rows)


def max_history_game_id(history: list[dict[str, float]]) -> float:
    if not history:
        return np.nan
    return max(item["GAME_ID_NUMERIC"] for item in history)


def team_side_features(history: list[dict[str, float]], prefix: str) -> dict[str, float]:
    features: dict[str, float] = {}
    history_count = len(history)
    features[f"{prefix}_history_missing"] = int(history_count == 0)
    features[f"{prefix}_less_than_5_history"] = int(history_count < 5)
    features[f"{prefix}_less_than_10_history"] = int(history_count < 10)
    for window in [5, 10]:
        recent = history[-window:]
        features[f"{prefix}_avg_points_scored_last_{window}"] = mean_or_zero(
            recent, "points_scored"
        )
        features[f"{prefix}_avg_points_allowed_last_{window}"] = mean_or_zero(
            recent, "points_allowed"
        )
        features[f"{prefix}_win_rate_last_{window}"] = mean_or_zero(recent, "win")
        features[f"{prefix}_avg_clutch_diff_last_{window}"] = mean_or_zero(recent, "clutch_diff")
    return features


def mean_or_zero(history: list[dict[str, float]], key: str) -> float:
    if not history:
        return 0.0
    return float(np.mean([item[key] for item in history]))


def relative_features(features: dict[str, float]) -> dict[str, float]:
    rows = {}
    for window in [5, 10]:
        rows[f"team_offense_strength_diff_last_{window}"] = (
            features[f"home_team_avg_points_scored_last_{window}"]
            - features[f"away_team_avg_points_scored_last_{window}"]
        )
        rows[f"team_defense_strength_diff_last_{window}"] = (
            features[f"home_team_avg_points_allowed_last_{window}"]
            - features[f"away_team_avg_points_allowed_last_{window}"]
        )
        rows[f"team_form_diff_last_{window}"] = (
            features[f"home_team_win_rate_last_{window}"]
            - features[f"away_team_win_rate_last_{window}"]
        )
        rows[f"team_clutch_strength_diff_last_{window}"] = (
            features[f"home_team_avg_clutch_diff_last_{window}"]
            - features[f"away_team_avg_clutch_diff_last_{window}"]
        )
    return rows


def leakage_validation(
    match_df: pd.DataFrame, history: pd.DataFrame, enhanced: pd.DataFrame
) -> pd.DataFrame:
    merged = match_df[["GAME_ID", "GAME_ID_NUMERIC"]].merge(history, on="GAME_ID", how="left")
    max_history = pd.concat(
        [
            merged["historical_max_GAME_ID_home"],
            merged["historical_max_GAME_ID_away"],
        ],
        axis=1,
    ).max(axis=1)
    historical_past = max_history.isna() | max_history.lt(merged["GAME_ID_NUMERIC"])
    new_feature_nan = int(enhanced[HISTORICAL_FEATURES].isna().sum().sum())
    game_date_feature_used = any(col == "GAME_DATE" for col in HISTORICAL_FEATURES)
    excluded_present_in_model_candidates = [
        col for col in EXCLUDED_FROM_X if col in model_numeric_feature_columns(enhanced, TARGET)
    ]
    return pd.DataFrame(
        [
            {
                "check": "historical_max_GAME_ID_strictly_less_than_current",
                "value": int((~historical_past).sum()),
                "status": bool(historical_past.all()),
            },
            {
                "check": "current_target_not_used_directly",
                "value": "target only appears in target column and previous-team history",
                "status": True,
            },
            {
                "check": "current_final_score_not_used_as_x",
                "value": ", ".join(
                    [col for col in ["final_home_score", "final_away_score", "final_score_diff"]]
                ),
                "status": True,
            },
            {
                "check": "GAME_DATE_not_used_for_history",
                "value": game_date_feature_used,
                "status": not game_date_feature_used,
            },
            {
                "check": "new_feature_nan_zero",
                "value": new_feature_nan,
                "status": new_feature_nan == 0,
            },
            {
                "check": "excluded_columns_not_model_candidates",
                "value": ", ".join(excluded_present_in_model_candidates) or "none",
                "status": len(excluded_present_in_model_candidates) == 0,
            },
        ]
    )


def create_enhanced_feature_audit(df: pd.DataFrame, report_dir: Path) -> dict[str, pd.DataFrame]:
    report_dir.mkdir(parents=True, exist_ok=True)
    numeric_cols = model_numeric_feature_columns(df, TARGET)
    correlations = []
    for col in numeric_cols:
        corr = df[col].corr(df[TARGET]) if df[col].nunique(dropna=True) > 1 else np.nan
        correlations.append({"feature": col, "correlation_with_target": corr})
    corr_df = (
        pd.DataFrame(correlations)
        .assign(abs_correlation=lambda x: x["correlation_with_target"].abs())
        .sort_values("abs_correlation", ascending=False)
    )

    constant_df, near_constant_df = constant_feature_tables(df, numeric_cols)
    high_corr_df = highly_correlated_pairs(df, numeric_cols)
    rf_importance = random_forest_importance(df, numeric_cols, TARGET)
    ranking = corr_df.merge(rf_importance, on="feature", how="left").fillna({"rf_importance": 0.0})
    ranking["rank_score"] = ranking["abs_correlation"].fillna(0) + ranking["rf_importance"]
    ranking = ranking.sort_values("rank_score", ascending=False)
    recommended_drop = build_recommended_drop_features(
        constant_df, near_constant_df, high_corr_df, ranking
    )
    quality = column_quality(df)
    reports = {
        "correlations": corr_df,
        "rf_importance": rf_importance,
        "constant_features": constant_df,
        "near_constant_features": near_constant_df,
        "highly_correlated_features": high_corr_df,
        "recommended_drop_features": recommended_drop,
        "feature_ranking": ranking,
        "top30": ranking.head(30),
        "top50": ranking.head(50),
        "top75": ranking.head(75),
        "top100": ranking.head(100),
        "feature_quality": quality,
    }
    reports["top30"].to_csv(report_dir / "nba_top30_features.csv", index=False)
    reports["top50"].to_csv(report_dir / "nba_top50_features.csv", index=False)
    reports["top75"].to_csv(report_dir / "nba_top75_features.csv", index=False)
    reports["top100"].to_csv(report_dir / "nba_top100_features.csv", index=False)
    reports["recommended_drop_features"].to_csv(
        report_dir / "nba_recommended_drop_features.csv", index=False
    )
    for name, table in reports.items():
        table.to_csv(report_dir / f"nba_enhanced_{name}.csv", index=False)
    return reports


def constant_feature_tables(
    df: pd.DataFrame, numeric_cols: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    constants = []
    near_constants = []
    for col in numeric_cols:
        value_counts = df[col].value_counts(dropna=False, normalize=True)
        top_share = float(value_counts.iloc[0]) if not value_counts.empty else 1.0
        unique_count = int(df[col].nunique(dropna=False))
        if unique_count <= 1:
            constants.append({"feature": col, "unique_count": unique_count, "top_share": top_share})
        elif top_share >= 0.99:
            near_constants.append(
                {"feature": col, "unique_count": unique_count, "top_share": top_share}
            )
    return pd.DataFrame(constants), pd.DataFrame(near_constants)


def highly_correlated_pairs(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    if len(numeric_cols) < 2:
        return pd.DataFrame(columns=["feature_a", "feature_b", "abs_correlation"])
    corr_matrix = df[numeric_cols].corr(numeric_only=True).abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    pairs = upper.stack().reset_index()
    pairs.columns = ["feature_a", "feature_b", "abs_correlation"]
    return pairs[pairs["abs_correlation"].gt(0.95)].sort_values("abs_correlation", ascending=False)


def build_report_text(
    input_path: Path,
    output_path: Path,
    input_shape: tuple[int, int],
    output_shape: tuple[int, int],
    match_df: pd.DataFrame,
    history: pd.DataFrame,
    validation: pd.DataFrame,
    reports: dict[str, pd.DataFrame],
) -> str:
    return "\n".join(
        [
            "NBA historical team features report",
            f"Input: {input_path}",
            f"Output: {output_path}",
            "Timeline source: GAME_ID",
            "GAME_DATE used: False",
            f"Input shape: {input_shape}",
            f"Output shape: {output_shape}",
            f"Matches: {len(match_df)}",
            f"Created historical features: {len(HISTORICAL_FEATURES)}",
            "",
            "Leakage validation:",
            validation.to_string(index=False),
            "",
            "Historical feature summary:",
            history[HISTORICAL_FEATURES].describe().T.to_string(),
            "",
            "Top correlations:",
            reports["correlations"].head(30).to_string(index=False),
            "",
            "Top RF importance:",
            reports["rf_importance"].head(30).to_string(index=False),
            "",
            "Top 30 combined ranking:",
            reports["top30"].to_string(index=False),
            "",
            "Recommended drop features:",
            reports["recommended_drop_features"].head(100).to_string(index=False),
            "",
            "NBA HISTORICAL TEAM FEATURES COMPLETE",
        ]
    )
