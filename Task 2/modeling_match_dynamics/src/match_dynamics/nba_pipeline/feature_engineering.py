from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


SCORE_COLUMNS = [
    "home_score_current",
    "away_score_current",
    "score_diff_home_current",
    "score_margin_numeric",
]

TEXT_METADATA_COLUMNS = [
    "PLAYER2_ID",
    "PLAYER2_NAME",
    "PLAYER2_TEAM_ID",
    "PLAYER2_TEAM_CITY",
    "PLAYER2_TEAM_NICKNAME",
    "PLAYER2_TEAM_ABBREVIATION",
    "PLAYER3_ID",
    "PLAYER3_NAME",
    "PLAYER3_TEAM_ID",
    "PLAYER3_TEAM_CITY",
    "PLAYER3_TEAM_NICKNAME",
    "PLAYER3_TEAM_ABBREVIATION",
    "HOMEDESCRIPTION",
    "VISITORDESCRIPTION",
    "SCORE",
    "SCOREMARGIN",
]

NUMERIC_METADATA_PREFIXES = ("PLAYER1_", "PLAYER2_", "PLAYER3_")
NUMERIC_METADATA_COLUMNS = {
    "GAME_ID",
    "EVENTNUM",
    "GAME_EVENT_ID",
    "TEAM_ID",
    "PLAYER_ID",
}

LEAKAGE_COLUMNS = [
    "target_score_diff_change_last_5min",
    "final_home_score",
    "final_away_score",
    "final_score_diff",
    "score_diff_at_5min_remaining",
    "cutoff_found",
]

FINAL_PREPROCESSING_EXCLUDE_FROM_MODEL = set(TEXT_METADATA_COLUMNS) | NUMERIC_METADATA_COLUMNS


def column_quality(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        null_count = int(df[col].isna().sum())
        rows.append(
            {
                "column": col,
                "dtype": str(df[col].dtype),
                "non_null_count": int(df[col].notna().sum()),
                "null_count": null_count,
                "null_percent": null_count / len(df) if len(df) else 0,
                "unique_count": int(df[col].nunique(dropna=True)),
            }
        )
    return pd.DataFrame(rows).sort_values("null_percent", ascending=False)


def parse_clock_seconds(value) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if ":" not in text:
        return np.nan
    parts = text.split(":")
    if len(parts) != 2:
        return np.nan
    try:
        return float(parts[0]) * 60 + float(parts[1])
    except ValueError:
        return np.nan


def ensure_nba_preprocessed_in_data_nba(source_path: Path, target_path: Path) -> None:
    if target_path.exists():
        return
    if not source_path.exists():
        raise FileNotFoundError(f"Neither {target_path} nor source fallback {source_path} exists.")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, target_path)


def is_metadata_numeric_column(column: str) -> bool:
    return column in NUMERIC_METADATA_COLUMNS or column.startswith(NUMERIC_METADATA_PREFIXES)


def run_nba_score_final_fix(
    input_path: Path, fallback_path: Path, output_path: Path, report_path: Path
) -> pd.DataFrame:
    ensure_nba_preprocessed_in_data_nba(fallback_path, input_path)
    df = pd.read_csv(input_path)
    input_shape = df.shape
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    game_count_before = int(df["GAME_ID"].nunique()) if "GAME_ID" in df else 0
    row_count_before = len(df)
    score_nan_before = {
        col: int(df[col].isna().sum()) for col in SCORE_COLUMNS if col in df.columns
    }
    numeric_nan_before = (
        df.select_dtypes(include=["number"]).isna().sum().loc[lambda x: x.gt(0)].reset_index()
    )
    numeric_nan_before.columns = ["column", "nan_before"]

    sort_cols = [
        col
        for col in ["GAME_ID", "game_seconds_elapsed", "PERIOD_event", "EVENTNUM"]
        if col in df.columns
    ]
    df = df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    for col in SCORE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df.groupby("GAME_ID")[col].ffill().fillna(0)
    if {"home_score_current", "away_score_current"}.issubset(df.columns):
        df["home_score_current"] = df.groupby("GAME_ID")["home_score_current"].cummax()
        df["away_score_current"] = df.groupby("GAME_ID")["away_score_current"].cummax()
        df["score_diff_home_current"] = df["home_score_current"] - df["away_score_current"]

    numeric_cols = list(df.select_dtypes(include=["number"]).columns)
    filled_columns: list[str] = []
    structural_numeric_nan_columns: list[str] = []
    for col in numeric_cols:
        nan_count = int(df[col].isna().sum())
        if nan_count == 0:
            continue
        if is_metadata_numeric_column(col):
            structural_numeric_nan_columns.append(col)
            continue
        if col in [
            "avg_distance",
            "std_distance",
            "spread_x",
            "spread_y",
            "ball_x",
            "ball_y",
            "ball_hoop_dist",
            "min_player_hoop_dist",
            "players_near_hoop",
            "low_shot_clock",
            "intensity",
        ]:
            if "movement_missing" not in df:
                df["movement_missing"] = df[col].isna().astype("int8")
            median = df[col].median()
            df[col] = df[col].fillna(0 if pd.isna(median) else median)
        else:
            df[col] = df[col].fillna(0)
        filled_columns.append(col)

    score_nan_after = {col: int(df[col].isna().sum()) for col in SCORE_COLUMNS if col in df.columns}
    numeric_nan_after = (
        df.select_dtypes(include=["number"]).isna().sum().loc[lambda x: x.gt(0)].reset_index()
    )
    numeric_nan_after.columns = ["column", "nan_after"]
    model_numeric_nan_after = {
        col: int(df[col].isna().sum())
        for col in df.select_dtypes(include=["number"]).columns
        if not is_metadata_numeric_column(col) and int(df[col].isna().sum()) > 0
    }
    home_decrease = (
        int(df.groupby("GAME_ID")["home_score_current"].diff().lt(0).sum())
        if "home_score_current" in df
        else 0
    )
    away_decrease = (
        int(df.groupby("GAME_ID")["away_score_current"].diff().lt(0).sum())
        if "away_score_current" in df
        else 0
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    report_lines = [
        "NBA preprocessing final fix report",
        f"Input: {input_path}",
        f"Output: {output_path}",
        f"Input shape: {input_shape}",
        f"Output shape: {df.shape}",
        f"Rows before: {row_count_before}",
        f"Rows after: {len(df)}",
        f"GAME_ID count before: {game_count_before}",
        f"GAME_ID count after: {int(df['GAME_ID'].nunique()) if 'GAME_ID' in df else 0}",
        "",
        "Score NaN before:",
        str(score_nan_before),
        "Score NaN after:",
        str(score_nan_after),
        "",
        "Numeric columns with NaN before:",
        numeric_nan_before.to_string(index=False) if not numeric_nan_before.empty else "none",
        "",
        "Numeric columns with NaN after:",
        numeric_nan_after.to_string(index=False) if not numeric_nan_after.empty else "none",
        "",
        "Model numeric NaN after:",
        str(model_numeric_nan_after) if model_numeric_nan_after else "none",
        "",
        "Filled numeric model columns:",
        ", ".join(filled_columns) if filled_columns else "none",
        "",
        "Structural numeric metadata NaN left as metadata:",
        ", ".join(structural_numeric_nan_columns) if structural_numeric_nan_columns else "none",
        "",
        "Monotonic score checks:",
        f"home_score_current decreases: {home_decrease}",
        f"away_score_current decreases: {away_decrease}",
        "",
        "Note: metadata/text structural missing values were intentionally not filled:",
        ", ".join([col for col in TEXT_METADATA_COLUMNS if col in df.columns]),
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    report_prefix = "nba_preprocessing_400_final"
    numeric_nan_before.to_csv(
        report_path.with_name(f"{report_prefix}_numeric_nan_before.csv"), index=False
    )
    numeric_nan_after.to_csv(
        report_path.with_name(f"{report_prefix}_numeric_nan_after.csv"), index=False
    )
    pd.DataFrame({"column": filled_columns}).to_csv(
        report_path.with_name(f"{report_prefix}_filled_numeric_columns.csv"), index=False
    )
    pd.DataFrame(
        [
            {
                "score_feature": col,
                "nan_before": score_nan_before.get(col, pd.NA),
                "nan_after": score_nan_after.get(col, pd.NA),
            }
            for col in SCORE_COLUMNS
        ]
    ).to_csv(report_path.with_name(f"{report_prefix}_fix_score_nan.csv"), index=False)
    pd.DataFrame(
        [
            {"check": "row_count_unchanged", "value": len(df) == row_count_before},
            {
                "check": "game_count_unchanged",
                "value": (int(df["GAME_ID"].nunique()) if "GAME_ID" in df else 0)
                == game_count_before,
            },
            {"check": "home_score_non_decreasing", "value": home_decrease == 0},
            {"check": "away_score_non_decreasing", "value": away_decrease == 0},
            {"check": "score_feature_nan_zero", "value": sum(score_nan_after.values()) == 0},
            {"check": "model_numeric_nan_zero", "value": len(model_numeric_nan_after) == 0},
        ]
    ).to_csv(report_path.with_name(f"{report_prefix}_fix_checks.csv"), index=False)
    return df


def add_game_clock_seconds(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "event_clock_remaining" in out:
        out["game_clock_seconds"] = pd.to_numeric(out["event_clock_remaining"], errors="coerce")
    elif "PCTIMESTRING" in out:
        out["game_clock_seconds"] = out["PCTIMESTRING"].apply(parse_clock_seconds)
    elif "game_clock_start" in out:
        out["game_clock_seconds"] = pd.to_numeric(out["game_clock_start"], errors="coerce")
    else:
        out["game_clock_seconds"] = np.nan
    return out


def infer_points(row: pd.Series) -> float:
    if row.get("is_made_shot", 0) != 1 and row.get("is_made_fg", 0) != 1:
        if row.get("is_free_throw", 0) == 1 and row.get("SCORE") is not np.nan:
            return 1.0
        return 0.0
    shot_type = str(row.get("SHOT_TYPE", "")).upper()
    if "3PT" in shot_type or "3 POINT" in shot_type:
        return 3.0
    return 2.0


def first_non_missing(series: pd.Series, default=np.nan):
    valid = series.dropna()
    return valid.iloc[0] if not valid.empty else default


def build_game_targets(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for game_id, game in df.groupby("GAME_ID", sort=False):
        sort_cols = [
            col
            for col in ["game_seconds_elapsed", "PERIOD_event", "EVENTNUM"]
            if col in game.columns
        ]
        game = game.sort_values(sort_cols, kind="mergesort")
        final_home = float(game["home_score_current"].iloc[-1])
        final_away = float(game["away_score_current"].iloc[-1])
        final_diff = final_home - final_away

        cutoff_rows = game[
            (pd.to_numeric(game["PERIOD_event"], errors="coerce") == 4)
            & (pd.to_numeric(game["game_clock_seconds"], errors="coerce") <= 300)
        ]
        if cutoff_rows.empty:
            before_cutoff = game[
                (pd.to_numeric(game["PERIOD_event"], errors="coerce") < 4)
                | (
                    (pd.to_numeric(game["PERIOD_event"], errors="coerce") == 4)
                    & (pd.to_numeric(game["game_clock_seconds"], errors="coerce") > 300)
                )
            ]
            score_at_cutoff = (
                float(before_cutoff["score_diff_home_current"].iloc[-1])
                if not before_cutoff.empty
                else 0.0
            )
            cutoff_found = 0
        else:
            score_at_cutoff = float(cutoff_rows["score_diff_home_current"].iloc[0])
            cutoff_found = 1

        rows.append(
            {
                "GAME_ID": game_id,
                "final_home_score": final_home,
                "final_away_score": final_away,
                "final_score_diff": final_diff,
                "score_diff_at_5min_remaining": score_at_cutoff,
                "target_score_diff_change_last_5min": final_diff - score_at_cutoff,
                "cutoff_found": cutoff_found,
            }
        )
    return pd.DataFrame(rows)


def add_clutch_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = add_game_clock_seconds(df)
    sort_cols = [
        col for col in ["GAME_ID", "game_seconds_elapsed", "PERIOD_event", "EVENTNUM"] if col in out
    ]
    out = out.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    created: list[str] = ["game_clock_seconds"]

    out["current_score_diff"] = out["score_diff_home_current"]
    out["absolute_score_diff"] = out["current_score_diff"].abs()
    out["home_leading"] = out["current_score_diff"].gt(0).astype("int8")
    out["away_leading"] = out["current_score_diff"].lt(0).astype("int8")
    out["tie_game"] = out["current_score_diff"].eq(0).astype("int8")
    out["game_is_close_3"] = out["absolute_score_diff"].le(3).astype("int8")
    out["game_is_close_5"] = out["absolute_score_diff"].le(5).astype("int8")
    out["game_is_close_10"] = out["absolute_score_diff"].le(10).astype("int8")
    created += [
        "current_score_diff",
        "absolute_score_diff",
        "home_leading",
        "away_leading",
        "tie_game",
        "game_is_close_3",
        "game_is_close_5",
        "game_is_close_10",
    ]

    out["total_score_current"] = out["home_score_current"] + out["away_score_current"]
    out["points_event"] = (
        out.groupby("GAME_ID")["total_score_current"].diff().fillna(out["total_score_current"])
    )
    out["points_event"] = out["points_event"].clip(lower=0)
    out["is_three_point"] = (
        out["SHOT_TYPE"].astype(str).str.upper().str.contains("3PT").astype("int8")
    )
    out["is_two_point"] = ((out["is_shot"].eq(1)) & (out["is_three_point"].eq(0))).astype("int8")
    out["shot_distance"] = pd.to_numeric(out.get("SHOT_DISTANCE", 0), errors="coerce").fillna(0)
    out["shot_distance_bucket"] = pd.cut(
        out["shot_distance"],
        bins=[-0.1, 3, 8, 16, 23.75, np.inf],
        labels=[0, 1, 2, 3, 4],
    ).astype("int8")
    if "SHOT_ZONE_BASIC" in out:
        out["shot_zone"] = out["SHOT_ZONE_BASIC"].fillna("no_shot").astype(str)
    else:
        out["shot_zone"] = "no_shot"
    created += [
        "total_score_current",
        "points_event",
        "is_three_point",
        "is_two_point",
        "shot_distance",
        "shot_distance_bucket",
        "shot_zone",
    ]

    rolling_sum_sources = {
        "made_fg": "is_made_fg",
        "missed_fg": "is_missed_fg",
        "rebounds": "is_rebound",
        "turnovers": "is_turnover",
        "fouls": "is_foul",
        "free_throws": "is_free_throw",
        "shots": "is_shot",
    }
    for window in [5, 10, 20]:
        for feature_name, src_col in rolling_sum_sources.items():
            if src_col in out:
                new_col = f"rolling_{feature_name}_last_{window}"
                out[new_col] = (
                    out.groupby("GAME_ID")[src_col]
                    .rolling(window=window, min_periods=1)
                    .sum()
                    .reset_index(level=0, drop=True)
                )
                created.append(new_col)

        out[f"rolling_points_scored_last_{window}"] = (
            out.groupby("GAME_ID")["points_event"]
            .rolling(window=window, min_periods=1)
            .sum()
            .reset_index(level=0, drop=True)
        )
        out[f"score_diff_change_last_{window}_events"] = (
            out.groupby("GAME_ID")["score_diff_home_current"].diff(window).fillna(0)
        )
        out[f"rolling_made_shots_last_{window}"] = (
            out.groupby("GAME_ID")["is_made_shot"]
            .rolling(window=window, min_periods=1)
            .sum()
            .reset_index(level=0, drop=True)
        )
        out[f"rolling_shot_distance_last_{window}"] = (
            out.groupby("GAME_ID")["shot_distance"]
            .rolling(window=window, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        created += [
            f"rolling_points_scored_last_{window}",
            f"score_diff_change_last_{window}_events",
            f"rolling_made_shots_last_{window}",
            f"rolling_shot_distance_last_{window}",
        ]

    out["positive_momentum"] = out["score_diff_change_last_10_events"].gt(0).astype("int8")
    out["negative_momentum"] = out["score_diff_change_last_10_events"].lt(0).astype("int8")
    created += ["positive_momentum", "negative_momentum"]

    out["spacing"] = pd.to_numeric(out.get("spread_x", 0), errors="coerce").fillna(
        0
    ) + pd.to_numeric(out.get("spread_y", 0), errors="coerce").fillna(0)
    out["ball_pressure"] = pd.to_numeric(
        out.get("min_player_hoop_dist", 0), errors="coerce"
    ).fillna(0) - pd.to_numeric(out.get("ball_hoop_dist", 0), errors="coerce").fillna(0)
    created += ["spacing", "ball_pressure"]
    movement_avg_sources = {
        "movement_intensity": "intensity",
        "spacing": "spacing",
        "ball_pressure": "ball_pressure",
    }
    for window in [5, 10, 20]:
        for feature_name, src_col in movement_avg_sources.items():
            if src_col in out:
                new_col = f"{feature_name}_last_{window}"
                out[new_col] = (
                    out.groupby("GAME_ID")[src_col]
                    .rolling(window=window, min_periods=1)
                    .mean()
                    .reset_index(level=0, drop=True)
                )
                created.append(new_col)

    period = pd.to_numeric(out.get("PERIOD_event", 0), errors="coerce").fillna(0)
    event_clock = pd.to_numeric(out["game_clock_seconds"], errors="coerce").fillna(0)
    out["game_progress_pct"] = (
        pd.to_numeric(out.get("game_seconds_elapsed", 0), errors="coerce").fillna(0) / (48 * 60)
    ).clip(0, 1.25)
    out["quarter_progress_pct"] = ((720 - event_clock.clip(upper=720)) / 720).clip(0, 1)
    out["seconds_remaining_normalized"] = (
        pd.to_numeric(out.get("game_seconds_remaining", 0), errors="coerce").fillna(0) / (48 * 60)
    ).clip(0, 1.25)
    out["early_game"] = period.le(1).astype("int8")
    out["mid_game"] = period.isin([2, 3]).astype("int8")
    out["late_game"] = period.ge(4).astype("int8")
    out["clutch_time"] = (period.eq(4) & event_clock.le(300)).astype("int8")
    created += [
        "game_progress_pct",
        "quarter_progress_pct",
        "seconds_remaining_normalized",
        "early_game",
        "mid_game",
        "late_game",
        "clutch_time",
    ]

    for col in created:
        if col in out and pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].replace([np.inf, -np.inf], np.nan).fillna(0)
    return out, created


def create_feature_engineering_reports(
    df: pd.DataFrame, created_features: list[str], report_dir: Path
) -> dict[str, pd.DataFrame]:
    report_dir.mkdir(parents=True, exist_ok=True)
    target = "target_score_diff_change_last_5min"

    numeric_cols = model_numeric_feature_columns(df, target)
    correlations = []
    for col in numeric_cols:
        corr = df[col].corr(df[target]) if df[col].nunique(dropna=True) > 1 else np.nan
        correlations.append({"feature": col, "correlation_with_target": corr})
    corr_df = (
        pd.DataFrame(correlations)
        .assign(abs_correlation=lambda x: x["correlation_with_target"].abs())
        .sort_values("abs_correlation", ascending=False)
    )

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
    constant_df = pd.DataFrame(constants)
    near_constant_df = pd.DataFrame(near_constants)

    high_corr_rows = []
    if len(numeric_cols) >= 2:
        corr_matrix = df[numeric_cols].corr(numeric_only=True).abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        pairs = upper.stack().reset_index()
        pairs.columns = ["feature_a", "feature_b", "abs_correlation"]
        high_corr_rows = pairs[pairs["abs_correlation"].gt(0.95)].sort_values(
            "abs_correlation", ascending=False
        )
    high_corr_df = (
        high_corr_rows
        if isinstance(high_corr_rows, pd.DataFrame)
        else pd.DataFrame(columns=["feature_a", "feature_b", "abs_correlation"])
    )

    rf_importance = random_forest_importance(df, numeric_cols, target)
    ranking = corr_df.merge(rf_importance, on="feature", how="left").fillna({"rf_importance": 0.0})
    ranking["rank_score"] = ranking["abs_correlation"].fillna(0) + ranking["rf_importance"]
    ranking = ranking.sort_values("rank_score", ascending=False)

    recommended_drop = build_recommended_drop_features(
        constant_df, near_constant_df, high_corr_df, ranking
    )

    target_distribution = df.groupby("GAME_ID")[target].first().describe().reset_index()
    target_distribution.columns = ["metric", "value"]
    target_by_game = df.groupby("GAME_ID")[target].first().reset_index()
    rows_per_game = df.groupby("GAME_ID").size().describe().reset_index()
    rows_per_game.columns = ["metric", "value"]
    seq_lengths = (
        df.groupby("GAME_ID").size().rename("events_before_cutoff").describe().reset_index()
    )
    seq_lengths.columns = ["metric", "value"]
    quality = column_quality(df)

    reports = {
        "created_features": pd.DataFrame({"feature": created_features}),
        "target_distribution": target_distribution,
        "target_by_game": target_by_game,
        "correlations": corr_df,
        "rf_importance": rf_importance,
        "feature_ranking": ranking,
        "top_30_features": ranking.head(30),
        "top_50_features": ranking.head(50),
        "top_75_features": ranking.head(75),
        "constant_features": constant_df,
        "near_constant_features": near_constant_df,
        "highly_correlated_features": high_corr_df,
        "recommended_drop_features": recommended_drop,
        "rows_per_game": rows_per_game,
        "sequence_statistics": seq_lengths,
        "top_missing_columns": quality.head(40),
        "feature_quality": quality,
        "head": df.head(50),
    }
    for name, table in reports.items():
        table.to_csv(report_dir / f"nba_feature_engineering_{name}.csv", index=False)
    return reports


def model_numeric_feature_columns(df: pd.DataFrame, target: str) -> list[str]:
    metadata_markers = ["DATE", "PLAYER", "TEAM", "DESCRIPTION", "SCOREMARGIN", "SCORE"]
    numeric_cols = []
    for col in df.columns:
        upper = col.upper()
        if col == target or col in LEAKAGE_COLUMNS:
            continue
        if is_metadata_numeric_column(col) or any(marker in upper for marker in metadata_markers):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
    return numeric_cols


def random_forest_importance(
    df: pd.DataFrame, numeric_cols: list[str], target: str
) -> pd.DataFrame:
    if not numeric_cols or target not in df:
        return pd.DataFrame(columns=["feature", "rf_importance"])
    sample = df[numeric_cols + [target]].replace([np.inf, -np.inf], np.nan).dropna()
    if sample.empty:
        return pd.DataFrame(columns=["feature", "rf_importance"])
    if len(sample) > 100_000:
        sample = sample.sample(100_000, random_state=42)
    x = sample[numeric_cols]
    y = sample[target]
    rf = RandomForestRegressor(
        n_estimators=120,
        max_depth=12,
        min_samples_leaf=20,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(x, y)
    return pd.DataFrame(
        {"feature": numeric_cols, "rf_importance": rf.feature_importances_}
    ).sort_values("rf_importance", ascending=False)


def build_recommended_drop_features(
    constant_df: pd.DataFrame,
    near_constant_df: pd.DataFrame,
    high_corr_df: pd.DataFrame,
    ranking: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for _, row in constant_df.iterrows():
        rows.append({"feature": row["feature"], "reason": "constant"})
    for _, row in near_constant_df.iterrows():
        rows.append({"feature": row["feature"], "reason": "near_constant_top_share_ge_0.99"})
    rank_map = {row.feature: i for i, row in enumerate(ranking.itertuples(index=False))}
    for _, row in high_corr_df.iterrows():
        feature_a = row["feature_a"]
        feature_b = row["feature_b"]
        drop_feature = (
            feature_a
            if rank_map.get(feature_a, 10**9) > rank_map.get(feature_b, 10**9)
            else feature_b
        )
        rows.append(
            {
                "feature": drop_feature,
                "reason": f"highly_correlated_with_{feature_b if drop_feature == feature_a else feature_a}",
            }
        )
    if not rows:
        return pd.DataFrame(columns=["feature", "reason"])
    return pd.DataFrame(rows).drop_duplicates().sort_values(["reason", "feature"])


def run_nba_feature_engineering(
    input_path: Path,
    fallback_input_path: Path,
    final_preprocessing_path: Path,
    output_path: Path,
    fix_report_path: Path,
    feature_report_dir: Path,
) -> dict[str, pd.DataFrame]:
    fixed = run_nba_score_final_fix(
        input_path, fallback_input_path, final_preprocessing_path, fix_report_path
    )
    engineered, created = add_clutch_features(fixed)
    targets = build_game_targets(engineered)
    engineered = engineered.merge(targets, on="GAME_ID", how="left")

    period = pd.to_numeric(engineered["PERIOD_event"], errors="coerce")
    clock = pd.to_numeric(engineered["game_clock_seconds"], errors="coerce")
    before_cutoff_mask = period.lt(4) | (period.eq(4) & clock.gt(300))
    model_rows = engineered.loc[before_cutoff_mask].copy()
    model_rows = model_rows.sort_values(
        ["GAME_ID", "game_seconds_elapsed", "PERIOD_event", "EVENTNUM"], kind="mergesort"
    ).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_rows.to_csv(output_path, index=False)
    reports = create_feature_engineering_reports(model_rows, created, feature_report_dir)

    summary_lines = [
        "NBA feature engineering report",
        f"Input: {input_path}",
        f"Output: {output_path}",
        f"Rows before cutoff: {len(model_rows)}",
        f"Games: {model_rows['GAME_ID'].nunique()}",
        f"Created features: {len(created)}",
        f"Cutoff found games: {int(targets['cutoff_found'].sum())}/{len(targets)}",
        "",
        "Target distribution:",
        reports["target_distribution"].to_string(index=False),
        "",
        "Top correlations:",
        reports["correlations"].head(20).to_string(index=False),
        "",
        "Top RF importance:",
        reports["rf_importance"].head(20).to_string(index=False),
        "",
        "Recommended drop features:",
        reports["recommended_drop_features"].head(50).to_string(index=False),
    ]
    (feature_report_dir / "nba_feature_engineering_report.txt").write_text(
        "\n".join(summary_lines), encoding="utf-8"
    )
    audit_report = [
        "NBA feature audit 400",
        f"Input final preprocessed: {final_preprocessing_path}",
        f"Output features: {output_path}",
        f"Rows before cutoff: {len(model_rows)}",
        f"Games: {model_rows['GAME_ID'].nunique()}",
        "",
        "Target distribution:",
        reports["target_distribution"].to_string(index=False),
        "",
        "Top-30 features:",
        reports["top_30_features"].to_string(index=False),
        "",
        "Constant features:",
        reports["constant_features"].to_string(index=False),
        "",
        "Near-constant features:",
        reports["near_constant_features"].to_string(index=False),
        "",
        "Highly correlated feature pairs:",
        reports["highly_correlated_features"].head(100).to_string(index=False),
        "",
        "Recommended drop features:",
        reports["recommended_drop_features"].head(100).to_string(index=False),
    ]
    (feature_report_dir.parent / "nba_feature_audit_400.txt").write_text(
        "\n".join(audit_report), encoding="utf-8"
    )
    return reports
