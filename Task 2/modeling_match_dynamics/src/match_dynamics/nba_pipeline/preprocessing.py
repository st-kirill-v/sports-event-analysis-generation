from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


DROP_COLUMNS = ["Unnamed: 0", "NEUTRALDESCRIPTION", "source_file"]

METADATA_COLUMNS = [
    "GAME_ID",
    "GAME_DATE",
    "HTM",
    "VTM",
    "HOMEDESCRIPTION",
    "VISITORDESCRIPTION",
    "PLAYER1_NAME",
    "PLAYER2_NAME",
    "PLAYER3_NAME",
    "TEAM_NAME",
    "WCTIMESTRING",
    "PCTIMESTRING",
]

MODEL_EXCLUDED_IDENTIFIERS = [
    "EVENTNUM",
    "GAME_EVENT_ID",
]

MOVEMENT_COLUMNS = [
    "movement_moments_total",
    "movement_moments_sampled",
    "game_clock_start",
    "game_clock_end",
    "shot_clock_start",
    "shot_clock_end",
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
]

SHOT_NUMERIC_COLUMNS = [
    "LOC_X",
    "LOC_Y",
    "SHOT_DISTANCE",
    "SHOT_MADE_FLAG",
    "SHOT_ATTEMPTED_FLAG",
    "MINUTES_REMAINING",
    "SECONDS_REMAINING",
]

SHOT_CATEGORICAL_COLUMNS = [
    "SHOT_TYPE",
    "SHOT_ZONE_AREA",
    "SHOT_ZONE_BASIC",
    "SHOT_ZONE_RANGE",
]

EVENT_FLAG_MAP = {
    "is_made_fg": 1,
    "is_missed_fg": 2,
    "is_free_throw": 3,
    "is_rebound": 4,
    "is_turnover": 5,
    "is_foul": 6,
    "is_violation": 7,
    "is_substitution": 8,
    "is_timeout": 9,
    "is_jump_ball": 10,
    "is_ejection": 11,
    "is_period_start": 12,
    "is_period_end": 13,
}

MODEL_READY_NUMERIC_COLUMNS = [
    "EVENTMSGTYPE",
    "EVENTMSGACTIONTYPE",
    "PERIOD_event",
    "home_score_current",
    "away_score_current",
    "score_diff_home_current",
    "score_margin_numeric",
    "is_shot",
    "is_made_shot",
    "is_missed_shot",
    *EVENT_FLAG_MAP.keys(),
    *SHOT_NUMERIC_COLUMNS,
    *MOVEMENT_COLUMNS,
    "movement_missing",
    "event_clock_remaining",
    "game_seconds_remaining",
    "game_seconds_elapsed",
    "is_fourth_quarter",
    "is_clutch_time",
]


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


def rows_per_game_stats(df: pd.DataFrame) -> pd.DataFrame:
    counts = df.groupby("GAME_ID").size()
    stats = counts.describe()
    return pd.DataFrame([{"metric": key, "value": value} for key, value in stats.items()])


def duplicate_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if {"GAME_ID", "EVENTNUM"}.issubset(df.columns):
        rows.append(
            {
                "key": "GAME_ID + EVENTNUM",
                "duplicate_rows": int(df.duplicated(["GAME_ID", "EVENTNUM"]).sum()),
                "rows_checked": len(df),
            }
        )
    if {"GAME_ID", "GAME_EVENT_ID"}.issubset(df.columns):
        subset = df[df["GAME_EVENT_ID"].notna()]
        rows.append(
            {
                "key": "GAME_ID + GAME_EVENT_ID",
                "duplicate_rows": int(subset.duplicated(["GAME_ID", "GAME_EVENT_ID"]).sum()),
                "rows_checked": len(subset),
            }
        )
    return pd.DataFrame(rows)


def parse_score(score) -> tuple[float, float]:
    if pd.isna(score):
        return np.nan, np.nan
    parts = str(score).replace(" ", "").split("-")
    if len(parts) != 2:
        return np.nan, np.nan
    try:
        # NBA play-by-play SCORE is visitor-away first, home second.
        away_score = float(parts[0])
        home_score = float(parts[1])
        return home_score, away_score
    except ValueError:
        return np.nan, np.nan


def parse_pctimestring(value) -> float:
    if pd.isna(value):
        return np.nan
    parts = str(value).split(":")
    if len(parts) != 2:
        return np.nan
    try:
        return float(parts[0]) * 60 + float(parts[1])
    except ValueError:
        return np.nan


def coerce_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "GAME_ID" in out:
        out["GAME_ID"] = out["GAME_ID"].astype("Int64").astype(str).str.zfill(10)
    for col in ["EVENTNUM", "EVENTMSGTYPE", "EVENTMSGACTIONTYPE", "PERIOD_event", "period"]:
        if col in out:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")
    for col in MOVEMENT_COLUMNS + SHOT_NUMERIC_COLUMNS:
        if col in out:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def add_score_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "SCORE" in out:
        parsed = out["SCORE"].apply(parse_score)
        out["home_score_current_raw"] = [score[0] for score in parsed]
        out["away_score_current_raw"] = [score[1] for score in parsed]
        if "EVENTMSGTYPE" in out:
            scoring_event = pd.to_numeric(out["EVENTMSGTYPE"], errors="coerce").isin([1, 3])
            out.loc[~scoring_event, ["home_score_current_raw", "away_score_current_raw"]] = np.nan
        sort_cols = (
            ["GAME_ID", "game_seconds_elapsed", "EVENTNUM"]
            if "game_seconds_elapsed" in out
            else ["GAME_ID", "EVENTNUM"]
        )
        out = out.sort_values(sort_cols, kind="mergesort")
        out["home_score_current"] = (
            out.groupby("GAME_ID")["home_score_current_raw"].ffill().fillna(0)
        )
        out["away_score_current"] = (
            out.groupby("GAME_ID")["away_score_current_raw"].ffill().fillna(0)
        )
        # Rare same-clock play-by-play rows can be ordered inconsistently. A score cannot
        # decrease, so keep the causal maximum reached so far inside each game.
        out["home_score_current"] = out.groupby("GAME_ID")["home_score_current"].cummax()
        out["away_score_current"] = out.groupby("GAME_ID")["away_score_current"].cummax()
        out["score_diff_home_current"] = out["home_score_current"] - out["away_score_current"]
        out = out.drop(columns=["home_score_current_raw", "away_score_current_raw"])
    if "SCOREMARGIN" in out:
        out["score_margin_numeric"] = (
            out["SCOREMARGIN"].replace({"TIE": 0}).pipe(pd.to_numeric, errors="coerce")
        )
        out["score_margin_numeric"] = (
            out.groupby("GAME_ID")["score_margin_numeric"].ffill().fillna(0)
        )
    for col in [
        "home_score_current",
        "away_score_current",
        "score_diff_home_current",
        "score_margin_numeric",
    ]:
        if col not in out:
            out[col] = 0
    return out


def add_shot_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    game_event_exists = out["GAME_EVENT_ID"].notna() if "GAME_EVENT_ID" in out else False
    shot_attempt = out["SHOT_ATTEMPTED_FLAG"].eq(1) if "SHOT_ATTEMPTED_FLAG" in out else False
    out["is_shot"] = (shot_attempt | game_event_exists).astype("int8")
    out["is_made_shot"] = (
        out["SHOT_MADE_FLAG"].fillna(0).eq(1) if "SHOT_MADE_FLAG" in out else False
    ).astype("int8")
    out["is_missed_shot"] = (
        out["is_shot"].eq(1) & out["SHOT_MADE_FLAG"].fillna(0).eq(0)
        if "SHOT_MADE_FLAG" in out
        else False
    ).astype("int8")
    for col in SHOT_NUMERIC_COLUMNS:
        if col in out:
            out[col] = out[col].fillna(0)
    for col in SHOT_CATEGORICAL_COLUMNS:
        if col in out:
            out[col] = out[col].fillna("no_shot").astype(str)
    return out


def add_event_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    event_type = (
        pd.to_numeric(out["EVENTMSGTYPE"], errors="coerce")
        if "EVENTMSGTYPE" in out
        else pd.Series()
    )
    for feature, value in EVENT_FLAG_MAP.items():
        out[feature] = event_type.eq(value).astype("int8")
    return out


def add_movement_missing_and_fill(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    present_cols = [col for col in MOVEMENT_COLUMNS if col in out.columns]
    if present_cols:
        out["movement_missing"] = out[present_cols].isna().any(axis=1).astype("int8")
        for col in present_cols:
            median = out[col].median()
            out[col] = out[col].fillna(0 if pd.isna(median) else median)
    else:
        out["movement_missing"] = 1
    if "low_shot_clock" in out:
        out["low_shot_clock"] = out["low_shot_clock"].fillna(0).clip(0, 1).astype("int8")
    return out


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "PERIOD_event" in out:
        out["PERIOD_event"] = pd.to_numeric(out["PERIOD_event"], errors="coerce")
    if "period" in out:
        out["period"] = pd.to_numeric(out["period"], errors="coerce")

    for col in [
        "game_clock_start",
        "game_clock_end",
        "shot_clock_start",
        "shot_clock_end",
        "MINUTES_REMAINING",
        "SECONDS_REMAINING",
    ]:
        if col in out:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    parsed_clock = (
        out["PCTIMESTRING"].apply(parse_pctimestring)
        if "PCTIMESTRING" in out
        else pd.Series(np.nan, index=out.index)
    )
    if "game_clock_start" not in out:
        out["game_clock_start"] = parsed_clock
    else:
        out["game_clock_start"] = out["game_clock_start"].fillna(parsed_clock)
    if "game_clock_end" not in out:
        out["game_clock_end"] = out["game_clock_start"]
    else:
        out["game_clock_end"] = out["game_clock_end"].fillna(out["game_clock_start"])

    period = pd.to_numeric(
        out.get("PERIOD_event", pd.Series(np.nan, index=out.index)), errors="coerce"
    )
    movement_clock = pd.to_numeric(out["game_clock_start"], errors="coerce")
    # PCTIMESTRING is the authoritative play-by-play event clock. Movement summaries can
    # be slightly offset, so they are kept as movement features but not used for event ordering.
    game_clock = parsed_clock.combine_first(movement_clock).fillna(0)
    out["event_clock_remaining"] = game_clock
    out["game_seconds_remaining"] = np.where(
        period.le(4),
        (4 - period.clip(lower=1, upper=4)) * 720 + game_clock,
        game_clock,
    )
    out["game_seconds_elapsed"] = np.where(
        period.le(4),
        (period.clip(lower=1, upper=4) - 1) * 720 + (720 - game_clock),
        4 * 720 + (period - 5).clip(lower=0) * 300 + (300 - game_clock.clip(upper=300)),
    )
    out["is_fourth_quarter"] = period.eq(4).astype("int8")
    out["is_clutch_time"] = (period.eq(4) & game_clock.le(300)).astype("int8")

    for col in ["game_seconds_remaining", "game_seconds_elapsed"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)
    return out


def score_monotonicity_checks(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in ["home_score_current", "away_score_current"]:
        if col not in df:
            continue
        diffs = df.groupby("GAME_ID")[col].diff()
        bad = int(diffs.lt(0).sum())
        rows.append(
            {"check": f"{col}_non_decreasing_by_game", "violations": bad, "status": bad == 0}
        )
    return pd.DataFrame(rows)


def quality_checks(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    duplicate_keys = (
        int(df.duplicated(["GAME_ID", "EVENTNUM"]).sum())
        if {"GAME_ID", "EVENTNUM"}.issubset(df.columns)
        else pd.NA
    )
    rows.append(
        {
            "check": "duplicate_GAME_ID_EVENTNUM",
            "value": duplicate_keys,
            "status": duplicate_keys == 0,
        }
    )
    if {"GAME_ID", "GAME_EVENT_ID"}.issubset(df.columns):
        subset = df[df["GAME_EVENT_ID"].notna()]
        duplicate_shot_keys = int(subset.duplicated(["GAME_ID", "GAME_EVENT_ID"]).sum())
        rows.append(
            {
                "check": "duplicate_GAME_ID_GAME_EVENT_ID_non_null",
                "value": duplicate_shot_keys,
                "status": duplicate_shot_keys == 0,
            }
        )
    if "game_clock_start" in df:
        bad = int(df["game_clock_start"].lt(0).sum())
        rows.append({"check": "negative_game_clock_start", "value": bad, "status": bad == 0})
    if "PERIOD_event" in df:
        bad = int((~df["PERIOD_event"].dropna().between(1, 10)).sum())
        rows.append({"check": "PERIOD_event_valid_range", "value": bad, "status": bad == 0})
    binary_cols = [col for col in df.columns if col.startswith("is_")] + ["movement_missing"]
    if "low_shot_clock" in df:
        binary_cols.append("low_shot_clock")
    for col in sorted(set(binary_cols)):
        values = set(df[col].dropna().unique().tolist())
        rows.append(
            {
                "check": f"{col}_binary_0_1",
                "value": str(sorted(values)),
                "status": values.issubset({0, 1}),
            }
        )
    model_numeric = [
        col
        for col in MODEL_READY_NUMERIC_COLUMNS
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col])
    ]
    nan_numeric = int(df[model_numeric].isna().sum().sum()) if model_numeric else 0
    rows.append(
        {
            "check": "nan_in_numeric_model_ready_columns",
            "value": nan_numeric,
            "status": nan_numeric == 0,
        }
    )
    inf_numeric = int(np.isinf(df[model_numeric]).sum().sum()) if model_numeric else 0
    rows.append(
        {
            "check": "infinite_in_numeric_model_ready_columns",
            "value": inf_numeric,
            "status": inf_numeric == 0,
        }
    )
    return pd.DataFrame(rows)


def run_nba_preprocessing(
    input_path: Path,
    fallback_input_path: Path,
    output_path: Path,
    report_dir: Path,
) -> dict[str, pd.DataFrame]:
    source_path = input_path if input_path.exists() else fallback_input_path
    df = pd.read_csv(source_path)
    before_shape = df.shape
    before_quality = column_quality(df)
    before_rows_per_game = rows_per_game_stats(df)
    before_duplicates = duplicate_diagnostics(df)

    dropped = [col for col in DROP_COLUMNS if col in df.columns]
    out = df.drop(columns=dropped)
    out = coerce_columns(out)
    out = add_time_features(out)
    out = add_score_features(out)
    out = add_shot_features(out)
    out = add_event_flags(out)
    out = add_movement_missing_and_fill(out)
    out = add_time_features(out)
    out = out.sort_values(
        ["GAME_ID", "game_seconds_elapsed", "EVENTNUM"], kind="mergesort"
    ).reset_index(drop=True)

    created_features = [
        "home_score_current",
        "away_score_current",
        "score_diff_home_current",
        "score_margin_numeric",
        "is_shot",
        "is_made_shot",
        "is_missed_shot",
        *EVENT_FLAG_MAP.keys(),
        "movement_missing",
        "event_clock_remaining",
        "game_seconds_remaining",
        "game_seconds_elapsed",
        "is_fourth_quarter",
        "is_clutch_time",
    ]
    created_features = [col for col in created_features if col in out.columns]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)

    after_quality = column_quality(out)
    checks = quality_checks(out)
    score_checks = score_monotonicity_checks(out)
    after_duplicates = duplicate_diagnostics(out)
    rows_per_game_after = rows_per_game_stats(out)
    summary = pd.DataFrame(
        [
            {"metric": "input_path", "value": str(source_path)},
            {"metric": "output_path", "value": str(output_path)},
            {"metric": "input_shape", "value": str(before_shape)},
            {"metric": "output_shape", "value": str(out.shape)},
            {"metric": "unique_GAME_ID", "value": int(out["GAME_ID"].nunique())},
            {"metric": "rows_per_game_min", "value": int(out.groupby("GAME_ID").size().min())},
            {"metric": "rows_per_game_max", "value": int(out.groupby("GAME_ID").size().max())},
            {"metric": "rows_per_game_mean", "value": float(out.groupby("GAME_ID").size().mean())},
            {"metric": "created_features", "value": len(created_features)},
            {"metric": "dropped_columns", "value": len(dropped)},
        ]
    )
    role_rows = []
    for col in out.columns:
        if col in METADATA_COLUMNS:
            role = "metadata_not_model_feature"
        elif col in MODEL_EXCLUDED_IDENTIFIERS:
            role = "identifier_not_model_feature"
        elif col in MODEL_READY_NUMERIC_COLUMNS and pd.api.types.is_numeric_dtype(out[col]):
            role = "model_numeric_candidate"
        elif pd.api.types.is_numeric_dtype(out[col]):
            role = "numeric_review_before_model"
        else:
            role = "object_or_text_not_model_feature"
        role_rows.append({"column": col, "role": role, "dtype": str(out[col].dtype)})

    reports = {
        "summary": summary,
        "before_quality": before_quality,
        "after_quality": after_quality,
        "top_missing_before": before_quality.head(40),
        "top_missing_after": after_quality.head(40),
        "rows_per_game_stats": before_rows_per_game,
        "rows_per_game_stats_after": rows_per_game_after,
        "duplicate_diagnostics_before": before_duplicates,
        "duplicate_diagnostics_after": after_duplicates,
        "score_monotonicity_checks": score_checks,
        "created_features": pd.DataFrame({"feature": created_features}),
        "dropped_columns": pd.DataFrame({"column": dropped}),
        "metadata_columns": pd.DataFrame({"column": [c for c in METADATA_COLUMNS if c in out]}),
        "column_roles": pd.DataFrame(role_rows),
        "quality_checks": checks,
        "head": out.head(50),
    }
    for name, table in reports.items():
        table.to_csv(report_dir / f"nba_preprocessing_{name}.csv", index=False)

    report_text = [
        "NBA 400 preprocessing report",
        f"Input: {source_path}",
        f"Output: {output_path}",
        f"Input shape: {before_shape}",
        f"Output shape: {out.shape}",
        f"Unique GAME_ID: {out['GAME_ID'].nunique()}",
        f"Dropped columns: {', '.join(dropped) if dropped else 'none'}",
        f"Created features: {', '.join(created_features)}",
        "",
        "Duplicate diagnostics before:",
        before_duplicates.to_string(index=False),
        "",
        "Duplicate diagnostics after:",
        after_duplicates.to_string(index=False),
        "",
        "Score monotonicity:",
        score_checks.to_string(index=False),
        "",
        "Quality checks:",
        checks.to_string(index=False),
    ]
    (report_dir / "nba_preprocessing_report.txt").write_text(
        "\n".join(report_text), encoding="utf-8"
    )
    (report_dir.parent / "nba_preprocessing_400_report.txt").write_text(
        "\n".join(report_text), encoding="utf-8"
    )
    return reports
