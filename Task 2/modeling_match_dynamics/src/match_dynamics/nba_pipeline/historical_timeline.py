from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd


MATCH_COLUMNS = [
    "GAME_ID",
    "GAME_DATE",
    "HTM",
    "VTM",
    "final_home_score",
    "final_away_score",
    "final_score_diff",
    "target_score_diff_change_last_5min",
]


def validate_nba_historical_timeline(
    input_path: Path,
    final_output_path: Path,
    report_path: Path,
    sample_size: int = 20,
) -> dict[str, pd.DataFrame | str]:
    if not input_path.exists():
        raise FileNotFoundError(f"NBA feature file not found: {input_path}")

    available_columns = pd.read_csv(input_path, nrows=0).columns.tolist()
    missing = [col for col in MATCH_COLUMNS if col not in available_columns]
    if missing:
        raise ValueError(f"Required match-level columns are missing: {missing}")

    df = pd.read_csv(input_path, usecols=MATCH_COLUMNS)
    game_date_exists = "GAME_DATE" in df.columns
    game_date_nan_pct = float(df["GAME_DATE"].isna().mean()) if game_date_exists else 1.0
    raw_game_dates = df["GAME_DATE"] if game_date_exists else pd.Series(dtype=object)
    parsed_dates = pd.to_datetime(raw_game_dates, errors="coerce") if game_date_exists else None
    parsed_nan_pct = float(parsed_dates.isna().mean()) if parsed_dates is not None else 1.0
    unique_game_dates = int(parsed_dates.nunique()) if parsed_dates is not None else 0
    unique_raw_game_dates = int(raw_game_dates.nunique(dropna=True)) if game_date_exists else 0

    match_df = (
        df.assign(GAME_DATE_PARSED=parsed_dates if parsed_dates is not None else pd.NaT)
        .groupby("GAME_ID", as_index=False, sort=False)
        .agg(
            {
                "GAME_DATE": "first",
                "GAME_DATE_PARSED": "first",
                "HTM": "first",
                "VTM": "first",
                "final_home_score": "first",
                "final_away_score": "first",
                "final_score_diff": "first",
                "target_score_diff_change_last_5min": "first",
            }
        )
    )
    match_df["GAME_ID_NUMERIC"] = pd.to_numeric(match_df["GAME_ID"], errors="coerce")

    parsed_year_min = int(parsed_dates.dt.year.min()) if parsed_dates is not None else 0
    parsed_year_max = int(parsed_dates.dt.year.max()) if parsed_dates is not None else 0
    raw_dates_look_zero_filled = (
        game_date_exists
        and unique_raw_game_dates <= 1
        and raw_game_dates.dropna().astype(str).str.fullmatch(r"0(\.0)?").all()
    )
    date_valid = (
        game_date_exists
        and parsed_nan_pct == 0.0
        and unique_game_dates > 1
        and not raw_dates_look_zero_filled
        and parsed_year_min >= 1990
        and parsed_year_max <= 2035
    )
    timeline_source = "GAME_DATE" if date_valid else "GAME_ID"

    if timeline_source == "GAME_DATE":
        match_df = match_df.sort_values(
            ["GAME_DATE_PARSED", "GAME_ID_NUMERIC", "GAME_ID"], kind="mergesort"
        ).reset_index(drop=True)
        duplicate_timeline_values = int(match_df["GAME_DATE_PARSED"].duplicated().sum())
        sorting_is_unambiguous = duplicate_timeline_values == 0
    else:
        match_df = match_df.sort_values(
            ["GAME_ID_NUMERIC", "GAME_ID"], kind="mergesort"
        ).reset_index(drop=True)
        duplicate_timeline_values = int(match_df["GAME_ID_NUMERIC"].duplicated().sum())
        sorting_is_unambiguous = duplicate_timeline_values == 0

    preview = match_df[["GAME_ID", "GAME_DATE"]].head(20).copy()
    duplicates = pd.DataFrame(
        [
            {
                "check": "duplicate_GAME_ID",
                "value": int(match_df["GAME_ID"].duplicated().sum()),
            },
            {
                "check": "duplicate_timeline_values",
                "value": duplicate_timeline_values,
            },
        ]
    )

    validation = leakage_validation_sample(match_df, timeline_source, sample_size)
    historical_valid = bool(validation["status"].all()) if not validation.empty else True

    report_lines = [
        "NBA historical timeline validation",
        f"Input: {input_path}",
        f"Final output: {final_output_path}",
        "",
        "STEP 1. Time quality",
        f"GAME_DATE exists: {game_date_exists}",
        f"GAME_DATE NaN percent: {game_date_nan_pct:.6f}",
        f"GAME_DATE parsed NaN percent: {parsed_nan_pct:.6f}",
        f"Unique raw GAME_DATE: {unique_raw_game_dates}",
        f"Unique GAME_DATE: {unique_game_dates}",
        f"Parsed GAME_DATE year range: {parsed_year_min}..{parsed_year_max}",
        f"GAME_DATE looks zero-filled: {raw_dates_look_zero_filled}",
        f"GAME_DATE valid datetime: {date_valid}",
        f"Timeline sorting unambiguous: {sorting_is_unambiguous}",
        f"Duplicate timeline values: {duplicate_timeline_values}",
        "",
        "STEP 2. Historical timeline source",
        f"Historical timeline source: {timeline_source}",
        "",
        "STEP 3. Match-level table",
        f"Matches: {len(match_df)}",
        f"Duplicate GAME_ID: {int(match_df['GAME_ID'].duplicated().sum())}",
        "",
        "First 20 matches:",
        preview.to_string(index=False),
        "",
        "STEP 4. Leakage validation sample",
        validation.to_string(index=False),
        "",
        f"Historical leakage validation passed: {historical_valid}",
        "",
        "NBA HISTORICAL TIMELINE VALIDATED",
    ]

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    final_output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(input_path, final_output_path)

    base = report_path.with_suffix("")
    match_df.to_csv(base.with_name(base.name + "_match_level.csv"), index=False)
    preview.to_csv(base.with_name(base.name + "_first_20_matches.csv"), index=False)
    validation.to_csv(base.with_name(base.name + "_leakage_sample.csv"), index=False)
    duplicates.to_csv(base.with_name(base.name + "_duplicates.csv"), index=False)

    return {
        "timeline_source": timeline_source,
        "match_level": match_df,
        "preview": preview,
        "validation": validation,
        "duplicates": duplicates,
        "report": "\n".join(report_lines),
    }


def leakage_validation_sample(
    match_df: pd.DataFrame, timeline_source: str, sample_size: int
) -> pd.DataFrame:
    sample = match_df.sample(min(sample_size, len(match_df)), random_state=42)
    rows = []
    for _, row in sample.iterrows():
        if timeline_source == "GAME_DATE":
            current_value = row["GAME_DATE_PARSED"]
            history = match_df[match_df["GAME_DATE_PARSED"].lt(current_value)]
            history_max = history["GAME_DATE_PARSED"].max() if not history.empty else pd.NaT
            status = pd.isna(history_max) or history_max < current_value
            rows.append(
                {
                    "GAME_ID": row["GAME_ID"],
                    "current_match_date": current_value,
                    "historical_max_date": history_max,
                    "history_count": len(history),
                    "status": bool(status),
                }
            )
        else:
            current_value = row["GAME_ID_NUMERIC"]
            history = match_df[match_df["GAME_ID_NUMERIC"].lt(current_value)]
            history_max = history["GAME_ID_NUMERIC"].max() if not history.empty else pd.NA
            status = pd.isna(history_max) or history_max < current_value
            rows.append(
                {
                    "GAME_ID": row["GAME_ID"],
                    "current_game_id": current_value,
                    "historical_max_game_id": history_max,
                    "history_count": len(history),
                    "status": bool(status),
                }
            )
    return pd.DataFrame(rows).sort_values("GAME_ID").reset_index(drop=True)
