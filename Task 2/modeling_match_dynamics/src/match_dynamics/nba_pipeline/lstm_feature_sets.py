from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Masking
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


TARGET_COLUMN = "target_score_diff_change_last_5min"

EXCLUDE_COLUMNS = {
    "final_home_score",
    "final_away_score",
    "final_score_diff",
    "score_diff_at_5min_remaining",
    TARGET_COLUMN,
    "cutoff_found",
    "GAME_ID",
    "GAME_DATE",
    "EVENTNUM",
    "GAME_EVENT_ID",
    "PLAYER1_NAME",
    "PLAYER2_NAME",
    "PLAYER3_NAME",
    "HTM",
    "VTM",
    "TEAM_NAME",
    "HOMEDESCRIPTION",
    "VISITORDESCRIPTION",
    "SCORE",
    "SCOREMARGIN",
    "PCTIMESTRING",
    "WCTIMESTRING",
}

FEATURE_SET_FILES = {
    "top30": "nba_top30_features.csv",
    "top50": "nba_top50_features.csv",
    "top75": "nba_top75_features.csv",
}

FULL_TRAINING_COMMANDS = [
    r"uv run python scripts\nba\train_nba_lstm_feature_sets.py --feature-set top30",
    r"uv run python scripts\nba\train_nba_lstm_feature_sets.py --feature-set top50",
    r"uv run python scripts\nba\train_nba_lstm_feature_sets.py --feature-set top75",
    r"uv run python scripts\nba\train_nba_lstm_feature_sets.py --feature-set all_features",
    r"uv run python scripts\nba\train_nba_lstm_feature_sets.py --feature-set all",
]


def run_nba_lstm_feature_set_pipeline(
    input_path: Path,
    reports_dir: Path,
    sequences_dir: Path,
    scalers_dir: Path,
    metrics_dir: Path,
    models_dir: Path,
    feature_set: str,
    smoke_test: bool = False,
) -> dict[str, pd.DataFrame]:
    df = pd.read_csv(input_path)
    df["GAME_ID"] = df["GAME_ID"].astype(str)
    selected_sets = (
        ["top30", "top50", "top75", "all_features"] if feature_set == "all" else [feature_set]
    )
    invalid = [
        name for name in selected_sets if name not in {"top30", "top50", "top75", "all_features"}
    ]
    if invalid:
        raise ValueError(f"Unknown feature set(s): {invalid}")

    all_features = select_all_numeric_features(df)
    validation = validate_feature_matrix(df, all_features)
    game_ids = sorted_game_ids(df)
    splits = split_games(game_ids)
    split_summary = build_split_summary(splits)

    sequence_reports = []
    feature_set_rows = []
    feature_warnings = []
    baselines = []
    metrics_dir.mkdir(parents=True, exist_ok=True)
    sequences_dir.mkdir(parents=True, exist_ok=True)
    scalers_dir.mkdir(parents=True, exist_ok=True)

    prepared = {}
    for name in selected_sets:
        features, warnings = feature_set_columns(name, all_features, reports_dir)
        feature_warnings.extend(warnings)
        arrays, scaler, shapes, baseline_metrics = prepare_feature_set_sequences(
            feature_set=name,
            df=df,
            splits=splits,
            features=features,
        )
        save_feature_set_sequences(name, arrays, features, sequences_dir)
        with (scalers_dir / f"nba_{name}_scaler.pkl").open("wb") as f:
            pickle.dump(scaler, f)
        prepared[name] = {"arrays": arrays, "features": features, "shapes": shapes}
        sequence_reports.extend(shapes)
        baselines.extend(baseline_metrics)
        feature_set_rows.append(
            {
                "feature_set": name,
                "features": len(features),
                "missing_requested_features": len(warnings),
                "max_sequence_length": shapes[0]["max_sequence_length"],
            }
        )

    shapes_df = pd.DataFrame(sequence_reports)
    feature_sets_df = pd.DataFrame(feature_set_rows)
    warnings_df = pd.DataFrame(feature_warnings, columns=["feature_set", "missing_feature"])
    baselines_df = pd.DataFrame(baselines)

    shapes_df.to_csv(metrics_dir / "feature_set_sequence_shapes.csv", index=False)
    split_summary.to_csv(metrics_dir / "feature_set_split_summary.csv", index=False)
    feature_sets_df.to_csv(metrics_dir / "feature_sets_summary.csv", index=False)
    validation.to_csv(metrics_dir / "feature_set_leakage_validation.csv", index=False)
    warnings_df.to_csv(metrics_dir / "feature_set_missing_requested_features.csv", index=False)
    baselines_df.to_csv(metrics_dir / "feature_set_baselines.csv", index=False)
    pd.DataFrame({"command": FULL_TRAINING_COMMANDS}).to_csv(
        metrics_dir / "feature_set_full_training_commands.csv", index=False
    )

    result = {
        "shapes": shapes_df,
        "split_summary": split_summary,
        "feature_sets": feature_sets_df,
        "validation": validation,
        "warnings": warnings_df,
        "baselines": baselines_df,
    }
    if smoke_test:
        if "top50" not in prepared:
            raise ValueError("Smoke test requires feature_set='top50' or 'all'.")
        smoke = train_feature_set(
            feature_set="top50",
            arrays=prepared["top50"]["arrays"],
            features=prepared["top50"]["features"],
            models_dir=models_dir,
            metrics_dir=metrics_dir,
            epochs=1,
            use_callbacks=False,
            output_prefix="smoke_test",
        )
        result.update(smoke)
    else:
        for name in selected_sets:
            train_feature_set(
                feature_set=name,
                arrays=prepared[name]["arrays"],
                features=prepared[name]["features"],
                models_dir=models_dir,
                metrics_dir=metrics_dir,
                epochs=50,
                use_callbacks=True,
                output_prefix=f"nba_lstm_{name}",
            )
    return result


def select_all_numeric_features(df: pd.DataFrame) -> list[str]:
    features = []
    for col in df.columns:
        if col in EXCLUDE_COLUMNS:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        features.append(col)
    return features


def validate_feature_matrix(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    leakage_in_x = sorted(set(features).intersection(EXCLUDE_COLUMNS))
    object_in_x = [col for col in features if not pd.api.types.is_numeric_dtype(df[col])]
    raw_x = df[features].replace([np.inf, -np.inf], np.nan)
    x = raw_x.fillna(0)
    return pd.DataFrame(
        [
            {
                "check": "leakage_columns_in_X_empty",
                "value": ", ".join(leakage_in_x) or "empty",
                "status": len(leakage_in_x) == 0,
            },
            {
                "check": "object_columns_in_X_empty",
                "value": ", ".join(object_in_x) or "empty",
                "status": len(object_in_x) == 0,
            },
            {
                "check": "nan_in_X_zero",
                "value": int(x.isna().sum().sum()),
                "status": int(x.isna().sum().sum()) == 0,
            },
            {
                "check": "infinite_values_in_X_zero",
                "value": int(np.isinf(df[features].to_numpy(dtype=float)).sum()),
                "status": int(np.isinf(df[features].to_numpy(dtype=float)).sum()) == 0,
            },
            {
                "check": "cutoff_rows_absent",
                "value": int(
                    (
                        pd.to_numeric(df.get("PERIOD_event", 0), errors="coerce").eq(4)
                        & pd.to_numeric(df.get("event_clock_remaining", 999), errors="coerce").le(
                            300
                        )
                    ).sum()
                ),
                "status": bool(
                    not (
                        pd.to_numeric(df.get("PERIOD_event", 0), errors="coerce").eq(4)
                        & pd.to_numeric(df.get("event_clock_remaining", 999), errors="coerce").le(
                            300
                        )
                    ).any()
                ),
            },
        ]
    )


def sorted_game_ids(df: pd.DataFrame) -> list[str]:
    game_ids = pd.to_numeric(df["GAME_ID"], errors="coerce").dropna().unique().tolist()
    return [str(int(game_id)) for game_id in sorted(game_ids)]


def split_games(game_ids: list[str]) -> dict[str, list[str]]:
    if len(game_ids) != 400:
        raise ValueError(f"Expected 400 games, got {len(game_ids)}.")
    return {
        "train": game_ids[:280],
        "val": game_ids[280:340],
        "test": game_ids[340:],
    }


def build_split_summary(splits: dict[str, list[str]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"split": split, "GAME_ID": game_id}
            for split, games in splits.items()
            for game_id in games
        ]
    )


def feature_set_columns(
    name: str, all_features: list[str], reports_dir: Path
) -> tuple[list[str], list[tuple[str, str]]]:
    if name == "all_features":
        return all_features, []
    feature_file = reports_dir / FEATURE_SET_FILES[name]
    ranking = pd.read_csv(feature_file)
    requested = ranking["feature"].astype(str).tolist()
    available = set(all_features)
    missing = [(name, feature) for feature in requested if feature not in available]
    return [feature for feature in requested if feature in available], missing


def prepare_feature_set_sequences(
    feature_set: str, df: pd.DataFrame, splits: dict[str, list[str]], features: list[str]
) -> tuple[
    dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    StandardScaler,
    list[dict],
    list[dict],
]:
    train_rows = (
        df[df["GAME_ID"].isin(splits["train"])][features]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )
    scaler = StandardScaler()
    scaler.fit(train_rows)
    train_lengths = df[df["GAME_ID"].isin(splits["train"])].groupby("GAME_ID").size()
    max_sequence_length = int(train_lengths.max())
    arrays = {}
    shapes = []
    for split, games in splits.items():
        arrays[split] = build_sequences(df, games, features, scaler, max_sequence_length)
        x, y, ids, lengths = arrays[split]
        shapes.append(
            {
                "feature_set": "pending",
                "split": split,
                "games": len(games),
                "X_shape": str(x.shape),
                "y_shape": str(y.shape),
                "min_sequence_length": int(lengths.min()),
                "max_sequence_length": int(lengths.max()),
                "max_sequence_length_train": max_sequence_length,
            }
        )
    for row in shapes:
        row["feature_set"] = feature_set
    baselines = baseline_metrics(arrays["train"][1], arrays["test"][1], feature_set)
    return arrays, scaler, shapes, baselines


def build_sequences(
    df: pd.DataFrame,
    games: list[str],
    features: list[str],
    scaler: StandardScaler,
    max_sequence_length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sequences = []
    targets = []
    ids = []
    lengths = []
    for game_id in games:
        game = df[df["GAME_ID"].eq(game_id)].sort_values(
            ["game_seconds_elapsed", "PERIOD_event", "EVENTNUM"], kind="mergesort"
        )
        x = game[features].replace([np.inf, -np.inf], np.nan).fillna(0)
        arr = scaler.transform(x)
        if len(arr) > max_sequence_length:
            arr = arr[-max_sequence_length:]
        length = len(arr)
        padded = np.zeros((max_sequence_length, len(features)), dtype=np.float32)
        padded[-length:, :] = arr.astype(np.float32)
        sequences.append(padded)
        targets.append(float(game[TARGET_COLUMN].iloc[0]))
        ids.append(game_id)
        lengths.append(length)
    return (
        np.stack(sequences),
        np.asarray(targets, dtype=np.float32),
        np.asarray(ids),
        np.asarray(lengths, dtype=np.int32),
    )


def save_feature_set_sequences(
    name: str,
    arrays: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    features: list[str],
    sequences_dir: Path,
) -> None:
    for split, (x, y, ids, lengths) in arrays.items():
        np.savez_compressed(
            sequences_dir / f"nba_{name}_{split}_sequences.npz",
            X=x,
            y=y,
            game_ids=ids,
            sequence_lengths=lengths,
            feature_columns=np.asarray(features),
        )


def baseline_metrics(y_train: np.ndarray, y_test: np.ndarray, feature_set: str) -> list[dict]:
    zero_pred = np.zeros_like(y_test)
    mean_pred = np.full_like(y_test, float(y_train.mean()))
    rows = [
        regression_metrics(y_test, zero_pred, "constant_zero_baseline", "test"),
        regression_metrics(y_test, mean_pred, "train_mean_baseline", "test"),
    ]
    for row in rows:
        row["feature_set"] = feature_set
    return rows


def build_model(max_sequence_length: int, num_features: int) -> Sequential:
    model = Sequential(
        [
            Input(shape=(max_sequence_length, num_features)),
            Masking(mask_value=0.0),
            LSTM(64),
            Dropout(0.3),
            Dense(32, activation="relu"),
            Dense(1),
        ]
    )
    model.compile(optimizer=Adam(), loss="huber", metrics=["mae", "mse"])
    return model


def train_feature_set(
    feature_set: str,
    arrays: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    features: list[str],
    models_dir: Path,
    metrics_dir: Path,
    epochs: int,
    use_callbacks: bool,
    output_prefix: str,
) -> dict[str, pd.DataFrame]:
    x_train, y_train, train_ids, _ = arrays["train"]
    x_val, y_val, val_ids, _ = arrays["val"]
    x_test, y_test, test_ids, _ = arrays["test"]
    model = build_model(x_train.shape[1], len(features))
    callbacks = (
        [EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)]
        if use_callbacks
        else []
    )
    history_obj = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=8,
        callbacks=callbacks,
        verbose=1,
    )
    models_dir.mkdir(parents=True, exist_ok=True)
    model.save(models_dir / f"{output_prefix}.keras")
    history = pd.DataFrame(history_obj.history)
    history.insert(0, "epoch", np.arange(1, len(history) + 1))
    metrics_rows = []
    prediction_rows = []
    for split, x, y, ids in [
        ("train", x_train, y_train, train_ids),
        ("val", x_val, y_val, val_ids),
        ("test", x_test, y_test, test_ids),
    ]:
        pred = model.predict(x, verbose=0).ravel()
        metrics_rows.append(regression_metrics(y, pred, f"NBA_LSTM_{feature_set}", split))
        for game_id, actual, predicted in zip(ids, y, pred, strict=False):
            prediction_rows.append(
                {
                    "feature_set": feature_set,
                    "split": split,
                    "GAME_ID": game_id,
                    "y_true": actual,
                    "y_pred": predicted,
                    "error": predicted - actual,
                    "abs_error": abs(predicted - actual),
                }
            )
    metrics = pd.DataFrame(metrics_rows)
    predictions = pd.DataFrame(prediction_rows)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(metrics_dir / f"{output_prefix}_metrics.csv", index=False)
    predictions.to_csv(metrics_dir / f"{output_prefix}_predictions.csv", index=False)
    history.to_csv(metrics_dir / f"{output_prefix}_history.csv", index=False)
    pd.DataFrame({"feature": features}).to_csv(
        metrics_dir / f"{output_prefix}_features.csv", index=False
    )
    return {"metrics": metrics, "predictions": predictions, "history": history}


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, model: str, split: str) -> dict:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "model": model,
        "split": split,
        "mae": mean_absolute_error(y_true, y_pred),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "r2": r2_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else np.nan,
    }
