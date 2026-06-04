from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Masking
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


TARGET_COLUMN = "target_score_diff_change_last_5min"
LEAKAGE_COLUMNS = {
    TARGET_COLUMN,
    "final_score_diff",
    "final_home_score",
    "final_away_score",
    "score_diff_at_5min_remaining",
}
EXCLUDED_NUMERIC_COLUMNS = {
    "GAME_EVENT_ID",
    "PERIOD_shot",
    "EVENTTIME",
    "SHOT_TIME",
    "QUARTER",
    "cutoff_found",
}
METADATA_MARKERS = ["ID", "DATE", "PLAYER", "TEAM"]


def _parse_game_date(value) -> pd.Timestamp:
    if pd.isna(value):
        return pd.NaT
    return pd.to_datetime(value, errors="coerce")


def _game_order(df: pd.DataFrame) -> list[str]:
    game_ids = pd.to_numeric(df["GAME_ID"], errors="coerce").dropna().unique().tolist()
    if game_ids:
        return [str(int(game_id)) for game_id in sorted(game_ids)]
    return sorted(df["GAME_ID"].astype(str).unique().tolist())


def _select_feature_columns(df: pd.DataFrame) -> list[str]:
    features = []
    for col in df.columns:
        upper = col.upper()
        if col in LEAKAGE_COLUMNS or col in EXCLUDED_NUMERIC_COLUMNS:
            continue
        if any(marker in upper for marker in METADATA_MARKERS):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            features.append(col)
    return features


def _split_games(game_ids: list[str]) -> dict[str, list[str]]:
    if len(game_ids) < 50:
        raise ValueError(f"Expected at least 50 games, got {len(game_ids)}.")
    train_end = int(len(game_ids) * 0.70)
    val_end = int(len(game_ids) * 0.85)
    return {
        "train": game_ids[:train_end],
        "val": game_ids[train_end:val_end],
        "test": game_ids[val_end:],
    }


def _fit_scaler(
    df: pd.DataFrame, train_games: list[str], feature_columns: list[str]
) -> StandardScaler:
    train_rows = df[df["GAME_ID"].astype(str).isin(train_games)][feature_columns].copy()
    train_rows = train_rows.replace([np.inf, -np.inf], np.nan).fillna(0)
    scaler = StandardScaler()
    scaler.fit(train_rows)
    return scaler


def _build_sequences(
    df: pd.DataFrame,
    games: list[str],
    feature_columns: list[str],
    scaler: StandardScaler,
    max_sequence_length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sequences = []
    targets = []
    game_ids = []
    lengths = []
    for game_id in games:
        game = df[df["GAME_ID"].astype(str).eq(str(game_id))].sort_values(
            ["PERIOD_event", "EVENTNUM"], kind="mergesort"
        )
        features = game[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0)
        arr = scaler.transform(features)
        if len(arr) > max_sequence_length:
            arr = arr[-max_sequence_length:]
        length = len(arr)
        padded = np.zeros((max_sequence_length, len(feature_columns)), dtype=np.float32)
        padded[-length:, :] = arr.astype(np.float32)
        sequences.append(padded)
        targets.append(float(game[TARGET_COLUMN].iloc[0]))
        game_ids.append(str(game_id))
        lengths.append(length)
    return (
        np.stack(sequences),
        np.asarray(targets, dtype=np.float32),
        np.asarray(game_ids),
        np.asarray(lengths, dtype=np.int32),
    )


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


def build_model(max_sequence_length: int, num_features: int) -> Sequential:
    model = Sequential(
        [
            Input(shape=(max_sequence_length, num_features)),
            Masking(mask_value=0.0),
            LSTM(64, return_sequences=False),
            Dropout(0.3),
            Dense(32, activation="relu"),
            Dense(1),
        ]
    )
    model.compile(optimizer=Adam(), loss="huber", metrics=["mae", "mse"])
    return model


def save_training_plots(
    history: pd.DataFrame,
    predictions: pd.DataFrame,
    figures_dir: Path,
) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    for col in ["loss", "val_loss"]:
        if col in history:
            ax.plot(history["epoch"], history[col], label=col)
    ax.set_title("NBA LSTM training curves")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Huber loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "nba_lstm_training_curves.png", dpi=160)
    plt.close(fig)

    test_pred = predictions[predictions["split"].eq("test")]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(test_pred["y_true"], test_pred["y_pred"], alpha=0.8)
    low = min(test_pred["y_true"].min(), test_pred["y_pred"].min())
    high = max(test_pred["y_true"].max(), test_pred["y_pred"].max())
    ax.plot([low, high], [low, high], linestyle="--", color="black")
    ax.set_title("NBA LSTM prediction vs actual")
    ax.set_xlabel("Actual score diff change")
    ax.set_ylabel("Predicted score diff change")
    fig.tight_layout()
    fig.savefig(figures_dir / "nba_lstm_prediction_vs_actual.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhline(0, linestyle="--", color="black", linewidth=1)
    ax.scatter(test_pred["y_pred"], test_pred["error"], alpha=0.8)
    ax.set_title("NBA LSTM residuals")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Prediction error")
    fig.tight_layout()
    fig.savefig(figures_dir / "nba_lstm_residuals.png", dpi=160)
    plt.close(fig)


def run_nba_lstm_clutch_regression(
    input_path: Path,
    data_dir: Path,
    models_dir: Path,
    metrics_dir: Path,
    figures_dir: Path,
) -> dict[str, pd.DataFrame]:
    df = pd.read_csv(input_path)
    df["GAME_ID"] = df["GAME_ID"].astype(str).str.zfill(10)
    game_ids = _game_order(df)
    splits = _split_games(game_ids)
    feature_columns = _select_feature_columns(df)

    train_lengths = df[df["GAME_ID"].isin(splits["train"])].groupby("GAME_ID").size()
    max_sequence_length = int(train_lengths.max())
    scaler = _fit_scaler(df, splits["train"], feature_columns)

    arrays = {}
    for split, games in splits.items():
        arrays[split] = _build_sequences(
            df=df,
            games=games,
            feature_columns=feature_columns,
            scaler=scaler,
            max_sequence_length=max_sequence_length,
        )

    data_dir.mkdir(parents=True, exist_ok=True)
    for split, (x, y, ids, lengths) in arrays.items():
        np.savez_compressed(
            data_dir / f"nba_{split}_sequences.npz",
            X=x,
            y=y,
            game_ids=ids,
            sequence_lengths=lengths,
            feature_columns=np.asarray(feature_columns),
        )

    models_dir.mkdir(parents=True, exist_ok=True)
    with (models_dir / "nba_lstm_scaler.pkl").open("wb") as f:
        pickle.dump(scaler, f)

    x_train, y_train, train_ids, train_seq_lengths = arrays["train"]
    x_val, y_val, val_ids, val_seq_lengths = arrays["val"]
    x_test, y_test, test_ids, test_seq_lengths = arrays["test"]

    model = build_model(max_sequence_length, len(feature_columns))
    model_summary_lines: list[str] = []
    model.summary(print_fn=model_summary_lines.append)
    callbacks = [EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)]
    history_obj = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=50,
        batch_size=8,
        callbacks=callbacks,
        verbose=1,
    )

    models_dir.mkdir(parents=True, exist_ok=True)
    model.save(models_dir / "nba_lstm_clutch_regression.keras")

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
        metrics_rows.append(regression_metrics(y, pred, "NBA_LSTM_clutch_regression", split))
        for game_id, actual, predicted in zip(ids, y, pred, strict=False):
            prediction_rows.append(
                {
                    "split": split,
                    "GAME_ID": game_id,
                    "y_true": actual,
                    "y_pred": predicted,
                    "error": predicted - actual,
                    "abs_error": abs(predicted - actual),
                }
            )

    zero_pred = np.zeros_like(y_test)
    mean_pred = np.full_like(y_test, y_train.mean())
    metrics_rows.append(regression_metrics(y_test, zero_pred, "constant_zero_baseline", "test"))
    metrics_rows.append(regression_metrics(y_test, mean_pred, "mean_train_baseline", "test"))

    metrics = pd.DataFrame(metrics_rows)
    predictions = pd.DataFrame(prediction_rows)
    shapes = pd.DataFrame(
        [
            {
                "split": "train",
                "games": len(splits["train"]),
                "X_shape": str(x_train.shape),
                "y_shape": str(y_train.shape),
                "min_sequence_length": int(train_seq_lengths.min()),
                "max_sequence_length": int(train_seq_lengths.max()),
            },
            {
                "split": "val",
                "games": len(splits["val"]),
                "X_shape": str(x_val.shape),
                "y_shape": str(y_val.shape),
                "min_sequence_length": int(val_seq_lengths.min()),
                "max_sequence_length": int(val_seq_lengths.max()),
            },
            {
                "split": "test",
                "games": len(splits["test"]),
                "X_shape": str(x_test.shape),
                "y_shape": str(y_test.shape),
                "min_sequence_length": int(test_seq_lengths.min()),
                "max_sequence_length": int(test_seq_lengths.max()),
            },
        ]
    )
    split_summary = pd.DataFrame(
        [
            {"split": split, "GAME_ID": game_id}
            for split, games in splits.items()
            for game_id in games
        ]
    )
    feature_df = pd.DataFrame({"feature": feature_columns})
    best_epoch = int(history.loc[history["val_loss"].idxmin(), "epoch"])
    training_summary = pd.DataFrame(
        [
            {
                "epochs_run": len(history),
                "best_epoch": best_epoch,
                "best_val_loss": float(history["val_loss"].min()),
                "final_train_loss": float(history["loss"].iloc[-1]),
                "final_val_loss": float(history["val_loss"].iloc[-1]),
                "overfitting_gap_final": float(
                    history["val_loss"].iloc[-1] - history["loss"].iloc[-1]
                ),
            }
        ]
    )

    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(metrics_dir / "nba_lstm_clutch_metrics.csv", index=False)
    predictions.to_csv(metrics_dir / "nba_lstm_clutch_predictions.csv", index=False)
    history.to_csv(metrics_dir / "nba_lstm_clutch_history.csv", index=False)
    shapes.to_csv(metrics_dir / "nba_lstm_clutch_shapes.csv", index=False)
    split_summary.to_csv(metrics_dir / "nba_lstm_clutch_split_summary.csv", index=False)
    feature_df.to_csv(metrics_dir / "nba_lstm_clutch_feature_columns.csv", index=False)
    training_summary.to_csv(metrics_dir / "nba_lstm_clutch_training_summary.csv", index=False)
    (models_dir / "nba_lstm_clutch_model_summary.txt").write_text(
        "\n".join(model_summary_lines), encoding="utf-8"
    )
    save_training_plots(history, predictions, figures_dir)

    return {
        "metrics": metrics,
        "predictions": predictions,
        "history": history,
        "shapes": shapes,
        "split_summary": split_summary,
        "features": feature_df,
        "training_summary": training_summary,
    }
