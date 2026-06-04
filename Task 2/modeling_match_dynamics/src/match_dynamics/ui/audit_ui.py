from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[3]
AUDIT_DIR = PROJECT_ROOT / "outputs" / "audits" / "data_quality"
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"


def csv_path(name: str) -> Path:
    return AUDIT_DIR / name


def file_signature(path: Path) -> str:
    if not path.exists():
        return "missing"
    stat = path.stat()
    return f"{stat.st_mtime_ns}:{stat.st_size}"


@st.cache_data(show_spinner=False)
def read_csv(path: str, stamp: str) -> pd.DataFrame:
    fpath = Path(path)
    if not fpath.exists():
        return pd.DataFrame()
    return pd.read_csv(fpath)


@st.cache_data(show_spinner=False)
def read_csv_head(path: str, nrows: int, stamp: str) -> pd.DataFrame:
    fpath = Path(path)
    if not fpath.exists():
        return pd.DataFrame()
    return pd.read_csv(fpath, nrows=nrows)


@st.cache_data(show_spinner=False)
def read_football_match_preview(path: str, nrows: int, stamp: str) -> pd.DataFrame:
    fpath = Path(path)
    if not fpath.exists():
        return pd.DataFrame()
    cols = [
        "id_odsp",
        "date",
        "league",
        "season",
        "country",
        "ht",
        "at",
        "fthg",
        "ftag",
        "final_score",
    ]
    df = pd.read_csv(fpath, usecols=lambda col: col in cols)
    rows_per_match = df.groupby("id_odsp", dropna=False).size().rename("event_rows")
    matches = df.drop_duplicates("id_odsp").merge(rows_per_match, on="id_odsp", how="left")
    return matches.head(nrows)


@st.cache_data(show_spinner=False)
def read_event_dataset_summary(path: str, dataset_name: str, stamp: str) -> pd.DataFrame:
    fpath = Path(path)
    if not fpath.exists():
        return pd.DataFrame()
    df = pd.read_csv(fpath, usecols=lambda col: col in ["id_odsp"])
    columns = len(pd.read_csv(fpath, nrows=0).columns)
    return pd.DataFrame(
        [
            {
                "dataset": dataset_name,
                "rows": len(df),
                "columns": columns,
                "unique_matches": df["id_odsp"].nunique(dropna=True)
                if "id_odsp" in df.columns
                else pd.NA,
                "null_id_odsp": int(df["id_odsp"].isna().sum())
                if "id_odsp" in df.columns
                else pd.NA,
            }
        ]
    )


def compare_column_profiles(before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
    if before.empty or after.empty:
        return pd.DataFrame()
    before_cols = before.set_index("column")
    after_cols = after.set_index("column")
    all_cols = sorted(set(before_cols.index) | set(after_cols.index))
    rows = []
    for col in all_cols:
        before_exists = col in before_cols.index
        after_exists = col in after_cols.index
        rows.append(
            {
                "column": col,
                "status": "created"
                if after_exists and not before_exists
                else "dropped"
                if before_exists and not after_exists
                else "kept",
                "dtype_before": before_cols.at[col, "dtype"] if before_exists else pd.NA,
                "dtype_after": after_cols.at[col, "dtype"] if after_exists else pd.NA,
                "missing_rate_before": before_cols.at[col, "missing_rate"]
                if before_exists
                else pd.NA,
                "missing_rate_after": after_cols.at[col, "missing_rate"] if after_exists else pd.NA,
                "n_unique_before": before_cols.at[col, "n_unique"] if before_exists else pd.NA,
                "n_unique_after": after_cols.at[col, "n_unique"] if after_exists else pd.NA,
            }
        )
    return pd.DataFrame(rows)


def load_table(name: str) -> pd.DataFrame:
    path = csv_path(name)
    return read_csv(str(path), file_signature(path))


def load_metric_table(name: str) -> pd.DataFrame:
    path = METRICS_DIR / name
    return read_csv(str(path), file_signature(path))


def load_football_metric_table(name: str) -> pd.DataFrame:
    path = METRICS_DIR / "football" / name
    return read_csv(str(path), file_signature(path))


def load_nba_metric_table(name: str) -> pd.DataFrame:
    path = METRICS_DIR / "nba" / name
    return read_csv(str(path), file_signature(path))


def load_report_table(name: str) -> pd.DataFrame:
    path = REPORTS_DIR / name
    return read_csv(str(path), file_signature(path))


@st.cache_data(show_spinner=False)
def read_direct_csv_head(path: str, nrows: int, stamp: str) -> pd.DataFrame:
    fpath = Path(path)
    if not fpath.exists():
        return pd.DataFrame()
    return pd.read_csv(fpath, nrows=nrows)


@st.cache_data(show_spinner=False)
def direct_nba_file_inventory(stamp: str) -> pd.DataFrame:
    nba_dir = PROJECT_ROOT / "data" / "nba"
    rows = []
    for source, folder, pattern in [
        ("movement", nba_dir / "movement", "*.7z"),
        ("events", nba_dir / "events", "*.csv"),
        ("shots", nba_dir / "shots", "*.csv"),
    ]:
        files = sorted(folder.glob(pattern)) if folder.exists() else []
        rows.append(
            {
                "source": source,
                "folder": str(folder),
                "files": len(files),
                "total_size_mb": sum(path.stat().st_size for path in files) / 1024 / 1024,
            }
        )
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def inspect_direct_movement_archive(stamp: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    movement_dir = PROJECT_ROOT / "data" / "nba" / "movement"
    archive_files = sorted(movement_dir.glob("*.7z")) if movement_dir.exists() else []
    if not archive_files:
        return pd.DataFrame(), pd.DataFrame()
    archive_path = archive_files[0]
    try:
        import json
        import tempfile

        import py7zr

        with py7zr.SevenZipFile(archive_path, mode="r") as archive:
            names = archive.getnames()
            json_names = [name for name in names if name.lower().endswith(".json")]
            inventory = pd.DataFrame(
                [
                    {
                        "archive_name": archive_path.name,
                        "archive_path": str(archive_path),
                        "files_inside": ", ".join(names),
                        "readable": True,
                        "warning": "",
                    }
                ]
            )
            if not json_names:
                return inventory, pd.DataFrame()
            with tempfile.TemporaryDirectory() as tmpdir:
                archive.extract(path=tmpdir, targets=[json_names[0]])
                extracted = Path(tmpdir) / json_names[0]
                with extracted.open(encoding="utf-8") as f:
                    payload = json.load(f)
            events = payload.get("events", [])
            sample_event = events[0] if events else {}
            moments = sample_event.get("moments", []) if isinstance(sample_event, dict) else []
            head = pd.DataFrame(
                [
                    {
                        "archive_name": archive_path.name,
                        "file_inside": json_names[0],
                        "gameid": payload.get("gameid"),
                        "events_count": len(events),
                        "sample_event_id": sample_event.get("eventId")
                        if isinstance(sample_event, dict)
                        else None,
                        "sample_moments_count": len(moments),
                        "sample_first_moment": str(moments[0])[:500] if moments else "",
                    }
                ]
            )
            return inventory, head
    except Exception as exc:
        inventory = pd.DataFrame(
            [
                {
                    "archive_name": archive_path.name,
                    "archive_path": str(archive_path),
                    "files_inside": "",
                    "readable": False,
                    "warning": str(exc),
                }
            ]
        )
        return inventory, pd.DataFrame()


def direct_column_quality(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
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


def show_missing_bar(profile: pd.DataFrame, title: str, limit: int = 30) -> None:
    if profile.empty or "missing_rate" not in profile.columns:
        st.info("РќРµС‚ РґР°РЅРЅС‹С… РґР»СЏ РіСЂР°С„РёРєР° РїСЂРѕРїСѓСЃРєРѕРІ.")
        return
    plot_df = (
        profile.sort_values("missing_rate", ascending=False)
        .head(limit)
        .sort_values("missing_rate", ascending=True)
    )
    fig = px.bar(
        plot_df,
        x="missing_rate",
        y="column",
        orientation="h",
        title=title,
        hover_data=[c for c in ["missing", "non_null", "dtype", "n_unique"] if c in plot_df],
    )
    fig.update_layout(height=max(420, 24 * len(plot_df)), xaxis_tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)


def show_dtype_bar(profile: pd.DataFrame, title: str) -> None:
    if profile.empty or "dtype" not in profile.columns:
        return
    counts = profile["dtype"].value_counts().reset_index()
    counts.columns = ["dtype", "columns"]
    fig = px.bar(counts, x="dtype", y="columns", title=title, text="columns")
    st.plotly_chart(fig, use_container_width=True)


def show_profile_table(profile: pd.DataFrame) -> None:
    if profile.empty:
        st.warning(
            "РўР°Р±Р»РёС†Р° РЅРµ РЅР°Р№РґРµРЅР°. РЎРЅР°С‡Р°Р»Р° Р·Р°РїСѓСЃС‚Рё Р°СѓРґРёС‚ РґР°РЅРЅС‹С…."
        )
        return
    search = st.text_input("Р¤РёР»СЊС‚СЂ РїРѕ РЅР°Р·РІР°РЅРёСЋ РєРѕР»РѕРЅРєРё", "")
    view = profile
    if search:
        view = view[view["column"].astype(str).str.contains(search, case=False, na=False)]
    st.dataframe(view, use_container_width=True, height=420)


def show_head_table(name: str, title: str) -> None:
    df = load_table(name)
    st.subheader(title)
    if df.empty:
        st.warning(f"Р¤Р°Р№Р» {name} РЅРµ РЅР°Р№РґРµРЅ.")
        return
    st.dataframe(df, use_container_width=True, height=360)


def show_overview() -> None:
    st.header("Анализ и моделирование динамики спортивных событий")
    st.markdown(
        """
        ### Цель работы

        Задача проекта - подготовить спортивные события как временные ряды и построить
        LSTM-модели, которые по уже произошедшей части матча прогнозируют будущие события
        или изменение счета.

        В работе рассматриваются два вида спорта и две постановки задачи:

        - футбол: классификация событий второго тайма;
        - NBA: регрессия изменения разницы счета в концовке матча.
        """
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Football Events")
        st.markdown(
            """
            **Источник данных:** [Kaggle: Football Events](https://www.kaggle.com/datasets/secareanualin/football-events)

            **Какие файлы используются:**

            - `events.csv` - event-level таблица футбольных событий: матч, минута, сторона,
              тип события, игрок, команда, соперник, текст события, признаки удара, передачи,
              карточек и других игровых действий.
            - `ginf.csv` - match-level таблица: команды, лига, страна, сезон, дата матча,
              финальный счет и служебная информация о матче.

            **Постановка задачи:**

            По событиям первого тайма предсказать, будет ли гол во втором тайме отдельно
            для хозяев и гостей.

            **Что модель видит:**

            - только минуты `1-45`;
            - события и агрегаты первого тайма;
            - исторические признаки команд, рассчитанные только по прошлым матчам.

            **Targets:**

            - `home_scores_next_half` - хозяева забьют после 45 минуты;
            - `away_scores_next_half` - гости забьют после 45 минуты.

            **Тип задачи:** multi-output binary classification.
            """
        )

    with col2:
        st.subheader("NBA Player Tracking")
        st.markdown(
            """
            **Источник данных:** [GitHub: nba-movement-data](https://github.com/sealneaward/nba-movement-data)

            **Какие источники используются:**

            - movement `.7z` archives - tracking-данные: координаты игроков и мяча по моментам;
            - `data/events/*.csv` - play-by-play события матча;
            - `data/shots/shots_fixed.csv` - shot chart: броски, зоны, дистанция, координаты,
              факт попадания/промаха.

            **Постановка задачи:**

            По событиям матча до отметки `Q4 05:00 remaining` предсказать, как изменится
            разница счета за последние 5 минут.

            **Что модель видит:**

            - play-by-play события до cutoff;
            - текущий счет на момент события;
            - признаки бросков;
            - агрегированные movement-признаки;
            - rolling tempo/momentum features до cutoff.

            **Target:**

            - `target_score_diff_change_last_5min` = финальная разница счета минус разница
              счета на отметке Q4 05:00.

            **Тип задачи:** LSTM regression.
            """
        )


def show_conclusion() -> None:
    st.header("Итоговые выводы")

    st.subheader("Общий результат")
    st.markdown(
        """
        Проект вынесен из notebook-style формата в локальный VS Code проект с двумя
        самостоятельными pipeline: Football и NBA. Оба направления используют LSTM как
        модель последовательностей, но решают разные прогнозные задачи.

        Главный принцип всего проекта: **никакого leakage**. Модель должна видеть только
        ту информацию, которая была доступна до прогнозной точки.
        """
    )

    st.subheader("Football")
    st.markdown(
        """
        Был построен полноценный pipeline прогнозирования событий второго тайма по событиям
        первого тайма футбольного матча.

        Pipeline включает:

        - merge и preprocessing event data;
        - leakage-safe temporal split;
        - advanced feature engineering;
        - historical team-strength features;
        - rolling/momentum/context features;
        - sequence construction для LSTM;
        - feature ablation;
        - threshold tuning;
        - calibration;
        - error analysis.

        Финальная модель:

        - multi-output LSTM;
        - top-50 features;
        - sequence length = 45 минут;
        - targets: `home_scores_next_half`, `away_scores_next_half`.

        Финальные thresholds:

        - HOME = 0.47
        - AWAY = 0.49

        Финальные test результаты:

        **HOME**

        - ROC-AUC ≈ 0.65
        - PR-AUC ≈ 0.72
        - MAE = 0.4678
        - MSE = 0.2313
        - RMSE = 0.4810

        **AWAY**

        - ROC-AUC ≈ 0.58-0.59
        - PR-AUC ≈ 0.57-0.60
        - MAE = 0.4893
        - MSE = 0.2466
        - RMSE = 0.4966

        Так как задача бинарная, MSE по вероятностям эквивалентен Brier score.
        Чем ниже значение, тем лучше калиброваны вероятности модели.

        Главный вывод по Football: модель извлекает реальный signal из match dynamics и
        historical team context. Результаты не идеальные, но реалистичные для noisy football
        forecasting задачи без xG провайдера, player tracking, составов и live odds.
        """
    )

    st.subheader("NBA")
    st.markdown(
        """
        NBA pipeline переведен в отдельную regression-задачу clutch-time forecasting
        на 400 матчах.

        Использованные данные:

        - `movement` archives из `nba-movement-data` - координаты игроков и мяча;
        - `events/*.csv` - play-by-play события;
        - `shots_fixed.csv` - броски, координаты, дистанция, зона и факт попадания.

        Pipeline включает:

        - merge трех источников по `GAME_ID` и event id;
        - preprocessing score, shot, event и movement признаков;
        - clutch-oriented feature engineering;
        - historical team-strength features по `GAME_ID` timeline;
        - leakage-safe split по матчам;
        - event-level sequence construction;
        - feature-set selection: top30, top50, top75, all_features;
        - baseline comparison и LSTM regression training.

        Финальная постановка:

        - один матч = одна event-level sequence;
        - модель видит события только до `Q4 05:00 remaining`;
        - target: `target_score_diff_change_last_5min`;
        - тип задачи: regression;
        - split: 280 train games, 60 validation games, 60 test games.

        Модель:

        - `Masking`;
        - `LSTM(64)`;
        - `Dropout(0.3)`;
        - `Dense(32, relu)`;
        - `Dense(1)`;
        - loss: Huber;
        - metrics: MAE, MSE, RMSE, R2.

        Baselines на test:

        - constant-zero baseline: MAE = 5.0167, MSE = 40.0167, RMSE = 6.3259;
        - train-mean baseline: MAE = 5.0274, MSE = 40.1263, RMSE = 6.3345.

        LSTM test results:

        - top30: MAE = 4.8358, MSE = 37.4949, RMSE = 6.1233, R2 = 0.0059;
        - top50: MAE = 4.9387, MSE = 39.0005, RMSE = 6.2450, R2 = -0.0340;
        - top75: MAE = 4.5989, MSE = 35.4153, RMSE = 5.9511, R2 = 0.0610;
        - all_features: MAE = 4.6073, MSE = 34.5804, RMSE = 5.8805, R2 = 0.0831.

        Главный вывод по NBA:

        - все LSTM-варианты лучше простых baseline по MAE/MSE;
        - `top75` лучший по MAE;
        - `all_features` лучший по MSE/RMSE/R2;
        - качество умеренное, но для noisy clutch-time regression это уже осмысленный baseline;
        - дальнейшее улучшение вероятнее всего даст больше матчей, более точный possession-level
          target, home/away possession ownership и richer player/team context.
        """
    )


def show_football_merge() -> None:
    st.header("Football Merge: events.csv + ginf.csv")
    st.caption(
        "One match row from ginf.csv is left-joined to many event rows from events.csv by id_odsp. "
        "Duplicated match-level values per event are expected for event-level ML pipelines."
    )

    summary = load_table("football_merge_summary.csv")
    profile = load_table("football_merged_event_match_columns.csv")

    if summary.empty:
        st.warning("Merge tables not found. Click `Refresh audit tables` in the sidebar.")
        return

    st.subheader("Merge summary")
    st.dataframe(summary, use_container_width=True)

    merged_row = summary[summary["dataset"].eq("merged")]
    if not merged_row.empty:
        cols = st.columns(4)
        cols[0].metric("Merged rows", f"{int(merged_row['rows'].iloc[0]):,}".replace(",", " "))
        cols[1].metric("Merged columns", int(merged_row["columns"].iloc[0]))
        cols[2].metric("Unique matches", int(merged_row["unique_matches"].iloc[0]))
        if "matched_rate" in merged_row:
            cols[3].metric("Matched rows", f"{float(merged_row['matched_rate'].iloc[0]):.1%}")

    st.subheader("Merged event-level head() with all columns")
    head_rows = st.slider("Rows to show from data/football_merged.csv", 5, 200, 30)
    merged_path = PROJECT_ROOT / "data" / "football_merged.csv"
    merged_head = read_csv_head(str(merged_path), head_rows, file_signature(merged_path))
    if merged_head.empty:
        st.warning(
            "data/football_merged.csv РЅРµ РЅР°Р№РґРµРЅ. РЎРЅР°С‡Р°Р»Р° Р·Р°РїСѓСЃС‚Рё merge РёР»Рё Refresh audit tables."
        )
    else:
        st.caption(
            "Р­С‚Рѕ РїРµСЂРІС‹Рµ СЃС‚СЂРѕРєРё РїРѕР»РЅРѕРіРѕ event-level merge: РІСЃРµ РєРѕР»РѕРЅРєРё events.csv + match-level РєРѕР»РѕРЅРєРё ginf.csv."
        )
        st.dataframe(merged_head, use_container_width=True, height=560)

    st.subheader("Compact merged preview: matches")
    match_rows = st.slider(
        "Rows to show from match-level preview",
        30,
        200,
        50,
        key="football_merge_match_preview_rows",
    )
    match_preview = read_football_match_preview(
        str(merged_path), match_rows, file_signature(merged_path)
    )
    if match_preview.empty:
        st.warning(
            "Match preview is unavailable. Refresh audit tables or rebuild football_merged.csv."
        )
    else:
        st.caption(
            "РљР°Р¶РґР°СЏ СЃС‚СЂРѕРєР° Р·РґРµСЃСЊ - РѕРґРёРЅ РјР°С‚С‡, Р° `event_rows` РїРѕРєР°Р·С‹РІР°РµС‚ С‡РёСЃР»Рѕ СЃРѕР±С‹С‚РёР№ РІ РјР°С‚С‡Рµ."
        )
        st.dataframe(match_preview, use_container_width=True, height=460)

    st.subheader("Merged columns: data types and quality")
    if profile.empty:
        st.warning("РџСЂРѕС„РёР»СЊ РєРѕР»РѕРЅРѕРє merge РЅРµ РЅР°Р№РґРµРЅ.")
    else:
        dtype_cols = [
            c
            for c in [
                "column",
                "dtype",
                "non_null",
                "missing",
                "missing_rate",
                "n_unique",
                "zero_count",
                "zero_rate",
                "mean",
                "std",
                "min",
                "median",
                "max",
            ]
            if c in profile.columns
        ]
        st.dataframe(profile[dtype_cols], use_container_width=True, height=520)

    show_missing_bar(profile, "Football Merged: Top Missing Columns")
    show_dtype_bar(profile, "Football Merged: Column Types")


def show_football_merged_processed() -> None:
    st.header("Football Merged Processed: football_merged_processed.csv")
    st.caption(
        "Same view structure as Football Merge, but using the processed minute-level dataset."
    )

    processed_path = PROJECT_ROOT / "data" / "football_merged_processed.csv"
    processed_stamp = file_signature(processed_path)
    summary = read_event_dataset_summary(
        str(processed_path), "football_merged_processed", processed_stamp
    )
    profile = load_table("football_merged_processed_columns.csv")

    if summary.empty:
        st.warning(
            "data/football_merged_processed.csv not found. Run processing or click `Refresh audit tables`."
        )
        return

    st.subheader("Processed summary")
    st.dataframe(summary, use_container_width=True)

    processed_row = summary.iloc[0]
    cols = st.columns(4)
    cols[0].metric("Processed rows", f"{int(processed_row['rows']):,}".replace(",", " "))
    cols[1].metric("Processed columns", int(processed_row["columns"]))
    cols[2].metric("Unique matches", int(processed_row["unique_matches"]))
    cols[3].metric("Null match ids", int(processed_row["null_id_odsp"]))

    st.subheader("Processed minute-level head() with all columns")
    head_rows = st.slider(
        "Rows to show from data/football_merged_processed.csv",
        5,
        200,
        30,
        key="football_merged_processed_head_rows",
    )
    processed_head = read_csv_head(str(processed_path), head_rows, processed_stamp)
    if processed_head.empty:
        st.warning("data/football_merged_processed.csv not found. Run processing or refresh audit.")
    else:
        st.caption("First rows of the full processed minute-level dataset with all columns.")
        st.dataframe(processed_head, use_container_width=True, height=560)

    st.subheader("Compact processed preview: matches")
    match_rows = st.slider(
        "Rows to show from processed match-level preview",
        30,
        200,
        50,
        key="football_processed_match_preview_rows",
    )
    match_preview = read_football_match_preview(str(processed_path), match_rows, processed_stamp)
    if match_preview.empty:
        st.warning(
            "Processed match preview is unavailable. Refresh audit tables or rebuild processed file."
        )
    else:
        st.caption("Each row is one match; `event_rows` is the number of first-half minute rows.")
        st.dataframe(match_preview, use_container_width=True, height=460)

    st.subheader("Processed columns: data types and quality")
    if profile.empty:
        st.warning("Processed column profile not found.")
    else:
        dtype_cols = [
            c
            for c in [
                "column",
                "dtype",
                "non_null",
                "missing",
                "missing_rate",
                "n_unique",
                "zero_count",
                "zero_rate",
                "mean",
                "std",
                "min",
                "median",
                "max",
            ]
            if c in profile.columns
        ]
        st.dataframe(profile[dtype_cols], use_container_width=True, height=520)

    show_missing_bar(profile, "Football Merged Processed: Top Missing Columns")
    show_dtype_bar(profile, "Football Merged Processed: Column Types")

    st.subheader("Target distribution")
    target_dist = load_report_table("football_target_distribution.csv")
    if target_dist.empty:
        st.info("Target distribution report is not ready. Run target analysis first.")
    else:
        st.dataframe(target_dist, use_container_width=True)
        fig = px.bar(
            target_dist,
            x="value",
            y="matches",
            color="target",
            barmode="group",
            title="Target Distribution By Matches",
            text="matches",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Feature-target correlations")
    corr = load_report_table("football_feature_target_correlations.csv")
    if corr.empty:
        st.info("Correlation report is not ready. Run target analysis first.")
    else:
        st.dataframe(corr, use_container_width=True, height=420)
        for target in ["home_scores_next_half", "away_scores_next_half"]:
            plot_df = (
                corr[corr["target"].eq(target)]
                .sort_values("abs_correlation", ascending=False)
                .head(20)
                .sort_values("abs_correlation", ascending=True)
            )
            if plot_df.empty:
                continue
            fig = px.bar(
                plot_df,
                x="correlation",
                y="feature",
                orientation="h",
                title=f"Top 20 Features vs {target}",
                hover_data=["abs_correlation"],
            )
            fig.update_layout(height=max(460, 24 * len(plot_df)))
            st.plotly_chart(fig, use_container_width=True)


def show_football_merged_feature_engineering() -> None:
    st.header("Football Merged Feature Engineering: football_merged_feature_engineering.csv")
    st.caption(
        "Working copy for feature engineering. From this point, feature changes should happen here, "
        "while football_merged_processed.csv stays as the clean processed baseline."
    )

    feature_path = PROJECT_ROOT / "data" / "football_merged_feature_engineering.csv"
    feature_stamp = file_signature(feature_path)
    summary = read_event_dataset_summary(
        str(feature_path), "football_merged_feature_engineering", feature_stamp
    )

    if summary.empty:
        st.warning(
            "data/football_merged_feature_engineering.csv not found. Create it from the processed dataset first."
        )
        return

    st.subheader("Feature engineering dataset summary")
    st.dataframe(summary, use_container_width=True)

    row = summary.iloc[0]
    cols = st.columns(4)
    cols[0].metric("Rows", f"{int(row['rows']):,}".replace(",", " "))
    cols[1].metric("Columns", int(row["columns"]))
    cols[2].metric("Unique matches", int(row["unique_matches"]))
    cols[3].metric("Null match ids", int(row["null_id_odsp"]))

    st.subheader("head() with all columns")
    head_rows = st.slider(
        "Rows to show from data/football_merged_feature_engineering.csv",
        5,
        200,
        30,
        key="football_merged_feature_engineering_head_rows",
    )
    head = read_csv_head(str(feature_path), head_rows, feature_stamp)
    st.dataframe(head, use_container_width=True, height=620)

    st.subheader("Feature engineering reports")
    summary = load_report_table("football_merged_feature_engineering_historical_summary.csv")
    validation = load_report_table("football_merged_feature_engineering_historical_validation.csv")
    created = load_report_table(
        "football_merged_feature_engineering_historical_created_features.csv"
    )
    if not summary.empty:
        st.markdown("**Historical team-strength summary**")
        st.dataframe(summary, use_container_width=True)
    if not validation.empty:
        st.markdown("**Historical leakage validation**")
        st.dataframe(validation, use_container_width=True)
    if not created.empty:
        with st.expander("Historical features created"):
            st.dataframe(created, use_container_width=True, height=360)

    st.subheader("Target distribution")
    target_dist = load_report_table("football_merged_feature_engineering_target_distribution.csv")
    if target_dist.empty:
        st.info("Target distribution is not ready. Run football feature engineering first.")
    else:
        st.dataframe(target_dist, use_container_width=True)
        fig = px.bar(
            target_dist,
            x="value",
            y="matches",
            color="target",
            barmode="group",
            title="Target Distribution By Matches",
            text="matches",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Feature importance proxy: correlation with targets")
    corr = load_report_table("football_merged_feature_engineering_feature_target_correlations.csv")
    if corr.empty:
        st.info("Feature-target correlation report is not ready.")
    else:
        st.dataframe(corr, use_container_width=True, height=420)
        for top_n in [20, 30, 40]:
            st.markdown(f"**Top {top_n} absolute correlations**")
            for target in ["home_scores_next_half", "away_scores_next_half"]:
                plot_df = (
                    corr[corr["target"].eq(target)]
                    .sort_values("abs_correlation", ascending=False)
                    .head(top_n)
                    .sort_values("abs_correlation", ascending=True)
                )
                if plot_df.empty:
                    continue
                fig = px.bar(
                    plot_df,
                    x="correlation",
                    y="feature",
                    orientation="h",
                    title=f"Top {top_n} Features vs {target}",
                    hover_data=["abs_correlation"],
                )
                fig.update_layout(height=max(520, 22 * len(plot_df)))
                st.plotly_chart(fig, use_container_width=True)


def show_nba_merge() -> None:
    st.header("NBA Merge")
    st.caption("NBA 400-game merge dataset: events + shots_fixed + event-level movement features.")

    nba_path = PROJECT_ROOT / "data" / "nba" / "nba_merged_400.csv"
    report_dir = PROJECT_ROOT / "outputs" / "reports" / "nba_merge_400"
    stamp = file_signature(nba_path)
    if not nba_path.exists():
        st.info(
            "NBA 400-match merged dataset not found. Build it first: "
            "`uv run python scripts/nba/merge_nba_400.py --skip-download`."
        )
        return

    sample = read_csv_head(str(nba_path), 200, stamp)
    row_count = sum(1 for _ in nba_path.open(encoding="utf-8")) - 1
    unique_games = 0
    if "GAME_ID" in sample:
        game_ids = pd.read_csv(nba_path, usecols=["GAME_ID"])
        unique_games = int(game_ids["GAME_ID"].nunique())
    report_summary_path = report_dir / "nba_merge_200_merged_summary.csv"
    report_summary = read_csv(str(report_summary_path), file_signature(report_summary_path))

    def report_metric(metric: str, default: str = "") -> str:
        if report_summary.empty or "metric" not in report_summary or "value" not in report_summary:
            return default
        rows = report_summary[report_summary["metric"].eq(metric)]
        if rows.empty:
            return default
        return str(rows["value"].iloc[0])

    movement_cols = [
        c
        for c in sample.columns
        if c
        in {
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
        }
    ]
    source_summary = pd.DataFrame(
        [
            {
                "dataset": "events CSV",
                "source": str(PROJECT_ROOT / "data" / "nba" / "events"),
                "role": "play-by-play события матча",
                "shape": report_metric("events_shape", "see data/nba/events"),
                "join_key": "GAME_ID + EVENTNUM",
            },
            {
                "dataset": "shots_fixed.csv",
                "source": str(PROJECT_ROOT / "data" / "nba" / "shots" / "shots_fixed.csv"),
                "role": "shot chart: броски, зоны, дистанция, попадание",
                "shape": report_metric("shots_fixed_selected_shape", "see shots_fixed.csv"),
                "join_key": "GAME_ID + GAME_EVENT_ID",
            },
            {
                "dataset": "movement archives",
                "source": str(PROJECT_ROOT / "data" / "nba" / "movement"),
                "role": "tracking summaries: координаты игроков и мяча",
                "shape": report_metric("movement_shape", "see movement JSON cache"),
                "join_key": "gameid + eventId -> GAME_ID + EVENTNUM",
            },
        ]
    )
    st.subheader("Source datasets")
    st.caption("Три источника, которые объединяются в NBA merged dataset.")
    st.dataframe(source_summary, use_container_width=True, height=220)

    merged_summary = pd.DataFrame(
        [
            {"metric": "merged_dataset", "value": str(nba_path)},
            {"metric": "merged_shape", "value": f"({row_count}, {len(sample.columns)})"},
            {"metric": "unique_GAME_ID", "value": unique_games},
            {
                "metric": "merge_logic",
                "value": "events + shots by GAME_ID/EVENTNUM, then movement by GAME_ID/EVENTNUM",
            },
            {"metric": "merge_reports", "value": str(report_dir)},
        ]
    )
    st.subheader("Merged output")
    st.dataframe(merged_summary, use_container_width=True, height=210)

    cols = st.columns(4)
    cols[0].metric("Rows", f"{row_count:,}".replace(",", " "))
    cols[1].metric("Columns", len(sample.columns))
    cols[2].metric("Games", int(unique_games))
    cols[3].metric("Movement columns", len(movement_cols))

    st.subheader("head() with all columns")
    head_rows = st.slider(
        "Rows to show from data/nba/nba_merged_400.csv",
        5,
        200,
        30,
        key="nba_merge_movement_head_rows",
    )
    head = read_csv_head(str(nba_path), head_rows, stamp)
    st.dataframe(head, use_container_width=True, height=620)

    st.subheader("Compact preview: key columns")
    key_cols = [
        "GAME_ID",
        "EVENTNUM",
        "EVENTMSGTYPE",
        "PERIOD_event",
        "PCTIMESTRING",
        "HOMEDESCRIPTION",
        "VISITORDESCRIPTION",
        "SCORE",
        "ACTION_TYPE",
        "SHOT_TYPE",
        "SHOT_MADE_FLAG",
        "movement_moments_total",
        "game_clock_start",
        "shot_clock_start",
        "ball_hoop_dist",
        "players_near_hoop",
        "intensity",
    ]
    st.dataframe(
        head[[c for c in key_cols if c in head.columns]], use_container_width=True, height=420
    )

    st.subheader("Merged columns: data types and quality")
    quality = direct_column_quality(sample)
    st.dataframe(quality, use_container_width=True, height=420)

    st.subheader("Top missing columns")
    top_missing = quality.sort_values("null_percent", ascending=False).head(40)
    st.dataframe(top_missing, use_container_width=True, height=320)
    if not top_missing.empty:
        plot_df = top_missing.sort_values("null_percent", ascending=True).tail(30)
        fig = px.bar(
            plot_df,
            x="null_percent",
            y="column",
            orientation="h",
            title="NBA merged top missing columns",
            text=plot_df["null_percent"].map(lambda x: f"{x:.1%}"),
        )
        fig.update_layout(height=max(520, 20 * len(plot_df)), xaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)


def show_nba_merge_processing() -> None:
    st.header("NBA Merge Processing")
    st.caption("Preprocessing result for `data/nba/nba_merged_preprocessed_400.csv`.")

    processed_path = PROJECT_ROOT / "data" / "nba" / "nba_merged_preprocessed_400.csv"
    stamp = file_signature(processed_path)
    if not processed_path.exists():
        st.info(
            "NBA 400 preprocessed dataset is not ready. Run "
            "`uv run python scripts/nba/run_nba_preprocessing.py`."
        )
        return

    report_base = REPORTS_DIR / "nba_preprocessing_400"
    summary = read_csv(
        str(report_base / "nba_preprocessing_summary.csv"),
        file_signature(report_base / "nba_preprocessing_summary.csv"),
    )
    before_quality = read_csv(
        str(report_base / "nba_preprocessing_before_quality.csv"),
        file_signature(report_base / "nba_preprocessing_before_quality.csv"),
    )
    after_quality = read_csv(
        str(report_base / "nba_preprocessing_after_quality.csv"),
        file_signature(report_base / "nba_preprocessing_after_quality.csv"),
    )
    top_before = read_csv(
        str(report_base / "nba_preprocessing_top_missing_before.csv"),
        file_signature(report_base / "nba_preprocessing_top_missing_before.csv"),
    )
    top_after = read_csv(
        str(report_base / "nba_preprocessing_top_missing_after.csv"),
        file_signature(report_base / "nba_preprocessing_top_missing_after.csv"),
    )
    created = read_csv(
        str(report_base / "nba_preprocessing_created_features.csv"),
        file_signature(report_base / "nba_preprocessing_created_features.csv"),
    )
    dropped = read_csv(
        str(report_base / "nba_preprocessing_dropped_columns.csv"),
        file_signature(report_base / "nba_preprocessing_dropped_columns.csv"),
    )
    metadata = read_csv(
        str(report_base / "nba_preprocessing_metadata_columns.csv"),
        file_signature(report_base / "nba_preprocessing_metadata_columns.csv"),
    )
    checks = read_csv(
        str(report_base / "nba_preprocessing_quality_checks.csv"),
        file_signature(report_base / "nba_preprocessing_quality_checks.csv"),
    )
    rows_stats = read_csv(
        str(report_base / "nba_preprocessing_rows_per_game_stats.csv"),
        file_signature(report_base / "nba_preprocessing_rows_per_game_stats.csv"),
    )
    duplicates_before = read_csv(
        str(report_base / "nba_preprocessing_duplicate_diagnostics_before.csv"),
        file_signature(report_base / "nba_preprocessing_duplicate_diagnostics_before.csv"),
    )
    duplicates_after = read_csv(
        str(report_base / "nba_preprocessing_duplicate_diagnostics_after.csv"),
        file_signature(report_base / "nba_preprocessing_duplicate_diagnostics_after.csv"),
    )
    score_checks = read_csv(
        str(report_base / "nba_preprocessing_score_monotonicity_checks.csv"),
        file_signature(report_base / "nba_preprocessing_score_monotonicity_checks.csv"),
    )
    column_roles = read_csv(
        str(report_base / "nba_preprocessing_column_roles.csv"),
        file_signature(report_base / "nba_preprocessing_column_roles.csv"),
    )

    st.subheader("Processing summary")
    st.dataframe(summary, use_container_width=True)

    head_rows = st.slider(
        "Rows to show from data/nba/nba_merged_preprocessed_400.csv",
        5,
        200,
        30,
        key="nba_preprocessing_head_rows",
    )
    head = read_csv_head(str(processed_path), head_rows, stamp)
    cols = st.columns(4)
    cols[0].metric("Rows shown", len(head))
    cols[1].metric("Columns", len(head.columns))
    cols[2].metric("Games in preview", int(head["GAME_ID"].nunique()) if "GAME_ID" in head else 0)
    cols[3].metric("Created features", len(created))

    st.subheader("Preprocessed head() with all columns")
    st.dataframe(head, use_container_width=True, height=620)

    processed = read_csv(str(processed_path), stamp)

    st.subheader("Score and duplicate diagnostics")
    st.caption(
        "Score features are forward-filled inside each GAME_ID. Remaining NaN values in "
        "PLAYER2/PLAYER3 descriptions, SCORE/SCOREMARGIN and shot-only metadata are structural "
        "missing values and are not model features."
    )
    diag_a, diag_b, diag_c = st.columns(3)
    with diag_a:
        st.markdown("**Duplicate keys before**")
        st.dataframe(duplicates_before, use_container_width=True, height=220)
    with diag_b:
        st.markdown("**Duplicate keys after**")
        st.dataframe(duplicates_after, use_container_width=True, height=220)
    with diag_c:
        st.markdown("**Score monotonicity**")
        st.dataframe(score_checks, use_container_width=True, height=220)

    st.subheader("Compact model-ready preview")
    preview_cols = [
        "GAME_ID",
        "EVENTNUM",
        "EVENTMSGTYPE",
        "PERIOD_event",
        "home_score_current",
        "away_score_current",
        "score_diff_home_current",
        "is_shot",
        "is_made_shot",
        "is_missed_shot",
        "is_turnover",
        "is_foul",
        "SHOT_DISTANCE",
        "SHOT_TYPE",
        "ball_hoop_dist",
        "players_near_hoop",
        "intensity",
        "movement_missing",
        "game_seconds_remaining",
        "is_fourth_quarter",
        "is_clutch_time",
    ]
    st.dataframe(
        head[[c for c in preview_cols if c in head.columns]], use_container_width=True, height=420
    )

    st.subheader("Created / dropped / metadata columns")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("**Created features**")
        st.dataframe(created, use_container_width=True, height=320)
    with col_b:
        st.markdown("**Dropped columns**")
        st.dataframe(dropped, use_container_width=True, height=320)
    with col_c:
        st.markdown("**Metadata columns**")
        st.dataframe(metadata, use_container_width=True, height=320)

    st.subheader("Column roles")
    st.dataframe(column_roles, use_container_width=True, height=420)

    st.subheader("Quality checks")
    st.dataframe(checks, use_container_width=True, height=360)
    if not checks.empty and "status" in checks:
        fig = px.bar(
            checks,
            x="status",
            y="check",
            orientation="h",
            title="NBA preprocessing quality checks",
        )
        fig.update_layout(height=max(520, 20 * len(checks)))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("NBA preprocessing visual diagnostics")
    chart_a, chart_b = st.columns(2)
    with chart_a:
        if not processed.empty and "EVENTMSGTYPE" in processed:
            event_counts = (
                processed["EVENTMSGTYPE"]
                .value_counts(dropna=False)
                .rename_axis("EVENTMSGTYPE")
                .reset_index(name="count")
                .sort_values("EVENTMSGTYPE")
            )
            fig = px.bar(
                event_counts,
                x="EVENTMSGTYPE",
                y="count",
                title="Event type distribution after preprocessing",
            )
            st.plotly_chart(fig, use_container_width=True)
    with chart_b:
        shot_cols = [c for c in ["is_shot", "is_made_shot", "is_missed_shot"] if c in processed]
        if shot_cols:
            shot_counts = processed[shot_cols].sum().reset_index()
            shot_counts.columns = ["feature", "count"]
            fig = px.bar(
                shot_counts,
                x="feature",
                y="count",
                title="Shot feature counts",
            )
            st.plotly_chart(fig, use_container_width=True)

    chart_c, chart_d = st.columns(2)
    with chart_c:
        if not processed.empty and "movement_missing" in processed:
            movement_counts = (
                processed["movement_missing"]
                .value_counts(dropna=False)
                .rename_axis("movement_missing")
                .reset_index(name="count")
                .sort_values("movement_missing")
            )
            fig = px.bar(
                movement_counts,
                x="movement_missing",
                y="count",
                title="Movement availability flag",
            )
            st.plotly_chart(fig, use_container_width=True)
    with chart_d:
        if not processed.empty and "score_diff_home_current" in processed:
            fig = px.histogram(
                processed,
                x="score_diff_home_current",
                nbins=40,
                title="Current home score margin distribution",
            )
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Rows per game statistics")
    st.dataframe(rows_stats, use_container_width=True)

    st.subheader("Column quality after preprocessing")
    st.dataframe(after_quality, use_container_width=True, height=420)

    st.subheader("Top missing columns before / after")
    col_before, col_after = st.columns(2)
    with col_before:
        st.markdown("**Before**")
        st.dataframe(top_before, use_container_width=True, height=360)
    with col_after:
        st.markdown("**After**")
        st.dataframe(top_after, use_container_width=True, height=360)

    if not top_after.empty:
        plot_df = top_after.sort_values("null_percent", ascending=True).tail(30)
        fig = px.bar(
            plot_df,
            x="null_percent",
            y="column",
            orientation="h",
            title="NBA preprocessing: top missing columns after cleanup",
            text=plot_df["null_percent"].map(lambda x: f"{x:.1%}"),
        )
        fig.update_layout(height=max(520, 20 * len(plot_df)), xaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("Column quality before preprocessing"):
        st.dataframe(before_quality, use_container_width=True, height=420)


def show_nba_merge_feature_engineering() -> None:
    st.header("NBA Merge Feature Engineering")
    st.caption(
        "Clutch-time regression features plus historical team-strength features for "
        "`data/nba/nba_feature_engineering_400_enhanced.csv`."
    )

    feature_path = PROJECT_ROOT / "data" / "nba" / "nba_feature_engineering_400_enhanced.csv"
    base_feature_path = PROJECT_ROOT / "data" / "nba" / "nba_feature_engineering_400.csv"
    if not feature_path.exists() and base_feature_path.exists():
        feature_path = base_feature_path
    stamp = file_signature(feature_path)
    if not feature_path.exists():
        st.info(
            "NBA feature engineering dataset is not ready. Run "
            "`uv run python scripts/nba/run_nba_feature_engineering.py`, then "
            "`uv run python scripts/nba/add_historical_team_features.py`."
        )
        return

    report_base = REPORTS_DIR / "nba_feature_engineering_400"
    created = read_csv(
        str(report_base / "nba_feature_engineering_created_features.csv"),
        file_signature(report_base / "nba_feature_engineering_created_features.csv"),
    )
    target_distribution = read_csv(
        str(report_base / "nba_feature_engineering_target_distribution.csv"),
        file_signature(report_base / "nba_feature_engineering_target_distribution.csv"),
    )
    target_by_game = read_csv(
        str(report_base / "nba_feature_engineering_target_by_game.csv"),
        file_signature(report_base / "nba_feature_engineering_target_by_game.csv"),
    )
    correlations = read_csv(
        str(report_base / "nba_feature_engineering_correlations.csv"),
        file_signature(report_base / "nba_feature_engineering_correlations.csv"),
    )
    rows_per_game = read_csv(
        str(report_base / "nba_feature_engineering_rows_per_game.csv"),
        file_signature(report_base / "nba_feature_engineering_rows_per_game.csv"),
    )
    sequence_stats = read_csv(
        str(report_base / "nba_feature_engineering_sequence_statistics.csv"),
        file_signature(report_base / "nba_feature_engineering_sequence_statistics.csv"),
    )
    top_missing = read_csv(
        str(report_base / "nba_feature_engineering_top_missing_columns.csv"),
        file_signature(report_base / "nba_feature_engineering_top_missing_columns.csv"),
    )
    quality = read_csv(
        str(report_base / "nba_feature_engineering_feature_quality.csv"),
        file_signature(report_base / "nba_feature_engineering_feature_quality.csv"),
    )
    final_checks = read_csv(
        str(REPORTS_DIR / "nba_preprocessing_400_final_fix_checks.csv"),
        file_signature(REPORTS_DIR / "nba_preprocessing_400_final_fix_checks.csv"),
    )
    final_numeric_nan_before = read_csv(
        str(REPORTS_DIR / "nba_preprocessing_400_final_numeric_nan_before.csv"),
        file_signature(REPORTS_DIR / "nba_preprocessing_400_final_numeric_nan_before.csv"),
    )
    final_numeric_nan_after = read_csv(
        str(REPORTS_DIR / "nba_preprocessing_400_final_numeric_nan_after.csv"),
        file_signature(REPORTS_DIR / "nba_preprocessing_400_final_numeric_nan_after.csv"),
    )
    rf_importance = read_csv(
        str(report_base / "nba_feature_engineering_rf_importance.csv"),
        file_signature(report_base / "nba_feature_engineering_rf_importance.csv"),
    )
    top_30 = read_csv(
        str(report_base / "nba_feature_engineering_top_30_features.csv"),
        file_signature(report_base / "nba_feature_engineering_top_30_features.csv"),
    )
    top_50 = read_csv(
        str(report_base / "nba_feature_engineering_top_50_features.csv"),
        file_signature(report_base / "nba_feature_engineering_top_50_features.csv"),
    )
    top_75 = read_csv(
        str(report_base / "nba_feature_engineering_top_75_features.csv"),
        file_signature(report_base / "nba_feature_engineering_top_75_features.csv"),
    )
    recommended_drop = read_csv(
        str(report_base / "nba_feature_engineering_recommended_drop_features.csv"),
        file_signature(report_base / "nba_feature_engineering_recommended_drop_features.csv"),
    )
    constant_features = read_csv(
        str(report_base / "nba_feature_engineering_constant_features.csv"),
        file_signature(report_base / "nba_feature_engineering_constant_features.csv"),
    )
    near_constant_features = read_csv(
        str(report_base / "nba_feature_engineering_near_constant_features.csv"),
        file_signature(report_base / "nba_feature_engineering_near_constant_features.csv"),
    )
    highly_correlated = read_csv(
        str(report_base / "nba_feature_engineering_highly_correlated_features.csv"),
        file_signature(report_base / "nba_feature_engineering_highly_correlated_features.csv"),
    )
    historical_created = read_csv(
        str(REPORTS_DIR / "nba_historical_team_features_created.csv"),
        file_signature(REPORTS_DIR / "nba_historical_team_features_created.csv"),
    )
    historical_validation = read_csv(
        str(REPORTS_DIR / "nba_historical_team_features_leakage_validation.csv"),
        file_signature(REPORTS_DIR / "nba_historical_team_features_leakage_validation.csv"),
    )
    historical_match_level = read_csv(
        str(REPORTS_DIR / "nba_historical_team_features_match_level.csv"),
        file_signature(REPORTS_DIR / "nba_historical_team_features_match_level.csv"),
    )
    enhanced_correlations = read_csv(
        str(REPORTS_DIR / "nba_enhanced_correlations.csv"),
        file_signature(REPORTS_DIR / "nba_enhanced_correlations.csv"),
    )
    enhanced_rf_importance = read_csv(
        str(REPORTS_DIR / "nba_enhanced_rf_importance.csv"),
        file_signature(REPORTS_DIR / "nba_enhanced_rf_importance.csv"),
    )
    enhanced_quality = read_csv(
        str(REPORTS_DIR / "nba_enhanced_feature_quality.csv"),
        file_signature(REPORTS_DIR / "nba_enhanced_feature_quality.csv"),
    )
    top_30_enhanced = read_csv(
        str(REPORTS_DIR / "nba_top30_features.csv"),
        file_signature(REPORTS_DIR / "nba_top30_features.csv"),
    )
    top_50_enhanced = read_csv(
        str(REPORTS_DIR / "nba_top50_features.csv"),
        file_signature(REPORTS_DIR / "nba_top50_features.csv"),
    )
    top_75_enhanced = read_csv(
        str(REPORTS_DIR / "nba_top75_features.csv"),
        file_signature(REPORTS_DIR / "nba_top75_features.csv"),
    )
    top_100_enhanced = read_csv(
        str(REPORTS_DIR / "nba_top100_features.csv"),
        file_signature(REPORTS_DIR / "nba_top100_features.csv"),
    )
    recommended_drop_enhanced = read_csv(
        str(REPORTS_DIR / "nba_recommended_drop_features.csv"),
        file_signature(REPORTS_DIR / "nba_recommended_drop_features.csv"),
    )

    head_rows = st.slider(
        f"Rows to show from {feature_path.relative_to(PROJECT_ROOT)}",
        5,
        200,
        30,
        key="nba_feature_engineering_head_rows",
    )
    head = read_csv_head(str(feature_path), head_rows, stamp)
    cols = st.columns(4)
    cols[0].metric("Rows shown", len(head))
    cols[1].metric("Columns", len(head.columns))
    cols[2].metric("Games in preview", int(head["GAME_ID"].nunique()) if "GAME_ID" in head else 0)
    cols[3].metric("Created features", len(created))

    if feature_path.name.endswith("_enhanced.csv"):
        st.subheader("Historical team-strength features")
        hist_cols = st.columns(4)
        hist_cols[0].metric("Final feature count", len(head.columns))
        hist_cols[1].metric("Historical features", len(historical_created))
        hist_cols[2].metric(
            "Match-level games",
            int(historical_match_level["GAME_ID"].nunique())
            if "GAME_ID" in historical_match_level
            else 0,
        )
        hist_cols[3].metric(
            "Leakage checks passed",
            int(historical_validation["status"].sum()) if "status" in historical_validation else 0,
        )
        st.caption("Timeline source: GAME_ID. GAME_DATE is not used because it is invalid.")
        h_a, h_b = st.columns(2)
        with h_a:
            st.markdown("**Historical feature summary**")
            st.dataframe(historical_created, use_container_width=True, height=320)
        with h_b:
            st.markdown("**Leakage validation**")
            st.dataframe(historical_validation, use_container_width=True, height=320)

    st.subheader("Feature-engineered head() with all columns")
    st.dataframe(head, use_container_width=True, height=620)

    st.subheader("Preprocessing final checks")
    check_a, check_b, check_c = st.columns(3)
    with check_a:
        st.markdown("**Final checks**")
        st.dataframe(final_checks, use_container_width=True, height=240)
    with check_b:
        st.markdown("**Numeric NaN before final fix**")
        st.dataframe(final_numeric_nan_before, use_container_width=True, height=240)
    with check_c:
        st.markdown("**Numeric NaN after final fix**")
        st.dataframe(final_numeric_nan_after, use_container_width=True, height=240)

    st.subheader("Target distribution")
    dist_a, dist_b = st.columns(2)
    with dist_a:
        st.dataframe(target_distribution, use_container_width=True, height=260)
    with dist_b:
        if not target_by_game.empty and "target_score_diff_change_last_5min" in target_by_game:
            fig = px.histogram(
                target_by_game,
                x="target_score_diff_change_last_5min",
                nbins=25,
                title="Target: score diff change in last 5 minutes",
            )
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Feature correlation with target")
    active_correlations = enhanced_correlations if not enhanced_correlations.empty else correlations
    st.dataframe(active_correlations.head(50), use_container_width=True, height=420)
    if not active_correlations.empty:
        top_corr = active_correlations.head(25).sort_values("abs_correlation", ascending=True)
        fig = px.bar(
            top_corr,
            x="abs_correlation",
            y="feature",
            orientation="h",
            title="Top feature correlations with target",
            hover_data=["correlation_with_target"],
        )
        fig.update_layout(height=620)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("RandomForestRegressor feature importance")
    active_rf_importance = (
        enhanced_rf_importance if not enhanced_rf_importance.empty else rf_importance
    )
    st.dataframe(active_rf_importance.head(75), use_container_width=True, height=420)
    if not active_rf_importance.empty:
        top_rf = active_rf_importance.head(30).sort_values("rf_importance", ascending=True)
        fig = px.bar(
            top_rf,
            x="rf_importance",
            y="feature",
            orientation="h",
            title="Top RF feature importance",
        )
        fig.update_layout(height=680)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Feature rankings")
    active_top_30 = top_30_enhanced if not top_30_enhanced.empty else top_30
    active_top_50 = top_50_enhanced if not top_50_enhanced.empty else top_50
    active_top_75 = top_75_enhanced if not top_75_enhanced.empty else top_75
    active_recommended_drop = (
        recommended_drop_enhanced if not recommended_drop_enhanced.empty else recommended_drop
    )
    rank_a, rank_b, rank_c, rank_d = st.columns(4)
    with rank_a:
        st.markdown("**Top-30 features**")
        st.dataframe(active_top_30, use_container_width=True, height=420)
    with rank_b:
        st.markdown("**Top-50 features**")
        st.dataframe(active_top_50, use_container_width=True, height=420)
    with rank_c:
        st.markdown("**Top-75 features**")
        st.dataframe(active_top_75, use_container_width=True, height=420)
    with rank_d:
        st.markdown("**Top-100 features**")
        st.dataframe(top_100_enhanced, use_container_width=True, height=420)

    st.subheader("Features recommended to drop")
    st.dataframe(active_recommended_drop, use_container_width=True, height=360)
    audit_a, audit_b, audit_c = st.columns(3)
    with audit_a:
        st.markdown("**Constant features**")
        st.dataframe(constant_features, use_container_width=True, height=300)
    with audit_b:
        st.markdown("**Near-constant features**")
        st.dataframe(near_constant_features, use_container_width=True, height=300)
    with audit_c:
        st.markdown("**Highly correlated pairs**")
        st.dataframe(highly_correlated.head(100), use_container_width=True, height=300)

    st.subheader("Created features")
    st.dataframe(created, use_container_width=True, height=360)

    st.subheader("Rows per game / sequence statistics")
    stat_a, stat_b = st.columns(2)
    with stat_a:
        st.dataframe(rows_per_game, use_container_width=True)
    with stat_b:
        st.dataframe(sequence_stats, use_container_width=True)

    st.subheader("Top missing columns")
    st.caption(
        "Most remaining missing values are structural metadata fields: optional players, "
        "description text, SCORE/SCOREMARGIN rows, and shot-only metadata."
    )
    st.dataframe(top_missing, use_container_width=True, height=360)
    if not top_missing.empty:
        plot_df = top_missing.sort_values("null_percent", ascending=True).tail(30)
        fig = px.bar(
            plot_df,
            x="null_percent",
            y="column",
            orientation="h",
            title="NBA feature engineering: top missing columns",
            text=plot_df["null_percent"].map(lambda x: f"{x:.1%}"),
        )
        fig.update_layout(height=max(520, 20 * len(plot_df)), xaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Feature quality table")
    st.dataframe(
        enhanced_quality if not enhanced_quality.empty else quality,
        use_container_width=True,
        height=460,
    )


def show_football_metrics() -> None:
    st.header("Football Metrics")
    st.caption("Sequence dataset diagnostics and baseline multi-output LSTM metrics.")

    diagnostics = load_report_table("football_sequence_diagnostics.csv")
    target_dist = load_report_table("football_sequence_target_distribution.csv")
    seq_stats = load_report_table("football_sequence_sequence_length_stats.csv")
    features = load_report_table("football_sequence_feature_columns.csv")
    excluded = load_report_table("football_sequence_excluded_columns.csv")
    target_checks = load_report_table("football_sequence_target_checks.csv")
    lstm_metrics = load_football_metric_table("baseline_lstm_metrics.csv")
    lstm_history = load_football_metric_table("baseline_lstm_history.csv")
    lstm_shapes = load_football_metric_table("baseline_lstm_shapes.csv")
    overfit = load_football_metric_table("baseline_lstm_overfitting_report.csv")
    confusion = load_football_metric_table("baseline_lstm_confusion_matrices.csv")
    top50_retrain_confusion = load_football_metric_table("top50_retrain_confusion_matrices.csv")
    ablation_comparison = load_football_metric_table("feature_ablation_fast_comparison.csv")
    ablation_summary = load_football_metric_table("feature_ablation_fast_training_summary.csv")
    ablation_ranking = load_football_metric_table("feature_ablation_fast_feature_ranking.csv")
    if ablation_comparison.empty:
        ablation_comparison = load_football_metric_table("feature_ablation_comparison.csv")
    if ablation_summary.empty:
        ablation_summary = load_football_metric_table("feature_ablation_training_summary.csv")
    if ablation_ranking.empty:
        ablation_ranking = load_football_metric_table("feature_ablation_feature_ranking.csv")

    if not lstm_metrics.empty:
        st.subheader("Baseline multi-output LSTM metrics")
        st.dataframe(lstm_metrics, use_container_width=True, height=420)
        metric = st.selectbox(
            "Metric",
            [
                c
                for c in ["pr_auc", "roc_auc", "f1", "precision", "recall", "brier", "mae", "mse"]
                if c in lstm_metrics
            ],
            key="football_lstm_metric_select",
        )
        fig = px.bar(
            lstm_metrics.sort_values(metric, ascending=False),
            x=metric,
            y="target",
            color="split",
            barmode="group",
            orientation="h",
            title=f"Baseline Football LSTM by {metric}",
        )
        st.plotly_chart(fig, use_container_width=True)

    if not lstm_history.empty:
        st.subheader("Training history")
        st.dataframe(lstm_history, use_container_width=True, height=320)
        loss_cols = [
            c
            for c in ["loss", "val_loss", "home_output_loss", "away_output_loss"]
            if c in lstm_history
        ]
        hist_long = lstm_history.melt(
            id_vars="epoch", value_vars=loss_cols, var_name="curve", value_name="value"
        )
        fig = px.line(
            hist_long, x="epoch", y="value", color="curve", title="Train / Validation Loss"
        )
        st.plotly_chart(fig, use_container_width=True)

    display_confusion = top50_retrain_confusion if not top50_retrain_confusion.empty else confusion
    if not display_confusion.empty:
        st.subheader("Confusion matrices")
        st.caption(
            "Showing top-50 retrain with final fixed thresholds when available; otherwise baseline."
        )
        st.dataframe(display_confusion, use_container_width=True, height=260)
        split_values = sorted(display_confusion["split"].unique().tolist())
        split = st.selectbox(
            "Confusion matrix split",
            split_values,
            index=split_values.index("test") if "test" in split_values else 0,
            key="football_confusion_split",
        )
        cols = st.columns(2)
        for idx, target in enumerate(["home_scores_next_half", "away_scores_next_half"]):
            matrix_df = display_confusion[
                display_confusion["split"].eq(split) & display_confusion["target"].eq(target)
            ].pivot(index="true_label", columns="predicted_label", values="count")
            matrix_df = matrix_df.reindex(index=[0, 1], columns=[0, 1]).fillna(0).astype(int)
            with cols[idx]:
                st.markdown(f"**{split}: {target}**")
                st.dataframe(matrix_df, use_container_width=True)
                fig = px.imshow(
                    matrix_df,
                    text_auto=True,
                    color_continuous_scale="Blues",
                    labels={"x": "Predicted", "y": "True", "color": "Count"},
                    title=f"Confusion matrix: {target}",
                )
                st.plotly_chart(fig, use_container_width=True)

    if not ablation_comparison.empty:
        st.subheader("Feature-count ablation")
        st.dataframe(ablation_comparison, use_container_width=True, height=420)
        metric = st.selectbox(
            "Ablation metric",
            [c for c in ["pr_auc", "roc_auc", "f1", "mae", "mse"] if c in ablation_comparison],
            key="football_ablation_metric_select",
        )
        fig = px.bar(
            ablation_comparison.sort_values(metric, ascending=False),
            x=metric,
            y="feature_set",
            color="target",
            barmode="group",
            orientation="h",
            title=f"Feature-count ablation by {metric}",
        )
        st.plotly_chart(fig, use_container_width=True)

    if not ablation_summary.empty:
        st.subheader("Feature-count ablation training summary")
        st.dataframe(ablation_summary, use_container_width=True)

    if not ablation_ranking.empty:
        with st.expander("Feature ranking by aggregate absolute correlation"):
            st.dataframe(ablation_ranking, use_container_width=True, height=520)

    top50_retrain_metrics = load_football_metric_table("top50_retrain_metrics.csv")
    top50_retrain_final = load_football_metric_table("top50_retrain_final_threshold_metrics.csv")
    top50_retrain_summary = load_football_metric_table("top50_retrain_training_summary.csv")
    top50_retrain_history = load_football_metric_table("top50_retrain_top_50_history.csv")
    if not top50_retrain_metrics.empty:
        st.subheader("Top-50 LSTM retrain")
        st.caption("Fresh retrain of the selected top-50 feature model.")
        st.dataframe(top50_retrain_metrics, use_container_width=True, height=320)
    if not top50_retrain_final.empty:
        st.markdown("**Top-50 retrain with final fixed thresholds**")
        st.dataframe(top50_retrain_final, use_container_width=True, height=320)
    if not top50_retrain_summary.empty:
        st.dataframe(top50_retrain_summary, use_container_width=True)
    if not top50_retrain_history.empty:
        loss_cols = [c for c in ["loss", "val_loss"] if c in top50_retrain_history]
        if loss_cols:
            hist_long = top50_retrain_history.melt(
                id_vars="epoch", value_vars=loss_cols, var_name="curve", value_name="value"
            )
            fig = px.line(
                hist_long,
                x="epoch",
                y="value",
                color="curve",
                title="Top-50 retrain loss",
            )
            st.plotly_chart(fig, use_container_width=True)
    if not top50_retrain_confusion.empty:
        st.markdown("**Top-50 retrain confusion matrices with final fixed thresholds**")
        split_values = sorted(top50_retrain_confusion["split"].unique().tolist())
        split = st.selectbox(
            "Top-50 retrain confusion split",
            split_values,
            index=split_values.index("test") if "test" in split_values else 0,
            key="top50_retrain_confusion_split",
        )
        cols = st.columns(2)
        for idx, target in enumerate(["home_scores_next_half", "away_scores_next_half"]):
            matrix_df = top50_retrain_confusion[
                top50_retrain_confusion["split"].eq(split)
                & top50_retrain_confusion["target"].eq(target)
            ].pivot(index="true_label", columns="predicted_label", values="count")
            matrix_df = matrix_df.reindex(index=[0, 1], columns=[0, 1]).fillna(0).astype(int)
            with cols[idx]:
                st.markdown(f"**{split}: {target}**")
                st.dataframe(matrix_df, use_container_width=True)
                fig = px.imshow(
                    matrix_df,
                    text_auto=True,
                    color_continuous_scale="Blues",
                    labels={"x": "Predicted", "y": "True", "color": "Count"},
                    title=f"Top-50 retrain confusion matrix: {target}",
                )
                st.plotly_chart(fig, use_container_width=True)

    threshold_best = load_football_metric_table("threshold_tuning/best_thresholds.csv")
    threshold_metrics = load_football_metric_table("threshold_tuning/threshold_metrics.csv")
    threshold_comparison = load_football_metric_table(
        "threshold_tuning/threshold_0_5_vs_tuned_comparison.csv"
    )
    if not threshold_best.empty:
        st.subheader("Top-50 threshold tuning")
        st.caption("Best thresholds are selected on validation by F1 and then applied to test.")
        st.dataframe(threshold_best, use_container_width=True)
    if not threshold_metrics.empty:
        st.dataframe(threshold_metrics, use_container_width=True, height=320)
    if not threshold_comparison.empty:
        st.dataframe(threshold_comparison, use_container_width=True, height=240)
        fig = px.bar(
            threshold_comparison,
            x="delta",
            y="metric",
            color="target",
            barmode="group",
            orientation="h",
            title="Test metric delta: tuned threshold minus 0.5",
        )
        st.plotly_chart(fig, use_container_width=True)
    threshold_figures_dir = PROJECT_ROOT / "outputs" / "figures" / "football" / "threshold_tuning"
    threshold_figures = sorted(threshold_figures_dir.glob("threshold_curves_*.png"))
    if threshold_figures:
        cols = st.columns(2)
        for idx, fig_path in enumerate(threshold_figures):
            with cols[idx % 2]:
                st.image(str(fig_path), caption=fig_path.name, use_container_width=True)

    balanced_best = load_football_metric_table("threshold_tuning_balanced/balanced_thresholds.csv")
    balanced_metrics = load_football_metric_table(
        "threshold_tuning_balanced/balanced_threshold_metrics.csv"
    )
    balanced_comparison = load_football_metric_table(
        "threshold_tuning_balanced/balanced_threshold_comparison.csv"
    )
    if not balanced_best.empty:
        st.subheader("Balanced top-50 threshold tuning")
        st.caption(
            "Validation F1 is optimized under precision constraints: home >= 0.60, away >= 0.55."
        )
        st.dataframe(balanced_best, use_container_width=True)
    if not balanced_metrics.empty:
        st.dataframe(balanced_metrics, use_container_width=True, height=320)
    if not balanced_comparison.empty:
        st.dataframe(balanced_comparison, use_container_width=True, height=260)
        fig = px.bar(
            balanced_comparison,
            x="balanced_minus_0_5",
            y="metric",
            color="target",
            barmode="group",
            orientation="h",
            title="Balanced threshold test delta vs 0.5",
        )
        st.plotly_chart(fig, use_container_width=True)
    balanced_figures_dir = (
        PROJECT_ROOT / "outputs" / "figures" / "football" / "threshold_tuning_balanced"
    )
    balanced_figures = sorted(balanced_figures_dir.glob("balanced_threshold_curves_*.png"))
    if balanced_figures:
        cols = st.columns(2)
        for idx, fig_path in enumerate(balanced_figures):
            with cols[idx % 2]:
                st.image(str(fig_path), caption=fig_path.name, use_container_width=True)

    final_thresholds = load_football_metric_table(
        "threshold_tuning_final/final_fixed_thresholds.csv"
    )
    final_metrics = load_football_metric_table(
        "threshold_tuning_final/final_fixed_threshold_metrics.csv"
    )
    final_comparison = load_football_metric_table(
        "threshold_tuning_final/final_fixed_threshold_comparison.csv"
    )
    calibration_metrics = load_football_metric_table("calibration/calibration_metrics.csv")
    calibration_comparison = load_football_metric_table("calibration/calibration_comparison.csv")
    calibration_diagnostics = load_football_metric_table("calibration/calibration_diagnostics.csv")
    if not final_thresholds.empty:
        st.subheader("Final fixed thresholds")
        st.dataframe(final_thresholds, use_container_width=True)
    if not final_metrics.empty:
        st.dataframe(final_metrics, use_container_width=True, height=280)
    if not final_comparison.empty:
        st.dataframe(final_comparison, use_container_width=True, height=260)

    if not calibration_metrics.empty:
        st.subheader("Top-50 probability calibration")
        st.caption(
            "Calibrators are fitted on validation predictions only; test is used only for final evaluation."
        )
        if not calibration_diagnostics.empty:
            st.dataframe(calibration_diagnostics, use_container_width=True)
        st.dataframe(calibration_metrics, use_container_width=True, height=320)
        metric = st.selectbox(
            "Calibration metric",
            [
                c
                for c in [
                    "brier",
                    "log_loss",
                    "roc_auc",
                    "pr_auc",
                    "accuracy",
                    "precision",
                    "recall",
                    "f1",
                ]
                if c in calibration_metrics
            ],
            key="football_calibration_metric_select",
        )
        fig = px.bar(
            calibration_metrics,
            x="calibration_method",
            y=metric,
            color="target",
            barmode="group",
            title=f"Raw vs calibrated probabilities by {metric}",
        )
        st.plotly_chart(fig, use_container_width=True)
    if not calibration_comparison.empty:
        st.markdown("**Calibration delta vs raw**")
        st.dataframe(calibration_comparison, use_container_width=True, height=320)
        main_calib = calibration_comparison[
            calibration_comparison["metric"].isin(["brier", "log_loss"])
        ].copy()
        if not main_calib.empty:
            fig = px.bar(
                main_calib,
                x="delta_calibrated_minus_raw",
                y="metric",
                color="calibrated_method",
                facet_col="target",
                barmode="group",
                orientation="h",
                title="Calibration delta for Brier/log_loss. Lower is better, so negative is improvement.",
            )
            st.plotly_chart(fig, use_container_width=True)
    calibration_figures_dir = PROJECT_ROOT / "outputs" / "figures" / "football" / "calibration"
    calibration_figures = sorted(calibration_figures_dir.glob("calibration_curve_*.png"))
    if calibration_figures:
        cols = st.columns(2)
        for idx, fig_path in enumerate(calibration_figures):
            with cols[idx % 2]:
                st.image(str(fig_path), caption=fig_path.name, use_container_width=True)

    figures_dir = PROJECT_ROOT / "outputs" / "figures" / "football"
    loss_fig = figures_dir / "baseline_lstm_loss_curves.png"
    if loss_fig.exists():
        st.subheader("Saved figures")
        st.image(str(loss_fig), caption="Train/validation loss curves", use_container_width=True)
        figure_files = sorted(figures_dir.glob("test_*_*.png"))
        cols = st.columns(2)
        for idx, fig_path in enumerate(figure_files):
            with cols[idx % 2]:
                st.image(str(fig_path), caption=fig_path.name, use_container_width=True)

    if not lstm_shapes.empty:
        st.subheader("LSTM tensor shapes")
        st.dataframe(lstm_shapes, use_container_width=True)

    if not overfit.empty:
        st.subheader("Overfitting analysis")
        st.dataframe(overfit, use_container_width=True)

    if diagnostics.empty:
        st.info(
            "Football sequence reports are not ready. Run `uv run python scripts/build_football_sequences.py`."
        )
        return

    st.subheader("Sequence diagnostics")
    st.dataframe(diagnostics, use_container_width=True)

    st.subheader("Target checks")
    st.dataframe(target_checks, use_container_width=True)

    st.subheader("Target distribution")
    st.dataframe(target_dist, use_container_width=True)
    if not target_dist.empty:
        fig = px.bar(
            target_dist,
            x="value",
            y="matches",
            color="target",
            facet_col="split",
            barmode="group",
            title="Football Target Distribution By Split",
            text="matches",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Sequence length stats")
    st.dataframe(seq_stats, use_container_width=True)

    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("Feature columns")
        st.dataframe(features, use_container_width=True, height=520)
    with col_right:
        st.subheader("Excluded leakage / metadata columns")
        st.dataframe(excluded, use_container_width=True, height=520)


def show_nba_metric() -> None:
    st.header("NBA Metric")
    st.caption("Feature-set LSTM regression pipeline for clutch-time score-difference change.")

    feature_set_summary = load_nba_metric_table("feature_sets_summary.csv")
    feature_set_shapes = load_nba_metric_table("feature_set_sequence_shapes.csv")
    feature_set_split = load_nba_metric_table("feature_set_split_summary.csv")
    feature_set_validation = load_nba_metric_table("feature_set_leakage_validation.csv")
    feature_set_baselines = load_nba_metric_table("feature_set_baselines.csv")
    feature_set_warnings = load_nba_metric_table("feature_set_missing_requested_features.csv")
    full_training_commands = load_nba_metric_table("feature_set_full_training_commands.csv")
    full_training_comparison = load_nba_metric_table("feature_set_full_training_comparison.csv")
    target_describe = load_nba_metric_table("nba_target_describe.csv")
    target_abs_stats = load_nba_metric_table("nba_target_abs_statistics.csv")
    target_percentages = load_nba_metric_table("nba_target_threshold_percentages.csv")
    target_mae_context = load_nba_metric_table("nba_target_mae_context.csv")
    smoke_metrics = load_nba_metric_table("smoke_test_metrics.csv")
    smoke_predictions = load_nba_metric_table("smoke_test_predictions.csv")
    smoke_history = load_nba_metric_table("smoke_test_history.csv")
    metrics = load_nba_metric_table("nba_lstm_clutch_metrics.csv")
    predictions = load_nba_metric_table("nba_lstm_clutch_predictions.csv")
    history = load_nba_metric_table("nba_lstm_clutch_history.csv")
    shapes = load_nba_metric_table("nba_lstm_clutch_shapes.csv")
    split_summary = load_nba_metric_table("nba_lstm_clutch_split_summary.csv")
    training_summary = load_nba_metric_table("nba_lstm_clutch_training_summary.csv")
    features = load_nba_metric_table("nba_lstm_clutch_feature_columns.csv")
    figures_dir = PROJECT_ROOT / "outputs" / "figures" / "nba"

    if feature_set_summary.empty and metrics.empty:
        st.info(
            "NBA LSTM metrics are not ready. Run "
            "`uv run python scripts/nba/train_nba_lstm_feature_sets.py --feature-set top50 --smoke-test`."
        )
        return

    if not feature_set_summary.empty:
        st.subheader("Feature-set training pipeline")
        split_counts = (
            feature_set_split["split"].value_counts().to_dict()
            if not feature_set_split.empty and "split" in feature_set_split
            else {}
        )
        cols = st.columns(4)
        cols[0].metric("Feature sets", len(feature_set_summary))
        cols[1].metric("Train games", int(split_counts.get("train", 0)))
        cols[2].metric("Validation games", int(split_counts.get("val", 0)))
        cols[3].metric("Test games", int(split_counts.get("test", 0)))

        st.markdown("**Feature sets summary**")
        st.dataframe(feature_set_summary, use_container_width=True)

        st.markdown("**Leakage / X validation**")
        st.dataframe(feature_set_validation, use_container_width=True)
        if not feature_set_validation.empty and "status" in feature_set_validation:
            fig = px.bar(
                feature_set_validation,
                x="status",
                y="check",
                orientation="h",
                title="NBA feature-set validation checks",
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Sequence shapes**")
        st.dataframe(feature_set_shapes, use_container_width=True, height=360)

        st.markdown("**Baselines on test**")
        st.dataframe(feature_set_baselines, use_container_width=True, height=300)

        if not target_describe.empty:
            st.subheader("Target analysis: target_score_diff_change_last_5min")
            st.caption(
                "Analysis is calculated per GAME_ID, because the target is repeated across "
                "event rows inside each match."
            )
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**target.describe()**")
                st.dataframe(target_describe, use_container_width=True, height=260)
            with c2:
                st.markdown("**abs(target) statistics**")
                st.dataframe(target_abs_stats, use_container_width=True, height=260)

            st.markdown("**Match share by absolute target size**")
            st.dataframe(target_percentages, use_container_width=True, height=220)
            if not target_percentages.empty:
                pct_plot = target_percentages.copy()
                pct_plot["percent_label"] = pct_plot["percent"] * 100
                fig = px.bar(
                    pct_plot,
                    x="segment",
                    y="percent_label",
                    text="percent_label",
                    title="Share of games by abs(target) threshold",
                )
                fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                fig.update_layout(yaxis_title="Percent of games", xaxis_title="")
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("**Best MAE compared with target scale**")
            st.dataframe(target_mae_context, use_container_width=True)
            if not target_mae_context.empty:
                row = target_mae_context.iloc[0]
                m1, m2, m3 = st.columns(3)
                m1.metric("Best MAE", f"{float(row['mae']):.3f}")
                m2.metric(
                    "MAE / mean abs(target)",
                    f"{float(row['mae_percent_of_mean_abs_target']) * 100:.1f}%",
                )
                m3.metric(
                    "MAE / median abs(target)",
                    f"{float(row['mae_percent_of_median_abs_target']) * 100:.1f}%",
                )

            target_figures = [
                figures_dir / "nba_target_histogram.png",
                figures_dir / "nba_abs_target_histogram.png",
            ]
            fig_cols = st.columns(2)
            for idx, fig_path in enumerate(target_figures):
                if fig_path.exists():
                    with fig_cols[idx % 2]:
                        st.image(str(fig_path), caption=fig_path.name, use_container_width=True)

        if not feature_set_warnings.empty:
            st.markdown("**Missing requested features**")
            st.dataframe(feature_set_warnings, use_container_width=True)

        if not smoke_metrics.empty:
            st.subheader("Smoke-test metrics: top50, 1 epoch")
            st.dataframe(smoke_metrics, use_container_width=True)
            metric = st.selectbox(
                "Smoke-test metric",
                [c for c in ["mae", "mse", "rmse", "r2"] if c in smoke_metrics],
                key="nba_smoke_metric_select",
            )
            fig = px.bar(
                smoke_metrics,
                x=metric,
                y="split",
                orientation="h",
                title=f"NBA smoke-test top50 by {metric}",
            )
            st.plotly_chart(fig, use_container_width=True)

        if not smoke_history.empty:
            st.markdown("**Smoke-test history**")
            st.dataframe(smoke_history, use_container_width=True)

        if not smoke_predictions.empty:
            st.markdown("**Smoke-test predictions**")
            st.dataframe(smoke_predictions, use_container_width=True, height=320)

        st.subheader("Full training commands")
        if not full_training_commands.empty:
            for command in full_training_commands["command"].tolist():
                st.code(command, language="powershell")
        else:
            st.code(
                "\n".join(
                    [
                        r"uv run python scripts\nba\train_nba_lstm_feature_sets.py --feature-set top30",
                        r"uv run python scripts\nba\train_nba_lstm_feature_sets.py --feature-set top50",
                        r"uv run python scripts\nba\train_nba_lstm_feature_sets.py --feature-set top75",
                        r"uv run python scripts\nba\train_nba_lstm_feature_sets.py --feature-set all_features",
                        r"uv run python scripts\nba\train_nba_lstm_feature_sets.py --feature-set all",
                    ]
                ),
                language="powershell",
            )

        st.subheader("Full-training comparison")
        if not full_training_comparison.empty:
            st.caption(
                "Full NBA LSTM regression training on top30/top50/top75/all_features. "
                "Lower MAE/MSE/RMSE is better; higher R2 is better."
            )
            st.dataframe(full_training_comparison, use_container_width=True, height=360)

            test_comparison = full_training_comparison[
                full_training_comparison["split"].eq("test")
            ].copy()
            if not test_comparison.empty:
                metric = st.selectbox(
                    "Full-training test metric",
                    [c for c in ["mae", "mse", "rmse", "r2"] if c in test_comparison],
                    key="nba_full_training_metric_select",
                )
                fig = px.bar(
                    test_comparison.sort_values(metric),
                    x="feature_set",
                    y=metric,
                    color="feature_set",
                    title=f"NBA LSTM feature-set comparison on test by {metric}",
                    text=metric,
                )
                fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
                st.plotly_chart(fig, use_container_width=True)

                best_mae = test_comparison.sort_values("mae").iloc[0]
                best_mse = test_comparison.sort_values("mse").iloc[0]
                c1, c2 = st.columns(2)
                c1.metric(
                    "Best by MAE",
                    str(best_mae["feature_set"]),
                    f"MAE {float(best_mae['mae']):.3f}",
                )
                c2.metric(
                    "Best by MSE/RMSE/R2",
                    str(best_mse["feature_set"]),
                    f"MSE {float(best_mse['mse']):.3f}",
                )
        else:
            st.caption(
                "After running full training, metrics for top30/top50/top75/all_features "
                "will be shown here."
            )

    if metrics.empty:
        return

    st.subheader("Split summary")
    split_a, split_b = st.columns(2)
    with split_a:
        st.dataframe(shapes, use_container_width=True, height=220)
    with split_b:
        if not split_summary.empty:
            split_counts = split_summary["split"].value_counts().reset_index()
            split_counts.columns = ["split", "games"]
            fig = px.bar(split_counts, x="split", y="games", title="Games by split")
            st.plotly_chart(fig, use_container_width=True)

    with st.expander("GAME_ID by split"):
        st.dataframe(split_summary, use_container_width=True, height=360)

    st.subheader("Baseline and LSTM metrics")
    st.caption(
        "Regression target: target_score_diff_change_last_5min. Lower MAE/MSE/RMSE is better; "
        "higher R2 is better."
    )
    st.dataframe(metrics, use_container_width=True, height=320)
    metric = st.selectbox(
        "NBA regression metric",
        [c for c in ["mae", "mse", "rmse", "r2"] if c in metrics],
        key="nba_lstm_metric_select",
    )
    plot_metrics = metrics[metrics["split"].eq("test")].copy()
    ascending = metric != "r2"
    fig = px.bar(
        plot_metrics.sort_values(metric, ascending=ascending),
        x=metric,
        y="model",
        orientation="h",
        title=f"NBA test models by {metric}",
    )
    st.plotly_chart(fig, use_container_width=True)

    if not training_summary.empty:
        st.subheader("Training summary")
        st.dataframe(training_summary, use_container_width=True)

    if not history.empty:
        st.subheader("Training history")
        st.dataframe(history, use_container_width=True, height=300)
        loss_cols = [
            c for c in ["loss", "val_loss", "mae", "val_mae", "mse", "val_mse"] if c in history
        ]
        hist_long = history.melt(
            id_vars="epoch", value_vars=loss_cols, var_name="curve", value_name="value"
        )
        fig = px.line(hist_long, x="epoch", y="value", color="curve", title="NBA LSTM curves")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Predictions")
    st.dataframe(predictions, use_container_width=True, height=360)
    test_predictions = predictions[predictions["split"].eq("test")]
    if not test_predictions.empty:
        col_a, col_b = st.columns(2)
        with col_a:
            fig = px.scatter(
                test_predictions,
                x="y_true",
                y="y_pred",
                hover_data=["GAME_ID", "error", "abs_error"],
                title="Prediction vs actual on test",
            )
            diagonal_min = test_predictions[["y_true", "y_pred"]].min().min()
            diagonal_max = test_predictions[["y_true", "y_pred"]].max().max()
            fig.add_shape(
                type="line",
                x0=diagonal_min,
                y0=diagonal_min,
                x1=diagonal_max,
                y1=diagonal_max,
                line={"dash": "dash", "color": "black"},
            )
            st.plotly_chart(fig, use_container_width=True)
        with col_b:
            fig = px.scatter(
                test_predictions,
                x="y_pred",
                y="error",
                hover_data=["GAME_ID", "y_true", "abs_error"],
                title="Residuals on test",
            )
            fig.add_hline(y=0, line_dash="dash", line_color="black")
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Saved figures")
    figure_files = [
        figures_dir / "nba_lstm_prediction_vs_actual.png",
        figures_dir / "nba_lstm_residuals.png",
        figures_dir / "nba_lstm_training_curves.png",
    ]
    cols = st.columns(3)
    for idx, fig_path in enumerate(figure_files):
        if fig_path.exists():
            with cols[idx % 3]:
                st.image(str(fig_path), caption=fig_path.name, use_container_width=True)

    with st.expander("Feature columns"):
        st.dataframe(features, use_container_width=True, height=520)


def refresh_audit() -> None:
    cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "common" / "audit_data_quality.py")]
    completed = subprocess.run(cmd, cwd=PROJECT_ROOT, check=False, capture_output=True, text=True)
    if completed.returncode == 0:
        st.cache_data.clear()
        st.success("РђСѓРґРёС‚ РѕР±РЅРѕРІР»РµРЅ.")
        if completed.stdout:
            st.code(completed.stdout)
    else:
        st.error("РќРµ СѓРґР°Р»РѕСЃСЊ РѕР±РЅРѕРІРёС‚СЊ Р°СѓРґРёС‚.")
        st.code(completed.stderr or completed.stdout)


def main() -> None:
    st.set_page_config(page_title="Match Dynamics Data Audit", layout="wide")
    st.title("Match Dynamics Data Audit")

    with st.sidebar:
        st.caption(f"Project: `{PROJECT_ROOT}`")
        st.caption(f"Audit files: `{AUDIT_DIR}`")
        if st.button("Refresh audit tables"):
            refresh_audit()
        st.divider()
        page = st.radio(
            "Page",
            [
                "Overview",
                "Football Merge",
                "Football Merged Processed",
                "Football Merged Feature Engineering",
                "NBA Merge",
                "NBA Merge Processing",
                "NBA Merge Feature Engineering",
                "Football Metrics",
                "NBA Metric",
                "Conclusion",
            ],
        )

    if page == "Overview":
        show_overview()
    elif page == "Football Merge":
        show_football_merge()
    elif page == "Football Merged Processed":
        show_football_merged_processed()
    elif page == "Football Merged Feature Engineering":
        show_football_merged_feature_engineering()
    elif page == "NBA Merge":
        show_nba_merge()
    elif page == "NBA Merge Processing":
        show_nba_merge_processing()
    elif page == "NBA Merge Feature Engineering":
        show_nba_merge_feature_engineering()
    elif page == "Football Metrics":
        show_football_metrics()
    elif page == "NBA Metric":
        show_nba_metric()
    elif page == "Conclusion":
        show_conclusion()


if __name__ == "__main__":
    main()
