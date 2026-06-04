# Modeling Match Dynamics

[← К Task 2](../README.md) · [К главному README](../../README.md)

Локальный VS Code проект для лабораторной работы по глубокому обучению:
**моделирование динамики спортивных матчей**.

Проект содержит два независимых pipeline:

- **Football Events**: прогноз голов во втором тайме по событиям первого тайма.
- **NBA Player Tracking**: прогноз изменения разницы счета за последние 5 минут матча.

Исходный notebook был вынесен в модульную структуру Python-проекта. Notebook-style код убран из активного pipeline.

## Данные

### Football Events

Источник: [Kaggle: Football Events](https://www.kaggle.com/datasets/secareanualin/football-events)

Используемые файлы:

| File | Content |
|---|---|
| `events.csv` | события матча: голы, удары, передачи, карточки, игровые действия |
| `ginf.csv` | метаданные матча: команды, лига, страна, сезон, дата, итоговый счет |

Постановка:

- модель видит только минуты `1-45`;
- один матч = sequence из 45 минут;
- targets:
  - `home_scores_next_half`;
  - `away_scores_next_half`;
- тип задачи: multi-output binary classification.

### NBA Player Tracking

Источник: [GitHub: sealneaward/nba-movement-data](https://github.com/sealneaward/nba-movement-data)

Используемые источники:

| File / Folder | Content |
|---|---|
| `movement .7z` | координаты игроков и мяча во времени |
| `events/*.csv` | play-by-play события матча |
| `shots_fixed.csv` | броски: координаты, дистанция, зона, попадание/промах |

Постановка:

- модель видит события только до `Q4 05:00 remaining`;
- один матч = event-level sequence;
- target: `target_score_diff_change_last_5min`;
- тип задачи: LSTM regression.

## Установка

Проект использует `uv`. `requirements.txt` не нужен.

```powershell
uv sync --python 3.13
```

Проверка:

```powershell
uv run python --version
```

## Быстрый запуск

### Streamlit UI

```powershell
uv run streamlit run scripts\run_data_audit_ui.py
```

UI содержит:

- Overview;
- Football Merge;
- Football Merged Processed;
- Football Merged Feature Engineering;
- NBA Merge;
- NBA Merge Processing;
- NBA Merge Feature Engineering;
- Football Metrics;
- NBA Metric;
- Conclusion.

### Football

Полный football pipeline:

```powershell
uv run python scripts\run_football.py
```

Отдельные football-команды лежат в:

```text
scripts/football/
```

Ключевые этапы:

```powershell
uv run python scripts\football\merge_football_data.py
uv run python scripts\football\process_football_merged.py
uv run python scripts\football\run_football_feature_engineering.py
uv run python scripts\football\build_football_sequences.py
uv run python scripts\football\train_football_lstm.py
uv run python scripts\football\run_football_feature_ablation.py
uv run python scripts\football\run_football_top50_retrain.py
```

### NBA

Активная NBA-версия работает с 400 матчами.

Полный NBA pipeline:

```powershell
uv run python scripts\run_nba.py
```

Если `data/nba/nba_merged_400.csv` уже собран:

```powershell
uv run python scripts\run_nba.py --skip-merge
```

Отдельные NBA-команды лежат в:

```text
scripts/nba/
```

Ключевые этапы:

```powershell
uv run python scripts\nba\merge_nba_400.py --skip-download
uv run python scripts\nba\run_nba_preprocessing.py
uv run python scripts\nba\run_nba_feature_engineering.py
uv run python scripts\nba\validate_historical_timeline.py
uv run python scripts\nba\add_historical_team_features.py
uv run python scripts\nba\train_nba_lstm_feature_sets.py --feature-set all
```

## Структура проекта

```text
src/match_dynamics/
  common/
    config.py
    data_loading.py
    evaluation.py
    models.py
    pipeline.py
    sequences.py
    visualization.py

  football_pipeline/
    core.py
    merge.py
    event_processing.py
    feature_engineering.py
    sequence_dataset.py
    lstm_training.py
    lstm_ablation.py
    threshold_tuning.py
    calibration.py
    error_analysis.py
    targeted_features.py
    targeted_training.py

  nba_pipeline/
    core.py
    merge.py
    preprocessing.py
    feature_engineering.py
    historical_timeline.py
    historical_team_features.py
    lstm_training.py
    lstm_feature_sets.py

  ui/
    audit.py
    audit_ui.py

scripts/
  run_football.py
  run_nba.py
  run_pipeline.py
  run_data_audit_ui.py
  common/
  football/
  nba/

archive/
  scripts_legacy/
```

В корне `src/match_dynamics` больше нет compatibility wrapper modules вида
`from .football_pipeline.merge import *`. Активный код лежит в namespace-папках.

## Финальные результаты

### Football

Test metrics:

| Target | Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | MAE | MSE |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `home_scores_next_half` | Baseline LSTM | 0.614 | 0.640 | 0.796 | 0.709 | 0.635 | 0.702 | 0.464 | 0.230 |
| `home_scores_next_half` | Top50 LSTM | 0.606 | 0.624 | 0.843 | 0.717 | 0.609 | 0.695 | 0.468 | 0.234 |
| `away_scores_next_half` | Baseline LSTM | 0.557 | 0.585 | 0.349 | 0.437 | 0.580 | 0.588 | 0.487 | 0.246 |
| `away_scores_next_half` | Top50 LSTM | 0.559 | 0.579 | 0.387 | 0.464 | 0.576 | 0.587 | 0.487 | 0.246 |

### NBA

Test metrics:

| Model | MAE | MSE | RMSE | R2 |
|---|---:|---:|---:|---:|
| Constant zero baseline | 5.017 | 40.017 | 6.326 | -0.061 |
| Train mean baseline | 5.027 | 40.126 | 6.335 | -0.064 |
| LSTM top75 | 4.599 | 35.415 | 5.951 | 0.061 |
| LSTM all_features | 4.607 | 34.580 | 5.881 | 0.083 |

## Colab note

Google Colab не имеет доступа к локальному диску `D:\`.
Для Colab данные нужно загружать вручную, подключать Google Drive или запускать проект локально.
