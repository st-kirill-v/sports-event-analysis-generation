# Task 2 - Моделирование динамики спортивных матчей

[← К главному README](../README.md)

Task 2 моделирует динамику спортивных матчей по последовательностям событий. В задаче реализованы два pipeline: Football Events для прогноза голов во втором тайме и NBA Player Tracking для прогноза изменения разницы счета в концовке матча.

## Цель

Построить модели, которые используют раннюю или частичную историю матча для прогноза дальнейшего развития игры.

## Данные

### Football Events

Источник: [Kaggle: Football Events](https://www.kaggle.com/datasets/secareanualin/football-events)

| Файл | Содержание |
|---|---|
| `events.csv` | события матча: голы, удары, передачи, карточки, игровые действия |
| `ginf.csv` | метаданные матча: команды, лига, страна, сезон, дата, итоговый счет |

Постановка:

- модель видит только минуты `1-45`;
- один матч представлен последовательностью из 45 минут;
- targets: `home_scores_next_half`, `away_scores_next_half`;
- тип задачи: multi-output binary classification.

### NBA Player Tracking

Источник: [GitHub: sealneaward/nba-movement-data](https://github.com/sealneaward/nba-movement-data)

| Источник | Содержание |
|---|---|
| `movement .7z` | координаты игроков и мяча во времени |
| `events/*.csv` | play-by-play события матча |
| `shots_fixed.csv` | броски: координаты, дистанция, зона, попадание/промах |

Постановка:

- модель видит события только до `Q4 05:00 remaining`;
- один матч представлен event-level sequence;
- target: `target_score_diff_change_last_5min`;
- тип задачи: LSTM regression.

## Методы

- merge и очистка событийных таблиц;
- feature engineering для футбола и NBA;
- построение последовательностей матчей;
- LSTM для бинарной классификации футбольных target-переменных;
- LSTM-регрессия для изменения разницы счета в NBA;
- threshold tuning, calibration, ablation и анализ ошибок;
- Streamlit UI для аудита данных и результатов.

## Структура папки

```text
Task 2/
├── README.md
└── modeling_match_dynamics/
    ├── pyproject.toml
    ├── src/match_dynamics/
    │   ├── common/
    │   ├── football_pipeline/
    │   ├── nba_pipeline/
    │   └── ui/
    ├── scripts/
    │   ├── run_football.py
    │   ├── run_nba.py
    │   ├── football/
    │   ├── nba/
    │   └── common/
    └── archive/
```

## Запуск

Команды выполняются из `Task 2/modeling_match_dynamics`.

### Установка

```powershell
uv sync --python 3.13
uv run python --version
```

### Streamlit UI

```powershell
uv run streamlit run scripts\run_data_audit_ui.py
```

UI содержит обзор данных, этапы merge/processing/feature engineering, метрики Football/NBA и итоговые выводы.

### Football pipeline

```powershell
uv run python scripts\run_football.py
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

### NBA pipeline

```powershell
uv run python scripts\run_nba.py
```

Если `data/nba/nba_merged_400.csv` уже собран:

```powershell
uv run python scripts\run_nba.py --skip-merge
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

## Результаты и метрики

### Football

| Target | Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | MAE | MSE |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `home_scores_next_half` | Baseline LSTM | 0.614 | 0.640 | 0.796 | 0.709 | 0.635 | 0.702 | 0.464 | 0.230 |
| `home_scores_next_half` | Top50 LSTM | 0.606 | 0.624 | 0.843 | 0.717 | 0.609 | 0.695 | 0.468 | 0.234 |
| `away_scores_next_half` | Baseline LSTM | 0.557 | 0.585 | 0.349 | 0.437 | 0.580 | 0.588 | 0.487 | 0.246 |
| `away_scores_next_half` | Top50 LSTM | 0.559 | 0.579 | 0.387 | 0.464 | 0.576 | 0.587 | 0.487 | 0.246 |

### NBA

| Model | MAE | MSE | RMSE | R2 |
|---|---:|---:|---:|---:|
| Constant zero baseline | 5.017 | 40.017 | 6.326 | -0.061 |
| Train mean baseline | 5.027 | 40.126 | 6.335 | -0.064 |
| LSTM top75 | 4.599 | 35.415 | 5.951 | 0.061 |
| LSTM all_features | 4.607 | 34.580 | 5.881 | 0.083 |

## Вывод

Последовательностные модели извлекают полезный сигнал из событийной истории матчей: для футбола лучше всего работает home-target LSTM, а для NBA LSTM с полным набором признаков превосходит константные baseline по RMSE и R2.
