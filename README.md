# Анализ и генерация данных о спортивных событиях

[![Repository Check](https://github.com/st-kirill-v/sports-event-analysis-generation/actions/workflows/repository-check.yml/badge.svg)](https://github.com/st-kirill-v/sports-event-analysis-generation/actions/workflows/repository-check.yml)
![Python](https://img.shields.io/badge/python-3.12+-blue)
![Jupyter](https://img.shields.io/badge/Jupyter-notebooks-orange)
![PyTorch](https://img.shields.io/badge/PyTorch-deep_learning-ee4c2c)
![Keras](https://img.shields.io/badge/Keras-CNN-red)
![GPT--2](https://img.shields.io/badge/GPT--2-text_generation-7057ff)
![GAN](https://img.shields.io/badge/GAN-trajectory_generation-00a86b)
![LSTM](https://img.shields.io/badge/LSTM-sequence_modeling-0096c7)
![NetworkX](https://img.shields.io/badge/NetworkX-graph_analysis-4b8bbe)

Проект посвящен применению методов глубокого обучения к анализу спортивных событий: от классификации типов игр до моделирования динамики матчей и генерации новых сценариев. В репозитории собраны пять независимых задач, охватывающих компьютерное зрение, последовательностное моделирование, генерацию текста, генерацию траекторий и графовый анализ взаимодействий игроков.

## Цель

Применить методы глубокого обучения для анализа спортивных событий, классификации типов игр, моделирования динамики матчей, генерации сценариев игр и визуализации траекторий игроков.

## Задачи

- классификация спортивных событий и типов игр по видеокадрам;
- моделирование динамики матчей по последовательностям событий;
- генерация текстовых сценариев футбольных матчей;
- генерация и визуальная оценка траекторий игроков NBA;
- сетевой анализ взаимодействий игроков по play-by-play и tracking-данным.

## Навигация

| Задача | Цель | Методы | Данные | Метрики |
|---|---|---|---|---|
| [Task 1 - Классификация типов спортивных событий](./Task%201/README.md) | Подготовить кадры Sports-1M и обучить классификатор видов спорта. | CNN, аугментации, supervised learning | Sports-1M, выбранные sport classes | Accuracy, classification report, confusion matrix |
| [Task 2 - Моделирование динамики матчей](./Task%202/README.md) | Предсказать развитие футбольных и NBA-матчей по событиям. | LSTM, sequence modeling, feature engineering | Football Events, NBA SportVU | Accuracy, F1, ROC-AUC, PR-AUC, MAE, RMSE, R2 |
| [Task 3 - Генерация текстовых сценариев](./Task%203/README.md) | Сгенерировать связные описания футбольных событий. | GPT-2 fine-tuning, language modeling | Football Events text corpus | Perplexity, BLEU, MOS |
| [Task 4 - Генерация траекторий NBA](./Task%204/README.md) | Сгенерировать карты движения игроков NBA. | DCGAN, Coordinate-VAE, rendering | NBA SportVU 2015/16 | Inception Score, FID, reconstruction MSE |
| [Task 5 - Сетевой анализ NBA](./Task%205/README.md) | Построить графы взаимодействий игроков. | NetworkX, graph centrality, pass heuristics | NBA play-by-play, SportVU, shots | in/out strength, betweenness, PageRank |

## Методы

Проект объединяет несколько направлений анализа спортивных данных:

- **Computer Vision** - извлечение кадров и классификация видов спорта с помощью CNN;
- **Sequence Modeling** - LSTM-модели для временной динамики матчей;
- **Text Generation** - дообучение GPT-2 на корпусе футбольных событий;
- **Generative Modeling** - DCGAN и VAE для синтеза траекторий игроков;
- **Graph Analysis** - построение сетей передач и метрик центральности.

## Данные

| Источник | Использование |
|---|---|
| Sports-1M | видеокадры для классификации спортивных классов |
| Football Events | события матчей, текстовые описания и метаданные |
| NBA SportVU movement data | координаты игроков и мяча во времени |
| NBA play-by-play / shots | события матчей, броски и ассисты |

## Результаты

| Задача | Основной результат |
|---|---|
| Task 1 | Подготовлен pipeline извлечения `128x128` кадров и обучения CNN-классификатора. |
| Task 2 | Football LSTM достигает F1 до `0.717`; NBA LSTM показывает RMSE до `5.881` и R2 до `0.083`. |
| Task 3 | Дообученная GPT-2 Medium улучшает Perplexity с `6.41` до `1.57`, BLEU с `0.0187` до `0.6323`, MOS с `2.91` до `4.48`. |
| Task 4 | DCGAN достигает IS `2.24` и FID `37.9`; Coordinate-VAE показывает reconstruction MSE около `5e-4`. |
| Task 5 | Реализованы графы ассистов и восстановленных передач, метрики NetworkX и интерактивные Plotly-визуализации. |

## Структура

```text
sports-event-analysis-generation/
├── README.md
├── pyproject.toml
├── requirements.txt
├── .gitignore
├── Task 1/
│   ├── README.md
│   ├── configs/
│   └── scripts/
├── Task 2/
│   ├── README.md
│   └── modeling_match_dynamics/
├── Task 3/
│   ├── README.md
│   └── task_3_dl_project/
├── Task 4/
│   ├── README.md
│   └── nba-motions/
└── Task 5/
    ├── README.md
    ├── movement/
    └── scripts/
```

## Быстрый старт

Для базовой установки зависимостей из корня:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Для Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Каждая задача имеет собственные команды запуска и описание окружения:

- [Task 1 - Sports-1M classification](./Task%201/README.md)
- [Task 2 - Match dynamics modeling](./Task%202/README.md)
- [Task 3 - GPT-2 football text generation](./Task%203/README.md)
- [Task 4 - NBA trajectory generation](./Task%204/README.md)
- [Task 5 - NBA movement network analysis](./Task%205/README.md)
