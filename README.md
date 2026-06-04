# Sports Event Analysis and Generation

[![Repository Check](https://github.com/st-kirill-v/sports-event-analysis-generation/actions/workflows/repository-check.yml/badge.svg)](https://github.com/st-kirill-v/sports-event-analysis-generation/actions/workflows/repository-check.yml)

Учебный ML/DL-репозиторий с пятью независимыми задачами по анализу и генерации спортивных событий. Главный README служит точкой входа: здесь собраны навигация, краткие описания подпроектов, данные, методы, метрики и инструкции по запуску.

Проект оформлен как единая лабораторная работа по глубокому обучению: каждая задача остается самостоятельным подпроектом со своим README, кодом, данными и результатами.

## Цель лабораторной работы

Исследовать несколько классов задач спортивной аналитики и генеративного моделирования:

- классификация типов спортивных событий по видеокадрам;
- моделирование динамики матчей по последовательностям событий;
- генерация текстовых футбольных сценариев;
- генерация карт траекторий игроков NBA;
- сетевой анализ взаимодействий игроков по SportVU и play-by-play данным.

## Навигация по задачам

| Task | Подпроект | Краткое описание | Методы | Данные | Основные метрики |
|---|---|---|---|---|---|
| [Task 1](./Task%201/README.md) | Классификация типов спортивных событий | Подготовка кадров Sports-1M и обучение CNN для классификации видов спорта. | CNN, image augmentation, supervised learning | Sports-1M, выбранные sport classes | Accuracy, classification report, confusion matrix |
| [Task 2](./Task%202/README.md) | Моделирование динамики спортивных матчей | Прогноз голов во втором тайме и изменения разницы счета в концовке NBA-матча. | LSTM, sequence modeling, feature engineering, Streamlit audit UI | Football Events, NBA SportVU/player tracking | Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, MAE, MSE, RMSE, R2 |
| [Task 3](./Task%203/README.md) | Генерация текстовых сценариев футбольных матчей | Дообучение GPT-2 Medium на событиях футбольных матчей. | GPT-2 fine-tuning, language modeling, MOS evaluation | Football Events, очищенный корпус событий | Perplexity, BLEU, MOS |
| [Task 4](./Task%204/README.md) | Генерация карт траекторий игроков NBA | Генерация траекторий движения игроков NBA в пиксельном и координатном представлениях. | DCGAN, Coordinate-VAE, trajectory rendering | NBA SportVU 2015/16 movement data | Inception Score, FID, reconstruction MSE |
| [Task 5](./Task%205/README.md) | NBA Movement Network Analysis | Построение графов взаимодействий игроков по ассистам и восстановленным передачам. | NetworkX, graph centrality, SportVU pass heuristics | NBA SportVU, play-by-play events, shots data | in/out strength, betweenness, PageRank |

## Используемые методы

- классические пайплайны подготовки данных: очистка, агрегация, feature engineering, sequence building;
- сверточные нейронные сети для классификации изображений;
- LSTM-модели для последовательностей матчей;
- GPT-2 fine-tuning для генерации текстовых событий;
- GAN/VAE-подходы для генерации траекторий;
- графовый анализ взаимодействий игроков.

## Датасеты

В подпроектах используются публичные спортивные датасеты:

- **Sports-1M** - видео и кадры для классификации видов спорта;
- **Football Events** - события футбольных матчей и метаданные матчей;
- **NBA SportVU / nba-movement-data** - координаты игроков и мяча во времени;
- **NBA play-by-play / shots** - события матчей и данные о бросках.

Большие датасеты, веса моделей, архивы и результаты не должны попадать в Git. Правила исключения добавлены в корневой [`.gitignore`](./.gitignore).

## Результаты

| Task | Ключевой результат |
|---|---|
| Task 1 | Подготовлен pipeline извлечения `128x128` кадров и обучения CNN; результаты сохраняются в `runs/cnn/`. |
| Task 2 | Football LSTM: F1 до `0.717` для `home_scores_next_half`; NBA LSTM: RMSE до `5.881`, R2 до `0.083`. |
| Task 3 | Дообученная GPT-2 Medium улучшила Perplexity с `6.41` до `1.57`, BLEU с `0.0187` до `0.6323`, MOS с `2.91` до `4.48`. |
| Task 4 | DCGAN достиг IS `2.24` и FID `37.9`; Coordinate-VAE дает reconstruction MSE около `5e-4`. |
| Task 5 | Реализованы графы ассистов и tracking-передач, центральности NetworkX и интерактивная Plotly-визуализация. |

Подробные таблицы, интерпретации и команды запуска находятся в README соответствующих задач.

## Структура репозитория

```text
sports-event-analysis-generation/
├── README.md
├── pyproject.toml
├── requirements.txt
├── .gitignore
├── Task 1/
│   ├── README.md
│   ├── scripts/
│   └── configs/
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

Рекомендуется запускать каждую задачу из её подпапки, потому что версии Python и окружения могут отличаться.

### Общая установка зависимостей

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

### Запуск подпроектов

- Task 1: см. [инструкцию подготовки кадров и обучения CNN](./Task%201/README.md).
- Task 2: см. [команды Football/NBA pipeline и Streamlit UI](./Task%202/README.md).
- Task 3: см. [Kaggle-инструкцию для GPT-2 fine-tuning](./Task%203/README.md).
- Task 4: см. [Kaggle-инструкцию для DCGAN и Coordinate-VAE](./Task%204/README.md).
- Task 5: см. [подготовку NBA movement data и построение графов](./Task%205/README.md).

## Команда проекта

Студенческий проект по курсу глубокого обучения.

Для оформления персонального состава команды можно заполнить таблицу:

| Участник | Роль |
|---|---|
| Участник 1 | Task 1 / classification |
| Участник 2 | Task 2 / match dynamics |
| Участник 3 | Task 3 / text generation |
| Участник 4 | Task 4 / trajectory generation |
| Участник 5 | Task 5 / network analysis |

## Примечания

- Исследовательский код, ноутбуки, данные, веса и результаты экспериментов не изменялись при оформлении репозитория.
- Корневые зависимости собраны для удобной навигации и базового локального запуска; для воспроизведения экспериментов используйте README конкретной задачи.
