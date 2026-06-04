# Task 5 - NBA Movement Network Analysis

[← К главному README](../README.md)

Task 5 строит графы взаимодействий игроков NBA по SportVU и play-by-play данным. Узлы графа - игроки, ребра - передачи мяча между игроками.

## Цель

Подготовить данные NBA movement/play-by-play и построить сетевое представление взаимодействий игроков для анализа ролей, связности и центральности в командной игре.

## Данные

Исходные данные:

- `data/*.7z` - сжатые JSON-файлы SportVU tracking;
- `data/events/*.csv` - play-by-play события матчей;
- `data/shots/shots.csv` - исходные броски;
- `data/shots/shots_fixed.csv` - броски с уточненными временем и координатами.

После подготовки создаются:

- `data/raw_json/*.json` - распакованные SportVU JSON;
- `data/processed/assist_edges.csv` - все ассистирующие передачи;
- `data/processed/assist_edges_weighted.csv` - агрегированный граф ассистов;
- `data/processed/tracking_pass_edges.csv` - передачи, восстановленные из tracking;
- `data/processed/tracking_pass_edges_weighted.csv` - агрегированный tracking-граф передач;
- `outputs/assist_graph/*` - граф ассистов, метрики центральности и визуализация.

## Методы

### Ассисты из play-by-play

Быстрый подход использует только таблицы `data/events/*.csv`. Для результативных бросков берется пара:

```text
passer = PLAYER2
receiver = PLAYER1
edge = PLAYER2 -> PLAYER1
```

Одинаковые пары игроков агрегируются в weighted graph.

### Передачи из SportVU tracking

Tracking-подход восстанавливает передачи по смене владельца мяча:

1. Для каждого момента находится мяч.
2. Определяется ближайший игрок.
3. Владение считается устойчивым при выполнении порогов `--max-distance`, `--max-ball-radius`, `--min-frames`.
4. Смена владения между игроками одной команды фиксируется как передача.
5. Передача засчитывается в окне `--max-pass-gap`.

## Структура папки

```text
Task 5/
├── README.md
├── pyproject.toml
├── setup.py
├── movement/
│   ├── convert_movement.py
│   ├── fix_shot_times.py
│   ├── json_to_csv.py
│   └── utils.py
├── scripts/
│   ├── extract_archives.py
│   ├── prepare_datasets.py
│   └── build_assist_graph.py
└── outputs/
```

## Запуск

### Установка

```bash
uv sync
```

Основные библиотеки:

- `pandas`;
- `numpy`;
- `networkx`;
- `matplotlib`;
- `py7zr`;
- `scipy`;
- `tqdm`.

### Подготовка данных

Распаковать архивы:

```bash
uv run python scripts/extract_archives.py
```

Для быстрой проверки:

```bash
uv run python scripts/extract_archives.py --limit 3
```

Сформировать оба датасета:

```bash
uv run python scripts/prepare_datasets.py
```

Только быстрый датасет ассистов:

```bash
uv run python scripts/prepare_datasets.py --skip-tracking
```

Проверка tracking-алгоритма на нескольких играх:

```bash
uv run python scripts/prepare_datasets.py --tracking-limit 3
```

Настройка tracking-эвристики:

```bash
uv run python scripts/prepare_datasets.py \
  --max-distance 3.0 \
  --max-ball-radius 4.0 \
  --min-frames 3 \
  --max-pass-gap 3.0
```

### Построение графа ассистов

Для Golden State Warriors:

```bash
uv run python scripts/build_assist_graph.py --team GSW
```

Для другой команды:

```bash
uv run python scripts/build_assist_graph.py --team ATL
```

## Результаты и метрики

Результаты сохраняются в `outputs/assist_graph/`:

- `<TEAM>_assist_graph.png` - статичная визуализация;
- `<TEAM>_assist_graph.html` - интерактивная Plotly-визуализация;
- `<TEAM>_assist_graph.graphml` - граф для Gephi или NetworkX;
- `<TEAM>_centrality.csv` - метрики центральности.

Метрики NetworkX:

- `in_strength` - сколько передач игрок получил как завершитель атак;
- `out_strength` - сколько передач игрок отдал;
- `betweenness` - насколько игрок связывает разные части сети;
- `pagerank` - интегральная важность игрока в направленном графе.

## Вывод

Графовый подход позволяет сравнивать роли игроков в атаке: `out_strength` выделяет создателей моментов, `in_strength` - основных адресатов передач, а `betweenness` и `pagerank` помогают находить структурно важных игроков.
