# NBA Movement Network Analysis

[← К главному README](../README.md)

Проект готовит данные NBA SportVU и play-by-play для сетевого анализа взаимодействий игроков. Узлы графа - игроки, ребра - передачи мяча между игроками. Реализованы два подхода:

- быстрый подход: граф ассистов по play-by-play событиям из `data/events`;
- более сильный подход: граф передач, восстановленных из SportVU tracking по координатам игроков и мяча.

## Установка

Требуется `uv`.

```bash
uv sync
```

Команда создаст виртуальное окружение `.venv` и установит библиотеки:

- `pandas`;
- `numpy`;
- `networkx`;
- `matplotlib`;
- `py7zr`;
- `scipy`;
- `tqdm`.

## Структура данных

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

## Два подхода к расчету взаимодействий

В работе используются два способа получить ребра графа. Они решают одну задачу - построить сеть взаимодействий игроков, но отличаются источниками данных и смыслом ребра.

### Подход 1: ассисты из play-by-play

Это быстрый и надежный вариант. Он использует только таблицы `data/events/*.csv`.

Каждый файл в `data/events` соответствует одному матчу и содержит play-by-play события: броски, подборы, фолы, замены, потери и другие игровые действия. Все эти CSV имеют одинаковую структуру, поэтому сначала они объединяются вертикально в одну общую таблицу событий:

```text
events_game_1
events_game_2
events_game_3
...
=> all_events
```

Это не `join` по ключу, а обычное объединение строк через `concat`, потому что каждая строка - отдельное событие матча.

Для расчета взаимодействий берутся только результативные броски:

```text
EVENTMSGTYPE = 1
```

В таких событиях:

- `PLAYER1_ID`, `PLAYER1_NAME` - игрок, который забил бросок;
- `PLAYER2_ID`, `PLAYER2_NAME` - игрок, который отдал ассист;
- `PLAYER1_TEAM_ID` и `PLAYER2_TEAM_ID` проверяются, чтобы оба игрока были из одной команды.

Правило построения ребра:

```text
passer = PLAYER2
receiver = PLAYER1
edge = PLAYER2 -> PLAYER1
```

То есть если в play-by-play записано:

```text
Millsap Jump Shot (Korver 1 AST)
```

то в граф добавляется направленное ребро:

```text
Kyle Korver -> Paul Millsap
```

После извлечения всех ассистов одинаковые пары игроков агрегируются:

```text
passer_id, receiver_id, team_id
=> weight = количество ассистов между этой парой
```

Результаты подхода:

- `assist_edges.csv` - каждая строка соответствует одному ассисту;
- `assist_edges_weighted.csv` - одна строка соответствует паре игроков, а `weight` показывает число ассистов.

Ограничение подхода: он учитывает только передачи, которые закончились результативным броском. Обычные передачи в атаке, которые не стали ассистами, сюда не попадают.

### Подход 2: передачи из SportVU tracking

Это более сильный, но эвристический вариант. Он использует координаты игроков и мяча из SportVU JSON.

Сначала архивы `data/*.7z` распаковываются в `data/raw_json/*.json`:

```bash
uv run python scripts/extract_archives.py
```

Каждый JSON содержит события матча и последовательность моментов (`moments`). В каждом моменте есть:

- номер четверти;
- игровое время;
- shot clock;
- координаты мяча;
- координаты всех игроков на площадке.

После преобразования movement-данные имеют логическую структуру:

```text
team_id, player_id, x_loc, y_loc, radius,
game_clock, shot_clock, quarter, game_id, event_id
```

Для мяча используется специальное значение:

```text
team_id = -1
```

SportVU не содержит готового поля `pass`, поэтому передача восстанавливается по смене владельца мяча. Алгоритм:

1. Для каждого момента находится мяч.
2. Среди игроков ищется ближайший к мячу.
3. Игрок считается владельцем мяча, если:
   - расстояние до мяча меньше порога `--max-distance`;
   - высота мяча `radius` меньше порога `--max-ball-radius`;
   - этот владелец стабилен несколько кадров подряд (`--min-frames`).
4. Если владение меняется с игрока A на игрока B той же команды, фиксируется передача:

```text
A -> B
```

5. Передача засчитывается только если смена владения произошла в разумном временном окне `--max-pass-gap`.

Для добавления имен игроков и аббревиатур команд tracking-датасет сопоставляется с `data/events/*.csv` как со справочником игроков. В `events` берутся поля:

```text
PLAYER1_ID, PLAYER1_NAME, PLAYER1_TEAM_ID, PLAYER1_TEAM_ABBREVIATION
PLAYER2_ID, PLAYER2_NAME, PLAYER2_TEAM_ID, PLAYER2_TEAM_ABBREVIATION
PLAYER3_ID, PLAYER3_NAME, PLAYER3_TEAM_ID, PLAYER3_TEAM_ABBREVIATION
```

Ключ сопоставления:

```text
tracking.player_id = events.PLAYER*_ID
```

Также используются общие идентификаторы игры и события:

```text
movement.game_id  = events.GAME_ID
movement.event_id = events.EVENTNUM
```

Результаты подхода:

- `tracking_pass_edges.csv` - каждая строка соответствует одной восстановленной передаче;
- `tracking_pass_edges_weighted.csv` - агрегированные пары игроков с весом `weight`.

Преимущество подхода: он учитывает не только ассисты, а большее число передач в атаке. Ограничение: это не официальная разметка передач, а расчетная эвристика, поэтому пороги нужно проверять и при необходимости калибровать.

## Соединение таблиц

Основные ключи:

```text
movement.game_id      = events.GAME_ID
movement.event_id     = events.EVENTNUM
shots.GAME_ID         = events.GAME_ID
shots.GAME_EVENT_ID   = events.EVENTNUM
tracking.player_id    = events.PLAYER1_ID / PLAYER2_ID / PLAYER3_ID
```

Для быстрого графа ассистов фактически не требуется соединение разных таблиц: все нужные поля уже есть в `events`. Таблицы матчей из `data/events` только склеиваются в один общий датасет.

Для tracking-подхода SportVU JSON дает координаты и смену владения, а `events` используется для расшифровки `player_id` в имена игроков и команды.

## Подготовка данных

Распаковать все `.7z` в `data/raw_json`:

```bash
uv run python scripts/extract_archives.py
```

Для быстрой проверки можно распаковать несколько матчей:

```bash
uv run python scripts/extract_archives.py --limit 3
```

Сформировать оба датасета:

```bash
uv run python scripts/prepare_datasets.py
```

Если нужен только быстрый датасет ассистов:

```bash
uv run python scripts/prepare_datasets.py --skip-tracking
```

Для проверки tracking-алгоритма на нескольких играх:

```bash
uv run python scripts/prepare_datasets.py --tracking-limit 3
```

Параметры tracking-эвристики можно настраивать:

```bash
uv run python scripts/prepare_datasets.py \
  --max-distance 3.0 \
  --max-ball-radius 4.0 \
  --min-frames 3 \
  --max-pass-gap 3.0
```

## Быстрый граф ассистов

Построить граф ассистов для команды, например Golden State Warriors:

```bash
uv run python scripts/build_assist_graph.py --team GSW
```

Результаты:

- `outputs/assist_graph/GSW_assist_graph.png` - визуализация графа;
- `outputs/assist_graph/GSW_assist_graph.html` - интерактивная Plotly-визуализация;
- `outputs/assist_graph/GSW_assist_graph.graphml` - граф для Gephi или NetworkX;
- `outputs/assist_graph/GSW_centrality.csv` - метрики центральности.

Можно указать любую аббревиатуру команды:

```bash
uv run python scripts/build_assist_graph.py --team ATL
```

Граф строится не по одной игре, а по всем подготовленным событиям выбранной команды. Если нужен анализ одного матча, датасет нужно дополнительно отфильтровать по `game_id`.

### Интерактивная визуализация

Скрипт дополнительно сохраняет HTML-файл с Plotly-графом:

```text
outputs/assist_graph/<TEAM>_assist_graph.html
```

Например:

```text
outputs/assist_graph/ATL_assist_graph.html
```

В интерактивном графе:

- кружки - игроки выбранной команды;
- размер кружка зависит от `out_strength`, то есть от числа отданных ассистов;
- линии со стрелками - направленные связи `ассистент -> забивший игрок`;
- цифры на линиях - количество ассистов между парой игроков;
- при наведении на игрока показываются команда, отданные ассисты, полученные ассисты, `betweenness` и `pagerank`;
- при наведении на линию показывается пара игроков и вес ребра.

## Метрики NetworkX

В скрипте рассчитываются:

- `in_strength` - сколько передач игрок получил как завершитель атак;
- `out_strength` - сколько передач игрок отдал;
- `betweenness` - насколько игрок связывает разные части сети;
- `pagerank` - интегральная важность игрока в направленном графе.

Для интерпретации:

- высокий `out_strength` показывает основных создателей моментов;
- высокий `in_strength` показывает основных адресатов передач;
- высокий `betweenness` выделяет игроков-посредников;
- высокий `pagerank` выделяет игроков, связанных с другими важными игроками.
