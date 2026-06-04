# Task 1: Sports-1M frame preparation

[← К главному README](../README.md)

Подготовка кадров для классификации видов спорта по Sports-1M.

## Окружение

```powershell
$env:UV_CACHE_DIR=".uv-cache"
uv venv
uv sync
```

Если `.venv` уже создан, достаточно:

```powershell
$env:UV_CACHE_DIR=".uv-cache"
uv sync
```

## Проверка разметки без скачивания

```powershell
$env:UV_CACHE_DIR=".uv-cache"
uv run python scripts/prepare_frames.py --dry-run
```

## Сбор train-кадров

```powershell
$env:UV_CACHE_DIR=".uv-cache"
uv run python scripts/prepare_frames.py `
  --partition train_partition.txt `
  --output data/frames/train `
  --target-per-class 10000
```

## Сбор test-кадров

Для тестовой выборки обычно лучше брать меньше кадров на класс, чтобы оценка не была слишком тяжелой:

```powershell
$env:UV_CACHE_DIR=".uv-cache"
uv run python scripts/prepare_frames.py `
  --partition test_partition.txt `
  --output data/frames/test `
  --target-per-class 1000
```

Кадры сохраняются как `128x128` JPEG в папки классов:

```text
data/frames/train/
  basketball/
  football/
  tennis/
  ...
```

Логи сохраняются в `data/logs/manifest.csv` и `data/logs/errors.csv`. Скрипт можно прерывать и запускать снова: он считает уже существующие изображения в папках классов и продолжает до нужного лимита.

## Выбранные классы

Маппинг лежит в `configs/selected_sports.json`. В `labels.txt` нет точной метки `football`, поэтому используется `association football`; для плавания используется `swimming (sport)`.

## Обучение CNN

После подготовки и ручной фильтрации данных структура должна быть такой:

```text
data/frames/train/
  baseball/
  basketball/
  ...

data/frames/test/
  baseball/
  basketball/
  ...
```

Запуск обучения:

```powershell
$env:UV_CACHE_DIR=".uv-cache"
uv sync
uv run python scripts/train_cnn.py `
  --train-dir data/frames/train `
  --test-dir data/frames/test `
  --output-dir runs/cnn `
  --epochs 20 `
  --batch-size 64
```

Скрипт сам определяет классы по папкам. Если после фильтрации какой-то класс отсутствует, количество выходов CNN будет уменьшено автоматически.

Аугментации применяются только к train:

- случайный кроп до `128x128`;
- горизонтальное отражение;
- небольшой поворот;
- изменение яркости, контраста, насыщенности и оттенка;
- небольшой сдвиг и масштабирование.

Результаты сохраняются в `runs/cnn/`:

```text
best_model.pt
metrics.json
history.json
classification_report.txt
confusion_matrix.png
```
