# Task 1 - Классификация типов спортивных событий

[← К главному README](../README.md)

Task 1 посвящена подготовке кадров Sports-1M и обучению CNN-классификатора для распознавания видов спорта по изображениям.

## Цель

Сформировать датасет кадров из видео Sports-1M, выбрать целевые спортивные классы и обучить сверточную модель для классификации типов спортивных событий.

## Данные

- **Источник:** Sports-1M.
- **Разметка:** `train_partition.txt`, `test_partition.txt`, `labels.txt`.
- **Классы:** выбранные виды спорта из `configs/selected_sports.json`.
- **Формат кадров:** JPEG `128x128`, разложенные по папкам классов.

Маппинг классов хранится в `configs/selected_sports.json`. Для футбола используется метка `association football`, для плавания - `swimming (sport)`.

## Методы

- извлечение кадров из видео;
- балансировка количества кадров на класс;
- ручная фильтрация подготовленных изображений;
- CNN-классификатор;
- train-аугментации: crop, horizontal flip, rotation, color jitter, shift/scale.

## Структура папки

```text
Task 1/
├── README.md
├── pyproject.toml
├── configs/
│   └── selected_sports.json
├── scripts/
│   ├── prepare_frames.py
│   └── train_cnn.py
├── train_partition.txt
├── test_partition.txt
└── labels.txt
```

После подготовки данные имеют вид:

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

## Запуск

### Окружение

```powershell
$env:UV_CACHE_DIR=".uv-cache"
uv venv
uv sync
```

Если окружение уже создано:

```powershell
$env:UV_CACHE_DIR=".uv-cache"
uv sync
```

### Проверка разметки

```powershell
$env:UV_CACHE_DIR=".uv-cache"
uv run python scripts/prepare_frames.py --dry-run
```

### Подготовка train-кадров

```powershell
$env:UV_CACHE_DIR=".uv-cache"
uv run python scripts/prepare_frames.py `
  --partition train_partition.txt `
  --output data/frames/train `
  --target-per-class 10000
```

### Подготовка test-кадров

```powershell
$env:UV_CACHE_DIR=".uv-cache"
uv run python scripts/prepare_frames.py `
  --partition test_partition.txt `
  --output data/frames/test `
  --target-per-class 1000
```

### Обучение CNN

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

Скрипт определяет классы по папкам. Если после фильтрации какой-то класс отсутствует, размер выходного слоя уменьшается автоматически.

## Результаты и метрики

Результаты обучения сохраняются в `runs/cnn/`:

```text
best_model.pt
metrics.json
history.json
classification_report.txt
confusion_matrix.png
```

Основные метрики:

- accuracy;
- classification report;
- confusion matrix.

## Вывод

Task 1 формирует воспроизводимый pipeline для перехода от видео Sports-1M к набору изображений и CNN-модели классификации видов спорта.
