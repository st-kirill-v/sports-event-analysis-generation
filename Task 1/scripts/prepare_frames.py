from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
from tqdm import tqdm
import yt_dlp


@dataclass(frozen=True)
class VideoRow:
    url: str
    class_name: str


def read_labels(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_selected_classes(config_path: Path, labels: list[str]) -> dict[int, str]:
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    label_to_id = {label: idx for idx, label in enumerate(labels)}

    selected: dict[int, str] = {}
    missing: list[str] = []
    for output_name, sports1m_label in raw.items():
        label_id = label_to_id.get(sports1m_label)
        if label_id is None:
            missing.append(f"{output_name}: {sports1m_label}")
            continue
        selected[label_id] = output_name

    if missing:
        raise ValueError("Labels are missing in labels.txt: " + ", ".join(missing))
    return selected


def parse_partition(path: Path, selected: dict[int, str], shuffle_seed: int) -> list[VideoRow]:
    rows: list[VideoRow] = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) != 2:
                continue

            url, label_part = parts
            try:
                label_ids = [int(item) for item in label_part.split(",")]
            except ValueError:
                continue

            hits = [label_id for label_id in label_ids if label_id in selected]
            if len(hits) != 1:
                continue

            rows.append(VideoRow(url=url, class_name=selected[hits[0]]))

    rng = random.Random(shuffle_seed)
    rng.shuffle(rows)
    return rows


def count_existing_images(class_dir: Path) -> int:
    if not class_dir.exists():
        return 0
    return sum(1 for path in class_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})


def safe_video_id(url: str) -> str:
    if "v=" in url:
        return url.split("v=", 1)[1].split("&", 1)[0]
    return "".join(char if char.isalnum() else "_" for char in url)[-32:]


def write_csv_row(path: Path, fieldnames: list[str], row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def read_attempted_urls(manifest_path: Path, errors_path: Path) -> set[tuple[str, str]]:
    attempted: set[tuple[str, str]] = set()

    for path in (manifest_path, errors_path):
        if not path.exists():
            continue
        with path.open("r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                class_name = row.get("class_name")
                url = row.get("url")
                if class_name and url:
                    attempted.add((class_name, url))

    return attempted


def download_video(url: str, download_dir: Path, retries: int, sleep_seconds: float) -> Path | None:
    download_dir.mkdir(parents=True, exist_ok=True)
    before = set(download_dir.iterdir())

    opts = {
        "outtmpl": str(download_dir / "%(id)s.%(ext)s"),
        "format": "bestvideo[height<=360]/best[height<=360]/worst",
        "quiet": True,
        "no_warnings": True,
        "ignoreerrors": True,
        "retries": retries,
        "fragment_retries": retries,
        "noplaylist": True,
    }

    with yt_dlp.YoutubeDL(opts) as ydl:
        try:
            result = ydl.extract_info(url, download=True)
        except Exception:
            result = None

    if sleep_seconds > 0:
        time.sleep(sleep_seconds)

    if not result:
        return None

    after = set(download_dir.iterdir())
    created = sorted(after - before, key=lambda item: item.stat().st_mtime, reverse=True)
    if created:
        return created[0]

    video_id = result.get("id") if isinstance(result, dict) else safe_video_id(url)
    matches = sorted(download_dir.glob(f"{video_id}.*"), key=lambda item: item.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def center_crop_resize(frame, size: int):
    height, width = frame.shape[:2]
    side = min(height, width)
    y0 = max((height - side) // 2, 0)
    x0 = max((width - side) // 2, 0)
    cropped = frame[y0 : y0 + side, x0 : x0 + side]
    return cv2.resize(cropped, (size, size), interpolation=cv2.INTER_AREA)


def frame_indices(total_frames: int, needed: int, start_ratio: float, end_ratio: float) -> Iterable[int]:
    start = int(total_frames * start_ratio)
    end = int(total_frames * end_ratio)
    if end <= start:
        start, end = 0, total_frames - 1

    if needed <= 1:
        yield max(start, 0)
        return

    for idx in range(needed):
        value = start + (end - start) * idx / (needed - 1)
        yield int(round(value))


def extract_frames(
    video_path: Path,
    output_dir: Path,
    class_name: str,
    video_id: str,
    needed: int,
    image_size: int,
    jpeg_quality: int,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return 0

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        capture.release()
        return 0

    written = 0
    for source_idx in frame_indices(total_frames, needed, start_ratio=0.05, end_ratio=0.95):
        capture.set(cv2.CAP_PROP_POS_FRAMES, source_idx)
        ok, frame = capture.read()
        if not ok or frame is None:
            continue

        resized = center_crop_resize(frame, image_size)
        file_name = f"{class_name}_{video_id}_{source_idx:07d}.jpg"
        output_path = output_dir / file_name
        saved = cv2.imwrite(str(output_path), resized, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        written += int(saved)

    capture.release()
    return written


def build_plan(rows: list[VideoRow]) -> dict[str, list[VideoRow]]:
    by_class: dict[str, list[VideoRow]] = {}
    for row in rows:
        by_class.setdefault(row.class_name, []).append(row)
    return by_class


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Sports-1M videos and extract 128x128 frames.")
    parser.add_argument("--partition", default="train_partition.txt", help="Sports-1M partition file.")
    parser.add_argument("--labels", default="labels.txt", help="Sports-1M labels.txt file.")
    parser.add_argument("--classes", default="configs/selected_sports.json", help="JSON mapping output class to label.")
    parser.add_argument("--output", default="data/frames/train", help="Output directory with class subfolders.")
    parser.add_argument("--tmp", default="data/tmp_videos", help="Temporary video download directory.")
    parser.add_argument("--log-dir", default="data/logs", help="Directory for manifest and error logs.")
    parser.add_argument("--target-per-class", type=int, default=10_000)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--max-frames-per-video", type=int, default=12)
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--sleep", type=float, default=0.5, help="Pause after each YouTube request.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--keep-videos", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    labels = read_labels(Path(args.labels))
    selected = load_selected_classes(Path(args.classes), labels)
    rows = parse_partition(Path(args.partition), selected, args.seed)
    by_class = build_plan(rows)

    print("Selected classes and available clean videos:")
    for class_name in sorted(by_class):
        print(f"  {class_name}: {len(by_class[class_name])}")

    if args.dry_run:
        return

    output_root = Path(args.output)
    tmp_root = Path(args.tmp)
    log_dir = Path(args.log_dir)
    manifest_path = log_dir / "manifest.csv"
    errors_path = log_dir / "errors.csv"
    attempted_urls = read_attempted_urls(manifest_path, errors_path)

    for class_name in sorted(by_class):
        class_dir = output_root / class_name
        existing = count_existing_images(class_dir)
        if existing >= args.target_per_class:
            print(f"{class_name}: already has {existing} frames, skipping.")
            continue

        videos = by_class[class_name]
        skipped_attempted = sum(1 for row in videos if (row.class_name, row.url) in attempted_urls)
        videos = [row for row in videos if (row.class_name, row.url) not in attempted_urls]
        if skipped_attempted:
            print(f"{class_name}: skipping {skipped_attempted} already attempted urls.")
        progress = tqdm(total=args.target_per_class, initial=existing, desc=class_name)

        for idx, row in enumerate(videos):
            current = count_existing_images(class_dir)
            remaining_frames = args.target_per_class - current
            if remaining_frames <= 0:
                break

            remaining_videos = max(len(videos) - idx, 1)
            needed = min(args.max_frames_per_video, max(1, math.ceil(remaining_frames / remaining_videos)))
            video_id = safe_video_id(row.url)
            video_dir = tmp_root / class_name

            video_path = download_video(row.url, video_dir, args.retries, args.sleep)
            if video_path is None:
                write_csv_row(
                    errors_path,
                    ["class_name", "url", "reason"],
                    {"class_name": class_name, "url": row.url, "reason": "download_failed"},
                )
                continue

            written = extract_frames(
                video_path=video_path,
                output_dir=class_dir,
                class_name=class_name,
                video_id=video_id,
                needed=needed,
                image_size=args.image_size,
                jpeg_quality=args.jpeg_quality,
            )

            write_csv_row(
                manifest_path,
                ["class_name", "url", "video_path", "frames_written"],
                {
                    "class_name": class_name,
                    "url": row.url,
                    "video_path": str(video_path),
                    "frames_written": written,
                },
            )

            if written == 0:
                write_csv_row(
                    errors_path,
                    ["class_name", "url", "reason"],
                    {"class_name": class_name, "url": row.url, "reason": "frame_extraction_failed"},
                )

            if not args.keep_videos:
                try:
                    video_path.unlink(missing_ok=True)
                except OSError:
                    shutil.rmtree(video_path, ignore_errors=True)

            progress.update(max(count_existing_images(class_dir) - current, 0))

        progress.close()


if __name__ == "__main__":
    main()
