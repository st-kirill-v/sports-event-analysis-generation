from __future__ import annotations

import argparse
from pathlib import Path

import py7zr
from tqdm import tqdm


def extract_archives(data_dir: Path, output_dir: Path, limit: int | None, overwrite: bool) -> None:
    archives = sorted(data_dir.glob("*.7z"))
    if limit is not None:
        archives = archives[:limit]

    output_dir.mkdir(parents=True, exist_ok=True)
    extracted = 0
    skipped = 0

    for archive in tqdm(archives, desc="Extracting SportVU archives"):
        expected_json = output_dir / f"{archive.stem}.json"
        if expected_json.exists() and not overwrite:
            skipped += 1
            continue

        with py7zr.SevenZipFile(archive, mode="r") as zf:
            zf.extractall(path=output_dir)
        extracted += 1

    print(f"Archives found: {len(archives)}")
    print(f"Extracted: {extracted}")
    print(f"Skipped existing: {skipped}")
    print(f"Output directory: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract .7z SportVU archives to JSON files.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw_json"))
    parser.add_argument("--limit", type=int, default=None, help="Extract only the first N archives.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    extract_archives(args.data_dir, args.output_dir, args.limit, args.overwrite)

