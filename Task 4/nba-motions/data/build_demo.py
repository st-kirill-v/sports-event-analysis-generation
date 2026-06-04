"""Build a small demo dataset for the first Kaggle run.

The full set is ~578k images and slow to upload. For the first run we only need
to prove the GAN learns the new GAN-friendly rendering, so we render a small,
match-distributed sample instead.

Steps:
1. List game JSONs, shuffle by seed, split 85% train games / 15% test games
   (disjoint by game, so no episode leaks between train and test).
2. Walk train games in shuffled order and render up to `--per-game` trajectories
   from each, stopping once `--n-train` images exist. Same for test with
   `--n-test`. Capping per game spreads the sample across many matches.
3. Optionally pack everything into a single zip ready to upload as a Kaggle
   Dataset. The zip has two folders: train/ and test/.

Usage:
    python data/build_demo.py \
        --json-dir D:/nba-motions-data/raw/nba-movement-data/data \
        --events-dir D:/nba-motions-data/raw/nba-movement-data/data/events \
        --out-dir D:/nba-motions-data/demo \
        --n-train 40000 --n-test 3000 --per-game 100 --zip
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import zipfile
from pathlib import Path

# allow running both as "python -m data.build_demo" and "python data/build_demo.py"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.pipeline import (
    KEEP_EVENT_TYPES,
    load_event_types,
    parse_event_trajectories,
    render_trajectory,
)


def game_ids_from_jsons(json_dir: Path) -> list[str]:
    return sorted(p.stem for p in json_dir.glob("*.json"))


def split_games(games: list[str], test_frac: float, seed: int) -> tuple[list[str], list[str]]:
    """Shuffle and split game ids into (train_games, test_games)."""
    rng = random.Random(seed)
    shuffled = list(games)
    rng.shuffle(shuffled)
    n_test = max(1, int(round(len(shuffled) * test_frac)))
    test_games = shuffled[:n_test]
    train_games = shuffled[n_test:]
    return train_games, test_games


def render_subset(
    games: list[str],
    json_dir: Path,
    events_dir: Path,
    dst_dir: Path,
    target: int,
    per_game: int,
) -> int:
    """Render up to `per_game` trajectories from each game until `target` total."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for gi, game_id in enumerate(games, 1):
        if written >= target:
            break
        json_path = json_dir / f"{game_id}.json"
        events_csv = events_dir / f"{game_id}.csv"
        if not json_path.exists() or not events_csv.exists():
            continue
        try:
            event_types = load_event_types(events_csv)
            data = json.load(open(json_path, "r", encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  ERROR on {game_id}: {exc}", file=sys.stderr)
            continue

        from_game = 0
        for event in data.get("events", []):
            if from_game >= per_game or written >= target:
                break
            eid = str(event.get("eventId", ""))
            meta = event_types.get(eid)
            if meta is None or meta["msgtype"] not in KEEP_EVENT_TYPES:
                continue
            traj = parse_event_trajectories(event, meta)
            if not traj:
                continue
            for player_id, xy in traj:
                if from_game >= per_game or written >= target:
                    break
                img = render_trajectory(xy)
                img.save(dst_dir / f"{game_id}_e{eid}_p{player_id}.png", format="PNG")
                from_game += 1
                written += 1
        print(f"  [{gi}/{len(games)}] {game_id}: total {written}/{target}", flush=True)
    return written


def pack_zip(out_dir: Path, zip_path: Path) -> None:
    """Zip out_dir/train and out_dir/test into one archive (stored, no recompress)."""
    print(f"Packing {zip_path} ...")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as zf:
        for sub in ("train", "test"):
            sub_dir = out_dir / sub
            if not sub_dir.exists():
                continue
            for png in sorted(sub_dir.glob("*.png")):
                zf.write(png, arcname=f"{sub}/{png.name}")
    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"Wrote {zip_path} ({size_mb:.0f} MB)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-dir", required=True, type=Path)
    ap.add_argument("--events-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--n-train", type=int, default=40000)
    ap.add_argument("--n-test", type=int, default=3000)
    ap.add_argument("--per-game", type=int, default=100)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--zip", action="store_true", help="pack a Kaggle-ready zip when done")
    args = ap.parse_args()

    games = game_ids_from_jsons(args.json_dir)
    if not games:
        print(f"No JSONs in {args.json_dir}", file=sys.stderr)
        return 1
    train_games, test_games = split_games(games, args.test_frac, args.seed)
    print(f"{len(games)} games -> {len(train_games)} train / {len(test_games)} test")

    n_tr = render_subset(train_games, args.json_dir, args.events_dir,
                         args.out_dir / "train", args.n_train, args.per_game)
    n_te = render_subset(test_games, args.json_dir, args.events_dir,
                        args.out_dir / "test", args.n_test, args.per_game)
    print(f"Rendered {n_tr} train + {n_te} test images into {args.out_dir}")

    if args.zip:
        zip_path = args.out_dir.parent / f"trajectories_demo_{n_tr // 1000}k.zip"
        pack_zip(args.out_dir, zip_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
