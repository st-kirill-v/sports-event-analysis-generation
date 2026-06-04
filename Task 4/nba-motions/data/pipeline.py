"""NBA SportVU -> 64x128 RGB trajectory cards (GAN-friendly rendering).

For each game JSON in `data/`:

1. Load play-by-play CSV `data/events/<game_id>.csv` for the same game and use
   EVENTMSGTYPE to keep only meaningful episodes:
       1 = Made Shot, 2 = Missed Shot, 4 = Rebound, 5 = Turnover, 6 = Foul
   Garbage is filtered by episode length instead of by type: only long episodes
   (>= 7 seconds) are kept, short fragments are dropped. Discarded types:
   3 (free throw), 7 (violation), 8 (substitution), 9 (timeout), 10 (jump ball),
   11 (ejection), 12/13 (period start/end), 18 (instant replay).
2. For each kept event:
   a. Detect the attacking team from PLAYER1_TEAM_ID in the matching row of
      the events CSV (this is the player whose shot/turnover anchors the event).
   b. Determine the target basket: mean ball x over the last ~1s of the episode.
      If x_mean < 47 -> attack went LEFT -> mirror x: x_new = 94 - x.
      After mirroring, all attacks point right (basket near x=94).
   c. For each of the 5 attacking players present throughout the episode,
      collect their (x, y) trajectory.
3. Render each trajectory as a 64x128 RGB PNG. The previous version drew a thin
   2 px line plus a faint court schematic; a pixel GAN cannot reproduce that
   sparse high-frequency signal and collapses it into grey mush. So this
   rendering is deliberately GAN-friendly:
   - White background, NO court schematic.
   - One thick antialiased trajectory line. We draw at 4x resolution with a
     wide line, then downscale with LANCZOS. The downscale turns the thick line
     into a smooth grey-to-black gradient that the generator can actually learn.
   - No end marker. A separate filled circle is not learned as an endpoint by a
     pixel GAN, it only shows up as stray dots, so the line is drawn alone.
4. Split rendered PNGs by GAME so test episodes never leak into train.

Usage:
    python data/pipeline.py \
        --json-dir D:/nba-motions-data/raw/nba-movement-data/data \
        --events-dir D:/nba-motions-data/raw/nba-movement-data/data/events \
        --out-dir D:/nba-motions-data/trajectories \
        --splits-dir splits
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw

# ----------------------------------------------------------------- constants

COURT_LENGTH_FT = 94.0
COURT_WIDTH_FT = 50.0
IMG_W = 128
IMG_H = 64

# Rendering knobs. We draw at SS x resolution and downscale with LANCZOS so the
# thick line becomes a smooth antialiased gradient (easy for a GAN to learn).
SS = 4                                    # supersampling factor
LINE_WIDTH_PX = 3                         # line width at the final resolution
                                          # (thinner than 5: less mush where the
                                          #  path crosses itself, still filled)

BALL_TEAM_ID = -1
# Made / Missed / Turnover, плюс Rebound и Foul. Мусор отсекаем длиной эпизода:
# берем только длинные эпизоды (>= 7 секунд), короткие обрывки отбрасываются.
KEEP_EVENT_TYPES = {1, 2, 4, 5, 6}
MIN_MOMENTS = 175                         # >= 7 seconds at 25 Hz
MIN_TRAJ_POINTS = 50                      # per player
MIN_RANGE_FT = 5.0                        # bbox side of the path


# ----------------------------------------------------------------- event CSV

def load_event_types(events_csv: Path) -> dict[str, dict]:
    """Return {eventnum_str: {msgtype, team_id}} for one game."""
    out: dict[str, dict] = {}
    with open(events_csv, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                eventnum = str(int(row["EVENTNUM"]))
            except (ValueError, KeyError, TypeError):
                continue
            try:
                msgtype = int(row["EVENTMSGTYPE"])
            except (ValueError, KeyError, TypeError):
                continue
            try:
                team_id = int(float(row["PLAYER1_TEAM_ID"])) if row.get("PLAYER1_TEAM_ID") else 0
            except (ValueError, TypeError):
                team_id = 0
            out[eventnum] = {"msgtype": msgtype, "team_id": team_id}
    return out


# ----------------------------------------------------------------- per-event parsing

def parse_event_trajectories(event: dict, event_meta: dict) -> list[tuple[int, np.ndarray]] | None:
    """Parse one event from a SportVU JSON.

    Returns list of (player_id, xy[N,2]) for the 5 attacking players, with all
    coordinates mirrored so the attack points right (basket at x=94).
    Returns None if the event is unusable.
    """
    moments = event.get("moments") or []
    if len(moments) < MIN_MOMENTS:
        return None
    attacking_team = event_meta["team_id"]
    if attacking_team == 0:
        return None

    # per-player trajectory dict
    per_player: dict[int, list[tuple[float, float]]] = {}
    ball_xs: list[float] = []
    player_team: dict[int, int] = {}
    for moment in moments:
        if not isinstance(moment, list) or len(moment) < 6:
            continue
        entries = moment[5]
        if not isinstance(entries, list) or len(entries) < 6:
            continue
        for entry in entries:
            if not isinstance(entry, list) or len(entry) < 4:
                continue
            team_id, player_id, x, y = entry[0], entry[1], entry[2], entry[3]
            if x is None or y is None:
                continue
            if not (0.0 <= x <= COURT_LENGTH_FT and 0.0 <= y <= COURT_WIDTH_FT):
                continue
            if team_id == BALL_TEAM_ID:
                ball_xs.append(float(x))
            else:
                per_player.setdefault(int(player_id), []).append((float(x), float(y)))
                player_team[int(player_id)] = int(team_id)

    if len(ball_xs) < 25:
        return None

    # Determine target basket from mean ball x in the last ~1s (25 moments).
    last_x_mean = float(np.mean(ball_xs[-25:]))
    mirror = last_x_mean < (COURT_LENGTH_FT / 2)   # attack went left -> mirror

    attackers = [pid for pid, tid in player_team.items() if tid == attacking_team]
    if len(attackers) < 1:
        return None

    out: list[tuple[int, np.ndarray]] = []
    for pid in attackers:
        pts = per_player.get(pid, [])
        if len(pts) < MIN_TRAJ_POINTS:
            continue
        arr = np.asarray(pts, dtype=np.float32)
        if (arr[:, 0].ptp() < MIN_RANGE_FT) and (arr[:, 1].ptp() < MIN_RANGE_FT):
            continue
        if mirror:
            arr = arr.copy()
            arr[:, 0] = COURT_LENGTH_FT - arr[:, 0]
        out.append((pid, arr))

    return out if out else None


# ----------------------------------------------------------------- rendering

def render_trajectory(xy: np.ndarray) -> Image.Image:
    """Render one (N,2) trajectory in feet as a 64x128 RGB image.

    No court schematic, no end marker. We draw one thick line at SS x resolution,
    then downscale with LANCZOS so the result is a smooth antialiased gradient.
    The end marker (a separate filled circle) was removed: a disconnected blob is
    not learned as an endpoint by a pixel GAN, it just shows up as stray dots.
    """
    big_w, big_h = IMG_W * SS, IMG_H * SS
    canvas = Image.new("RGB", (big_w, big_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    sx = big_w / COURT_LENGTH_FT
    sy = big_h / COURT_WIDTH_FT
    pixels = [(float(p[0]) * sx, float(p[1]) * sy) for p in xy]
    width = max(1, LINE_WIDTH_PX * SS)
    draw.line(pixels, fill=(0, 0, 0), width=width, joint="curve")
    return canvas.resize((IMG_W, IMG_H), Image.LANCZOS)


# ----------------------------------------------------------------- driver

def process_one_game(
    json_path: Path,
    events_dir: Path,
    out_dir: Path,
) -> int:
    """Render PNGs for one game. Returns number of trajectories written."""
    game_id = json_path.stem
    events_csv = events_dir / f"{game_id}.csv"
    if not events_csv.exists():
        print(f"  no events CSV for {game_id}, skipping", file=sys.stderr)
        return 0
    event_types = load_event_types(events_csv)

    with open(json_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    events = data.get("events", [])

    n_written = 0
    for event in events:
        eid = str(event.get("eventId", ""))
        meta = event_types.get(eid)
        if meta is None:
            continue
        if meta["msgtype"] not in KEEP_EVENT_TYPES:
            continue
        traj = parse_event_trajectories(event, meta)
        if not traj:
            continue
        for player_id, xy in traj:
            img = render_trajectory(xy)
            out_name = f"{game_id}_e{eid}_p{player_id}.png"
            img.save(out_dir / out_name, format="PNG", optimize=False)
            n_written += 1
    return n_written


def write_splits_by_game(out_dir: Path, splits_dir: Path, rng: random.Random) -> dict[str, int]:
    """Split PNGs by game: 100 train / 15 val / 15 test."""
    splits_dir.mkdir(parents=True, exist_ok=True)
    all_pngs = sorted(out_dir.glob("*.png"))
    games = sorted({p.name.split("_e")[0] for p in all_pngs})
    rng.shuffle(games)
    n = len(games)
    n_train = max(1, int(round(n * 100 / 130)))
    n_val = max(1, int(round(n * 15 / 130)))
    train_games = set(games[:n_train])
    val_games = set(games[n_train : n_train + n_val])
    test_games = set(games[n_train + n_val :])
    subsets = {
        "train": [p for p in all_pngs if p.name.split("_e")[0] in train_games],
        "val":   [p for p in all_pngs if p.name.split("_e")[0] in val_games],
        "test":  [p for p in all_pngs if p.name.split("_e")[0] in test_games],
    }
    for name, files in subsets.items():
        with open(splits_dir / f"{name}.txt", "w", encoding="utf-8") as fh:
            for p in files:
                fh.write(str(p) + "\n")
    return {name: len(files) for name, files in subsets.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-dir", required=True, type=Path)
    ap.add_argument("--events-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--splits-dir", required=True, type=Path)
    ap.add_argument("--limit-games", type=int, default=0, help=">0 = process only N games (debug)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    json_files = sorted(args.json_dir.glob("*.json"))
    if args.limit_games:
        json_files = json_files[: args.limit_games]
    print(f"Found {len(json_files)} JSON files in {args.json_dir}")
    if not json_files:
        return 1

    total = 0
    for i, jp in enumerate(json_files, 1):
        try:
            n = process_one_game(jp, args.events_dir, args.out_dir)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  ERROR on {jp.name}: {exc}", file=sys.stderr)
            continue
        total += n
        print(f"  [{i}/{len(json_files)}] {jp.name}: +{n} (total {total})", flush=True)
    print(f"Done. Wrote {total} PNGs to {args.out_dir}")

    counts = write_splits_by_game(args.out_dir, args.splits_dir, rng)
    print(f"Splits: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
