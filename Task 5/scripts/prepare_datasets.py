from __future__ import annotations

import argparse
import json
import math
from collections import deque
from pathlib import Path
from typing import Iterable

import pandas as pd
from tqdm import tqdm


VALID_ASSIST_EVENT = 1
BALL_TEAM_ID = -1


def load_events(events_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(events_dir.glob("*.csv")):
        frames.append(pd.read_csv(path, dtype={"GAME_ID": str}))
    if not frames:
        raise FileNotFoundError(f"No event CSV files found in {events_dir}")
    return pd.concat(frames, ignore_index=True)


def build_assist_edges(events: pd.DataFrame) -> pd.DataFrame:
    events = events.copy()
    numeric_cols = ["EVENTMSGTYPE", "PLAYER1_ID", "PLAYER2_ID", "PLAYER1_TEAM_ID", "PLAYER2_TEAM_ID"]
    for col in numeric_cols:
        events[col] = pd.to_numeric(events[col], errors="coerce")

    assists = events[
        (events["EVENTMSGTYPE"] == VALID_ASSIST_EVENT)
        & events["PLAYER2_ID"].notna()
        & (events["PLAYER2_ID"] != 0)
        & events["PLAYER1_ID"].notna()
        & (events["PLAYER1_ID"] != 0)
        & (events["PLAYER1_TEAM_ID"] == events["PLAYER2_TEAM_ID"])
    ].copy()

    result = pd.DataFrame(
        {
            "game_id": assists["GAME_ID"].astype(str),
            "eventnum": assists["EVENTNUM"],
            "period": assists["PERIOD"],
            "pctimestring": assists["PCTIMESTRING"],
            "team_id": assists["PLAYER2_TEAM_ID"].astype("Int64"),
            "team_abbreviation": assists["PLAYER2_TEAM_ABBREVIATION"],
            "passer_id": assists["PLAYER2_ID"].astype("Int64"),
            "passer_name": assists["PLAYER2_NAME"],
            "receiver_id": assists["PLAYER1_ID"].astype("Int64"),
            "receiver_name": assists["PLAYER1_NAME"],
            "weight": 1,
            "source": "play_by_play_assist",
        }
    )
    return result.sort_values(["game_id", "eventnum"]).reset_index(drop=True)


def aggregate_edges(edges: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "team_id",
        "team_abbreviation",
        "passer_id",
        "passer_name",
        "receiver_id",
        "receiver_name",
        "source",
    ]
    return (
        edges.groupby(group_cols, dropna=False)
        .agg(weight=("weight", "sum"), games=("game_id", "nunique"))
        .reset_index()
        .sort_values(["team_abbreviation", "weight"], ascending=[True, False])
    )


def iter_json_files(raw_json_dir: Path, limit: int | None) -> Iterable[Path]:
    files = sorted(raw_json_dir.glob("*.json"))
    if limit is not None:
        files = files[:limit]
    return files


def nearest_owner(players: list[list[float]], ball: list[float], max_distance: float, max_ball_radius: float) -> tuple[int, int] | None:
    if ball[4] > max_ball_radius:
        return None

    best_player = None
    best_distance = float("inf")
    for player in players:
        team_id, player_id, x_loc, y_loc = int(player[0]), int(player[1]), float(player[2]), float(player[3])
        if player_id <= 0 or team_id == BALL_TEAM_ID:
            continue
        distance = math.dist((float(ball[2]), float(ball[3])), (x_loc, y_loc))
        if distance < best_distance:
            best_player = (team_id, player_id)
            best_distance = distance

    if best_player is None or best_distance > max_distance:
        return None
    return best_player


def stable_owner(recent: deque[tuple[int, int] | None], min_frames: int) -> tuple[int, int] | None:
    candidates = [owner for owner in recent if owner is not None]
    if len(candidates) < min_frames:
        return None
    last = candidates[-1]
    if candidates[-min_frames:].count(last) == min_frames:
        return last
    return None


def build_tracking_pass_edges(
    raw_json_dir: Path,
    limit: int | None,
    max_distance: float,
    max_ball_radius: float,
    min_frames: int,
    max_pass_gap: float,
) -> pd.DataFrame:
    rows = []
    files = list(iter_json_files(raw_json_dir, limit))
    if not files:
        return pd.DataFrame(
            columns=[
                "game_id",
                "event_id",
                "quarter",
                "start_game_clock",
                "end_game_clock",
                "team_id",
                "passer_id",
                "receiver_id",
                "weight",
                "source",
            ]
        )

    for path in tqdm(files, desc="Inferring tracking passes"):
        game_id = path.stem
        with path.open("r", encoding="utf-8") as handle:
            game = json.load(handle)

        last_owner = None
        last_owner_clock = None
        last_owner_event = None
        last_owner_quarter = None
        recent: deque[tuple[int, int] | None] = deque(maxlen=max(min_frames, 1) * 2)

        for event in game.get("events", []):
            event_id = event.get("eventId")
            for moment in event.get("moments", []):
                quarter = int(moment[0])
                game_clock = float(moment[2])
                entities = moment[5]
                balls = [entity for entity in entities if int(entity[0]) == BALL_TEAM_ID]
                if not balls:
                    recent.append(None)
                    continue

                ball = balls[0]
                players = [entity for entity in entities if int(entity[0]) != BALL_TEAM_ID]
                owner = nearest_owner(players, ball, max_distance=max_distance, max_ball_radius=max_ball_radius)
                recent.append(owner)
                current_owner = stable_owner(recent, min_frames=min_frames)
                if current_owner is None:
                    continue

                if last_owner is None:
                    last_owner = current_owner
                    last_owner_clock = game_clock
                    last_owner_event = event_id
                    last_owner_quarter = quarter
                    continue

                same_player = current_owner == last_owner
                same_team = current_owner[0] == last_owner[0]
                clock_gap = abs(float(last_owner_clock) - game_clock)
                if not same_player and same_team and clock_gap <= max_pass_gap:
                    rows.append(
                        {
                            "game_id": game_id,
                            "event_id": last_owner_event,
                            "quarter": last_owner_quarter,
                            "start_game_clock": last_owner_clock,
                            "end_game_clock": game_clock,
                            "team_id": current_owner[0],
                            "passer_id": last_owner[1],
                            "receiver_id": current_owner[1],
                            "weight": 1,
                            "source": "tracking_inferred_pass",
                        }
                    )

                last_owner = current_owner
                last_owner_clock = game_clock
                last_owner_event = event_id
                last_owner_quarter = quarter

    return pd.DataFrame(rows)


def attach_player_names(edges: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    if edges.empty or "passer_name" in edges.columns:
        return edges

    players = []
    for prefix in ["PLAYER1", "PLAYER2", "PLAYER3"]:
        players.append(
            events[
                [
                    f"{prefix}_ID",
                    f"{prefix}_NAME",
                    f"{prefix}_TEAM_ID",
                    f"{prefix}_TEAM_ABBREVIATION",
                ]
            ].rename(
                columns={
                    f"{prefix}_ID": "player_id",
                    f"{prefix}_NAME": "player_name",
                    f"{prefix}_TEAM_ID": "team_id",
                    f"{prefix}_TEAM_ABBREVIATION": "team_abbreviation",
                }
            )
        )

    player_lookup = pd.concat(players, ignore_index=True)
    player_lookup["player_id"] = pd.to_numeric(player_lookup["player_id"], errors="coerce").astype("Int64")
    player_lookup["team_id"] = pd.to_numeric(player_lookup["team_id"], errors="coerce").astype("Int64")
    player_lookup = player_lookup.dropna(subset=["player_id"]).drop_duplicates("player_id")

    edges = edges.copy()
    edges["passer_id"] = pd.to_numeric(edges["passer_id"], errors="coerce").astype("Int64")
    edges["receiver_id"] = pd.to_numeric(edges["receiver_id"], errors="coerce").astype("Int64")
    edges["team_id"] = pd.to_numeric(edges["team_id"], errors="coerce").astype("Int64")

    passer_lookup = player_lookup.rename(
        columns={
            "player_id": "passer_id",
            "player_name": "passer_player_name",
            "team_id": "passer_team_id",
            "team_abbreviation": "passer_team_abbreviation",
        }
    )
    receiver_lookup = player_lookup.rename(
        columns={
            "player_id": "receiver_id",
            "player_name": "receiver_player_name",
            "team_id": "receiver_team_id",
            "team_abbreviation": "receiver_team_abbreviation",
        }
    )

    edges = edges.merge(passer_lookup, on="passer_id", how="left")
    edges = edges.merge(receiver_lookup, on="receiver_id", how="left")
    edges["team_abbreviation"] = edges["passer_team_abbreviation"].fillna(edges["receiver_team_abbreviation"])
    edges["passer_name"] = edges["passer_player_name"].fillna(edges["passer_id"].astype(str))
    edges["receiver_name"] = edges["receiver_player_name"].fillna(edges["receiver_id"].astype(str))
    return edges.drop(
        columns=[
            "passer_player_name",
            "passer_team_id",
            "passer_team_abbreviation",
            "receiver_player_name",
            "receiver_team_id",
            "receiver_team_abbreviation",
        ],
        errors="ignore",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare assist and tracking-inferred passing datasets.")
    parser.add_argument("--events-dir", type=Path, default=Path("data/events"))
    parser.add_argument("--raw-json-dir", type=Path, default=Path("data/raw_json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--tracking-limit", type=int, default=None, help="Process only the first N JSON games.")
    parser.add_argument("--skip-tracking", action="store_true")
    parser.add_argument("--max-distance", type=float, default=3.0)
    parser.add_argument("--max-ball-radius", type=float, default=4.0)
    parser.add_argument("--min-frames", type=int, default=3)
    parser.add_argument("--max-pass-gap", type=float, default=3.0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    event_data = load_events(args.events_dir)
    assist_edges = build_assist_edges(event_data)
    assist_weighted = aggregate_edges(assist_edges)
    assist_edges.to_csv(args.output_dir / "assist_edges.csv", index=False)
    assist_weighted.to_csv(args.output_dir / "assist_edges_weighted.csv", index=False)

    print(f"Assist rows: {len(assist_edges):,}")
    print(f"Weighted assist edges: {len(assist_weighted):,}")

    if not args.skip_tracking:
        tracking_edges = build_tracking_pass_edges(
            raw_json_dir=args.raw_json_dir,
            limit=args.tracking_limit,
            max_distance=args.max_distance,
            max_ball_radius=args.max_ball_radius,
            min_frames=args.min_frames,
            max_pass_gap=args.max_pass_gap,
        )
        tracking_edges = attach_player_names(tracking_edges, event_data)
        tracking_edges.to_csv(args.output_dir / "tracking_pass_edges.csv", index=False)
        if tracking_edges.empty:
            tracking_weighted = aggregate_edges(
                pd.DataFrame(
                    columns=[
                        "team_id",
                        "team_abbreviation",
                        "passer_id",
                        "passer_name",
                        "receiver_id",
                        "receiver_name",
                        "source",
                        "weight",
                        "game_id",
                    ]
                )
            )
        else:
            tracking_weighted = aggregate_edges(tracking_edges)
        tracking_weighted.to_csv(args.output_dir / "tracking_pass_edges_weighted.csv", index=False)
        print(f"Tracking-inferred pass rows: {len(tracking_edges):,}")
        print(f"Weighted tracking edges: {len(tracking_weighted):,}")
