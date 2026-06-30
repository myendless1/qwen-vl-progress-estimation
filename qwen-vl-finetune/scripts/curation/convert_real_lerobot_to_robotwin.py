#!/usr/bin/env python3
"""Convert real-robot LeRobot parquet repos to RobotWin-compatible layout.

Reads ``data/chunk-*/episode_*.parquet`` with puppet/master fields and writes
``data-lerobot/chunk-*/episode_*.parquet`` containing the 7-column RobotWin
schema:

    observation.state (16-dim), action (16-dim), timestamp, frame_index,
    episode_index, index, task_index

State layout matches RobotWin sim LeRobot:
    left xyz[0:3], left quat wxyz[3:7], left gripper[7],
    right xyz[8:11], right quat wxyz[11:15], right gripper[15]

Two source schemas are supported:
    pose: puppet.end_effector_{left,right}_pose_align.data + gripper scalars
    arm:  puppet.arm_{left,right}_position_align.data + gripper scalars
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np

POSE_LEFT = "puppet.end_effector_left_pose_align.data"
POSE_RIGHT = "puppet.end_effector_right_pose_align.data"
GRIP_LEFT = "puppet.end_effector_left_position_align.data"
GRIP_RIGHT = "puppet.end_effector_right_position_align.data"
ARM_LEFT = "puppet.arm_left_position_align.data"
ARM_RIGHT = "puppet.arm_right_position_align.data"
META_COLUMNS = ("timestamp", "frame_index", "episode_index", "index", "task_index")
ROBOTWIN_COLUMNS = ("observation.state", "action", *META_COLUMNS)
ConversionMode = Literal["pose", "arm"]
MOTION_EPS = 1e-12


@dataclass
class ConversionStats:
    episodes: int = 0
    skipped: int = 0
    failed: int = 0
    mode: ConversionMode | None = None


def normalize_quat_wxyz(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    norm = float(np.linalg.norm(quat))
    if norm <= MOTION_EPS:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return quat / norm


def segment_to_state(segment7: np.ndarray, gripper: float) -> np.ndarray:
    segment7 = np.asarray(segment7, dtype=np.float64)
    if segment7.shape != (7,):
        raise ValueError(f"expected 7-d segment, got shape {segment7.shape}")
    xyz = segment7[:3]
    quat = normalize_quat_wxyz(segment7[3:7])
    return np.concatenate([xyz, quat, np.asarray([gripper], dtype=np.float64)])


def build_observation_states(table, mode: ConversionMode) -> np.ndarray:
    if mode == "pose":
        left_segments = np.asarray(table.column(POSE_LEFT).to_pylist(), dtype=np.float64)
        right_segments = np.asarray(table.column(POSE_RIGHT).to_pylist(), dtype=np.float64)
    else:
        left_segments = np.asarray(table.column(ARM_LEFT).to_pylist(), dtype=np.float64)
        right_segments = np.asarray(table.column(ARM_RIGHT).to_pylist(), dtype=np.float64)

    left_grip = np.asarray(table.column(GRIP_LEFT).to_pylist(), dtype=np.float64)
    right_grip = np.asarray(table.column(GRIP_RIGHT).to_pylist(), dtype=np.float64)
    num_rows = table.num_rows
    states = np.empty((num_rows, 16), dtype=np.float64)
    for row in range(num_rows):
        left = segment_to_state(left_segments[row], left_grip[row])
        right = segment_to_state(right_segments[row], right_grip[row])
        states[row] = np.concatenate([left, right])
    return states


def build_actions(states: np.ndarray) -> np.ndarray:
    actions = np.empty_like(states)
    if len(states) == 0:
        return actions
    if len(states) == 1:
        actions[0] = states[0]
        return actions
    actions[:-1] = states[1:]
    actions[-1] = states[-1]
    return actions


def detect_conversion_mode(column_names: set[str]) -> ConversionMode:
    pose_fields = {POSE_LEFT, POSE_RIGHT, GRIP_LEFT, GRIP_RIGHT}
    arm_fields = {ARM_LEFT, ARM_RIGHT, GRIP_LEFT, GRIP_RIGHT}
    if pose_fields.issubset(column_names):
        return "pose"
    if arm_fields.issubset(column_names):
        return "arm"
    missing_pose = sorted(pose_fields - column_names)
    missing_arm = sorted(arm_fields - column_names)
    raise ValueError(
        "unsupported parquet schema; "
        f"missing pose fields={missing_pose}, missing arm fields={missing_arm}"
    )


def read_source_table(source_path: Path, mode: ConversionMode):
    import pyarrow.parquet as pq

    if mode == "pose":
        columns = [POSE_LEFT, POSE_RIGHT, GRIP_LEFT, GRIP_RIGHT, *META_COLUMNS]
    else:
        columns = [ARM_LEFT, ARM_RIGHT, GRIP_LEFT, GRIP_RIGHT, *META_COLUMNS]
    return pq.read_table(source_path, columns=columns)


def write_robotwin_parquet(
    output_path: Path,
    states: np.ndarray,
    actions: np.ndarray,
    timestamp: np.ndarray,
    frame_index: np.ndarray,
    episode_index: np.ndarray,
    index: np.ndarray,
    task_index: np.ndarray,
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    list_double = pa.list_(pa.float64())
    table = pa.table(
        {
            "observation.state": pa.array(states.tolist(), type=list_double),
            "action": pa.array(actions.tolist(), type=list_double),
            "timestamp": pa.array(timestamp, type=pa.float32()),
            "frame_index": pa.array(frame_index, type=pa.int64()),
            "episode_index": pa.array(episode_index, type=pa.int64()),
            "index": pa.array(index, type=pa.int64()),
            "task_index": pa.array(task_index, type=pa.int64()),
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output_path)


def validate_robotwin_parquet(path: Path) -> None:
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=list(ROBOTWIN_COLUMNS))
    if table.num_columns != len(ROBOTWIN_COLUMNS):
        raise ValueError(f"unexpected column count in {path}")
    if table.num_rows == 0:
        raise ValueError(f"empty parquet written: {path}")
    state0 = table.column("observation.state")[0].as_py()
    action0 = table.column("action")[0].as_py()
    if len(state0) != 16 or len(action0) != 16:
        raise ValueError(f"expected 16-d vectors in {path}")


def convert_episode(
    source_path: Path,
    output_path: Path,
    mode: ConversionMode,
    *,
    overwrite: bool,
    dry_run: bool,
) -> Literal["converted", "skipped"]:
    if output_path.exists() and not overwrite:
        return "skipped"

    table = read_source_table(source_path, mode)
    states = build_observation_states(table, mode)
    actions = build_actions(states)
    timestamp = np.asarray(table.column("timestamp").to_pylist(), dtype=np.float32)
    frame_index = np.asarray(table.column("frame_index").to_pylist(), dtype=np.int64)
    episode_index = np.asarray(table.column("episode_index").to_pylist(), dtype=np.int64)
    index = np.asarray(table.column("index").to_pylist(), dtype=np.int64)
    task_index = np.asarray(table.column("task_index").to_pylist(), dtype=np.int64)

    if dry_run:
        return "converted"

    write_robotwin_parquet(
        output_path,
        states,
        actions,
        timestamp,
        frame_index,
        episode_index,
        index,
        task_index,
    )
    validate_robotwin_parquet(output_path)
    return "converted"


def iter_source_episodes(repo_dir: Path) -> list[Path]:
    data_dir = repo_dir / "data"
    if not data_dir.exists():
        return []
    return sorted(data_dir.glob("chunk-*/episode_*.parquet"))


def output_episode_path(repo_dir: Path, source_path: Path) -> Path:
    rel = source_path.relative_to(repo_dir / "data")
    return repo_dir / "data-lerobot" / rel


def convert_repo(
    repo_dir: Path,
    *,
    overwrite: bool,
    dry_run: bool,
) -> ConversionStats:
    import pyarrow.parquet as pq

    stats = ConversionStats()
    episodes = iter_source_episodes(repo_dir)
    if not episodes:
        return stats

    mode = detect_conversion_mode(set(pq.read_schema(episodes[0]).names))
    stats.mode = mode

    for source_path in episodes:
        output_path = output_episode_path(repo_dir, source_path)
        try:
            result = convert_episode(
                source_path,
                output_path,
                mode,
                overwrite=overwrite,
                dry_run=dry_run,
            )
        except Exception:
            stats.failed += 1
            raise
        if result == "skipped":
            stats.skipped += 1
        else:
            stats.episodes += 1
    return stats


def discover_repos(root: Path, repo_filter: str | None = None) -> list[Path]:
    repos = []
    for candidate in sorted(root.iterdir()):
        if not candidate.is_dir():
            continue
        if repo_filter and candidate.name != repo_filter:
            continue
        if iter_source_episodes(candidate):
            repos.append(candidate)
    return repos


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/media/damoxing/datasets/VLN-CE/cogwam_data/20260629"),
        help="Root directory containing real-robot repos.",
    )
    parser.add_argument(
        "--repo",
        default=None,
        help="Convert a single repo directory name under --root.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing data-lerobot parquet files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan and validate conversion without writing parquet files.",
    )
    parser.add_argument(
        "--max-repos",
        type=int,
        default=None,
        help="Optional limit for debugging.",
    )
    parser.add_argument(
        "--log-name",
        default=None,
        help="Summary JSON basename. Defaults to convert_real_lerobot_<timestamp>.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repos = discover_repos(args.root, repo_filter=args.repo)
    if args.max_repos is not None:
        repos = repos[: args.max_repos]
    if not repos:
        raise SystemExit(f"No repos with data/chunk-*/episode_*.parquet found under {args.root}")

    summary: dict[str, Any] = {
        "root": str(args.root.resolve()),
        "dry_run": args.dry_run,
        "overwrite": args.overwrite,
        "num_repos": len(repos),
        "repos": {},
        "totals": {"episodes": 0, "skipped": 0, "failed": 0},
    }

    for repo in repos:
        repo_stats = convert_repo(repo, overwrite=args.overwrite, dry_run=args.dry_run)
        summary["repos"][repo.name] = {
            "mode": repo_stats.mode,
            "episodes": repo_stats.episodes,
            "skipped": repo_stats.skipped,
            "failed": repo_stats.failed,
        }
        summary["totals"]["episodes"] += repo_stats.episodes
        summary["totals"]["skipped"] += repo_stats.skipped
        summary["totals"]["failed"] += repo_stats.failed
        print(
            f"[{repo_stats.mode}] {repo.name}: "
            f"converted={repo_stats.episodes} skipped={repo_stats.skipped} failed={repo_stats.failed}"
        )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_name = args.log_name or f"convert_real_lerobot_{stamp}"
    summary_path = args.root / f"{log_name}.json"
    if not args.dry_run:
        write_summary(summary_path, summary)
        print(f"Wrote summary to {summary_path}")
    print(
        "Done: "
        f"repos={len(repos)} converted_episodes={summary['totals']['episodes']} "
        f"skipped={summary['totals']['skipped']} failed={summary['totals']['failed']}"
    )


if __name__ == "__main__":
    main()
