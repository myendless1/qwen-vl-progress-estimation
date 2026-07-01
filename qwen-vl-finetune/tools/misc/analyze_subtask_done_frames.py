#!/usr/bin/env python
"""Analyze trailing frames with motion progress ~= 1.0 per subtask."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from qwenvl.data.robotwin_processor import _load_chunks_size, _robotwin_repo_dirs
from qwenvl.data.robotwin_progress import (
    build_subtask_progress_lookup,
    episode_parquet_path,
    load_episode_states,
    progress_from_curve,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        default="/media/damoxing/datasets/vae4d/lerobot-vae4d-org/robotwin_gt_depth",
    )
    parser.add_argument("--split", choices=("train", "test", "all"), default="train")
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--max-episodes", type=int, default=None)
    return parser.parse_args()


def motion_progress_at(curve, frame: int, start: int, end: int) -> float:
    if frame <= start:
        return 0.0
    if frame >= end:
        offset = end - start
    else:
        offset = frame - start
    progress = progress_from_curve(curve, offset)
    if progress is None:
        return float((frame - start) / max(1, end - start))
    return float(progress)


def trailing_count(values, threshold: float) -> int:
    count = 0
    for value in reversed(values):
        if value + 1e-9 >= threshold:
            count += 1
        else:
            break
    return count


def summarize(name: str, values: list[int]) -> None:
    arr = np.asarray(values, dtype=np.int64)
    print(
        f"{name}: n={len(arr)} "
        f"min={int(arr.min())} max={int(arr.max())} "
        f"mean={arr.mean():.2f} median={float(np.median(arr)):.1f}"
    )


def main():
    args = parse_args()
    split_arg = None if args.split == "all" else args.split

    tail_eq1 = []
    tail_gt995 = []
    total_eq1 = []
    total_gt995 = []
    fixed_done3 = []
    no_state = 0
    episode_count = 0

    for repo_dir in _robotwin_repo_dirs(
        args.data_root,
        split=split_arg,
        test_ratio=args.test_ratio,
        split_seed=args.split_seed,
    ):
        anno_dir = repo_dir / "anno"
        if not anno_dir.is_dir():
            continue
        chunks_size = _load_chunks_size(repo_dir)
        for anno_path in sorted(anno_dir.glob("episode_*.json")):
            if args.max_episodes is not None and episode_count >= args.max_episodes:
                break
            with open(anno_path, "r") as f:
                anno = json.load(f)
            subtasks = anno.get("subtasks", [])
            if not subtasks:
                continue
            episode_index = int(anno["episode_index"])
            state_path = episode_parquet_path(repo_dir, episode_index, chunks_size)
            if not state_path.exists():
                no_state += 1
                continue
            try:
                states = load_episode_states(state_path)
                progress_lookup = build_subtask_progress_lookup(states, subtasks, anno)
            except Exception:
                no_state += 1
                continue

            num_frames = min(int(anno["num_frames"]), len(states))
            for subtask in subtasks:
                start = int(subtask["start_frame"])
                end = min(int(subtask["end_frame"]), num_frames - 1)
                if start >= num_frames or end < start:
                    continue
                curve = progress_lookup.get(start)
                if curve is None:
                    continue

                progresses = [
                    motion_progress_at(curve, frame, start, end)
                    for frame in range(start, end + 1)
                ]
                tail_eq1.append(trailing_count(progresses, 1.0))
                tail_gt995.append(trailing_count(progresses, 0.995))
                total_eq1.append(sum(1 for p in progresses if p + 1e-9 >= 1.0))
                total_gt995.append(sum(1 for p in progresses if p + 1e-9 >= 0.995))
                fixed_done3.append(min(3, end - start + 1))
            episode_count += 1
        if args.max_episodes is not None and episode_count >= args.max_episodes:
            break

    print(f"split={args.split} episodes={episode_count} subtasks={len(tail_eq1)} skipped_no_state={no_state}")
    print()
    print("Trailing consecutive frames from subtask end (motion progress):")
    summarize("  progress == 1.0", tail_eq1)
    summarize("  progress >= 0.995", tail_gt995)
    print()
    print("Total frames in subtask (motion progress):")
    summarize("  progress == 1.0", total_eq1)
    summarize("  progress >= 0.995", total_gt995)
    print()
    summarize("  fixed current_done window (last 3 frames)", fixed_done3)

    for label, values in (
        ("tail == 1.0", tail_eq1),
        ("tail >= 0.995", tail_gt995),
        ("total == 1.0", total_eq1),
        ("total >= 0.995", total_gt995),
    ):
        arr = np.asarray(values)
        print(f"\n{label} distribution:")
        for k in (1, 2, 3, 4, 5, 10, 20, 50):
            pct = 100.0 * np.mean(arr >= k)
            print(f"  >={k:2d} frames: {pct:5.1f}%")


if __name__ == "__main__":
    main()
