#!/usr/bin/env python
"""Visualize RobotWin Q2 raw training labels (undone progress buckets + all done frames)."""

import argparse
import json
import random
import sys
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

TOOLS_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TOOLS_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(TOOLS_ROOT))

from qwenvl.data.robotwin_processor import (
    PREV_DONE_MAX_FRAMES,
    _load_chunks_size,
    _robotwin_repo_dirs,
    parse_robotwin_views,
)
from qwenvl.data.robotwin_progress import (
    build_subtask_progress_lookup,
    current_done_frame_indices,
    episode_parquet_path,
    load_episode_states,
    progress_for_subtask,
    select_undone_frame_indices,
)
from tools.utils.robotwin_eval import write_csv, write_json
from tools.utils.robotwin_video import build_episode_samples, save_q2_video


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Visualize RobotWin Q2 raw training data aligned with build_robotwin_samples: "
            "undone frames sampled by progress bucket and all current_done frames."
        )
    )
    parser.add_argument("--data-root", default="/media/damoxing/datasets/vae4d/lerobot-vae4d-org/robotwin_gt_depth")
    parser.add_argument("--anno-root", default=None)
    parser.add_argument("--views", default="main,left_wrist,right_wrist")
    parser.add_argument("--split", choices=("train", "test", "all"), default="train")
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--q2-frame-stride", type=int, default=1)
    parser.add_argument(
        "--q2-progress-bucket-size",
        type=float,
        default=0.01,
        help="Undone progress bucket width; <=0 falls back to q2-frame-stride.",
    )
    parser.add_argument("--boundary-extra-frames", type=int, default=2)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--num-episodes", type=int, default=10, help="Number of random episodes to visualize.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--video-fps", type=float, default=2.0)
    parser.add_argument("--video-width", type=int, default=1280)
    parser.add_argument("--video-top-height", type=int, default=720)
    parser.add_argument("--video-curve-height", type=int, default=420)
    return parser.parse_args()


def safe_name(value):
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value))


def clipped_frames(frames: Sequence[int], num_frames: int) -> List[int]:
    return sorted({frame for frame in frames if 0 <= frame < num_frames})


def q2_trajectory_entries(
    subtask: Dict,
    subtask_index: int,
    num_frames: int,
    q2_frame_stride: int,
    q2_progress_bucket_size: float,
    states,
    anno: Dict,
    progress_lookup,
) -> List[Dict]:
    """Match build_robotwin_samples Q2 sampling for one subtask."""
    start = int(subtask["start_frame"])
    end = int(subtask["end_frame"])
    if start >= num_frames:
        return []
    end = min(end, num_frames - 1)
    curve = progress_lookup.get(start) if progress_lookup is not None else None
    current_done_frames = current_done_frame_indices(
        subtask, num_frames, states=states, anno=anno, curve=curve
    )
    not_done_end = min(end, min(current_done_frames) - 1) if current_done_frames else end

    entries = []
    for frame in select_undone_frame_indices(
        start,
        not_done_end,
        subtask=subtask,
        states=states,
        anno=anno,
        curve=curve,
        q2_frame_stride=q2_frame_stride,
        q2_progress_bucket_size=q2_progress_bucket_size,
    ):
        entries.append(
            {
                "frame_index": frame,
                "current_subtask_index": subtask_index,
                "current_subtask_goal": subtask["subtask_goal"],
                "q2_group": "undone",
                "done_label": 0.0,
                "progress_label": progress_for_subtask(
                    subtask,
                    frame,
                    states=states,
                    anno=anno,
                    curve=curve,
                ),
            }
        )
    for frame in current_done_frames:
        entries.append(
            {
                "frame_index": frame,
                "current_subtask_index": subtask_index,
                "current_subtask_goal": subtask["subtask_goal"],
                "q2_group": "current_done",
                "done_label": 1.0,
                "progress_label": 1.0,
            }
        )
    return entries


def resource_repo_dir(data_root: Path, anno_root: Optional[str], repo_name: str) -> Path:
    if anno_root and Path(data_root).resolve() != Path(anno_root).resolve():
        return Path(anno_root) / repo_name
    return data_root / repo_name


def resolve_image_repo_dir(args, repo_name: str) -> Optional[Path]:
    if args.anno_root and Path(args.data_root).resolve() != Path(args.anno_root).resolve():
        return Path(args.data_root) / repo_name
    return None


def collect_anno_paths(args) -> List[Path]:
    """Collect episode anno paths from split repos (glob only, no validation)."""
    data_root = Path(args.data_root)
    split_arg = None if args.split == "all" else args.split
    paths: List[Path] = []

    for image_repo_dir in _robotwin_repo_dirs(
        args.data_root,
        split=split_arg,
        test_ratio=args.test_ratio,
        split_seed=args.split_seed,
        anno_root=args.anno_root,
    ):
        resource_dir = resource_repo_dir(data_root, args.anno_root, image_repo_dir.name)
        anno_dir = resource_dir / "anno"
        if anno_dir.is_dir():
            paths.extend(sorted(anno_dir.glob("episode_*.json")))
    return paths


def ref_from_anno_path(anno_path: Path) -> Dict:
    resource_dir = anno_path.parent.parent
    episode_index = int(anno_path.stem.split("_", 1)[1])
    return {
        "repo": resource_dir.name,
        "resource_repo_dir": str(resource_dir),
        "anno_path": str(anno_path),
        "episode_index": episode_index,
    }


def load_episode_entries(args, ref: Dict, views: Sequence[str]) -> Optional[Dict]:
    """Build Q2 training-frame entries for one episode."""
    anno_path = Path(ref["anno_path"])
    with open(anno_path, "r") as f:
        anno = json.load(f)
    subtasks = anno.get("subtasks", [])
    if not subtasks:
        return None

    resource_dir = Path(ref["resource_repo_dir"])
    episode_index = int(ref["episode_index"])
    chunks_size = _load_chunks_size(resource_dir)
    num_frames = int(anno["num_frames"])
    states = None
    progress_lookup = None
    state_parquet_path = episode_parquet_path(resource_dir, episode_index, chunks_size)
    if state_parquet_path.exists():
        try:
            states = load_episode_states(state_parquet_path)
            num_frames = min(num_frames, len(states))
            progress_lookup = build_subtask_progress_lookup(states, subtasks, anno)
        except Exception:
            states = None
            progress_lookup = None
    if num_frames <= 0:
        return None

    entries = []
    for idx, subtask in enumerate(subtasks):
        entries.extend(
            q2_trajectory_entries(
                subtask,
                idx,
                num_frames,
                args.q2_frame_stride,
                args.q2_progress_bucket_size,
                states,
                anno,
                progress_lookup,
            )
        )
        if idx > 0:
            start = int(subtask["start_frame"])
            prev_subtask = subtasks[idx - 1]
            for frame in clipped_frames(range(start, start + PREV_DONE_MAX_FRAMES), num_frames):
                entries.append(
                    {
                        "frame_index": frame,
                        "current_subtask_index": idx - 1,
                        "current_subtask_goal": prev_subtask["subtask_goal"],
                        "q2_group": "prev_done",
                        "done_label": 1.0,
                        "progress_label": 1.0,
                    }
                )
    if not entries:
        return None
    entries.sort(key=lambda item: (int(item["frame_index"]), int(item["current_subtask_index"])))
    return {
        "repo": ref["repo"],
        "resource_repo_dir": ref["resource_repo_dir"],
        "anno_path": ref["anno_path"],
        "episode_index": episode_index,
        "task_goal": anno["task_goal"],
        "num_subtasks": len(subtasks),
        "entries": entries,
    }


def build_samples_for_rows(anno_path, rows, image_repo_dir, views):
    by_subtask = defaultdict(list)
    for row in rows:
        by_subtask[int(row["current_subtask_index"])].append(row)

    anno = None
    sample_by_frame = {}
    for subtask_index, subtask_rows in by_subtask.items():
        subtask_rows = sorted(subtask_rows, key=lambda row: int(row["frame_index"]))
        frame_indices = [int(row["frame_index"]) for row in subtask_rows]
        anno, samples = build_episode_samples(
            anno_path,
            frame_indices=frame_indices,
            fixed_subtask_index=subtask_index,
            image_repo_dir=image_repo_dir,
            views=views,
        )
        if len(samples) != len(subtask_rows):
            raise ValueError(
                f"Failed to build samples for subtask {subtask_index}: "
                f"expected {len(subtask_rows)}, got {len(samples)}"
            )
        for sample, row in zip(samples, subtask_rows):
            sample_by_frame[int(row["frame_index"]), subtask_index] = sample

    built_samples = [
        sample_by_frame[int(row["frame_index"]), int(row["current_subtask_index"])]
        for row in rows
    ]
    return anno, built_samples


def entries_to_rows(episode: Dict) -> List[Dict]:
    rows = []
    for idx, entry in enumerate(episode["entries"]):
        done = float(entry["done_label"])
        progress = float(entry["progress_label"])
        rows.append(
            {
                "sample_index": idx,
                "repo": episode["repo"],
                "repo_dir": episode["resource_repo_dir"],
                "anno_path": episode["anno_path"],
                "episode_index": episode["episode_index"],
                "frame_index": int(entry["frame_index"]),
                "current_subtask_index": int(entry["current_subtask_index"]),
                "current_subtask_goal": entry["current_subtask_goal"],
                "q2_group": entry["q2_group"],
                "done_label": done,
                "done_prob": done,
                "done_pred": int(done >= 0.5),
                "done_correct": 1,
                "progress_label": progress,
                "progress_pred": progress,
                "progress_abs_err": 0.0,
                "progress_sq_err": 0.0,
            }
        )
    return rows


def render_episode(args, episode: Dict, rank: int, output_dir: Path, views: Sequence[str]) -> Dict:
    rows = entries_to_rows(episode)
    anno_path = Path(episode["anno_path"])
    frame_indices = [int(row["frame_index"]) for row in rows]
    anno, built_samples = build_samples_for_rows(
        anno_path,
        rows,
        resolve_image_repo_dir(args, episode["repo"]),
        views,
    )
    if len(built_samples) != len(rows):
        raise ValueError(
            f"Failed to build image samples for episode {episode['repo']} "
            f"episode {episode['episode_index']}: expected {len(rows)}, got {len(built_samples)}"
        )

    stem = (
        f"{rank:03d}_{safe_name(episode['repo'])}_episode_{episode['episode_index']:06d}_"
        f"frames_{len(rows)}"
    )
    csv_path = output_dir / f"{stem}.csv"
    json_path = output_dir / f"{stem}.json"
    video_path = output_dir / f"{stem}.mp4"
    write_csv(csv_path, rows)
    write_json(
        json_path,
        {
            "episode_key": {
                "repo": episode["repo"],
                "episode_index": episode["episode_index"],
            },
            "task_goal": episode["task_goal"],
            "num_subtasks": episode["num_subtasks"],
            "anno_path": str(anno_path),
            "num_frames": len(rows),
            "frame_indices": frame_indices,
            "q2_groups": [row["q2_group"] for row in rows],
            "rows": rows,
        },
    )
    saved_video = save_q2_video(
        rows,
        built_samples,
        anno,
        video_path,
        fps=args.video_fps,
        width=args.video_width,
        top_height=args.video_top_height,
        curve_height=args.video_curve_height,
        timeline_num_frames=int(anno["num_frames"]),
    )
    return {
        "rank": rank,
        "repo": episode["repo"],
        "episode_index": episode["episode_index"],
        "task_goal": episode["task_goal"],
        "num_subtasks": episode["num_subtasks"],
        "num_frames": len(rows),
        "num_undone": sum(1 for row in rows if row["q2_group"] == "undone"),
        "num_done": sum(1 for row in rows if row["q2_group"] in {"current_done", "prev_done"}),
        "csv": str(csv_path),
        "json": str(json_path),
        "video": saved_video,
    }


def main():
    args = parse_args()
    if args.anno_root is None:
        args.anno_root = args.data_root
    views = parse_robotwin_views(args.views)

    anno_paths = collect_anno_paths(args)
    if not anno_paths:
        raise ValueError("No episode anno files found for the requested split/settings.")

    rng = random.Random(args.seed)
    rng.shuffle(anno_paths)
    if args.max_episodes is not None:
        anno_paths = anno_paths[: args.max_episodes]

    output_dir = Path(
        args.output_dir
        or Path(args.data_root).parent / f"q2_raw_data_vis_{args.split}_seed{args.seed}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "data_root": args.data_root,
        "anno_root": args.anno_root,
        "views": args.views,
        "split": args.split,
        "test_ratio": args.test_ratio,
        "split_seed": args.split_seed,
        "q2_frame_stride": args.q2_frame_stride,
        "q2_progress_bucket_size": args.q2_progress_bucket_size,
        "boundary_extra_frames": args.boundary_extra_frames,
        "num_episodes_requested": args.num_episodes,
        "num_anno_paths": len(anno_paths),
        "seed": args.seed,
        "saved": [],
        "skipped": [],
    }

    saved_count = 0
    for anno_path in anno_paths:
        if saved_count >= args.num_episodes:
            break
        label = f"{anno_path.parent.parent.name}/{anno_path.name}"
        print(f"Trying episode {saved_count + 1}/{args.num_episodes}: {label}", flush=True)
        try:
            ref = ref_from_anno_path(anno_path)
            episode = load_episode_entries(args, ref, views)
            if episode is None:
                print(f"skip (no entries): {label}", flush=True)
                manifest["skipped"].append({"anno_path": str(anno_path), "reason": "no entries"})
                continue
            item = render_episode(args, episode, saved_count, output_dir, views)
            manifest["saved"].append(item)
            saved_count += 1
            print(item["video"], flush=True)
        except Exception as exc:
            print(f"skip ({exc}): {label}", flush=True)
            manifest["skipped"].append(
                {
                    "anno_path": str(anno_path),
                    "reason": str(exc),
                    "traceback": traceback.format_exc(limit=3),
                }
            )

    if saved_count == 0:
        raise ValueError(f"Failed to save any episode videos (attempted {len(manifest['skipped'])}).")

    manifest_path = output_dir / "manifest.json"
    write_json(manifest_path, manifest)
    print(manifest_path)


if __name__ == "__main__":
    main()
