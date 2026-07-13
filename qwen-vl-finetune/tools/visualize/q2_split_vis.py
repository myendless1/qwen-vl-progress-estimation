#!/usr/bin/env python
"""Visualize Q2 on random episodes from a RobotWin split (e.g. train), with model inference."""

import argparse
import random
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence

TOOLS_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TOOLS_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(TOOLS_ROOT))
sys.path.insert(0, str(TOOLS_ROOT / "eval"))

from qwenvl.data.robotwin_processor import _robotwin_repo_dirs, parse_robotwin_views
from tools.utils.robotwin_eval import load_eval_context, run_q2_predictions, write_csv, write_json
from tools.visualize.q2_done_errors import (
    count_row_errors,
    episode_entries_to_samples,
)
from tools.visualize.q2_raw_data import (
    build_samples_for_rows,
    load_episode_entries,
    ref_from_anno_path,
    resolve_image_repo_dir,
    safe_name,
)
from tools.utils.robotwin_video import save_q2_video
from robotwin_eval_config import apply_config_argv


def _argv_has_option(argv, *names):
    for index, arg in enumerate(argv):
        if arg in names or any(arg.startswith(f"{name}=") for name in names):
            return True
        if index > 0 and argv[index - 1] in names:
            return True
    return False


def _resolve_split_output_dir(args, original_argv):
    if _argv_has_option(original_argv, "--output-dir"):
        return args
    split = "test" if args.split == "eval" else args.split
    for attr in (f"{split}_vis_output_dir", f"{split}_output_dir"):
        value = getattr(args, attr, None)
        if value:
            args.output_dir = value
            break
    return args


def parse_args():
    original_argv = sys.argv[1:]
    apply_config_argv("q2_vis")
    parser = argparse.ArgumentParser(
        description=(
            "Pick random episodes from one RobotWin split (default train), "
            "re-run Q2 inference on full training-aligned trajectories, and save videos."
        )
    )
    parser.add_argument("--data-root", default="/media/damoxing/datasets/vae4d/lerobot-vae4d-org/robotwin_gt_depth")
    parser.add_argument("--anno-root", default=None)
    parser.add_argument("--anno-dir", default="anno")
    parser.add_argument("--views", default="main,left_wrist,right_wrist")
    parser.add_argument("--split", choices=("train", "test", "all"), default="train")
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=15,
        help="Number of task repos to sample. <=0 uses every repo in the split.",
    )
    parser.add_argument(
        "--per-repo-episodes",
        type=int,
        default=1,
        help="Random episodes to sample from each selected task repo.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--train-output-dir", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--test-output-dir", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--train-vis-output-dir", default=None)
    parser.add_argument("--test-vis-output-dir", default=None)
    parser.add_argument("--train-sample-manifest", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--test-sample-manifest", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--max-samples", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--shuffle-samples", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--base-model", default="/media/damoxing/ckp/qwen_ft/Qwen3-VL-2B-Instruct")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--memory-frames", type=int, default=1)
    parser.add_argument("--memory-frame-stride", type=int, default=1)
    parser.add_argument("--q2-frame-stride", type=int, default=1)
    parser.add_argument("--q2-progress-bucket-size", type=float, default=0.01)
    parser.add_argument("--boundary-extra-frames", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--model-max-length", type=int, default=4096)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--voting-done", action="store_true")
    parser.add_argument("--done-vote-count", type=int, default=5)
    parser.add_argument("--done-vote-threshold", type=int, default=3)
    parser.add_argument("--video-fps", type=float, default=2.0)
    parser.add_argument("--video-width", type=int, default=1280)
    parser.add_argument("--video-top-height", type=int, default=720)
    parser.add_argument("--video-curve-height", type=int, default=420)
    return _resolve_split_output_dir(parser.parse_args(), original_argv)


def resource_repo_dir(data_root: Path, anno_root: Optional[str], repo_name: str) -> Path:
    if anno_root and Path(data_root).resolve() != Path(anno_root).resolve():
        return Path(anno_root) / repo_name
    return data_root / repo_name


def select_episode_refs(
    data_root: str,
    anno_root: Optional[str],
    anno_dir_name: str,
    split: str,
    test_ratio: float,
    split_seed: int,
    num_tasks: int,
    per_repo_episodes: int,
    seed: int,
) -> List[Dict]:
    split_arg = None if split == "all" else split
    repo_dirs = list(
        _robotwin_repo_dirs(
            data_root,
            split=split_arg,
            test_ratio=test_ratio,
            split_seed=split_seed,
            anno_root=anno_root,
            anno_dir_name=anno_dir_name,
        )
    )
    if not repo_dirs:
        raise ValueError(f"No repos found for split={split!r}.")

    rng = random.Random(seed)
    repo_dirs = sorted(repo_dirs, key=lambda path: path.name)
    rng.shuffle(repo_dirs)
    if num_tasks > 0:
        repo_dirs = repo_dirs[:num_tasks]

    refs: List[Dict] = []
    data_root_path = Path(data_root)
    per_repo_episodes = max(1, per_repo_episodes)
    for image_repo_dir in repo_dirs:
        resource_dir = resource_repo_dir(data_root_path, anno_root, image_repo_dir.name)
        anno_paths = sorted((resource_dir / anno_dir_name).glob("episode_*.json"))
        if not anno_paths:
            continue
        pick_count = min(per_repo_episodes, len(anno_paths))
        chosen_paths = (
            rng.sample(anno_paths, pick_count)
            if pick_count < len(anno_paths)
            else anno_paths[:pick_count]
        )
        for anno_path in sorted(chosen_paths, key=lambda path: path.name):
            refs.append(ref_from_anno_path(anno_path))
    return refs


def render_episode(
    args,
    episode: Dict,
    rows: List[Dict],
    rank: int,
    output_dir: Path,
    views: Sequence[str],
) -> Dict:
    anno_path = Path(episode["anno_path"])
    image_repo_dir = resolve_image_repo_dir(args, episode["repo"])
    anno, built_samples = build_samples_for_rows(anno_path, rows, image_repo_dir, views)
    if len(built_samples) != len(rows):
        raise ValueError(
            f"Failed to build image samples for episode {episode['repo']} "
            f"episode {episode['episode_index']}: expected {len(rows)}, got {len(built_samples)}"
        )

    err_counts = count_row_errors(rows, args.threshold)
    stem = (
        f"{rank:03d}_{safe_name(episode['repo'])}_episode_{episode['episode_index']:06d}_"
        f"frames_{len(rows)}_fp{err_counts['FP']}_fn{err_counts['FN']}"
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
            "error_counts": err_counts,
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
        "error_counts": err_counts,
        "csv": str(csv_path),
        "json": str(json_path),
        "video": saved_video,
    }


def main():
    args = parse_args()
    if args.anno_root is None:
        args.anno_root = args.data_root

    output_dir = Path(
        args.output_dir
        or Path(args.data_root).parent / f"q2_{args.split}_vis_seed{args.seed}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    views = parse_robotwin_views(args.views)

    episode_refs = select_episode_refs(
        args.data_root,
        args.anno_root,
        args.anno_dir,
        args.split,
        args.test_ratio,
        args.split_seed,
        args.num_tasks,
        args.per_repo_episodes,
        args.seed,
    )
    if not episode_refs:
        raise ValueError(f"No episodes selected for split={args.split!r}.")

    selected_repos = sorted({ref["repo"] for ref in episode_refs})

    vis_args = SimpleNamespace(
        data_root=args.data_root,
        anno_root=args.anno_root,
        anno_dir=args.anno_dir,
        q2_frame_stride=args.q2_frame_stride,
        q2_progress_bucket_size=args.q2_progress_bucket_size,
        boundary_extra_frames=args.boundary_extra_frames,
    )
    context = load_eval_context(args, prefer_checkpoint_processor=False)

    manifest = {
        "split": args.split,
        "test_ratio": args.test_ratio,
        "split_seed": args.split_seed,
        "num_tasks_requested": args.num_tasks,
        "per_repo_episodes": args.per_repo_episodes,
        "num_tasks_selected": len(selected_repos),
        "num_episodes_selected": len(episode_refs),
        "selected_repos": selected_repos,
        "seed": args.seed,
        "checkpoint": args.checkpoint,
        "base_model": args.base_model,
        "data_root": args.data_root,
        "anno_root": args.anno_root,
        "anno_dir": args.anno_dir,
        "views": args.views,
        "memory_frames": args.memory_frames,
        "memory_frame_stride": args.memory_frame_stride,
        "q2_frame_stride": args.q2_frame_stride,
        "q2_progress_bucket_size": args.q2_progress_bucket_size,
        "output_dir": str(output_dir),
        "saved": [],
        "skipped": [],
    }

    for rank, ref in enumerate(episode_refs):
        label = f"{ref['repo']}/episode_{ref['episode_index']:06d}"
        print(f"Episode {rank + 1}/{len(episode_refs)}: {label}", flush=True)
        try:
            episode = load_episode_entries(vis_args, ref, views)
            if episode is None:
                manifest["skipped"].append({"ref": ref, "reason": "no entries"})
                continue

            samples, _ = episode_entries_to_samples(
                episode,
                views,
                resolve_image_repo_dir(args, episode["repo"]),
            )
            for sample in samples:
                sample.memory_frames = args.memory_frames
                sample.memory_frame_stride = args.memory_frame_stride
            rows = run_q2_predictions(
                args,
                samples,
                context=context,
                progress_prefix=f"Q2 {args.split} vis {rank + 1}/{len(episode_refs)}",
            )
            item = render_episode(args, episode, rows, rank, output_dir, views)
            manifest["saved"].append(item)
            print(item["video"], flush=True)
        except Exception as exc:
            print(f"skip ({exc}): {label}", flush=True)
            manifest["skipped"].append(
                {
                    "ref": ref,
                    "reason": str(exc),
                    "traceback": traceback.format_exc(limit=3),
                }
            )

    if not manifest["saved"]:
        raise ValueError("Failed to save any episode videos.")

    manifest_path = output_dir / "manifest.json"
    write_json(manifest_path, manifest)
    print(manifest_path)


if __name__ == "__main__":
    main()
