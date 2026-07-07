#!/usr/bin/env python
"""Visualize Q2 done errors: dedupe episodes, re-infer full trajectories, render like q2_raw_data."""

import argparse
import json
import random
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

TOOLS_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TOOLS_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(TOOLS_ROOT))

from qwenvl.data.robotwin_processor import (
    ROBOTWIN_IGNORE_FLOAT,
    RobotWinSample,
    _load_chunks_size,
    _robotwin_repo_dirs,
    _view_hdf5_path,
    parse_robotwin_views,
)
from tools.utils.robotwin_eval import load_eval_context, run_q2_predictions, write_csv, write_json
from tools.utils.robotwin_video import anno_path_from_prediction, save_q2_video
from tools.visualize.q2_raw_data import (
    build_samples_for_rows,
    load_episode_entries,
    resolve_image_repo_dir,
    safe_name,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Pick deduped error episodes from q2_predictions.json, "
            "re-run Q2 inference on full training-aligned trajectories, and save videos."
        )
    )
    parser.add_argument("--predictions-json", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--error-types", nargs="+", choices=("FP", "FN"), default=("FP", "FN"))
    parser.add_argument(
        "--per-repo-episodes",
        type=int,
        default=2,
        help="Pick this many error episodes from each split repo (e.g. 5 test repos x 2 = 10 videos).",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Optional global cap after per-repo selection. Ignored when --per-repo-episodes <= 0.",
    )
    parser.add_argument("--selection", choices=("worst", "first", "random"), default="worst")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--anno-root", default=None)
    parser.add_argument("--views", default=None)
    parser.add_argument("--split", choices=("train", "test", "all"), default="test")
    parser.add_argument("--test-ratio", type=float, default=None)
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--q2-frame-stride", type=int, default=1)
    parser.add_argument(
        "--q2-progress-bucket-size",
        type=float,
        default=0.01,
        help="Undone progress bucket width for full-episode visualization sampling.",
    )
    parser.add_argument("--boundary-extra-frames", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--model-max-length", type=int, default=4096)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--video-fps", type=float, default=2.0)
    parser.add_argument("--video-width", type=int, default=1280)
    parser.add_argument("--video-top-height", type=int, default=720)
    parser.add_argument("--video-curve-height", type=int, default=420)
    return parser.parse_args()


def load_predictions(path: Path):
    with open(path, "r") as f:
        data = json.load(f)
    if isinstance(data, list):
        return {}, data
    return data, data.get("rows", [])


def apply_eval_defaults(args, meta):
    for name in (
        "base_model",
        "checkpoint",
        "data_root",
        "anno_root",
        "views",
        "split",
        "test_ratio",
        "split_seed",
        "threshold",
        "q2_frame_stride",
        "q2_progress_bucket_size",
        "boundary_extra_frames",
        "batch_size",
        "model_max_length",
    ):
        meta_key = name if name != "q2_frame_stride" else "q2_frame_stride"
        if name == "q2_progress_bucket_size" and "q2_progress_bucket_size" not in meta:
            meta_key = None
        if getattr(args, name, None) is None and meta_key and meta_key in meta:
            setattr(args, name, meta[meta_key])
    if args.data_root is None:
        raise ValueError("--data-root is required when it is not present in predictions JSON.")
    if args.checkpoint is None:
        raise ValueError("--checkpoint is required when it is not present in predictions JSON.")
    if args.base_model is None:
        raise ValueError("--base-model is required when it is not present in predictions JSON.")
    if args.threshold is None:
        args.threshold = 0.5
    if args.anno_root is None:
        args.anno_root = args.data_root
    if args.test_ratio is None:
        args.test_ratio = 0.05
    if args.split_seed is None:
        args.split_seed = 0


def error_type(row, threshold):
    label = int(float(row["done_label"]) >= threshold)
    pred = int(float(row["done_pred"]))
    if label == 0 and pred == 1:
        return "FP"
    if label == 1 and pred == 0:
        return "FN"
    return None


def episode_key(row):
    if row.get("anno_path"):
        return row["anno_path"]
    return (row.get("repo_dir") or row.get("repo"), int(row.get("episode_index", -1)))


def repo_name_from_row(row) -> str:
    if row.get("repo"):
        return str(row["repo"])
    if row.get("repo_dir"):
        return Path(row["repo_dir"]).name
    if row.get("anno_path"):
        return Path(row["anno_path"]).parent.parent.name
    raise ValueError(f"Cannot resolve repo name from row: {row}")


def collect_error_rows(rows, threshold, wanted_types) -> List[Dict]:
    errors = []
    for row in rows:
        kind = error_type(row, threshold)
        if kind in wanted_types:
            item = dict(row)
            item["error_type"] = kind
            errors.append(item)
    return errors


def best_error_per_episode(errors: Sequence[Dict]) -> Dict:
    best_by_episode: Dict = {}
    for row in errors:
        key = episode_key(row)
        if key not in best_by_episode or error_score(row) > error_score(best_by_episode[key]):
            best_by_episode[key] = row
    return best_by_episode


def error_score(row):
    prob = float(row["done_prob"])
    return prob if row["error_type"] == "FP" else 1.0 - prob


def rank_episode_errors(items, selection, seed):
    items = list(items)
    if selection == "worst":
        items.sort(key=error_score, reverse=True)
    elif selection == "random":
        random.Random(seed).shuffle(items)
    else:
        items.sort(key=lambda row: int(row.get("sample_index", 0)))
    return items


def select_error_episodes(rows, threshold, wanted_types, max_episodes, selection, seed):
    errors = collect_error_rows(rows, threshold, wanted_types)
    best_by_episode = best_error_per_episode(errors)
    ranked = rank_episode_errors(list(best_by_episode.values()), selection, seed)
    limit = max_episodes if max_episodes is not None else len(ranked)
    return ranked[:limit], len(errors), len(best_by_episode)


def split_repo_names(
    data_root: str,
    split: str,
    test_ratio: float,
    split_seed: int,
    anno_root: Optional[str],
) -> List[str]:
    repos = _robotwin_repo_dirs(
        data_root,
        split=split,
        test_ratio=test_ratio,
        split_seed=split_seed,
        anno_root=anno_root,
    )
    return sorted(repo.name for repo in repos)


def select_error_episodes_per_repo(
    rows,
    threshold,
    wanted_types,
    repo_names: Sequence[str],
    per_repo: int,
    selection: str,
    seed: int,
    max_episodes: Optional[int] = None,
) -> Tuple[List[Dict], int, int, Dict[str, int]]:
    errors = collect_error_rows(rows, threshold, wanted_types)
    best_by_episode = best_error_per_episode(errors)

    by_repo: Dict[str, List[Dict]] = defaultdict(list)
    for row in best_by_episode.values():
        by_repo[repo_name_from_row(row)].append(row)

    selected: List[Dict] = []
    picked_per_repo: Dict[str, int] = {}
    for repo in repo_names:
        ranked = rank_episode_errors(by_repo.get(repo, []), selection, seed)
        picked = ranked[: max(0, per_repo)]
        picked_per_repo[repo] = len(picked)
        selected.extend(picked)

    if max_episodes is not None:
        selected = selected[:max_episodes]
    return selected, len(errors), len(best_by_episode), picked_per_repo


def episode_ref_from_error(error: Dict) -> Dict:
    anno_path = error.get("anno_path")
    if not anno_path:
        anno_path = str(anno_path_from_prediction(error))
    resource_dir = Path(anno_path).parent.parent
    return {
        "repo": error.get("repo") or resource_dir.name,
        "resource_repo_dir": str(error.get("repo_dir") or resource_dir),
        "anno_path": str(anno_path),
        "episode_index": int(error["episode_index"]),
        "source_error": error,
    }


def episode_entries_to_samples(episode: Dict, views: Sequence[str], image_repo_dir: Optional[Path]):
    resource_dir = Path(episode["resource_repo_dir"])
    image_repo_dir = image_repo_dir or resource_dir
    episode_index = int(episode["episode_index"])
    chunks_size = _load_chunks_size(resource_dir)
    with open(episode["anno_path"], "r") as f:
        anno = json.load(f)
    subtasks = anno["subtasks"]
    image_hdf5_paths = {
        view: _view_hdf5_path(image_repo_dir, episode_index, chunks_size, view)
        for view in views
    }
    samples: List[RobotWinSample] = []
    for entry in episode["entries"]:
        subtask_index = int(entry["current_subtask_index"])
        samples.append(
            RobotWinSample(
                kind="q2",
                repo_dir=resource_dir,
                image_hdf5_paths=image_hdf5_paths,
                frame_index=int(entry["frame_index"]),
                frame_start=int(entry["frame_index"]),
                frame_end=int(entry["frame_index"]),
                task_goal=episode["task_goal"],
                subtasks=subtasks,
                current_subtask_index=subtask_index,
                views=tuple(views),
                image_repo_dir=image_repo_dir,
                current_done=float(entry["done_label"]),
                need_replan=ROBOTWIN_IGNORE_FLOAT,
                incident=ROBOTWIN_IGNORE_FLOAT,
                progress=float(entry["progress_label"]),
                q2_group=entry["q2_group"],
                state_values=entry.get("state_values"),
            )
        )
    return samples, anno


def count_row_errors(rows, threshold) -> Dict[str, int]:
    counts = {"FP": 0, "FN": 0}
    for row in rows:
        kind = error_type(row, threshold)
        if kind:
            counts[kind] += 1
    return counts


def render_episode_with_predictions(
    args,
    episode: Dict,
    rows: List[Dict],
    rank: int,
    output_dir: Path,
    views: Sequence[str],
    source_error: Dict,
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
            "source_error": source_error,
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
        "source_error_type": source_error.get("error_type"),
        "source_error_frame": int(source_error.get("frame_index", -1)),
        "csv": str(csv_path),
        "json": str(json_path),
        "video": saved_video,
    }


def main():
    args = parse_args()
    predictions_path = Path(args.predictions_json)
    meta, pred_rows = load_predictions(predictions_path)
    apply_eval_defaults(args, meta)
    if args.output_dir is None:
        args.output_dir = str(predictions_path.parent / "q2_done_error_videos")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    views = parse_robotwin_views(args.views or "main,left_wrist,right_wrist")

    if args.per_repo_episodes > 0:
        repo_names = split_repo_names(
            args.data_root,
            args.split,
            args.test_ratio,
            args.split_seed,
            args.anno_root,
        )
        episode_errors, total_errors, unique_error_episodes, picked_per_repo = select_error_episodes_per_repo(
            pred_rows,
            threshold=args.threshold,
            wanted_types=set(args.error_types),
            repo_names=repo_names,
            per_repo=args.per_repo_episodes,
            selection=args.selection,
            seed=args.seed,
            max_episodes=args.max_episodes,
        )
    else:
        repo_names = []
        picked_per_repo = {}
        episode_errors, total_errors, unique_error_episodes = select_error_episodes(
            pred_rows,
            threshold=args.threshold,
            wanted_types=set(args.error_types),
            max_episodes=args.max_episodes or 12,
            selection=args.selection,
            seed=args.seed,
        )
    if not episode_errors:
        raise ValueError("No matching FP/FN errors found in predictions JSON.")

    vis_args = SimpleNamespace(
        data_root=args.data_root,
        anno_root=args.anno_root,
        q2_frame_stride=args.q2_frame_stride,
        q2_progress_bucket_size=args.q2_progress_bucket_size,
        boundary_extra_frames=args.boundary_extra_frames,
    )
    context = load_eval_context(args, prefer_checkpoint_processor=False)

    manifest = {
        "predictions_json": str(predictions_path),
        "checkpoint": args.checkpoint,
        "base_model": args.base_model,
        "output_dir": str(output_dir),
        "threshold": args.threshold,
        "split": args.split,
        "test_ratio": args.test_ratio,
        "split_seed": args.split_seed,
        "per_repo_episodes": args.per_repo_episodes,
        "split_repos": repo_names,
        "picked_per_repo": picked_per_repo,
        "q2_frame_stride": args.q2_frame_stride,
        "q2_progress_bucket_size": args.q2_progress_bucket_size,
        "total_matching_errors": total_errors,
        "unique_error_episodes": unique_error_episodes,
        "saved": [],
        "skipped": [],
    }

    for rank, source_error in enumerate(episode_errors):
        ref = episode_ref_from_error(source_error)
        label = f"{ref['repo']}/episode_{ref['episode_index']:06d}"
        print(f"Episode {rank + 1}/{len(episode_errors)}: {label}", flush=True)
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
            rows = run_q2_predictions(
                args,
                samples,
                context=context,
                progress_prefix=f"Q2 error vis {rank + 1}/{len(episode_errors)}",
            )
            item = render_episode_with_predictions(
                args,
                episode,
                rows,
                rank,
                output_dir,
                views,
                source_error,
            )
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
        raise ValueError("Failed to save any error episode videos.")

    manifest_path = output_dir / "manifest.json"
    write_json(manifest_path, manifest)
    print(manifest_path)


if __name__ == "__main__":
    main()
