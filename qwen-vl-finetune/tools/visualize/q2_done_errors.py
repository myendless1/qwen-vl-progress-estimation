#!/usr/bin/env python
import argparse
import json
import os
import random
import sys
from pathlib import Path

import torch

TOOLS_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TOOLS_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(TOOLS_ROOT))

from tools.utils.robotwin_eval import load_eval_context, run_q2_predictions, write_csv, write_json
from tools.utils.robotwin_video import anno_path_from_prediction, build_episode_samples, save_q2_video


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize RobotWin Q2 done FP/FN errors in local time windows.")
    parser.add_argument("--predictions-json", required=True, help="Path to q2_predictions.json produced by tools/eval/q2.py.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--error-types", nargs="+", choices=("FP", "FN"), default=("FP", "FN"))
    parser.add_argument("--max-errors", type=int, default=12)
    parser.add_argument("--num-per-type", type=int, default=None)
    parser.add_argument("--dedupe-episode", action="store_true")
    parser.add_argument("--selection", choices=("worst", "first", "random"), default="worst")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--window-before", type=int, default=24)
    parser.add_argument("--window-after", type=int, default=24)
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--split", choices=("train", "test", "all"), default=None)
    parser.add_argument("--test-ratio", type=float, default=None)
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--q2-frame-stride", type=int, default=8)
    parser.add_argument("--boundary-extra-frames", type=int, default=2)
    parser.add_argument("--model-max-length", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--attn-implementation", default=os.environ.get("ATTN_IMPLEMENTATION", "sdpa"))
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--video-fps", type=float, default=10.0)
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
    for name in ("base_model", "checkpoint", "data_root", "split", "test_ratio", "split_seed", "threshold"):
        if getattr(args, name) is None and name in meta:
            setattr(args, name, meta[name])
    if args.base_model is None or args.checkpoint is None:
        raise ValueError("--base-model and --checkpoint are required when they are not present in predictions JSON.")
    if args.data_root is None:
        raise ValueError("--data-root is required when it is not present in predictions JSON.")
    if args.split is None:
        args.split = "test"
    if args.test_ratio is None:
        args.test_ratio = 0.05
    if args.split_seed is None:
        args.split_seed = 0
    if args.threshold is None:
        args.threshold = 0.5


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


def rank_errors(items, selection, seed):
    items = list(items)
    if selection == "worst":
        def key(row):
            prob = float(row["done_prob"])
            return prob if row["error_type"] == "FP" else 1.0 - prob
        items.sort(key=key, reverse=True)
    elif selection == "random":
        random.Random(seed).shuffle(items)
    else:
        items.sort(key=lambda row: int(row.get("sample_index", 0)))
    return items


def dedupe_by_episode(items):
    seen = set()
    deduped = []
    for row in items:
        key = episode_key(row)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def select_errors(rows, threshold, wanted_types, max_errors, num_per_type, dedupe_episode, selection, seed):
    selected = []
    for row in rows:
        kind = error_type(row, threshold)
        if kind in wanted_types:
            item = dict(row)
            item["error_type"] = kind
            selected.append(item)

    total = len(selected)
    if num_per_type is not None:
        final = []
        for kind in wanted_types:
            ranked = rank_errors([row for row in selected if row["error_type"] == kind], selection, seed)
            if dedupe_episode:
                ranked = dedupe_by_episode(ranked)
            final.extend(ranked[:num_per_type])
        return final, total

    ranked = rank_errors(selected, selection, seed)
    if dedupe_episode:
        ranked = dedupe_by_episode(ranked)
    return ranked[:max_errors], total


def safe_name(value):
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value))


def main():
    args = parse_args()
    predictions_path = Path(args.predictions_json)
    meta, rows = load_predictions(predictions_path)
    apply_eval_defaults(args, meta)
    if args.output_dir is None:
        args.output_dir = str(predictions_path.parent / "q2_done_error_videos")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    errors, total_errors = select_errors(
        rows,
        threshold=args.threshold,
        wanted_types=set(args.error_types),
        max_errors=args.max_errors,
        num_per_type=args.num_per_type,
        dedupe_episode=args.dedupe_episode,
        selection=args.selection,
        seed=args.seed,
    )
    if not errors:
        raise ValueError("No matching FP/FN errors found in predictions JSON.")

    context = load_eval_context(args)
    manifest = {
        "predictions_json": str(predictions_path),
        "output_dir": str(output_dir),
        "threshold": args.threshold,
        "num_per_type": args.num_per_type,
        "dedupe_episode": args.dedupe_episode,
        "total_matching_errors": total_errors,
        "saved": [],
    }
    for rank, error in enumerate(errors):
        anno_path = anno_path_from_prediction(error)
        center_frame = int(error["frame_index"])
        start_frame = max(0, center_frame - args.window_before)
        end_frame = center_frame + args.window_after
        anno, samples = build_episode_samples(
            anno_path,
            start_frame=start_frame,
            end_frame=end_frame,
            fixed_subtask_index=int(error["current_subtask_index"]),
        )
        rows_window = run_q2_predictions(args, samples, context=context, progress_prefix=f"{error['error_type']} {rank}")

        stem = (
            f"{rank:03d}_{error['error_type']}_"
            f"{safe_name(error.get('repo', Path(error.get('repo_dir', 'repo')).name))}_"
            f"episode_{int(anno['episode_index']):06d}_frame_{center_frame:06d}"
        )
        csv_path = output_dir / f"{stem}.csv"
        json_path = output_dir / f"{stem}.json"
        video_path = output_dir / f"{stem}.mp4"
        write_csv(csv_path, rows_window)
        write_json(
            json_path,
            {
                "source_error": error,
                "anno_path": str(anno_path),
                "window_before": args.window_before,
                "window_after": args.window_after,
                "rows": rows_window,
            },
        )
        saved_video = save_q2_video(
            rows_window,
            samples,
            anno,
            video_path,
            fps=args.video_fps,
            width=args.video_width,
            top_height=args.video_top_height,
            curve_height=args.video_curve_height,
            event_frame=center_frame,
        )
        item = {
            "error_type": error["error_type"],
            "source_sample_index": error.get("sample_index"),
            "repo": error.get("repo"),
            "anno_path": str(anno_path),
            "frame_index": center_frame,
            "done_label": error["done_label"],
            "done_pred": error["done_pred"],
            "done_prob": error["done_prob"],
            "csv": str(csv_path),
            "json": str(json_path),
            "video": saved_video,
        }
        manifest["saved"].append(item)
        print(saved_video, flush=True)

    manifest_path = output_dir / "manifest.json"
    write_json(manifest_path, manifest)
    print(manifest_path)


if __name__ == "__main__":
    main()
