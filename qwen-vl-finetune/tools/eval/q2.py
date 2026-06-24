#!/usr/bin/env python
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import torch

TOOLS_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TOOLS_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(TOOLS_ROOT))

from tools.utils.robotwin_eval import (
    build_samples,
    load_eval_context,
    run_q2_predictions,
    sample_items,
    summarize_q2,
    write_csv,
    write_json,
)
from tools.utils.robotwin_video import build_episode_samples, collect_episode_annos, save_q2_video


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate RobotWin Q2 current-done classification and progress regression.")
    parser.add_argument("--base-model", default="/media/damoxing/ckp/qwen_ft/Qwen3-VL-2B-Instruct")
    parser.add_argument("--checkpoint", default="/media/damoxing/ckp/qwen_ft/robotwin_qwen3vl_2b")
    parser.add_argument("--data-root", default="/media/damoxing/datasets/vae4d/lerobot-vae4d-org/robotwin_gt_depth")
    parser.add_argument("--output-dir", default="/media/damoxing/ckp/qwen_ft/robotwin_qwen3vl_2b/eval_q2")
    parser.add_argument("--split", choices=("train", "test", "all"), default="test")
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--q2-frame-stride", type=int, default=8)
    parser.add_argument("--boundary-extra-frames", type=int, default=2)
    parser.add_argument("--model-max-length", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--shuffle-samples", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--attn-implementation", default=os.environ.get("ATTN_IMPLEMENTATION", "sdpa"))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--save-videos", action="store_true")
    parser.add_argument("--video-cases", type=int, default=5)
    parser.add_argument("--video-case-seed", type=int, default=0)
    parser.add_argument("--video-case-selection", choices=("random", "first"), default="random")
    parser.add_argument("--video-max-frames", type=int, default=None)
    parser.add_argument("--video-fps", type=float, default=10.0)
    parser.add_argument("--video-width", type=int, default=1280)
    parser.add_argument("--video-top-height", type=int, default=720)
    parser.add_argument("--video-curve-height", type=int, default=420)
    return parser.parse_args()


def run_video_visualization(args, context, output_dir):
    if not args.save_videos:
        return []
    video_dir = output_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    annos = collect_episode_annos(
        data_root=args.data_root,
        split=args.split,
        test_ratio=args.test_ratio,
        split_seed=args.split_seed,
        limit=args.video_cases,
        selection=args.video_case_selection,
        seed=args.video_case_seed,
    )
    results = []
    for index, anno_path in enumerate(annos):
        anno, samples = build_episode_samples(anno_path, max_frames=args.video_max_frames)
        rows = run_q2_predictions(args, samples, context=context, progress_prefix=f"Q2 video {index}")
        stem = f"{samples[0].repo_dir.name}_episode_{int(anno['episode_index']):06d}"
        csv_path = video_dir / f"{stem}_trajectory.csv"
        json_path = video_dir / f"{stem}_trajectory.json"
        video_path = video_dir / f"{stem}.mp4"
        write_csv(csv_path, rows)
        write_json(json_path, {"anno_path": anno_path, "rows": rows})
        saved_video = save_q2_video(
            rows,
            samples,
            anno,
            video_path,
            fps=args.video_fps,
            width=args.video_width,
            top_height=args.video_top_height,
            curve_height=args.video_curve_height,
        )
        results.append({"anno_path": anno_path, "csv": str(csv_path), "json": str(json_path), "video": saved_video})
        print(saved_video, flush=True)
    return results


def main():
    args = parse_args()
    random.seed(args.sample_seed)
    torch.manual_seed(args.sample_seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = build_samples(args, kind="q2")
    samples = sample_items(samples, args.max_samples, seed=args.sample_seed, shuffle=args.shuffle_samples)
    if not samples:
        raise ValueError("No Q2 samples found for the requested split/settings.")

    context = load_eval_context(args)
    start = time.perf_counter()
    rows = run_q2_predictions(args, samples, context=context)
    metrics = summarize_q2(rows, threshold=args.threshold)
    predictions_csv = output_dir / "q2_predictions.csv"
    predictions_json = output_dir / "q2_predictions.json"
    metrics_json = output_dir / "q2_metrics.json"
    write_csv(predictions_csv, rows)
    write_json(
        predictions_json,
        {
            "checkpoint": args.checkpoint,
            "base_model": args.base_model,
            "data_root": args.data_root,
            "split": args.split,
            "test_ratio": args.test_ratio,
            "split_seed": args.split_seed,
            "threshold": args.threshold,
            "rows": rows,
        },
    )

    video_results = run_video_visualization(args, context, output_dir)
    summary = {
        "checkpoint": args.checkpoint,
        "base_model": args.base_model,
        "data_root": args.data_root,
        "split": args.split,
        "test_ratio": args.test_ratio,
        "split_seed": args.split_seed,
        "max_episodes": args.max_episodes,
        "q2_frame_stride": args.q2_frame_stride,
        "boundary_extra_frames": args.boundary_extra_frames,
        "threshold": args.threshold,
        "predictions_csv": str(predictions_csv),
        "predictions_json": str(predictions_json),
        "videos": video_results,
        "wall_s": time.perf_counter() - start,
        **metrics,
    }
    write_json(metrics_json, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(metrics_json)


if __name__ == "__main__":
    main()
