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
    move_to_device,
    normalize_generated_text,
    parse_json_or_none,
    prepare_q1_prompt,
    q1_ground_truth,
    sample_items,
    select_one_q1_sample_per_task,
    select_q1_samples,
    task_slug_from_repo_name,
    write_xlsx,
    write_json,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate RobotWin Q1 full task decomposition generation.")
    parser.add_argument("--base-model", default="/media/damoxing/ckp/qwen_ft/Qwen3-VL-2B-Instruct")
    parser.add_argument("--checkpoint", default="/media/damoxing/ckp/qwen_ft/robotwin_qwen3vl_2b")
    parser.add_argument("--data-root", default="/media/damoxing/datasets/vae4d/lerobot-vae4d-org/robotwin_gt_depth")
    parser.add_argument(
        "--anno-root",
        default=None,
        help="Optional anno/meta root. When set, images come from --data-root and annotations from --anno-root.",
    )
    parser.add_argument(
        "--views",
        default="main,left_wrist,right_wrist",
        help="Comma-separated camera views, e.g. 'main' or 'main,left_wrist,right_wrist'.",
    )
    parser.add_argument("--output-json", default="/media/damoxing/ckp/qwen_ft/robotwin_qwen3vl_2b/eval_q1.json")
    parser.add_argument("--output-xlsx", default=None)
    parser.add_argument("--split", choices=("train", "test", "all"), default="test")
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--q2-frame-stride", type=int, default=8)
    parser.add_argument("--boundary-extra-frames", type=int, default=2)
    parser.add_argument("--model-max-length", type=int, default=4096)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help="After Q1 state filtering, select one example per deduplicated task slug and keep up to this many tasks.",
    )
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--shuffle-samples", action="store_true")
    parser.add_argument(
        "--one-example-per-task",
        action="store_true",
        help="Select one Q1 example from each deduplicated task slug before applying --max-tasks.",
    )
    parser.add_argument(
        "--all-states",
        action="store_true",
        help="Evaluate Q1 prompts at every subtask boundary. By default only current_subtask_index=0 is used, i.e. full decomposition.",
    )
    parser.add_argument("--current-subtask-index", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--attn-implementation", default=os.environ.get("ATTN_IMPLEMENTATION", "sdpa"))
    parser.add_argument(
        "--voting-done",
        action="store_true",
        help="Load a RobotWin checkpoint trained with done voting heads.",
    )
    parser.add_argument("--done-vote-count", type=int, default=5)
    return parser.parse_args()


def _subtask_index(item, fallback):
    if isinstance(item, dict):
        value = item.get("subtask_index", fallback)
        try:
            return int(value)
        except (TypeError, ValueError):
            return fallback
    return fallback


def _subtask_goal(item):
    if not isinstance(item, dict):
        return None
    return item.get("subtask_goal")


def merge_q1_subtasks(gt, prediction):
    gt_items = gt if isinstance(gt, list) else []
    pred_items = prediction if isinstance(prediction, list) else []
    gt_by_index = {_subtask_index(item, idx): item for idx, item in enumerate(gt_items)}
    pred_by_index = {_subtask_index(item, idx): item for idx, item in enumerate(pred_items)}
    indices = sorted(set(gt_by_index) | set(pred_by_index))
    return [
        {
            "subtask_index": index,
            "subtask_goal": _subtask_goal(gt_by_index.get(index)),
            "subtask_pred": _subtask_goal(pred_by_index.get(index)),
        }
        for index in indices
    ]


def main():
    args = parse_args()
    random.seed(args.sample_seed)
    torch.manual_seed(args.sample_seed)
    output_json = Path(args.output_json)

    samples = build_samples(args, kind="q1")
    samples = select_q1_samples(
        samples,
        all_states=args.all_states,
        current_subtask_index=args.current_subtask_index,
    )
    if args.one_example_per_task:
        samples = select_one_q1_sample_per_task(
            samples,
            max_tasks=args.max_tasks,
            seed=args.sample_seed,
            shuffle_tasks=args.shuffle_samples,
        )
    else:
        samples = sample_items(samples, args.max_samples, seed=args.sample_seed, shuffle=args.shuffle_samples)
    if not samples:
        raise ValueError("No Q1 samples found for the requested split/settings.")

    context = load_eval_context(args, prefer_checkpoint_processor=True)
    model = context["model"]
    processor = context["processor"]
    tokenizer = context["tokenizer"]
    base_model = model.base_model
    base_model.eval()

    rows = []
    start = time.perf_counter()
    with torch.inference_mode():
        for index, sample in enumerate(samples, 1):
            prompt = move_to_device(prepare_q1_prompt(sample, processor), args.device)
            input_len = prompt["input_ids"].shape[1]
            generated = base_model.generate(
                **prompt,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            pred_text = tokenizer.decode(generated[0, input_len:], skip_special_tokens=True).strip()
            gt = q1_ground_truth(sample)
            gt_text = json.dumps(gt, ensure_ascii=False, separators=(",", ":"))
            prediction_json = parse_json_or_none(pred_text)
            pred_norm = normalize_generated_text(pred_text)
            gt_norm = normalize_generated_text(gt_text)
            rows.append(
                {
                    "sample_index": index - 1,
                    "repo": sample.repo_dir.name,
                    "task_slug": task_slug_from_repo_name(sample.repo_dir.name),
                    "frame_index": sample.frame_index,
                    "current_subtask_index": sample.current_subtask_index,
                    "task_goal": sample.task_goal,
                    "completed_subtasks": sample.subtasks[: sample.current_subtask_index],
                    "subtasks": merge_q1_subtasks(gt, prediction_json),
                    "normalized_exact_match": pred_norm == gt_norm,
                    "raw": {
                        "gt": gt,
                        "gt_text": gt_text,
                        "prediction_text": pred_text,
                        "prediction_json": prediction_json,
                    },
                }
            )
            if index == len(samples) or index % 20 == 0:
                print(f"Q1 generated {index}/{len(samples)}", flush=True)

    exact = sum(1 for row in rows if row["normalized_exact_match"])
    output_xlsx = Path(args.output_xlsx) if args.output_xlsx else output_json.with_suffix(".xlsx")
    summary = {
        "checkpoint": args.checkpoint,
        "base_model": args.base_model,
        "data_root": args.data_root,
        "split": args.split,
        "test_ratio": args.test_ratio,
        "split_seed": args.split_seed,
        "max_episodes": args.max_episodes,
        "one_example_per_task": args.one_example_per_task,
        "max_tasks": args.max_tasks,
        "num_q1_evaluated": len(rows),
        "num_unique_task_slugs": len({row["task_slug"] for row in rows}),
        "full_decomposition_only": not args.all_states,
        "current_subtask_index": None if args.all_states else args.current_subtask_index,
        "normalized_exact_match": exact / max(1, len(rows)),
        "wall_s": time.perf_counter() - start,
        "output_xlsx": str(output_xlsx),
        "results": rows,
    }
    write_json(output_json, summary)
    write_xlsx(output_xlsx, rows, sheet_name="q1_results")
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, ensure_ascii=False, indent=2))
    print(output_json)
    print(output_xlsx)


if __name__ == "__main__":
    main()
