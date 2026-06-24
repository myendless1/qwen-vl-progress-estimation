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
    select_q1_samples,
    write_json,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate RobotWin Q1 full task decomposition generation.")
    parser.add_argument("--base-model", default="/media/damoxing/ckp/qwen_ft/Qwen3-VL-2B-Instruct")
    parser.add_argument("--checkpoint", default="/media/damoxing/ckp/qwen_ft/robotwin_qwen3vl_2b")
    parser.add_argument("--data-root", default="/media/damoxing/datasets/vae4d/lerobot-vae4d-org/robotwin_gt_depth")
    parser.add_argument("--output-json", default="/media/damoxing/ckp/qwen_ft/robotwin_qwen3vl_2b/eval_q1.json")
    parser.add_argument("--split", choices=("train", "test", "all"), default="test")
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--q2-frame-stride", type=int, default=8)
    parser.add_argument("--boundary-extra-frames", type=int, default=2)
    parser.add_argument("--model-max-length", type=int, default=4096)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--shuffle-samples", action="store_true")
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
    return parser.parse_args()


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
            pred_norm = normalize_generated_text(pred_text)
            gt_norm = normalize_generated_text(gt_text)
            rows.append(
                {
                    "sample_index": index - 1,
                    "repo": sample.repo_dir.name,
                    "frame_index": sample.frame_index,
                    "current_subtask_index": sample.current_subtask_index,
                    "task_goal": sample.task_goal,
                    "completed_subtasks": sample.subtasks[: sample.current_subtask_index],
                    "gt": gt,
                    "gt_text": gt_text,
                    "prediction_text": pred_text,
                    "prediction_json": parse_json_or_none(pred_text),
                    "normalized_exact_match": pred_norm == gt_norm,
                }
            )
            if index == len(samples) or index % 20 == 0:
                print(f"Q1 generated {index}/{len(samples)}", flush=True)

    exact = sum(1 for row in rows if row["normalized_exact_match"])
    summary = {
        "checkpoint": args.checkpoint,
        "base_model": args.base_model,
        "data_root": args.data_root,
        "split": args.split,
        "test_ratio": args.test_ratio,
        "split_seed": args.split_seed,
        "max_episodes": args.max_episodes,
        "num_q1_evaluated": len(rows),
        "full_decomposition_only": not args.all_states,
        "current_subtask_index": None if args.all_states else args.current_subtask_index,
        "normalized_exact_match": exact / max(1, len(rows)),
        "wall_s": time.perf_counter() - start,
        "results": rows,
    }
    write_json(output_json, summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, ensure_ascii=False, indent=2))
    print(output_json)


if __name__ == "__main__":
    main()
