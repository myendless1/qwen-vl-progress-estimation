#!/usr/bin/env python
import argparse
import dataclasses
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

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
from robotwin_eval_config import apply_config_argv


def parse_args():
    apply_config_argv("q1")
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
        "--anno-dir",
        default="anno",
        help="Annotation directory name inside each RobotWin task directory.",
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
    parser.add_argument("--memory-frames", type=int, default=1)
    parser.add_argument("--memory-frame-stride", type=int, default=1)
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
    parser.add_argument(
        "--one-episode-per-task-all-undone-subtasks",
        action="store_true",
        help=(
            "For each deduplicated task slug, choose one episode, then evaluate one undone frame "
            "from every subtask in that episode."
        ),
    )
    parser.add_argument(
        "--subsequent-windows",
        default="1,3,5",
        help="Comma-separated numbers of subsequent subtasks to score from the current subtask.",
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


def _episode_key(sample) -> tuple:
    paths = tuple(str(sample.image_hdf5_paths[view]) for view in sorted(sample.image_hdf5_paths))
    return sample.repo_dir.name, paths


def _undone_end(sample) -> int:
    end = int(sample.frame_end)
    if sample.q1_done_start_frame is not None:
        end = min(end, int(sample.q1_done_start_frame) - 1)
    return end


def _fix_q1_sample_to_undone_frame(sample, rng: random.Random):
    start = int(sample.frame_start)
    end = _undone_end(sample)
    if end < start:
        return None
    frame = rng.randint(start, end)
    return dataclasses.replace(sample, frame_index=frame, frame_start=frame, frame_end=frame)


def select_one_episode_per_task_all_undone_subtasks(samples, max_tasks: Optional[int], seed: int, shuffle_tasks: bool):
    by_task: Dict[str, Dict[tuple, List[Any]]] = defaultdict(lambda: defaultdict(list))
    for sample in samples:
        if _undone_end(sample) < int(sample.frame_start):
            continue
        by_task[task_slug_from_repo_name(sample.repo_dir.name)][_episode_key(sample)].append(sample)

    rng = random.Random(seed)
    selected_tasks = sorted(by_task)
    if shuffle_tasks:
        rng.shuffle(selected_tasks)
    if max_tasks is not None:
        selected_tasks = selected_tasks[:max_tasks]

    selected = []
    for task_slug in selected_tasks:
        episodes = list(by_task[task_slug].values())
        episode_samples = sorted(rng.choice(episodes), key=lambda item: int(item.current_subtask_index))
        for sample in episode_samples:
            fixed = _fix_q1_sample_to_undone_frame(sample, rng)
            if fixed is not None:
                selected.append(fixed)
    return selected


def parse_windows(value: str) -> List[int]:
    windows = []
    for part in value.split(","):
        part = part.strip()
        if part:
            windows.append(int(part))
    return sorted(set(w for w in windows if w > 0))


def _normalize_goal_text(text: Any) -> str:
    return " ".join(str(text or "").strip().lower().split())


def _subtask_goals(items: Any) -> List[str]:
    if not isinstance(items, list):
        return []
    goals = []
    for item in items:
        if isinstance(item, dict):
            goals.append(_normalize_goal_text(item.get("subtask_goal", "")))
        else:
            goals.append(_normalize_goal_text(item))
    return goals


def _tokenize_metric_text(goals: Sequence[str]) -> List[str]:
    return " ".join(goals).split()


def edit_distance(a: Sequence[str], b: Sequence[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, av in enumerate(a, 1):
        curr = [i]
        for j, bv in enumerate(b, 1):
            curr.append(
                min(
                    prev[j] + 1,
                    curr[j - 1] + 1,
                    prev[j - 1] + (0 if av == bv else 1),
                )
            )
        prev = curr
    return prev[-1]


def sentence_bleu(reference: Sequence[str], prediction: Sequence[str], max_order: int = 4) -> float:
    if not prediction:
        return 0.0
    if not reference:
        return 1.0 if not prediction else 0.0
    precisions = []
    for order in range(1, max_order + 1):
        pred_ngrams = Counter(tuple(prediction[i : i + order]) for i in range(len(prediction) - order + 1))
        ref_ngrams = Counter(tuple(reference[i : i + order]) for i in range(len(reference) - order + 1))
        overlap = sum(min(count, ref_ngrams[ngram]) for ngram, count in pred_ngrams.items())
        total = sum(pred_ngrams.values())
        precisions.append((overlap + 1.0) / (total + 1.0))
    brevity = 1.0 if len(prediction) > len(reference) else pow(2.718281828459045, 1.0 - len(reference) / max(1, len(prediction)))
    return float(brevity * pow(max(1e-12, precisions[0] * precisions[1] * precisions[2] * precisions[3]), 0.25))


def score_subsequent_windows(gt, prediction, windows: Sequence[int]) -> Dict[str, Dict[str, Any]]:
    gt_goals = _subtask_goals(gt)
    pred_goals = _subtask_goals(prediction)
    scores = {}
    for window in windows:
        ref_goals = gt_goals[:window]
        hyp_goals = pred_goals[:window]
        ref_tokens = _tokenize_metric_text(ref_goals)
        hyp_tokens = _tokenize_metric_text(hyp_goals)
        dist = edit_distance(ref_tokens, hyp_tokens)
        denom = max(1, len(ref_tokens))
        scores[f"subsequent_{window}"] = {
            "bleu": sentence_bleu(ref_tokens, hyp_tokens),
            "edit_distance": dist,
            "normalized_edit_distance": dist / denom,
            "exact_match": ref_goals == hyp_goals,
            "gt_goals": ref_goals,
            "pred_goals": hyp_goals,
        }
    return scores


def summarize_window_scores(rows: Sequence[Dict[str, Any]], windows: Sequence[int]) -> Dict[str, Any]:
    summary = {}
    for window in windows:
        key = f"subsequent_{window}"
        values = [row["metrics"][key] for row in rows if key in row.get("metrics", {})]
        if not values:
            continue
        summary[key] = {
            "num_samples": len(values),
            "bleu": sum(float(item["bleu"]) for item in values) / len(values),
            "edit_distance": sum(float(item["edit_distance"]) for item in values) / len(values),
            "normalized_edit_distance": sum(float(item["normalized_edit_distance"]) for item in values) / len(values),
            "exact_match": sum(1 for item in values if item["exact_match"]) / len(values),
        }
    return summary


def _optional_output_xlsx(value: Optional[str], output_json: Path) -> Optional[Path]:
    if value is None:
        return output_json.with_suffix(".xlsx")
    if str(value).strip().lower() in {"", "false", "none", "null", "0"}:
        return None
    return Path(value)


def main():
    args = parse_args()
    random.seed(args.sample_seed)
    torch.manual_seed(args.sample_seed)
    output_json = Path(args.output_json)
    windows = parse_windows(args.subsequent_windows)

    samples = build_samples(args, kind="q1")
    if args.one_episode_per_task_all_undone_subtasks:
        samples = select_one_episode_per_task_all_undone_subtasks(
            samples,
            max_tasks=args.max_tasks,
            seed=args.sample_seed,
            shuffle_tasks=args.shuffle_samples,
        )
    else:
        samples = select_q1_samples(
            samples,
            all_states=args.all_states,
            current_subtask_index=args.current_subtask_index,
        )
    if args.one_example_per_task and not args.one_episode_per_task_all_undone_subtasks:
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
            metrics = score_subsequent_windows(gt, prediction_json, windows)
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
                    "metrics": metrics,
                    **{
                        f"{metric_key}_{name}": value
                        for metric_key, metric in metrics.items()
                        for name, value in metric.items()
                        if name in {"bleu", "edit_distance", "normalized_edit_distance", "exact_match"}
                    },
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
    output_xlsx = _optional_output_xlsx(args.output_xlsx, output_json)
    summary = {
        "checkpoint": args.checkpoint,
        "base_model": args.base_model,
        "data_root": args.data_root,
        "anno_root": args.anno_root,
        "anno_dir": args.anno_dir,
        "split": args.split,
        "test_ratio": args.test_ratio,
        "split_seed": args.split_seed,
        "max_episodes": args.max_episodes,
        "one_example_per_task": args.one_example_per_task,
        "one_episode_per_task_all_undone_subtasks": args.one_episode_per_task_all_undone_subtasks,
        "max_tasks": args.max_tasks,
        "subsequent_windows": windows,
        "num_q1_evaluated": len(rows),
        "num_unique_task_slugs": len({row["task_slug"] for row in rows}),
        "full_decomposition_only": not args.all_states,
        "current_subtask_index": None if args.all_states else args.current_subtask_index,
        "normalized_exact_match": exact / max(1, len(rows)),
        "subsequent_metrics": summarize_window_scores(rows, windows),
        "wall_s": time.perf_counter() - start,
        "output_xlsx": str(output_xlsx) if output_xlsx is not None else "",
        "results": rows,
    }
    write_json(output_json, summary)
    if output_xlsx is not None:
        write_xlsx(output_xlsx, rows, sheet_name="q1_results")
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, ensure_ascii=False, indent=2))
    print(output_json)
    if output_xlsx is not None:
        print(output_xlsx)


if __name__ == "__main__":
    main()
