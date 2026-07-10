#!/usr/bin/env python
import argparse
import copy
import gc
import json
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Sequence

import torch

TOOLS_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TOOLS_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(TOOLS_ROOT))
sys.path.insert(0, str(TOOLS_ROOT / "eval"))

from qwenvl.data.data_processor import get_rope_index_3
from qwenvl.data.robotwin_processor import (
    DONE_VOTING_QUERY_TOKENS,
    Q2_SYSTEM_PROMPT,
    QUERY_TOKENS,
    RobotWinDataCollator,
    VIEW_LABELS,
    _completed_subtasks,
    _load_observation_images,
    _load_observation_image_sequence,
    _memory_frame_indices,
    _messages_for_sample,
    _system_prompt,
    build_robotwin_query_attention_mask,
    parse_robotwin_views,
)
from tools.utils.robotwin_eval import (
    build_samples,
    load_eval_context,
    move_to_device,
    sample_items,
    write_json,
)
from robotwin_eval_config import apply_config_argv


def parse_args():
    apply_config_argv("q2_bench")
    parser = argparse.ArgumentParser(description="Benchmark RobotWin Q2 single done-query/progress-query latency.")
    parser.add_argument("--base-model", default="/media/damoxing/ckp/qwen_ft/Qwen3-VL-2B-Instruct")
    parser.add_argument("--checkpoint", default="/media/damoxing/ckp/qwen_ft/robotwin_qwen3vl_2b-voting-done/checkpoint-16000")
    parser.add_argument("--model-a-name", default="2b")
    parser.add_argument("--model-a-base-model", default=None)
    parser.add_argument("--model-a-checkpoint", default=None)
    parser.add_argument("--model-b-name", default="8b")
    parser.add_argument("--model-b-base-model", default="/media/damoxing/ckp/qwen_ft/Qwen3-VL-8B-Instruct")
    parser.add_argument("--model-b-checkpoint", default="/media/damoxing/ckp/qwen_ft/robotwin_qwen3vl_8b/checkpoint-44000")
    parser.add_argument("--data-root", default="/media/damoxing/datasets/vae4d/lerobot-vae4d-org/robotwin_gt_depth")
    parser.add_argument("--anno-root", default=None)
    parser.add_argument("--views", default="main,left_wrist,right_wrist")
    parser.add_argument("--split", choices=("train", "test", "all"), default="train")
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--q2-frame-stride", type=int, default=8)
    parser.add_argument("--memory-frames", type=int, default=1)
    parser.add_argument("--memory-frame-stride", type=int, default=1)
    parser.add_argument("--q2-progress-bucket-size", type=float, default=0.01)
    parser.add_argument("--boundary-extra-frames", type=int, default=2)
    parser.add_argument("--model-max-length", type=int, default=4096)
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--train-output-dir", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--test-output-dir", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--train-sample-manifest", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--test-sample-manifest", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--shuffle-samples", action="store_true")
    parser.add_argument("--warmup-iters", type=int, default=8)
    parser.add_argument("--timed-iters", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--voting-done", action="store_true")
    parser.add_argument("--done-vote-count", type=int, default=5)
    parser.add_argument("--done-vote-threshold", type=int, default=3, help=argparse.SUPPRESS)
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def _cached_text(sample, views: Sequence[str]) -> Dict[str, Any]:
    completed = _completed_subtasks(sample.subtasks, sample.current_subtask_index)
    current = sample.subtasks[sample.current_subtask_index]
    state_values = sample.state_values or {}
    state_text = ""
    if state_values:
        state_text = (
            "Robot state: the gripper values of left and right arms are currently "
            f"{state_values['left_gripper']:.3f} and {state_values['right_gripper']:.3f}, "
            "while the z values of left and right arms are "
            f"{state_values['left_z']:.3f} and {state_values['right_z']:.3f}, respectively.\n"
        )
    observation_label = "Image observation" if len(views) == 1 else "Image observations"
    return {
        "task_text": (
            f"Global task: {sample.task_goal}\n"
            f"Completed subtasks: {json.dumps(list(completed), ensure_ascii=False, separators=(',', ':'))}\n"
            f"Current subtask: {current['subtask_goal']}\n"
            f"{state_text}"
            f"{observation_label}:\n"
        ),
        "system": _system_prompt(Q2_SYSTEM_PROMPT, views),
    }


def _content_with_fresh_observation(sample, cached: Dict[str, Any], views: Sequence[str], query_suffix: str):
    memory_frame_indices = _memory_frame_indices(
        sample.frame_index,
        sample.memory_frames,
        sample.memory_frame_stride,
    )
    if len(memory_frame_indices) == 1:
        image_sequence = [(None, _load_observation_images(sample.image_hdf5_paths, sample.frame_index, views))]
    else:
        image_sequence = _load_observation_image_sequence(sample.image_hdf5_paths, memory_frame_indices, views)
    content = [{"type": "text", "text": cached["task_text"]}]

    for timestep, (image_frame, images) in enumerate(image_sequence):
        if len(image_sequence) > 1:
            content.append({"type": "text", "text": f"<time {timestep + 1} frame {image_frame}>\n"})
        for view in views:
            content.append({"type": "text", "text": f"{VIEW_LABELS[view]} "})
            content.append({"type": "image", "image": images[view]})
            content.append({"type": "text", "text": "\n"})
    content.append({"type": "text", "text": query_suffix})
    return content


def _prepare_single_query_batch(sample, cached, views, processor, merge_size, collator, query_suffix):
    messages = _messages_for_sample(
        cached["system"],
        _content_with_fresh_observation(sample, cached, views, query_suffix),
    )
    item = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    item["labels"] = torch.full_like(item["input_ids"], -100)
    grid_thw = item.get("image_grid_thw")
    if grid_thw is not None and not isinstance(grid_thw, (list, tuple)):
        grid_thw = [grid_thw]
    position_ids, _ = get_rope_index_3(
        merge_size,
        item["input_ids"],
        image_grid_thw=torch.cat(grid_thw, dim=0) if grid_thw else None,
    )
    item["position_ids"] = position_ids
    item["robotwin_current_done"] = torch.tensor(sample.current_done, dtype=torch.float32)
    item["robotwin_need_replan"] = torch.tensor(-100.0, dtype=torch.float32)
    item["robotwin_incident"] = torch.tensor(-100.0, dtype=torch.float32)
    item["robotwin_progress"] = torch.tensor(sample.progress, dtype=torch.float32)
    batch = collator([item])
    batch["attention_mask"] = build_robotwin_query_attention_mask(
        batch["input_ids"],
        collator.tokenizer.pad_token_id,
        collator.query_token_ids,
    )
    return batch


def _summarize(values: List[float]) -> Dict[str, float]:
    values = sorted(values)
    if not values:
        return {}
    return {
        "mean_ms": statistics.fmean(values),
        "median_ms": statistics.median(values),
        "min_ms": values[0],
        "max_ms": values[-1],
        "p90_ms": values[min(len(values) - 1, int(len(values) * 0.9))],
    }


def _model_specs(args):
    return [
        {
            "name": args.model_a_name,
            "base_model": args.model_a_base_model or args.base_model,
            "checkpoint": args.model_a_checkpoint or args.checkpoint,
        },
        {
            "name": args.model_b_name,
            "base_model": args.model_b_base_model,
            "checkpoint": args.model_b_checkpoint,
        },
    ]


def _bench_one_model(args, spec, samples, cached_texts, views):
    run_args = copy.copy(args)
    run_args.base_model = spec["base_model"]
    run_args.checkpoint = spec["checkpoint"]
    run_args.voting_done = True
    context = load_eval_context(run_args, prefer_checkpoint_processor=False)
    model = context["model"]
    processor = context["processor"]
    collator: RobotWinDataCollator = context["collator"]
    merge_size = context["merge_size"]
    query_suffix = f"{DONE_VOTING_QUERY_TOKENS[0]}{QUERY_TOKENS['value']}"

    preprocess_ms = []
    forward_ms = []
    total_iters = args.warmup_iters + args.timed_iters
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    with torch.inference_mode():
        for index in range(total_iters):
            sample = samples[index % len(samples)]
            cached = cached_texts[index % len(cached_texts)]

            t0 = time.perf_counter()
            batch = _prepare_single_query_batch(sample, cached, views, processor, merge_size, collator, query_suffix)
            batch = move_to_device(batch, args.device)
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            outputs = model(**batch)
            _ = outputs.robotwin_logits["current_done"]
            _ = outputs.robotwin_progress
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            t2 = time.perf_counter()
            if index >= args.warmup_iters:
                preprocess_ms.append((t1 - t0) * 1000.0)
                forward_ms.append((t2 - t1) * 1000.0)

    peak_memory_gib = None
    if args.device.startswith("cuda"):
        peak_memory_gib = torch.cuda.max_memory_allocated() / (1024**3)

    result = {
        **spec,
        "warmup_iters": args.warmup_iters,
        "timed_iters": args.timed_iters,
        "num_samples_cycled": len(samples),
        "preprocess": _summarize(preprocess_ms),
        "forward": _summarize(forward_ms),
        "end_to_end_preprocess_plus_forward": _summarize([a + b for a, b in zip(preprocess_ms, forward_ms)]),
        "peak_memory_gib": peak_memory_gib,
    }

    del model, processor, collator, context
    gc.collect()
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    return result


def main():
    args = parse_args()
    if args.anno_root is None:
        args.anno_root = args.data_root
    views = parse_robotwin_views(args.views)
    all_samples = build_samples(args, kind="q2")
    samples = sample_items(all_samples, args.max_samples, seed=args.sample_seed, shuffle=args.shuffle_samples)
    if not samples:
        raise ValueError("No Q2 samples found for latency benchmark.")
    cached_texts = [_cached_text(sample, views) for sample in samples]

    results = []
    for spec in _model_specs(args):
        print(f"Benchmarking {spec['name']}: {spec['checkpoint']}", flush=True)
        results.append(_bench_one_model(args, spec, samples, cached_texts, views))
        print(json.dumps(results[-1], ensure_ascii=False, indent=2), flush=True)

    summary = {
        "split": args.split,
        "test_ratio": args.test_ratio,
        "split_seed": args.split_seed,
        "views": args.views,
        "q2_frame_stride": args.q2_frame_stride,
        "boundary_extra_frames": args.boundary_extra_frames,
        "note": "Text content is cached per sample; each iteration reloads/reassigns observation images before one forward with one done query and one progress query.",
        "results": results,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)

    if args.output_json:
        write_json(Path(args.output_json), summary)
        print(args.output_json, flush=True)


if __name__ == "__main__":
    main()
