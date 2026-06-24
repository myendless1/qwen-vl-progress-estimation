#!/usr/bin/env python
import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path

import torch
from transformers import AutoProcessor, AutoTokenizer, Qwen3VLForConditionalGeneration

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from qwenvl.data.data_processor import get_rope_index_3, update_processor_pixels
from qwenvl.data.robotwin_processor import (
    QUERY_TOKENS,
    RobotWinDataCollator,
    build_robotwin_samples,
    preprocess_robotwin_sample,
)
from qwenvl.train.argument import DataArguments
from qwenvl.train.robotwin_model import RobotWinQwenWrapper


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate RobotWin Q2 heads from a fine-tuned Qwen3-VL checkpoint.")
    parser.add_argument("--base-model", default="/media/damoxing/ckp/qwen_ft/Qwen3-VL-2B-Instruct")
    parser.add_argument("--checkpoint", default="/media/damoxing/ckp/qwen_ft/robotwin_qwen3vl_2b")
    parser.add_argument("--data-root", default="/media/damoxing/datasets/vae4d/lerobot-vae4d-org/robotwin_gt_depth")
    parser.add_argument("--split", choices=("train", "test", "all"), default="test")
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--q2-frame-stride", type=int, default=8)
    parser.add_argument("--boundary-extra-frames", type=int, default=2)
    parser.add_argument("--model-max-length", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--shuffle-samples", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--attn-implementation", default=os.environ.get("ATTN_IMPLEMENTATION", "sdpa"))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--predictions-csv", default=None)
    return parser.parse_args()


def dtype_from_name(name: str):
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    return torch.float32


def load_robotwin_tokenizer(args):
    tokenizer_kwargs = {
        "model_max_length": args.model_max_length,
        "padding_side": "right",
        "use_fast": False,
    }
    try:
        return AutoTokenizer.from_pretrained(args.checkpoint, **tokenizer_kwargs)
    except Exception as exc:
        print(
            f"warning: failed to load tokenizer from checkpoint ({exc}); "
            "falling back to base tokenizer plus RobotWin query tokens.",
            file=sys.stderr,
        )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, **tokenizer_kwargs)
    special_tokens = list(QUERY_TOKENS.values())
    config_path = Path(args.checkpoint) / "tokenizer_config.json"
    if config_path.exists():
        with open(config_path, "r") as f:
            tokenizer_config = json.load(f)
        special_tokens = tokenizer_config.get("extra_special_tokens", special_tokens)
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    return tokenizer


def load_model(args, query_token_ids):
    dtype = dtype_from_name(args.dtype)
    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.base_model,
        attn_implementation=args.attn_implementation,
        dtype=dtype if args.device != "cpu" else torch.float32,
    )
    base_model.resize_token_embeddings(max(query_token_ids.values()) + 1)
    model = RobotWinQwenWrapper(base_model)

    state_path = Path(args.checkpoint) / "pytorch_model.bin"
    if not state_path.exists():
        state_path = Path(args.checkpoint)
    state_dict = torch.load(state_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(
            json.dumps(
                {
                    "load_warning": {
                        "missing": missing[:20],
                        "num_missing": len(missing),
                        "unexpected": unexpected[:20],
                        "num_unexpected": len(unexpected),
                    }
                },
                ensure_ascii=False,
            )
        )

    model.to(args.device)
    model.eval()
    return model


def move_to_device(batch, device):
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def batched(items, batch_size):
    for start in range(0, len(items), batch_size):
        yield start, items[start : start + batch_size]


def main():
    args = parse_args()
    random.seed(args.sample_seed)
    torch.manual_seed(args.sample_seed)

    processor = AutoProcessor.from_pretrained(args.base_model)
    tokenizer = load_robotwin_tokenizer(args)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if hasattr(processor, "tokenizer"):
        processor.tokenizer = tokenizer

    query_token_ids = {name: tokenizer.convert_tokens_to_ids(token) for name, token in QUERY_TOKENS.items()}
    missing_tokens = [token for token, token_id in zip(QUERY_TOKENS.values(), query_token_ids.values()) if token_id is None]
    if missing_tokens:
        raise ValueError(f"Checkpoint tokenizer is missing RobotWin query tokens: {missing_tokens}")

    data_args = DataArguments(
        robotwin_data_root=args.data_root,
        robotwin_test_ratio=args.test_ratio,
        robotwin_split_seed=args.split_seed,
        robotwin_q2_frame_stride=args.q2_frame_stride,
        robotwin_boundary_extra_frames=args.boundary_extra_frames,
    )
    data_args.model_type = "qwen3vl"
    data_args.max_pixels = getattr(data_args, "max_pixels", 28 * 28 * 576)
    data_args.min_pixels = getattr(data_args, "min_pixels", 28 * 28 * 16)
    processor = update_processor_pixels(processor, data_args)
    merge_size = getattr(processor.image_processor, "merge_size", 2)
    collator = RobotWinDataCollator(tokenizer, query_token_ids=query_token_ids)

    samples = [
        sample
        for sample in build_robotwin_samples(
            args.data_root,
            q2_frame_stride=args.q2_frame_stride,
            boundary_extra_frames=args.boundary_extra_frames,
            max_episodes=args.max_episodes,
            split=args.split,
            test_ratio=args.test_ratio,
            split_seed=args.split_seed,
        )
        if sample.kind == "q2"
    ]
    if args.shuffle_samples:
        random.Random(args.sample_seed).shuffle(samples)
    if args.max_samples is not None:
        samples = samples[: args.max_samples]
    if not samples:
        raise ValueError("No Q2 samples found.")

    model = load_model(args, query_token_ids)

    total = 0
    correct_done = 0
    progress_sqerr = 0.0
    progress_abserr_max = 0.0
    preprocess_s = 0.0
    forward_s = 0.0
    rows = []
    wall_start = time.perf_counter()

    with torch.inference_mode():
        for batch_start, batch_samples in batched(samples, args.batch_size):
            prepared = []
            preprocess_start = time.perf_counter()
            for sample in batch_samples:
                item = preprocess_robotwin_sample(sample, processor)
                grid_thw = item.get("image_grid_thw")
                if grid_thw is not None and not isinstance(grid_thw, (list, tuple)):
                    grid_thw = [grid_thw]
                position_ids, _ = get_rope_index_3(
                    merge_size,
                    item["input_ids"],
                    image_grid_thw=torch.cat(grid_thw, dim=0) if grid_thw else None,
                )
                item["position_ids"] = position_ids
                prepared.append(item)
            preprocess_s += time.perf_counter() - preprocess_start

            batch = move_to_device(collator(prepared), args.device)
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            forward_start = time.perf_counter()
            outputs = model(**batch)
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            forward_s += time.perf_counter() - forward_start

            done_probs = torch.sigmoid(outputs.robotwin_logits["current_done"]).detach().float().cpu()
            progress_preds = outputs.robotwin_progress.detach().float().cpu()
            done_labels = batch["robotwin_current_done"].detach().float().cpu()
            progress_labels = batch["robotwin_progress"].detach().float().cpu()

            done_pred = done_probs.ge(args.threshold)
            done_true = done_labels.ge(args.threshold)
            correct_done += int(done_pred.eq(done_true).sum().item())

            errors = progress_preds - progress_labels
            progress_sqerr += float((errors * errors).sum().item())
            progress_abserr_max = max(progress_abserr_max, float(errors.abs().max().item()))
            total += len(batch_samples)

            if args.predictions_csv:
                for offset, sample in enumerate(batch_samples):
                    rows.append(
                        {
                            "sample_index": batch_start + offset,
                            "repo": sample.repo_dir.name,
                            "frame_index": sample.frame_index,
                            "current_subtask_index": sample.current_subtask_index,
                            "done_label": float(done_labels[offset].item()),
                            "done_prob": float(done_probs[offset].item()),
                            "done_pred": int(done_pred[offset].item()),
                            "progress_label": float(progress_labels[offset].item()),
                            "progress_pred": float(progress_preds[offset].item()),
                            "progress_abs_err": abs(float(errors[offset].item())),
                        }
                    )

            if total % max(args.batch_size * 50, 50) == 0 or total == len(samples):
                print(f"evaluated {total}/{len(samples)}", flush=True)

    total_s = time.perf_counter() - wall_start
    metrics = {
        "checkpoint": str(args.checkpoint),
        "base_model": str(args.base_model),
        "data_root": str(args.data_root),
        "split": args.split,
        "test_ratio": args.test_ratio,
        "split_seed": args.split_seed,
        "num_q2_samples": total,
        "done_accuracy": correct_done / total,
        "progress_mse": progress_sqerr / total,
        "progress_max_abs_err": progress_abserr_max,
        "threshold": args.threshold,
        "avg_preprocess_s_per_sample": preprocess_s / total,
        "avg_forward_s_per_sample": forward_s / total,
        "avg_total_s_per_sample": total_s / total,
        "total_wall_s": total_s,
        "batch_size": args.batch_size,
    }
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
    if args.predictions_csv:
        Path(args.predictions_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.predictions_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
