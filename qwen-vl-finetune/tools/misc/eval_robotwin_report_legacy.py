#!/usr/bin/env python
import argparse
import csv
import difflib
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOOLS_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from eval_robotwin_q2_legacy import batched, load_model, move_to_device
from qwenvl.data.data_processor import get_rope_index_3, update_processor_pixels
from qwenvl.data.robotwin_processor import (
    Q1_SYSTEM_PROMPT,
    QUERY_TOKENS,
    RobotWinDataCollator,
    _completed_subtasks,
    _future_subtasks,
    _load_observation_images,
    _messages_for_sample,
    _user_content,
    build_robotwin_samples,
    preprocess_robotwin_sample,
)
from qwenvl.train.argument import DataArguments


def parse_args():
    parser = argparse.ArgumentParser(description="RobotWin Q1/Q2 evaluation report with video visualization.")
    parser.add_argument("--base-model", default="/media/damoxing/ckp/qwen_ft/Qwen3-VL-2B-Instruct")
    parser.add_argument("--checkpoint", default="/media/damoxing/ckp/qwen_ft/robotwin_qwen3vl_2b")
    parser.add_argument("--data-root", default="/media/damoxing/datasets/vae4d/lerobot-vae4d-org/robotwin_gt_depth")
    parser.add_argument("--output-dir", default="/media/damoxing/ckp/qwen_ft/robotwin_qwen3vl_2b/report_last_ckpt")
    parser.add_argument("--split", choices=("train", "test", "all"), default="all")
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--max-episodes", type=int, default=500)
    parser.add_argument("--q2-frame-stride", type=int, default=8)
    parser.add_argument("--boundary-extra-frames", type=int, default=2)
    parser.add_argument("--q1-max-samples", type=int, default=50)
    parser.add_argument("--q1-sample-seed", type=int, default=0)
    parser.add_argument("--q1-max-new-tokens", type=int, default=256)
    parser.add_argument("--model-max-length", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--attn-implementation", default=os.environ.get("ATTN_IMPLEMENTATION", "sdpa"))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--q2-predictions-csv", default="/media/damoxing/ckp/qwen_ft/robotwin_qwen3vl_2b/eval_q2_predictions.csv")
    parser.add_argument("--rerun-q2", action="store_true", help="Ignore --q2-predictions-csv and run Q2 inference.")
    parser.add_argument("--video-fps", type=int, default=2)
    parser.add_argument("--video-max-frames", type=int, default=120)
    parser.add_argument("--keep-video-frames", action="store_true")
    return parser.parse_args()


def json_dumps(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def safe_float(row, key):
    return float(row[key])


def safe_int(row, key):
    return int(float(row[key]))


def make_query_token_ids(tokenizer):
    ids = {name: tokenizer.convert_tokens_to_ids(token) for name, token in QUERY_TOKENS.items()}
    missing = [name for name, token_id in ids.items() if token_id is None]
    if missing:
        raise ValueError(f"Missing RobotWin query tokens in tokenizer: {missing}")
    return ids


def build_samples(args):
    samples = build_robotwin_samples(
        args.data_root,
        q2_frame_stride=args.q2_frame_stride,
        boundary_extra_frames=args.boundary_extra_frames,
        max_episodes=args.max_episodes,
        split=args.split,
        test_ratio=args.test_ratio,
        split_seed=args.split_seed,
    )
    return [s for s in samples if s.kind == "q1"], [s for s in samples if s.kind == "q2"]


def load_processor_tokenizer(args):
    processor = AutoProcessor.from_pretrained(args.checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if hasattr(processor, "tokenizer"):
        processor.tokenizer = tokenizer
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
    return update_processor_pixels(processor, data_args), tokenizer


def normalize_q1_text(text):
    text = text.strip()
    try:
        return json_dumps(json.loads(text))
    except Exception:
        return " ".join(text.split())


def prepare_q1_prompt(sample, processor):
    images = _load_observation_images(sample.image_hdf5_paths, sample.frame_index)
    completed = _completed_subtasks(sample.subtasks, sample.current_subtask_index)
    user_content = _user_content(sample.task_goal, completed, images)
    messages = _messages_for_sample(Q1_SYSTEM_PROMPT, user_content)
    return processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )


def evaluate_q1(args, q1_samples, processor, tokenizer, model, output_dir):
    if args.q1_max_samples is not None and args.q1_max_samples < len(q1_samples):
        rng = random.Random(args.q1_sample_seed)
        selected_indices = sorted(rng.sample(range(len(q1_samples)), args.q1_max_samples))
    else:
        selected_indices = list(range(len(q1_samples)))

    rows = []
    base_model = model.base_model
    base_model.eval()
    start = time.perf_counter()
    with torch.inference_mode():
        for done, sample_index in enumerate(selected_indices, 1):
            sample = q1_samples[sample_index]
            prompt = prepare_q1_prompt(sample, processor)
            prompt = move_to_device(prompt, args.device)
            input_len = prompt["input_ids"].shape[1]
            generated = base_model.generate(
                **prompt,
                max_new_tokens=args.q1_max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            pred_text = tokenizer.decode(generated[0, input_len:], skip_special_tokens=True).strip()
            gt = _future_subtasks(sample.subtasks, sample.current_subtask_index)
            gt_text = json_dumps(gt)
            pred_norm = normalize_q1_text(pred_text)
            gt_norm = normalize_q1_text(gt_text)
            diff = "\n".join(
                difflib.unified_diff(
                    gt_norm.splitlines(),
                    pred_norm.splitlines(),
                    fromfile="GT",
                    tofile="PRED",
                    lineterm="",
                )
            )
            rows.append(
                {
                    "sample_index": sample_index,
                    "repo": sample.repo_dir.name,
                    "frame_index": sample.frame_index,
                    "current_subtask_index": sample.current_subtask_index,
                    "task_goal": sample.task_goal,
                    "gt": gt_text,
                    "pred": pred_text,
                    "normalized_exact_match": int(pred_norm == gt_norm),
                    "diff": diff,
                }
            )
            if done == len(selected_indices) or done % 20 == 0:
                print(f"Q1 generated {done}/{len(selected_indices)}", flush=True)

    csv_path = output_dir / "q1_generation_diff.csv"
    json_path = output_dir / "q1_generation_diff.json"
    write_csv(csv_path, rows)
    with open(json_path, "w") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    exact = sum(row["normalized_exact_match"] for row in rows)
    return {
        "num_q1_total_available": len(q1_samples),
        "num_q1_evaluated": len(rows),
        "normalized_exact_match": exact / max(1, len(rows)),
        "q1_csv": str(csv_path),
        "q1_json": str(json_path),
        "q1_wall_s": time.perf_counter() - start,
    }


def prepare_q2_batch(samples, processor, merge_size, collator):
    prepared = []
    for sample in samples:
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
    return collator(prepared)


def run_q2(args, q2_samples, processor, tokenizer, query_token_ids, model, output_dir):
    merge_size = getattr(processor.image_processor, "merge_size", 2)
    collator = RobotWinDataCollator(tokenizer, query_token_ids=query_token_ids)
    rows = []
    start = time.perf_counter()
    with torch.inference_mode():
        for batch_start, batch_samples in batched(q2_samples, args.batch_size):
            batch = move_to_device(prepare_q2_batch(batch_samples, processor, merge_size, collator), args.device)
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            outputs = model(**batch)
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            done_probs = torch.sigmoid(outputs.robotwin_logits["current_done"]).detach().float().cpu()
            progress_preds = outputs.robotwin_progress.detach().float().cpu()
            done_labels = batch["robotwin_current_done"].detach().float().cpu()
            progress_labels = batch["robotwin_progress"].detach().float().cpu()
            done_pred = done_probs.ge(args.threshold)
            errors = progress_preds - progress_labels
            for offset, sample in enumerate(batch_samples):
                rows.append(
                    {
                        "sample_index": batch_start + offset,
                        "repo": sample.repo_dir.name,
                        "frame_index": sample.frame_index,
                        "current_subtask_index": sample.current_subtask_index,
                        "q2_group": sample.q2_group,
                        "done_label": float(done_labels[offset].item()),
                        "done_prob": float(done_probs[offset].item()),
                        "done_pred": int(done_pred[offset].item()),
                        "progress_label": float(progress_labels[offset].item()),
                        "progress_pred": float(progress_preds[offset].item()),
                        "progress_abs_err": abs(float(errors[offset].item())),
                    }
                )
            if len(rows) == len(q2_samples) or len(rows) % max(args.batch_size * 50, 200) == 0:
                print(f"Q2 evaluated {len(rows)}/{len(q2_samples)}", flush=True)
    csv_path = output_dir / "q2_predictions.csv"
    write_csv(csv_path, rows)
    return rows, {"q2_predictions_csv": str(csv_path), "q2_wall_s": time.perf_counter() - start}


def load_q2_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def summarize_q2(args, rows, output_dir):
    total = len(rows)
    confusion = Counter()
    by_group = defaultdict(Counter)
    sqerr = 0.0
    max_abs = 0.0
    for row in rows:
        label = int(safe_float(row, "done_label") >= args.threshold)
        pred = safe_int(row, "done_pred")
        if label == 1 and pred == 1:
            key = "TP"
        elif label == 0 and pred == 0:
            key = "TN"
        elif label == 0 and pred == 1:
            key = "FP"
        else:
            key = "FN"
        confusion[key] += 1
        group = row.get("q2_group") or "unknown"
        by_group[group][key] += 1
        err = safe_float(row, "progress_pred") - safe_float(row, "progress_label")
        sqerr += err * err
        max_abs = max(max_abs, abs(err))

    distribution_rows = []
    for name, counter in [("all", confusion), *sorted(by_group.items())]:
        n = sum(counter.values())
        distribution_rows.append(
            {
                "group": name,
                "total": n,
                "TP": counter["TP"],
                "TN": counter["TN"],
                "FP": counter["FP"],
                "FN": counter["FN"],
                "true_positive": counter["TP"] + counter["FN"],
                "true_negative": counter["TN"] + counter["FP"],
                "pred_positive": counter["TP"] + counter["FP"],
                "pred_negative": counter["TN"] + counter["FN"],
                "accuracy": (counter["TP"] + counter["TN"]) / max(1, n),
            }
        )

    progress_rows = sorted(
        [
            {
                **row,
                "progress_sq_err": (
                    safe_float(row, "progress_pred") - safe_float(row, "progress_label")
                )
                ** 2,
            }
            for row in rows
        ],
        key=lambda r: float(r["progress_abs_err"]),
        reverse=True,
    )
    write_csv(output_dir / "q2_tf_pn_distribution.csv", distribution_rows)
    write_csv(output_dir / "q2_progress_errors_sorted.csv", progress_rows)
    return {
        "num_q2_evaluated": total,
        "done_accuracy": (confusion["TP"] + confusion["TN"]) / max(1, total),
        "tf_pn": dict(confusion),
        "progress_mse": sqerr / max(1, total),
        "progress_max_abs_err": max_abs,
        "q2_distribution_csv": str(output_dir / "q2_tf_pn_distribution.csv"),
        "q2_progress_errors_csv": str(output_dir / "q2_progress_errors_sorted.csv"),
    }


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)


def wrap_text(text, width):
    lines = []
    for part in str(text).splitlines() or [""]:
        lines.extend(textwrap.wrap(part, width=width) or [""])
    return lines


def draw_text(draw, xy, text, font, fill=(20, 20, 20), width=90, line_h=16, max_lines=None):
    x, y = xy
    lines = wrap_text(text, width)
    if max_lines is not None:
        lines = lines[:max_lines]
    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        y += line_h
    return y


def resize_keep(image, width):
    height = max(1, round(image.height * width / image.width))
    return image.resize((width, height), Image.Resampling.BICUBIC)


def make_video_panel(sample, row, rank, reason, thumb_width=360):
    font = ImageFont.load_default()
    images = _load_observation_images(sample.image_hdf5_paths, sample.frame_index)
    main = resize_keep(images["main"], thumb_width)
    left = resize_keep(images["left_wrist"], thumb_width // 2)
    right = resize_keep(images["right_wrist"], thumb_width // 2)
    gap = 10
    header_h = 210
    image_h = max(main.height, left.height, right.height)
    width = thumb_width + gap + thumb_width // 2 + gap + thumb_width // 2
    height = header_h + image_h + 32
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    label = int(safe_float(row, "done_label") >= 0.5)
    pred = safe_int(row, "done_pred")
    color = (35, 145, 70) if label == pred else (210, 45, 45)
    draw.rectangle((0, 0, width - 1, height - 1), outline=color, width=5)
    current = sample.subtasks[sample.current_subtask_index]
    y = 12
    y = draw_text(draw, (14, y), f"#{rank:03d} {reason} | sample={row['sample_index']} repo={sample.repo_dir.name} frame={sample.frame_index}", font, fill=color, width=112)
    metric = (
        f"done GT={label} pred={pred} prob={safe_float(row, 'done_prob'):.4f} | "
        f"progress GT={safe_float(row, 'progress_label'):.3f} pred={safe_float(row, 'progress_pred'):.3f} "
        f"abs_err={safe_float(row, 'progress_abs_err'):.3f}"
    )
    y = draw_text(draw, (14, y + 4), metric, font, width=112)
    y = draw_text(draw, (14, y + 4), f"Task: {sample.task_goal}", font, width=112, max_lines=3)
    draw_text(draw, (14, y + 4), f"Current subtask: {current['subtask_goal']}", font, width=112, max_lines=3)

    image_y = header_h
    x = 0
    for label_name, image in (("main", main), ("left wrist", left), ("right wrist", right)):
        canvas.paste(image, (x, image_y))
        draw.rectangle((x, image_y, x + image.width - 1, image_y + image.height - 1), outline=(30, 30, 30), width=1)
        draw.text((x + 6, image_y + image.height + 8), label_name, font=font, fill=(20, 20, 20))
        x += image.width + gap
    return canvas


def make_q2_video(args, rows, q2_samples, output_dir):
    if not rows or shutil.which("ffmpeg") is None:
        return None
    max_index = max(safe_int(row, "sample_index") for row in rows)
    if max_index >= len(q2_samples):
        print(
            f"Skip video: predictions sample_index max={max_index} but rebuilt Q2 samples={len(q2_samples)}.",
            flush=True,
        )
        return None
    enriched = []
    for row in rows:
        idx = safe_int(row, "sample_index")
        if 0 <= idx < len(q2_samples):
            label = int(safe_float(row, "done_label") >= args.threshold)
            pred = safe_int(row, "done_pred")
            if label == 0 and pred == 1:
                reason = "FP done"
            elif label == 1 and pred == 0:
                reason = "FN done"
            else:
                reason = "largest progress error"
            enriched.append((safe_float(row, "progress_abs_err"), label != pred, reason, row, q2_samples[idx]))
    enriched.sort(key=lambda item: (item[1], item[0]), reverse=True)
    selected = enriched[: args.video_max_frames]
    if not selected:
        return None

    video_path = output_dir / "q2_errors_and_progress_video.mp4"
    frames_dir = output_dir / "q2_video_frames" if args.keep_video_frames else Path(tempfile.mkdtemp(prefix="robotwin_q2_video_"))
    frames_dir.mkdir(parents=True, exist_ok=True)
    try:
        for rank, (_, _, reason, row, sample) in enumerate(selected):
            panel = make_video_panel(sample, row, rank, reason)
            panel.save(frames_dir / f"frame_{rank:05d}.png")
        cmd = [
            "ffmpeg",
            "-y",
            "-framerate",
            str(args.video_fps),
            "-i",
            str(frames_dir / "frame_%05d.png"),
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(video_path),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    finally:
        if not args.keep_video_frames:
            shutil.rmtree(frames_dir, ignore_errors=True)
    return str(video_path)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.q1_sample_seed)
    torch.manual_seed(args.q1_sample_seed)

    q1_samples, q2_samples = build_samples(args)
    processor, tokenizer = load_processor_tokenizer(args)
    query_token_ids = make_query_token_ids(tokenizer)
    model = load_model(args, query_token_ids)

    q1_metrics = evaluate_q1(args, q1_samples, processor, tokenizer, model, output_dir)

    q2_extra = {}
    q2_csv = Path(args.q2_predictions_csv) if args.q2_predictions_csv else None
    if args.rerun_q2 or q2_csv is None or not q2_csv.exists():
        rows, q2_extra = run_q2(args, q2_samples, processor, tokenizer, query_token_ids, model, output_dir)
    else:
        rows = load_q2_rows(q2_csv)
        q2_extra = {"q2_predictions_csv": str(q2_csv), "q2_reused_existing_predictions": True}

    q2_metrics = summarize_q2(args, rows, output_dir)
    video_path = make_q2_video(args, rows, q2_samples, output_dir)
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
        "num_q1_available": len(q1_samples),
        "num_q2_available": len(q2_samples),
        **q1_metrics,
        **q2_extra,
        **q2_metrics,
        "q2_video": video_path,
    }
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(summary_path)


if __name__ == "__main__":
    main()
