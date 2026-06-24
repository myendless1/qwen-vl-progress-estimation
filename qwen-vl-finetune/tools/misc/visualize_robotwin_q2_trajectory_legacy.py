#!/usr/bin/env python
import argparse
import csv
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOOLS_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from eval_robotwin_q2_legacy import load_model, move_to_device
from qwenvl.data.data_processor import get_rope_index_3, update_processor_pixels
from qwenvl.data.robotwin_processor import (
    QUERY_TOKENS,
    RobotWinDataCollator,
    RobotWinSample,
    _load_chunks_size,
    _load_observation_images,
    _robotwin_repo_dirs,
    _view_hdf5_path,
    preprocess_robotwin_sample,
)
from qwenvl.train.argument import DataArguments


def parse_args():
    parser = argparse.ArgumentParser(description="Run Q2 on every frame of one RobotWin episode and visualize curves.")
    parser.add_argument("--anno-path", default="/media/damoxing/datasets/vae4d/lerobot-vae4d-org/robotwin_gt_depth/blocks_ranking_rgb-aloha-agilex_clean_50/anno/episode_000000.json")
    parser.add_argument("--prompt-mode", choices=("active", "fixed"), default="active")
    parser.add_argument("--subtask-index", type=int, default=0)
    parser.add_argument("--subtask-text", default=None)
    parser.add_argument("--base-model", default="/media/damoxing/ckp/qwen_ft/Qwen3-VL-2B-Instruct")
    parser.add_argument("--checkpoint", default="/media/damoxing/ckp/qwen_ft/robotwin_qwen3vl_2b")
    parser.add_argument("--output-dir", default="/media/damoxing/ckp/qwen_ft/robotwin_qwen3vl_2b/q2_trajectory_vis")
    parser.add_argument("--data-root", default="/media/damoxing/datasets/vae4d/lerobot-vae4d-org/robotwin_gt_depth")
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument(
        "--batch-visualize-splits",
        nargs="+",
        choices=("train", "test"),
        default=None,
        help="Batch mode: select cases from these RobotWin splits instead of using --anno-path.",
    )
    parser.add_argument("--cases-per-split", type=int, default=5)
    parser.add_argument("--case-seed", type=int, default=0)
    parser.add_argument("--case-selection", choices=("random", "first"), default="random")
    parser.add_argument("--case-min-frames", type=int, default=1)
    parser.add_argument("--case-max-frames", type=int, default=None)
    parser.add_argument("--model-max-length", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--boundary-extra-frames", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--attn-implementation", default=os.environ.get("ATTN_IMPLEMENTATION", "sdpa"))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--include-boundary-probes",
        action="store_true",
        help="At each transition, also query the previous and current subtask prompts on overlapping boundary frames.",
    )
    parser.add_argument("--save-frame-strip", action="store_true", help="Also save a thumbnail strip of all frames.")
    parser.add_argument("--save-video", action="store_true", help="Save a video with observations above and progress curves below.")
    parser.add_argument("--output-video", default=None, help="Optional explicit mp4 path for --save-video.")
    parser.add_argument("--video-fps", type=float, default=10.0)
    parser.add_argument("--video-width", type=int, default=1280)
    parser.add_argument("--video-top-height", type=int, default=720)
    parser.add_argument("--video-curve-height", type=int, default=360)
    return parser.parse_args()


def make_query_token_ids(tokenizer):
    query_token_ids = {name: tokenizer.convert_tokens_to_ids(token) for name, token in QUERY_TOKENS.items()}
    missing = [name for name, token_id in query_token_ids.items() if token_id is None]
    if missing:
        raise ValueError(f"Missing query tokens in tokenizer: {missing}")
    return query_token_ids


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


def subtask_index_for_frame(subtasks, frame):
    for idx, subtask in enumerate(subtasks):
        if int(subtask["start_frame"]) <= frame <= int(subtask["end_frame"]):
            return idx
    for idx in range(len(subtasks) - 1, -1, -1):
        if frame > int(subtasks[idx]["end_frame"]):
            return idx
    return 0


def make_q2_sample(
    repo_dir,
    image_hdf5_paths,
    frame,
    task_goal,
    subtasks,
    current_subtask_index,
    done,
    progress,
    probe_kind,
):
    sample = RobotWinSample(
        kind="q2",
        repo_dir=repo_dir,
        image_hdf5_paths=image_hdf5_paths,
        frame_index=frame,
        frame_start=frame,
        frame_end=frame,
        task_goal=task_goal,
        subtasks=subtasks,
        current_subtask_index=current_subtask_index,
        current_done=done,
        progress=progress,
    )
    sample.probe_kind = probe_kind
    return sample


def progress_for_subtask(subtask, frame):
    start = int(subtask["start_frame"])
    end = int(subtask["end_frame"])
    denom = max(1, end - start)
    if frame <= start:
        return 0.0
    if frame >= end:
        return 1.0
    return max(0.0, min(1.0, (frame - start) / denom))


def build_episode_samples(
    anno_path,
    prompt_mode,
    subtask_index,
    subtask_text,
    boundary_extra_frames,
    max_frames=None,
    include_boundary_probes=False,
):
    anno_path = Path(anno_path)
    repo_dir = anno_path.parent.parent
    with open(anno_path, "r") as f:
        anno = json.load(f)
    subtasks = [dict(item) for item in anno["subtasks"]]
    if subtask_index < 0 or subtask_index >= len(subtasks):
        raise IndexError(f"subtask_index {subtask_index} out of range for {len(subtasks)} subtasks.")
    if subtask_text is not None:
        subtasks[subtask_index]["subtask_goal"] = subtask_text

    episode_index = int(anno["episode_index"])
    chunks_size = _load_chunks_size(repo_dir)
    image_hdf5_paths = {
        view: _view_hdf5_path(repo_dir, episode_index, chunks_size, view)
        for view in ("main", "left_wrist", "right_wrist")
    }
    num_frames = int(anno["num_frames"])
    if max_frames is not None:
        num_frames = min(num_frames, max_frames)

    samples = []
    for frame in range(num_frames):
        current_subtask_index = subtask_index
        if prompt_mode == "active":
            current_subtask_index = subtask_index_for_frame(subtasks, frame)
        current = subtasks[current_subtask_index]
        end = int(current["end_frame"])
        progress = progress_for_subtask(current, frame)
        done = 1.0 if frame >= end - 1 else 0.0
        if done:
            progress = 1.0
        samples.append(
            make_q2_sample(
                repo_dir,
                image_hdf5_paths,
                frame,
                anno["task_goal"],
                subtasks,
                current_subtask_index,
                done,
                progress,
                "active",
            )
        )

    if include_boundary_probes and prompt_mode == "active":
        for current_idx in range(1, len(subtasks)):
            prev_idx = current_idx - 1
            prev = subtasks[prev_idx]
            current = subtasks[current_idx]
            prev_end = int(prev["end_frame"])
            current_start = int(current["start_frame"])
            frames = [prev_end - 1, prev_end, current_start, current_start + 1, current_start + 2]
            for frame in sorted(set(frames)):
                if frame < 0 or frame >= num_frames:
                    continue
                samples.append(
                    make_q2_sample(
                        repo_dir,
                        image_hdf5_paths,
                        frame,
                        anno["task_goal"],
                        subtasks,
                        prev_idx,
                        1.0,
                        1.0,
                        "prev_done_probe",
                    )
                )
                samples.append(
                    make_q2_sample(
                        repo_dir,
                        image_hdf5_paths,
                        frame,
                        anno["task_goal"],
                        subtasks,
                        current_idx,
                        0.0,
                        0.0,
                        "current_not_done_probe",
                    )
                )
    return anno, samples


def batched(items, batch_size):
    for start in range(0, len(items), batch_size):
        yield start, items[start : start + batch_size]


def prepare_batch(samples, processor, merge_size, collator):
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


def load_eval_context(args):
    processor = AutoProcessor.from_pretrained(args.base_model)
    tokenizer = load_robotwin_tokenizer(args)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if hasattr(processor, "tokenizer"):
        processor.tokenizer = tokenizer

    data_args = DataArguments()
    data_args.model_type = "qwen3vl"
    processor = update_processor_pixels(processor, data_args)
    merge_size = getattr(processor.image_processor, "merge_size", 2)
    query_token_ids = make_query_token_ids(tokenizer)
    collator = RobotWinDataCollator(tokenizer, query_token_ids=query_token_ids)
    model = load_model(args, query_token_ids)
    return processor, merge_size, collator, model


def run_predictions(args, samples, context=None):
    if context is None:
        context = load_eval_context(args)
    processor, merge_size, collator, model = context
    rows = []
    with torch.inference_mode():
        for start, batch_samples in batched(samples, args.batch_size):
            batch = move_to_device(prepare_batch(batch_samples, processor, merge_size, collator), args.device)
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            outputs = model(**batch)
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()

            done_probs = torch.sigmoid(outputs.robotwin_logits["current_done"]).detach().float().cpu()
            progress_preds = outputs.robotwin_progress.detach().float().cpu()
            done_labels = batch["robotwin_current_done"].detach().float().cpu()
            progress_labels = batch["robotwin_progress"].detach().float().cpu()
            done_preds = done_probs.ge(args.threshold)
            done_true = done_labels.ge(args.threshold)
            for offset, sample in enumerate(batch_samples):
                rows.append(
                    {
                        "frame_index": sample.frame_index,
                        "probe_kind": getattr(sample, "probe_kind", "active"),
                        "current_subtask_index": sample.current_subtask_index,
                        "current_subtask_goal": sample.subtasks[sample.current_subtask_index]["subtask_goal"],
                        "done_label": float(done_labels[offset].item()),
                        "done_prob": float(done_probs[offset].item()),
                        "done_pred": int(done_preds[offset].item()),
                        "done_correct": int(done_preds[offset].eq(done_true[offset]).item()),
                        "progress_label": float(progress_labels[offset].item()),
                        "progress_pred": float(progress_preds[offset].item()),
                        "progress_abs_err": abs(float(progress_preds[offset].item() - progress_labels[offset].item())),
                    }
                )
            print(f"predicted {len(rows)}/{len(samples)}", flush=True)
    return rows


def draw_axes(draw, box, title, font):
    x0, y0, x1, y1 = box
    draw.rectangle(box, outline=(40, 40, 40), width=1)
    draw.text((x0, y0 - 20), title, font=font, fill=(20, 20, 20))
    for value in (0.0, 0.5, 1.0):
        y = y1 - value * (y1 - y0)
        draw.line((x0, y, x1, y), fill=(225, 225, 225), width=1)
        draw.text((x0 - 32, y - 6), f"{value:.1f}", font=font, fill=(80, 80, 80))


def point_for(row, key, box, num_frames):
    x0, y0, x1, y1 = box
    frame = int(row["frame_index"])
    denom = max(1, num_frames - 1)
    x = x0 + frame / denom * (x1 - x0)
    value = max(0.0, min(1.0, float(row[key])))
    y = y1 - value * (y1 - y0)
    return x, y


def draw_polyline(draw, rows, key, box, color, width=3, num_frames=None):
    if len(rows) < 2:
        return
    if num_frames is None:
        num_frames = len(rows)
    points = [point_for(row, key, box, num_frames) for row in rows]
    draw.line(points, fill=color, width=width, joint="curve")


def resize_letterbox(image, size, fill=(18, 20, 24)):
    target_w, target_h = size
    scale = min(target_w / image.width, target_h / image.height)
    new_w = max(1, int(round(image.width * scale)))
    new_h = max(1, int(round(image.height * scale)))
    resized = image.resize((new_w, new_h), Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", size, fill)
    canvas.paste(resized, ((target_w - new_w) // 2, (target_h - new_h) // 2))
    return canvas


def draw_video_panel(sample, row, anno, width, height, font):
    canvas = Image.new("RGB", (width, height), (242, 244, 247))
    draw = ImageDraw.Draw(canvas)
    pad = 24
    title_h = 54
    gap = 14
    images = _load_observation_images(sample.image_hdf5_paths, sample.frame_index)

    main_w = int(width * 0.66)
    side_w = width - main_w - pad * 2 - gap
    image_y = title_h + pad
    image_h = height - image_y - pad
    wrist_h = (image_h - gap) // 2

    main_box = (pad, image_y, main_w, image_h)
    wrist_box = (pad + main_w + gap, image_y, side_w, wrist_h)
    wrist2_box = (pad + main_w + gap, image_y + wrist_h + gap, side_w, image_h - wrist_h - gap)

    title = (
        f"{sample.repo_dir.name} episode_{int(anno['episode_index']):06d} "
        f"frame={sample.frame_index} subtask={sample.current_subtask_index}"
    )
    draw.text((pad, 18), title[:180], font=font, fill=(20, 24, 32))
    metric = (
        f"progress gt={float(row['progress_label']):.3f} pred={float(row['progress_pred']):.3f} "
        f"done gt={int(float(row['done_label']) >= 0.5)} pred={int(row['done_pred'])} prob={float(row['done_prob']):.3f}"
    )
    draw.text((width - pad - 420, 18), metric, font=font, fill=(20, 24, 32))

    for label, image, box in (
        ("main", images["main"], main_box),
        ("left wrist", images["left_wrist"], wrist_box),
        ("right wrist", images["right_wrist"], wrist2_box),
    ):
        x, y, w, h = box
        panel = resize_letterbox(image, (w, h))
        canvas.paste(panel, (x, y))
        draw.rectangle((x, y, x + w - 1, y + h - 1), outline=(36, 42, 52), width=2)
        draw.rectangle((x + 8, y + 8, x + 104, y + 30), fill=(20, 24, 32))
        draw.text((x + 14, y + 13), label, font=font, fill=(245, 247, 250))
    return canvas


def draw_progress_panel(rows, sample, current_frame, width, height, font):
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    active_rows = sorted(
        [row for row in rows if row.get("probe_kind", "active") == "active"],
        key=lambda item: int(item["frame_index"]),
    )
    if not active_rows:
        active_rows = sorted(rows, key=lambda item: int(item["frame_index"]))
    num_frames = max(int(row["frame_index"]) for row in active_rows) + 1

    plot = (70, 58, width - 32, height - 58)
    draw.text((24, 18), "Progress: blue=prediction, gray=gt", font=font, fill=(20, 24, 32))
    draw.text((width - 300, 18), f"frame {current_frame}/{max(0, num_frames - 1)}", font=font, fill=(90, 96, 108))
    draw_axes(draw, plot, "", font)

    for idx, subtask in enumerate(sample.subtasks):
        start = int(subtask["start_frame"])
        if start < 0 or start >= num_frames:
            continue
        x = plot[0] + start / max(1, num_frames - 1) * (plot[2] - plot[0])
        draw.line((x, plot[1], x, plot[3]), fill=(232, 205, 116), width=1)
        draw.text((x + 4, plot[1] + 4), f"s{idx}", font=font, fill=(130, 98, 20))

    draw_polyline(draw, active_rows, "progress_label", plot, (145, 150, 160), width=3, num_frames=num_frames)
    draw_polyline(draw, active_rows, "progress_pred", plot, (32, 102, 204), width=4, num_frames=num_frames)

    current_x = plot[0] + current_frame / max(1, num_frames - 1) * (plot[2] - plot[0])
    draw.line((current_x, plot[1], current_x, plot[3]), fill=(210, 45, 45), width=3)
    for frame in (0, num_frames // 2, num_frames - 1):
        x = plot[0] + frame / max(1, num_frames - 1) * (plot[2] - plot[0])
        draw.text((x - 10, plot[3] + 16), str(frame), font=font, fill=(80, 86, 96))

    current_rows = [row for row in active_rows if int(row["frame_index"]) == current_frame]
    if current_rows:
        row = current_rows[0]
        text = f"gt={float(row['progress_label']):.3f}  pred={float(row['progress_pred']):.3f}  abs_err={float(row['progress_abs_err']):.3f}"
        draw.text((24, height - 28), text, font=font, fill=(20, 24, 32))
    return canvas


def save_progress_video(rows, samples, anno, output_path, fps, width, top_height, curve_height):
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to save the progress video.")

    active_pairs = [
        (sample, row)
        for sample, row in zip(samples, rows)
        if row.get("probe_kind", "active") == "active"
    ]
    if not active_pairs:
        active_pairs = list(zip(samples, rows))
    active_pairs.sort(key=lambda pair: int(pair[1]["frame_index"]))
    if not active_pairs:
        raise ValueError("No frames available for video.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    height = top_height + curve_height
    if width % 2:
        width += 1
    if height % 2:
        height += 1

    font = ImageFont.load_default()
    with tempfile.TemporaryDirectory(prefix="robotwin_progress_video_") as tmp_dir:
        tmp_dir = Path(tmp_dir)
        for idx, (sample, row) in enumerate(active_pairs):
            top = draw_video_panel(sample, row, anno, width, top_height, font)
            curve = draw_progress_panel(rows, sample, int(row["frame_index"]), width, curve_height, font)
            frame = Image.new("RGB", (width, height), "white")
            frame.paste(top, (0, 0))
            frame.paste(curve, (0, top_height))
            frame.save(tmp_dir / f"frame_{idx:06d}.png")

        cmd = [
            "ffmpeg",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(tmp_dir / "frame_%06d.png"),
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed while writing {output_path}:\n{result.stderr[-4000:]}")
    return output_path


def draw_done_curve(draw, rows, box, num_frames):
    if len(rows) < 2:
        return
    for prev, cur in zip(rows[:-1], rows[1:]):
        color = (20, 150, 70) if int(prev["done_correct"]) and int(cur["done_correct"]) else (210, 45, 45)
        draw.line((point_for(prev, "done_prob", box, num_frames), point_for(cur, "done_prob", box, num_frames)), fill=color, width=3)
    for row in rows:
        x, y = point_for(row, "done_prob", box, num_frames)
        color = (20, 150, 70) if int(row["done_correct"]) else (210, 45, 45)
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)


def draw_diamond(draw, x, y, radius, fill, outline=(40, 40, 40)):
    points = [(x, y - radius), (x + radius, y), (x, y + radius), (x - radius, y)]
    draw.polygon(points, fill=fill, outline=outline)


def draw_probe_markers(draw, rows, done_box, progress_box, num_frames, font):
    legend_x = done_box[0] + 520
    legend_y = done_box[1] - 22
    draw.rectangle((legend_x, legend_y, legend_x + 12, legend_y + 12), fill=(245, 145, 35), outline=(60, 60, 60))
    draw.text((legend_x + 18, legend_y), "prev prompt: should be done", font=font, fill=(60, 60, 60))
    draw_diamond(draw, legend_x + 220, legend_y + 6, 7, (80, 145, 235))
    draw.text((legend_x + 235, legend_y), "current prompt: should not be done", font=font, fill=(60, 60, 60))

    for row in rows:
        kind = row.get("probe_kind", "active")
        if kind == "active":
            continue
        correct = int(row["done_correct"])
        fill = (245, 145, 35) if kind == "prev_done_probe" else (80, 145, 235)
        outline = (20, 150, 70) if correct else (210, 45, 45)
        x_done, y_done = point_for(row, "done_prob", done_box, num_frames)
        x_prog, y_prog = point_for(row, "progress_pred", progress_box, num_frames)
        if kind == "prev_done_probe":
            draw.rectangle((x_done - 5, y_done - 5, x_done + 5, y_done + 5), fill=fill, outline=outline, width=2)
            draw.rectangle((x_prog - 5, y_prog - 5, x_prog + 5, y_prog + 5), fill=fill, outline=outline, width=2)
        else:
            draw_diamond(draw, x_done, y_done, 6, fill, outline=outline)
            draw_diamond(draw, x_prog, y_prog, 6, fill, outline=outline)


def make_curve_image(rows, anno, sample, output_path):
    font = ImageFont.load_default()
    width, height = 1200, 720
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    active_rows = [row for row in rows if row.get("probe_kind", "active") == "active"]
    if not active_rows:
        active_rows = rows
    num_frames = max(int(row["frame_index"]) for row in rows) + 1
    subtask = sample.subtasks[sample.current_subtask_index]
    subtask_indices = sorted({int(row["current_subtask_index"]) for row in active_rows})
    if len(subtask_indices) == 1:
        title = (
            f"{sample.repo_dir.name} episode_{int(anno['episode_index']):06d} "
            f"subtask={sample.current_subtask_index}: {subtask['subtask_goal']}"
        )
    else:
        title = (
            f"{sample.repo_dir.name} episode_{int(anno['episode_index']):06d} "
            f"active subtasks={','.join(str(i) for i in subtask_indices)}"
        )
    draw.text((24, 18), title[:180], font=font, fill=(20, 20, 20))
    draw.text((24, 38), f"Task: {sample.task_goal}"[:180], font=font, fill=(20, 20, 20))

    done_box = (70, 100, width - 40, 330)
    progress_box = (70, 420, width - 40, 650)
    draw_axes(draw, done_box, "Done probability: green=correct, red=wrong, gray=label", font)
    draw_axes(draw, progress_box, "Progress: blue=prediction, gray=label", font)

    draw_polyline(draw, active_rows, "done_label", done_box, (150, 150, 150), width=2, num_frames=num_frames)
    draw_done_curve(draw, active_rows, done_box, num_frames)
    draw_polyline(draw, active_rows, "progress_label", progress_box, (170, 170, 170), width=2, num_frames=num_frames)
    draw_polyline(draw, active_rows, "progress_pred", progress_box, (35, 105, 210), width=3, num_frames=num_frames)
    draw_probe_markers(draw, rows, done_box, progress_box, num_frames, font)

    for idx in subtask_indices:
        start = int(sample.subtasks[idx]["start_frame"])
        x = progress_box[0] + start / max(1, num_frames - 1) * (progress_box[2] - progress_box[0])
        draw.line((x, done_box[1], x, progress_box[3]), fill=(235, 210, 120), width=1)
        draw.text((x + 3, progress_box[1] - 16), f"s{idx}", font=font, fill=(120, 95, 20))

    for box in (done_box, progress_box):
        x0, y0, x1, y1 = box
        for frame in (0, num_frames // 2, num_frames - 1):
            x = x0 + frame / max(1, num_frames - 1) * (x1 - x0)
            draw.text((x - 10, y1 + 10), str(frame), font=font, fill=(80, 80, 80))
    canvas.save(output_path)


def make_frame_strip(samples, rows, output_path, thumb_width=120, cols=10, gap=8):
    thumbs = []
    font = ImageFont.load_default()
    for sample, row in zip(samples, rows):
        image = _load_observation_images(sample.image_hdf5_paths, sample.frame_index)["main"]
        thumb_h = max(1, round(image.height * thumb_width / image.width))
        thumb = image.resize((thumb_width, thumb_h), Image.Resampling.BICUBIC)
        panel = Image.new("RGB", (thumb_width, thumb_h + 28), "white")
        panel.paste(thumb, (0, 0))
        color = (20, 150, 70) if int(row["done_correct"]) else (210, 45, 45)
        draw = ImageDraw.Draw(panel)
        draw.rectangle((0, 0, thumb_width - 1, thumb_h - 1), outline=color, width=4)
        draw.text((4, thumb_h + 6), f"f{sample.frame_index} d{row['done_prob']:.2f} p{row['progress_pred']:.2f}", font=font, fill=(20, 20, 20))
        thumbs.append(panel)

    if not thumbs:
        return
    panel_w, panel_h = thumbs[0].size
    rows_count = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * panel_w + (cols - 1) * gap, rows_count * panel_h + (rows_count - 1) * gap), (245, 245, 245))
    for idx, thumb in enumerate(thumbs):
        x = (idx % cols) * (panel_w + gap)
        y = (idx // cols) * (panel_h + gap)
        sheet.paste(thumb, (x, y))
    sheet.save(output_path)


def save_csv(rows, output_path):
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)


def output_stem(args, anno, samples):
    if args.prompt_mode == "active":
        stem = f"{samples[0].repo_dir.name}_episode_{int(anno['episode_index']):06d}_active_subtasks"
    else:
        stem = f"{samples[0].repo_dir.name}_episode_{int(anno['episode_index']):06d}_subtask_{args.subtask_index}"
    if args.include_boundary_probes:
        stem = f"{stem}_boundary_probes"
    return stem


def visualize_episode(args, anno_path, output_dir, context=None, output_video=None):
    output_dir.mkdir(parents=True, exist_ok=True)
    anno, samples = build_episode_samples(
        anno_path,
        args.prompt_mode,
        args.subtask_index,
        args.subtask_text,
        args.boundary_extra_frames,
        max_frames=args.max_frames,
        include_boundary_probes=args.include_boundary_probes,
    )
    rows = run_predictions(args, samples, context=context)

    stem = output_stem(args, anno, samples)
    csv_path = output_dir / f"{stem}_q2_predictions.csv"
    plot_path = output_dir / f"{stem}_q2_curves.png"
    save_csv(rows, csv_path)
    make_curve_image(rows, anno, samples[0], plot_path)
    print(csv_path)
    print(plot_path)
    if args.save_video:
        video_path = Path(output_video) if output_video else output_dir / f"{stem}_progress_video.mp4"
        video_path = save_progress_video(
            rows,
            samples,
            anno,
            video_path,
            fps=args.video_fps,
            width=args.video_width,
            top_height=args.video_top_height,
            curve_height=args.video_curve_height,
        )
        print(video_path)
    if args.save_frame_strip:
        strip_path = output_dir / f"{stem}_frame_strip.png"
        make_frame_strip(samples, rows, strip_path)
        print(strip_path)
    return {
        "anno_path": str(anno_path),
        "csv": str(csv_path),
        "plot": str(plot_path),
        "video": str(video_path) if args.save_video else None,
        "frame_strip": str(strip_path) if args.save_frame_strip else None,
    }


def episode_has_images(repo_dir, anno):
    episode_index = int(anno["episode_index"])
    chunks_size = _load_chunks_size(repo_dir)
    return all(
        _view_hdf5_path(repo_dir, episode_index, chunks_size, view).exists()
        for view in ("main", "left_wrist", "right_wrist")
    )


def collect_split_annos(args, split):
    annos = []
    for repo_dir in _robotwin_repo_dirs(
        args.data_root,
        split=split,
        test_ratio=args.test_ratio,
        split_seed=args.split_seed,
    ):
        for anno_path in sorted((repo_dir / "anno").glob("episode_*.json")):
            with open(anno_path, "r") as f:
                anno = json.load(f)
            num_frames = int(anno.get("num_frames", 0))
            if num_frames < args.case_min_frames:
                continue
            if args.case_max_frames is not None and num_frames > args.case_max_frames:
                continue
            if not anno.get("subtasks"):
                continue
            if not episode_has_images(repo_dir, anno):
                continue
            annos.append(str(anno_path))
    if args.case_selection == "random":
        rng = random.Random(args.case_seed)
        rng.shuffle(annos)
    return annos[: args.cases_per_split]


def case_dir_name(index, anno_path):
    anno_path = Path(anno_path)
    task_name = anno_path.parent.parent.name
    episode_name = anno_path.stem
    safe_task_name = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in task_name)
    return f"{index:02d}_{safe_task_name}_{episode_name}"


def run_batch_visualization(args):
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    context = load_eval_context(args)
    manifest = {
        "data_root": args.data_root,
        "test_ratio": args.test_ratio,
        "split_seed": args.split_seed,
        "case_seed": args.case_seed,
        "case_selection": args.case_selection,
        "cases_per_split": args.cases_per_split,
        "splits": {},
    }
    for split in args.batch_visualize_splits:
        selected = collect_split_annos(args, split)
        if len(selected) < args.cases_per_split:
            print(
                f"warning: requested {args.cases_per_split} {split} cases, "
                f"but only found {len(selected)} matching episodes.",
                file=sys.stderr,
            )
        manifest["splits"][split] = []
        for index, anno_path in enumerate(selected):
            case_dir = output_root / split / case_dir_name(index, anno_path)
            print(f"=== {split} case {index}: {anno_path} ===", flush=True)
            result = visualize_episode(args, anno_path, case_dir, context=context)
            manifest["splits"][split].append(result)
    manifest_path = output_root / "batch_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(manifest_path)


def main():
    args = parse_args()
    if args.batch_visualize_splits:
        if args.output_video:
            raise ValueError("--output-video is only supported for single-episode visualization.")
        run_batch_visualization(args)
        return

    visualize_episode(args, args.anno_path, Path(args.output_dir), output_video=args.output_video)


if __name__ == "__main__":
    main()
