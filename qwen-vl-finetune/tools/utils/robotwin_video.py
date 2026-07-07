#!/usr/bin/env python
import json
import random
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

from qwenvl.data.robotwin_processor import (
    RobotWinSample,
    _load_chunks_size,
    _load_observation_images,
    _robotwin_repo_dirs,
    _view_hdf5_path,
)
from qwenvl.data.robotwin_progress import (
    build_subtask_progress_lookup,
    current_done_frame_indices,
    episode_parquet_path,
    load_episode_states,
    progress_for_subtask,
    state_prompt_values,
)


def subtask_index_for_frame(subtasks, frame: int) -> int:
    for idx, subtask in enumerate(subtasks):
        if int(subtask["start_frame"]) <= frame <= int(subtask["end_frame"]):
            return idx
    for idx in range(len(subtasks) - 1, -1, -1):
        if frame > int(subtasks[idx]["end_frame"]):
            return idx
    return 0


def build_episode_samples(
    anno_path,
    max_frames: Optional[int] = None,
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
    frame_indices: Optional[Sequence[int]] = None,
    fixed_subtask_index: Optional[int] = None,
    image_repo_dir: Optional[Path] = None,
    views: Sequence[str] = ("main", "left_wrist", "right_wrist"),
) -> Tuple[Dict, List[RobotWinSample]]:
    anno_path = Path(anno_path)
    repo_dir = anno_path.parent.parent
    image_repo_dir = Path(image_repo_dir) if image_repo_dir is not None else repo_dir
    with open(anno_path, "r") as f:
        anno = json.load(f)
    subtasks = [dict(item) for item in anno["subtasks"]]
    episode_index = int(anno["episode_index"])
    chunks_size = _load_chunks_size(repo_dir)
    active_views = tuple(views)
    image_hdf5_paths = {
        view: _view_hdf5_path(image_repo_dir, episode_index, chunks_size, view)
        for view in active_views
    }
    num_frames = int(anno["num_frames"])
    if max_frames is not None:
        num_frames = min(num_frames, max_frames)

    first_frame = max(0, start_frame or 0)
    last_frame = num_frames - 1 if end_frame is None else min(num_frames - 1, end_frame)
    if frame_indices is not None:
        frames_to_build = sorted({int(frame) for frame in frame_indices})
    else:
        frames_to_build = list(range(first_frame, last_frame + 1))

    progress_lookup = None
    states = None
    state_parquet_path = episode_parquet_path(repo_dir, episode_index, chunks_size)
    if state_parquet_path.exists():
        try:
            states = load_episode_states(state_parquet_path)
            progress_lookup = build_subtask_progress_lookup(states, subtasks, anno)
        except Exception:
            states = None
            progress_lookup = None

    samples = []
    done_frame_cache: Dict[int, set] = {}
    for frame in frames_to_build:
        current_idx = fixed_subtask_index
        if current_idx is None:
            current_idx = subtask_index_for_frame(subtasks, frame)
        if current_idx < 0 or current_idx >= len(subtasks):
            raise IndexError(f"fixed_subtask_index {current_idx} out of range for {len(subtasks)} subtasks.")
        current = subtasks[current_idx]
        start = int(current["start_frame"])
        if start not in done_frame_cache:
            curve = progress_lookup.get(start) if progress_lookup is not None else None
            done_frame_cache[start] = set(
                current_done_frame_indices(
                    current,
                    num_frames,
                    states=states,
                    anno=anno,
                    curve=curve,
                )
            )
        done = 1.0 if frame in done_frame_cache[start] else 0.0
        curve = progress_lookup.get(start) if progress_lookup is not None else None
        progress = 1.0 if done else progress_for_subtask(
            current,
            frame,
            states=states,
            anno=anno,
            curve=curve,
        )
        samples.append(
            RobotWinSample(
                kind="q2",
                repo_dir=repo_dir,
                image_hdf5_paths=image_hdf5_paths,
                frame_index=frame,
                frame_start=frame,
                frame_end=frame,
                task_goal=anno["task_goal"],
                subtasks=subtasks,
                current_subtask_index=current_idx,
                views=active_views,
                image_repo_dir=image_repo_dir,
                current_done=done,
                progress=progress,
                q2_group="trajectory",
                state_values=state_prompt_values(states, frame),
            )
        )
    return anno, samples


def episode_has_images(repo_dir, anno) -> bool:
    episode_index = int(anno["episode_index"])
    chunks_size = _load_chunks_size(repo_dir)
    return all(
        _view_hdf5_path(repo_dir, episode_index, chunks_size, view).exists()
        for view in ("main", "left_wrist", "right_wrist")
    )


def collect_episode_annos(
    data_root: str,
    split: str,
    test_ratio: float,
    split_seed: int,
    limit: int,
    selection: str = "random",
    seed: int = 0,
) -> List[str]:
    split_arg = None if split == "all" else split
    annos = []
    for repo_dir in _robotwin_repo_dirs(
        data_root,
        split=split_arg,
        test_ratio=test_ratio,
        split_seed=split_seed,
    ):
        for anno_path in sorted((repo_dir / "anno").glob("episode_*.json")):
            with open(anno_path, "r") as f:
                anno = json.load(f)
            if not anno.get("subtasks") or not episode_has_images(repo_dir, anno):
                continue
            annos.append(str(anno_path))
    if selection == "random":
        random.Random(seed).shuffle(annos)
    return annos[:limit]


def anno_path_from_prediction(row: Dict) -> Path:
    if row.get("anno_path"):
        return Path(row["anno_path"])
    repo_dir = Path(row["repo_dir"]) if row.get("repo_dir") else None
    if repo_dir is None:
        raise ValueError("Prediction row must contain either anno_path or repo_dir.")
    episode_index = int(row["episode_index"])
    return repo_dir / "anno" / f"episode_{episode_index:06d}.json"


def resize_letterbox(image, size, fill=(18, 20, 24)):
    target_w, target_h = size
    scale = min(target_w / image.width, target_h / image.height)
    new_w = max(1, int(round(image.width * scale)))
    new_h = max(1, int(round(image.height * scale)))
    resized = image.resize((new_w, new_h), Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", size, fill)
    canvas.paste(resized, ((target_w - new_w) // 2, (target_h - new_h) // 2))
    return canvas


def truncate_text(draw, text: str, font, max_width: int) -> str:
    text = str(text)
    if not text:
        return ""
    if draw.textlength(text, font=font) <= max_width:
        return text
    trimmed = text
    while trimmed and draw.textlength(trimmed + "...", font=font) > max_width:
        trimmed = trimmed[:-1]
    return f"{trimmed}..." if trimmed else "..."


def draw_top_panel(sample, row, anno, width, height, font):
    canvas = Image.new("RGB", (width, height), (242, 244, 247))
    draw = ImageDraw.Draw(canvas)
    pad = 24
    title_y = 14
    instruction_y = 32
    title_h = 62
    gap = 14
    views = tuple(sample.views) if getattr(sample, "views", None) else tuple(sample.image_hdf5_paths)
    images = _load_observation_images(sample.image_hdf5_paths, sample.frame_index, views)
    image_y = title_h + pad
    image_h = height - image_y - pad
    boxes = []
    if views == ("main",) or set(views) == {"main"}:
        boxes = (("main", images["main"], (pad, image_y, width - pad * 2, image_h)),)
    else:
        main_w = int(width * 0.66)
        side_w = width - main_w - pad * 2 - gap
        wrist_h = (image_h - gap) // 2
        if "main" in images:
            boxes.append(("main", images["main"], (pad, image_y, main_w, image_h)))
        if "left_wrist" in images:
            boxes.append(
                ("left wrist", images["left_wrist"], (pad + main_w + gap, image_y, side_w, wrist_h))
            )
        if "right_wrist" in images:
            boxes.append(
                (
                    "right wrist",
                    images["right_wrist"],
                    (pad + main_w + gap, image_y + wrist_h + gap, side_w, image_h - wrist_h - gap),
                )
            )
        if not boxes:
            view = views[0]
            boxes = ((view, images[view], (pad, image_y, width - pad * 2, image_h)),)
    title = f"{sample.repo_dir.name} episode_{int(anno['episode_index']):06d} frame={sample.frame_index} subtask={sample.current_subtask_index}"
    draw.text((pad, title_y), title[:180], font=font, fill=(20, 24, 32))
    instruction = row.get("current_subtask_goal")
    if not instruction and sample.subtasks:
        idx = int(sample.current_subtask_index)
        if 0 <= idx < len(sample.subtasks):
            instruction = sample.subtasks[idx].get("subtask_goal", "")
    instruction = truncate_text(draw, instruction or "", font, width - 2 * pad)
    if instruction:
        draw.text((pad, instruction_y), instruction, font=font, fill=(60, 68, 80))
    metric = (
        f"done gt={int(float(row['done_label']) >= 0.5)} pred={int(float(row['done_pred']))} prob={float(row['done_prob']):.3f}  "
        f"progress gt={float(row['progress_label']):.3f} pred={float(row['progress_pred']):.3f}"
    )
    draw.text((max(pad, width - pad - 560), title_y), metric, font=font, fill=(20, 24, 32))
    for label, image, box in boxes:
        x, y, w, h = box
        panel = resize_letterbox(image, (w, h))
        canvas.paste(panel, (x, y))
        draw.rectangle((x, y, x + w - 1, y + h - 1), outline=(36, 42, 52), width=2)
        draw.rectangle((x + 8, y + 8, x + 112, y + 30), fill=(20, 24, 32))
        draw.text((x + 14, y + 13), label, font=font, fill=(245, 247, 250))
    return canvas


def draw_axes(draw, box, title, font):
    x0, y0, x1, y1 = box
    draw.rectangle(box, outline=(40, 40, 40), width=1)
    draw.text((x0, y0 - 20), title, font=font, fill=(20, 24, 32))
    for value in (0.0, 0.5, 1.0):
        y = y1 - value * (y1 - y0)
        draw.line((x0, y, x1, y), fill=(225, 225, 225), width=1)
        draw.text((x0 - 32, y - 6), f"{value:.1f}", font=font, fill=(80, 86, 96))


def point_for(row, key, box, num_frames):
    x0, y0, x1, y1 = box
    frame = int(row["frame_index"])
    x = x0 + frame / max(1, num_frames - 1) * (x1 - x0)
    value = max(0.0, min(1.0, float(row[key])))
    y = y1 - value * (y1 - y0)
    return x, y


def without_prev_done(rows):
    return [row for row in rows if row.get("q2_group") != "prev_done"]


def only_prev_done(rows):
    return [row for row in rows if row.get("q2_group") == "prev_done"]


def _ordered_rows(rows):
    return sorted(rows, key=lambda row: (int(row["frame_index"]), int(row.get("current_subtask_index", 0))))


def _prev_done_groups(rows):
    groups = {}
    for row in only_prev_done(rows):
        key = int(row.get("current_subtask_index", 0))
        groups.setdefault(key, []).append(row)
    return list(groups.values())


def draw_points_with_polyline(
    draw,
    rows,
    key,
    box,
    line_color,
    point_color,
    point_radius=3,
    line_width=2,
    num_frames=None,
):
    if not rows:
        return
    if num_frames is None:
        num_frames = max(int(row["frame_index"]) for row in rows) + 1
    ordered = _ordered_rows(rows)
    if len(ordered) >= 2:
        draw.line(
            [point_for(row, key, box, num_frames) for row in ordered],
            fill=line_color,
            width=line_width,
            joint="curve",
        )
    for row in ordered:
        x, y = point_for(row, key, box, num_frames)
        draw.ellipse(
            (x - point_radius, y - point_radius, x + point_radius, y + point_radius),
            fill=point_color,
        )


def draw_polyline(draw, rows, key, box, color, width=3, num_frames=None):
    if len(rows) < 2:
        return
    if num_frames is None:
        num_frames = max(int(row["frame_index"]) for row in rows) + 1
    ordered = _ordered_rows(rows)
    draw.line(
        [point_for(row, key, box, num_frames) for row in ordered],
        fill=color,
        width=width,
        joint="curve",
    )


def draw_done_predictions(draw, rows, box, num_frames):
    main_rows = without_prev_done(rows)
    if len(main_rows) >= 2:
        ordered = _ordered_rows(main_rows)
        for prev, cur in zip(ordered[:-1], ordered[1:]):
            color = (20, 150, 70) if int(float(prev["done_correct"])) and int(float(cur["done_correct"])) else (210, 45, 45)
            draw.line(
                (point_for(prev, "done_prob", box, num_frames), point_for(cur, "done_prob", box, num_frames)),
                fill=color,
                width=3,
            )
    for row in main_rows:
        x, y = point_for(row, "done_prob", box, num_frames)
        color = (20, 150, 70) if int(float(row["done_correct"])) else (210, 45, 45)
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)

    for group in _prev_done_groups(rows):
        if len(group) >= 2:
            ordered = _ordered_rows(group)
            draw.line(
                [point_for(row, "done_prob", box, num_frames) for row in ordered],
                fill=(245, 145, 35),
                width=2,
                joint="curve",
            )
        for row in group:
            x, y = point_for(row, "done_prob", box, num_frames)
            point_color = (20, 150, 70) if int(float(row["done_correct"])) else (210, 45, 45)
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=point_color)


def draw_progress_predictions(draw, rows, box, num_frames):
    main_rows = without_prev_done(rows)
    draw_polyline(draw, main_rows, "progress_pred", box, (35, 105, 210), width=3, num_frames=num_frames)
    for group in _prev_done_groups(rows):
        draw_points_with_polyline(
            draw,
            group,
            "progress_pred",
            box,
            line_color=(245, 145, 35),
            point_color=(245, 145, 35),
            point_radius=3,
            line_width=2,
            num_frames=num_frames,
        )


def draw_curve_panel(
    rows,
    sample,
    current_frame,
    width,
    height,
    font,
    event_frame: Optional[int] = None,
    timeline_num_frames: Optional[int] = None,
):
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    rows = sorted(rows, key=lambda item: (int(item["frame_index"]), int(item.get("current_subtask_index", 0))))
    curve_rows = without_prev_done(rows)
    num_frames = timeline_num_frames
    if num_frames is None:
        num_frames = max(int(row["frame_index"]) for row in rows) + 1
    done_box = (70, 54, width - 36, height // 2 - 28)
    progress_box = (70, height // 2 + 44, width - 36, height - 58)
    draw_axes(draw, done_box, "Done: gray=label; pred green/red, prev_done orange", font)
    draw_axes(draw, progress_box, "Progress: gray=label; pred blue, prev_done orange", font)
    draw_polyline(draw, curve_rows, "done_label", done_box, (150, 150, 150), width=2, num_frames=num_frames)
    draw_done_predictions(draw, rows, done_box, num_frames)
    draw_polyline(draw, curve_rows, "progress_label", progress_box, (170, 170, 170), width=2, num_frames=num_frames)
    draw_progress_predictions(draw, rows, progress_box, num_frames)
    for idx, subtask in enumerate(sample.subtasks):
        start = int(subtask["start_frame"])
        if start < 0 or start >= num_frames:
            continue
        x = done_box[0] + start / max(1, num_frames - 1) * (done_box[2] - done_box[0])
        draw.line((x, done_box[1], x, progress_box[3]), fill=(235, 210, 120), width=1)
        draw.text((x + 4, progress_box[1] - 18), f"s{idx}", font=font, fill=(120, 95, 20))

    if event_frame is not None:
        event_x = done_box[0] + event_frame / max(1, num_frames - 1) * (done_box[2] - done_box[0])
        draw.line((event_x, done_box[1], event_x, progress_box[3]), fill=(210, 45, 45), width=5)
        draw.text((event_x + 5, done_box[1] + 5), "error", font=font, fill=(210, 45, 45))

    current_x = done_box[0] + current_frame / max(1, num_frames - 1) * (done_box[2] - done_box[0])
    draw.line((current_x, done_box[1], current_x, progress_box[3]), fill=(20, 24, 32), width=2)
    for box in (done_box, progress_box):
        for frame in (0, num_frames // 2, num_frames - 1):
            x = box[0] + frame / max(1, num_frames - 1) * (box[2] - box[0])
            draw.text((x - 10, box[3] + 12), str(frame), font=font, fill=(80, 86, 96))
    draw.text((24, height - 26), f"frame {current_frame}/{max(0, num_frames - 1)}", font=font, fill=(20, 24, 32))
    return canvas


def save_q2_video(
    rows: Sequence[Dict],
    samples: Sequence[RobotWinSample],
    anno: Dict,
    output_path,
    fps: float,
    width: int,
    top_height: int,
    curve_height: int,
    event_frame: Optional[int] = None,
    timeline_num_frames: Optional[int] = None,
) -> str:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to save the progress video.")
    pairs = sorted(zip(samples, rows), key=lambda pair: int(pair[1]["frame_index"]))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    height = top_height + curve_height
    if width % 2:
        width += 1
    if height % 2:
        height += 1
    font = ImageFont.load_default()
    with tempfile.TemporaryDirectory(prefix="robotwin_q2_video_") as tmp_dir:
        tmp_dir = Path(tmp_dir)
        for idx, (sample, row) in enumerate(pairs):
            top = draw_top_panel(sample, row, anno, width, top_height, font)
            curve = draw_curve_panel(
                rows,
                sample,
                int(row["frame_index"]),
                width,
                curve_height,
                font,
                event_frame=event_frame,
                timeline_num_frames=timeline_num_frames,
            )
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
    return str(output_path)
