#!/usr/bin/env python3
"""Export RoboTwin subtask review videos or per-subtask clips."""

from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path("/media/damoxing/datasets/vae4d/lerobot-vae4d-org/robotwin_gt_depth")
PREFERRED_VIDEO_KEYS = (
    "observation.images.cam_high",
    "observation.images.cam_main",
    "observation.images.cam_front",
    "observation.images.head_camera",
)
SUBTASK_COLORS = (
    (62, 153, 255),
    (255, 132, 82),
    (92, 204, 137),
    (208, 142, 255),
    (255, 205, 72),
    (82, 216, 225),
    (255, 108, 155),
    (165, 190, 90),
    (150, 165, 255),
    (225, 160, 100),
)
TEXT_WHITE = (238, 238, 238)
TEXT_MUTED = (172, 178, 188)
PANEL_BG = (23, 24, 28)
CANVAS_BG = (11, 12, 14)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def iter_repos(root: Path, only: str | None) -> list[Path]:
    repos: list[Path] = []
    for repo in sorted(root.iterdir()):
        if not repo.is_dir():
            continue
        if only and only not in repo.name:
            continue
        if (repo / "meta" / "info.json").exists() and (repo / "anno").exists():
            repos.append(repo)
    return repos


def select_one_repo_per_task(repos: list[Path]) -> list[Path]:
    """Choose one deterministic repo for each task, preferring clean splits."""
    selected: dict[str, Path] = {}
    for repo in repos:
        slug = task_slug(repo.name)
        current = selected.get(slug)
        if current is None:
            selected[slug] = repo
            continue
        current_is_clean = "clean_50" in current.name
        candidate_is_clean = "clean_50" in repo.name
        if candidate_is_clean and not current_is_clean:
            selected[slug] = repo
    return [selected[slug] for slug in sorted(selected)]


def video_feature_keys(info: dict[str, Any]) -> list[str]:
    features = info.get("features", {})
    return [key for key, spec in features.items() if isinstance(spec, dict) and spec.get("dtype") == "video"]


def select_video_key(info: dict[str, Any], requested: str | None) -> str:
    keys = video_feature_keys(info)
    if not keys:
        raise ValueError("no video feature found in meta/info.json")
    if requested:
        if requested in keys:
            return requested
        matches = [key for key in keys if key.endswith(requested) or key.split(".")[-1] == requested]
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise ValueError(f"requested video key {requested!r} not found; available={keys}")
        raise ValueError(f"requested video key {requested!r} is ambiguous; matches={matches}")
    for key in PREFERRED_VIDEO_KEYS:
        if key in keys:
            return key
    return keys[0]


def episode_chunk(episode_index: int, info: dict[str, Any]) -> int:
    return episode_index // int(info.get("chunks_size", 1000))


def resolve_video_path(repo: Path, info: dict[str, Any], episode_index: int, video_key: str) -> Path:
    pattern = info.get("video_path")
    if not pattern:
        raise ValueError("meta/info.json does not contain video_path")
    rel = pattern.format(episode_chunk=episode_chunk(episode_index, info), episode_index=episode_index, video_key=video_key)
    path = repo / rel
    if path.exists():
        return path

    # Some converted repos use a different directory name but keep the same
    # feature key. Search narrowly before giving up.
    candidates = sorted(repo.glob(f"videos*/chunk-*/{video_key}/episode_{episode_index:06d}.mp4"))
    if candidates:
        return candidates[0]
    raise FileNotFoundError(path)


def sanitize_filename(text: str, max_len: int = 96) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    if not text:
        text = "subtask"
    return text[:max_len].rstrip("_")


def sample_annos(repo: Path, sample_count: int, seed: int, selection: str) -> list[Path]:
    annos = sorted((repo / "anno").glob("episode_*.json"))
    if sample_count <= 0 or sample_count >= len(annos):
        return annos
    if selection == "first":
        return annos[:sample_count]
    rng = random.Random(f"{seed}:{repo.name}")
    return sorted(rng.sample(annos, sample_count))


def sample_issue_records(
    issues: list[dict[str, Any]],
    sample_count: int,
    seed: int,
    selection: str,
) -> list[dict[str, Any]]:
    if sample_count <= 0 or sample_count >= len(issues):
        return issues
    if selection == "first":
        return issues[:sample_count]
    rng = random.Random(seed)
    return rng.sample(issues, sample_count)


def issue_subtask_index(issue: dict[str, Any], side: str) -> int | None:
    context = issue.get(side)
    if not isinstance(context, dict):
        return None
    try:
        return int(context.get("subtask_index"))
    except (TypeError, ValueError):
        return None


def issue_summary(issue: dict[str, Any]) -> str:
    left = issue.get("left_subtask") if isinstance(issue.get("left_subtask"), dict) else {}
    current = issue.get("current_subtask") if isinstance(issue.get("current_subtask"), dict) else {}
    raw_subtask = int(issue.get("subtask_index", -1))
    display_subtask = raw_subtask + 1 if raw_subtask >= 0 else "?"
    left_text = (
        "none"
        if not left
        else f"{left.get('subtask_type', '?')}/{left.get('truncation_rule', '?')}"
    )
    current_text = (
        "none"
        if not current
        else f"{current.get('subtask_type', '?')}/{current.get('truncation_rule', '?')}"
    )
    return (
        f"issue: {issue.get('rule', '')}  |  "
        f"left {left_text} -> current {current_text}  |  "
        f"subtask {display_subtask} (index {raw_subtask})"
    )


def issue_output_name(issue: dict[str, Any], sample_index: int) -> str:
    repo = sanitize_filename(str(issue.get("repo", "repo")), max_len=60)
    rule = sanitize_filename(str(issue.get("rule", "issue")), max_len=40)
    episode = int(issue.get("episode_index", -1))
    subtask = int(issue.get("subtask_index", -1))
    display_subtask = subtask + 1 if subtask >= 0 else subtask
    return f"issue_{sample_index:02d}_{repo}_episode_{episode:06d}_subtask_{display_subtask:02d}_idx{subtask:02d}_{rule}_review.mp4"


def run_ffmpeg(
    input_video: Path,
    output_video: Path,
    start_frame: int,
    end_frame: int,
    dry_run: bool,
) -> None:
    output_video.parent.mkdir(parents=True, exist_ok=True)
    filters = [
        f"trim=start_frame={start_frame}:end_frame={end_frame + 1}",
        "setpts=PTS-STARTPTS",
    ]
    filters.append("colorchannelmixer=rr=0:rb=1:bb=0:br=1")
    vf = ",".join(filters)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(input_video),
        "-vf",
        vf,
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        str(output_video),
    ]
    if dry_run:
        print(" ".join(cmd))
        return
    subprocess.run(cmd, check=True)


def export_episode(
    repo: Path,
    anno_path: Path,
    info: dict[str, Any],
    video_key: str,
    output_root: Path,
    dry_run: bool,
) -> list[dict[str, Any]]:
    anno = read_json(anno_path)
    episode_index = int(anno["episode_index"])
    input_video = resolve_video_path(repo, info, episode_index, video_key)
    episode_dir = output_root / repo.name / f"episode_{episode_index:06d}"
    records: list[dict[str, Any]] = []

    for subtask in anno["subtasks"]:
        subtask_index = int(subtask["subtask_index"])
        goal = str(subtask["subtask_goal"])
        start_frame = int(subtask["start_frame"])
        end_frame = int(subtask["end_frame"])
        clip_name = f"{subtask_index:02d}_{sanitize_filename(goal)}.mp4"
        output_video = episode_dir / clip_name
        run_ffmpeg(
            input_video,
            output_video,
            start_frame,
            end_frame,
            dry_run=dry_run,
        )
        records.append(
            {
                "repo": repo.name,
                "episode_index": episode_index,
                "task_goal": anno.get("task_goal", ""),
                "video_key": video_key,
                "source_video": str(input_video),
                "clip": str(output_video),
                "subtask_index": subtask_index,
                "subtask_goal": goal,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "boundary_source": subtask.get("boundary_source", ""),
            }
        )

    write_json(episode_dir / "manifest.json", records)
    return records


def clamp_frame(frame_index: int, end_frame: int) -> int:
    return min(max(frame_index, 0), max(end_frame, 0))


def current_subtask(subtasks: list[dict[str, Any]], frame_index: int) -> dict[str, Any]:
    if not subtasks:
        raise ValueError("annotation does not contain subtasks")
    for subtask in subtasks:
        if int(subtask["start_frame"]) <= frame_index <= int(subtask["end_frame"]):
            return subtask
    if frame_index < int(subtasks[0]["start_frame"]):
        return subtasks[0]
    return subtasks[-1]


def text_size(cv2: Any, text: str, scale: float, thickness: int) -> tuple[int, int]:
    (width, height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    return width, height


def wrap_text(cv2: Any, text: str, max_width: int, scale: float, thickness: int) -> list[str]:
    words = text.strip().split()
    if not words:
        return [""]
    lines: list[str] = []
    current = ""
    for word in words:
        trial = word if not current else f"{current} {word}"
        if text_size(cv2, trial, scale, thickness)[0] <= max_width:
            current = trial
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def draw_text(
    cv2: Any,
    image: Any,
    text: str,
    origin: tuple[int, int],
    scale: float,
    color: tuple[int, int, int] = TEXT_WHITE,
    thickness: int = 1,
) -> None:
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def subtask_color(index: int) -> tuple[int, int, int]:
    return SUBTASK_COLORS[index % len(SUBTASK_COLORS)]


def draw_review_frame(
    frame: Any,
    anno: dict[str, Any],
    frame_index: int,
    canvas_width: int,
    review_scale: int,
) -> Any:
    import cv2
    import numpy as np

    subtasks = anno["subtasks"]
    active = current_subtask(subtasks, frame_index)
    active_index = int(active["subtask_index"])
    episode_end = max(int(subtask["end_frame"]) for subtask in subtasks)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    video = cv2.resize(frame, None, fx=review_scale, fy=review_scale, interpolation=cv2.INTER_AREA)
    video_h, video_w = video.shape[:2]
    split_rows = len(subtasks)
    issue = anno.get("_issue_context")
    issue_header_h = 46 if isinstance(issue, dict) else 0
    top_height = 126 + issue_header_h + split_rows * 27
    timeline_height = 70
    canvas_width += canvas_width % 2
    canvas_height = top_height + timeline_height + video_h
    canvas_height += canvas_height % 2
    canvas = np.full((canvas_height, canvas_width, 3), CANVAS_BG, dtype=np.uint8)

    cv2.rectangle(canvas, (0, 0), (canvas_width, top_height), PANEL_BG, -1)
    cv2.rectangle(canvas, (0, top_height), (canvas_width, top_height + timeline_height), PANEL_BG, -1)

    margin = 18
    text_width = canvas_width - margin * 2
    repo = anno.get("repo", "")
    episode_index = int(anno["episode_index"])
    task_goal = str(anno.get("task_goal", "")).strip()
    active_start = int(active["start_frame"])
    active_end = int(active["end_frame"])
    active_goal = str(active["subtask_goal"])
    boundary = str(active.get("boundary_source", ""))
    active_type = str(active.get("subtask_type", ""))
    active_rule = str(active.get("truncation_rule", ""))

    draw_text(
        cv2,
        canvas,
        f"{repo}  |  episode_{episode_index:06d}  |  subtask {active_index + 1}/{len(subtasks)}",
        (margin, 26),
        0.58,
        TEXT_WHITE,
        1,
    )
    for line_i, line in enumerate(wrap_text(cv2, f"Task: {task_goal}", text_width, 0.48, 1)[:2]):
        draw_text(cv2, canvas, line, (margin, 52 + line_i * 20), 0.48, TEXT_MUTED, 1)

    current_y = 96
    issue_current_index: int | None = None
    issue_left_index: int | None = None
    if isinstance(issue, dict):
        issue_current_index = issue_subtask_index(issue, "current_subtask")
        issue_left_index = issue_subtask_index(issue, "left_subtask")
        issue_y = 90
        cv2.rectangle(canvas, (margin - 6, issue_y - 17), (canvas_width - margin + 6, issue_y + 27), (58, 43, 34), -1)
        for line_i, line in enumerate(wrap_text(cv2, issue_summary(issue), text_width - 12, 0.45, 1)[:2]):
            draw_text(cv2, canvas, line, (margin, issue_y + line_i * 18), 0.45, (255, 205, 135), 1)
        current_y += issue_header_h

    cv2.rectangle(canvas, (margin - 6, current_y - 17), (canvas_width - margin + 6, current_y + 10), (43, 48, 55), -1)
    cv2.rectangle(canvas, (margin - 6, current_y - 17), (margin - 1, current_y + 10), subtask_color(active_index), -1)
    current_meta = "/".join(item for item in (active_type, active_rule) if item)
    current_text = f"Current: {active_start}-{active_end}  {current_meta}  {active_goal}  [{boundary}]"
    for line_i, line in enumerate(wrap_text(cv2, current_text, text_width - 12, 0.46, 1)[:2]):
        draw_text(cv2, canvas, line, (margin, current_y + line_i * 19), 0.46, TEXT_WHITE, 1)

    split_y = 130 + issue_header_h
    for subtask in subtasks:
        index = int(subtask["subtask_index"])
        y = split_y + index * 27
        row_bg = (34, 37, 42) if index == active_index else PANEL_BG
        if index == issue_current_index:
            row_bg = (72, 47, 35)
        elif index == issue_left_index:
            row_bg = (48, 49, 63)
        cv2.rectangle(canvas, (margin - 6, y - 16), (canvas_width - margin + 6, y + 7), row_bg, -1)
        color = subtask_color(index)
        cv2.rectangle(canvas, (margin, y - 12), (margin + 14, y + 2), color, -1)
        marker = "!" if index == issue_current_index else "<" if index == issue_left_index else " "
        label = (
            f"{marker}{index + 1:02d}  "
            f"{int(subtask['start_frame']):04d}-{int(subtask['end_frame']):04d}  "
            f"{subtask.get('subtask_type', '')}/{subtask.get('truncation_rule', '')}  "
            f"{subtask['subtask_goal']}"
        )
        lines = wrap_text(cv2, label, text_width - 28, 0.42, 1)
        draw_text(cv2, canvas, lines[0], (margin + 24, y), 0.42, TEXT_WHITE if index == active_index else TEXT_MUTED, 1)

    timeline_x = margin
    timeline_y = top_height + 17
    timeline_w = canvas_width - margin * 2
    timeline_h = 24
    cv2.rectangle(canvas, (timeline_x, timeline_y), (timeline_x + timeline_w, timeline_y + timeline_h), (48, 50, 56), -1)
    denominator = max(1, episode_end + 1)
    for subtask in subtasks:
        index = int(subtask["subtask_index"])
        x0 = timeline_x + int(timeline_w * int(subtask["start_frame"]) / denominator)
        x1 = timeline_x + int(timeline_w * (int(subtask["end_frame"]) + 1) / denominator)
        x1 = max(x0 + 2, x1)
        cv2.rectangle(canvas, (x0, timeline_y), (x1, timeline_y + timeline_h), subtask_color(index), -1)
        if x1 - x0 >= 26:
            draw_text(cv2, canvas, str(index + 1), (x0 + 7, timeline_y + 17), 0.45, (10, 10, 10), 1)
    cursor_x = timeline_x + int(timeline_w * clamp_frame(frame_index, episode_end) / denominator)
    cv2.line(canvas, (cursor_x, timeline_y - 7), (cursor_x, timeline_y + timeline_h + 7), (250, 250, 250), 2)
    draw_text(cv2, canvas, "color split", (timeline_x, timeline_y + timeline_h + 30), 0.44, TEXT_MUTED, 1)
    draw_text(
        cv2,
        canvas,
        f"frame {frame_index:04d}/{episode_end:04d}",
        (timeline_x + timeline_w - 134, timeline_y + timeline_h + 30),
        0.44,
        TEXT_MUTED,
        1,
    )

    video_x = (canvas_width - video_w) // 2
    video_y = top_height + timeline_height
    canvas[video_y : video_y + video_h, video_x : video_x + video_w] = video
    cv2.rectangle(canvas, (video_x, video_y), (video_x + video_w - 1, video_y + video_h - 1), (56, 60, 68), 1)
    return canvas


def read_video_frames(input_video: Path, frame_indices: list[int]) -> dict[int, Any]:
    import cv2

    wanted = sorted(set(frame_indices))
    frames: dict[int, Any] = {}
    if not wanted:
        return frames
    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video {input_video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    for frame_index in wanted:
        if total > 0 and frame_index >= total:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if ok:
            frames[frame_index] = frame
    cap.release()
    return frames


def draw_contact_sheet(
    repo: Path,
    anno: dict[str, Any],
    subtask: dict[str, Any],
    frames: dict[int, Any],
    frame_indices: list[int],
    *,
    scale: float,
    cell_width: int,
) -> Any:
    import cv2
    import numpy as np

    available = [index for index in frame_indices if index in frames]
    if not available:
        raise RuntimeError("no frames were available for contact sheet")
    first = frames[available[0]]
    thumb_w = cell_width
    thumb_h = max(1, int(first.shape[0] * thumb_w / max(1, first.shape[1]) * scale))
    header_h = 112
    label_h = 30
    margin = 14
    gap = 8
    canvas_w = margin * 2 + len(available) * thumb_w + (len(available) - 1) * gap
    canvas_h = header_h + thumb_h + label_h + margin
    canvas_w += canvas_w % 2
    canvas_h += canvas_h % 2
    canvas = np.full((canvas_h, canvas_w, 3), CANVAS_BG, dtype=np.uint8)
    cv2.rectangle(canvas, (0, 0), (canvas_w, header_h), PANEL_BG, -1)

    prev_goal = ""
    idx = int(subtask["subtask_index"])
    subtasks = anno.get("subtasks", [])
    if idx > 0 and idx - 1 < len(subtasks):
        prev_goal = str(subtasks[idx - 1].get("subtask_goal", ""))
    current_goal = str(subtask.get("subtask_goal", ""))
    boundary = str(subtask.get("boundary_source", ""))
    title = f"{repo.name}  episode_{int(anno['episode_index']):06d}  boundary before subtask {idx + 1}"
    draw_text(cv2, canvas, title, (margin, 26), 0.55, TEXT_WHITE, 1)
    draw_text(cv2, canvas, f"source: {boundary}", (margin, 50), 0.46, TEXT_MUTED, 1)
    if prev_goal:
        for line_i, line in enumerate(wrap_text(cv2, f"prev: {prev_goal}", canvas_w - margin * 2, 0.42, 1)[:1]):
            draw_text(cv2, canvas, line, (margin, 75 + line_i * 18), 0.42, TEXT_MUTED, 1)
    for line_i, line in enumerate(wrap_text(cv2, f"next: {current_goal}", canvas_w - margin * 2, 0.42, 1)[:1]):
        draw_text(cv2, canvas, line, (margin, 96 + line_i * 18), 0.42, TEXT_WHITE, 1)

    y = header_h
    for col, frame_index in enumerate(available):
        x = margin + col * (thumb_w + gap)
        frame = cv2.cvtColor(frames[frame_index], cv2.COLOR_BGR2RGB)
        thumb = cv2.resize(frame, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
        canvas[y : y + thumb_h, x : x + thumb_w] = thumb
        color = (246, 246, 246) if frame_index == int(subtask["start_frame"]) else TEXT_MUTED
        cv2.rectangle(canvas, (x, y), (x + thumb_w - 1, y + thumb_h - 1), color, 1)
        draw_text(cv2, canvas, f"f{frame_index}", (x + 6, y + thumb_h + 21), 0.44, color, 1)
    return canvas


def export_contact_sheets(
    repo: Path,
    anno_path: Path,
    info: dict[str, Any],
    video_key: str,
    output_root: Path,
    dry_run: bool,
    *,
    window: int,
    cell_width: int,
    scale: float,
) -> list[dict[str, Any]]:
    import cv2

    anno = read_json(anno_path)
    anno["repo"] = repo.name
    episode_index = int(anno["episode_index"])
    input_video = resolve_video_path(repo, info, episode_index, video_key)
    episode_dir = output_root / repo.name / f"episode_{episode_index:06d}"
    records: list[dict[str, Any]] = []
    for subtask in anno["subtasks"][1:]:
        start_frame = int(subtask["start_frame"])
        frame_indices = [clamp_frame(start_frame + offset, int(anno.get("num_frames", start_frame + window + 1)) - 1) for offset in range(-window, window + 1)]
        output_image = episode_dir / f"boundary_{int(subtask['subtask_index']):02d}_frame_{start_frame:04d}.png"
        if dry_run:
            print(f"contact-sheet {input_video} frames={frame_indices} -> {output_image}")
        else:
            frames = read_video_frames(input_video, frame_indices)
            sheet = draw_contact_sheet(
                repo,
                anno,
                subtask,
                frames,
                frame_indices,
                scale=scale,
                cell_width=cell_width,
            )
            output_image.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_image), sheet)
        records.append(
            {
                "repo": repo.name,
                "episode_index": episode_index,
                "video_key": video_key,
                "source_video": str(input_video),
                "contact_sheet": str(output_image),
                "subtask_index": int(subtask["subtask_index"]),
                "start_frame": start_frame,
                "boundary_source": subtask.get("boundary_source", ""),
                "subtask_goal": subtask.get("subtask_goal", ""),
                "frame_indices": frame_indices,
            }
        )
    write_json(episode_dir / "contact_sheet_manifest.json", records)
    return records


def render_review_episode(
    repo: Path,
    anno_path: Path,
    info: dict[str, Any],
    video_key: str,
    output_root: Path,
    dry_run: bool,
    review_scale: int,
    review_width: int,
    crf: int,
    issue: dict[str, Any] | None = None,
    output_name: str | None = None,
) -> dict[str, Any]:
    import cv2
    import numpy as np

    anno = read_json(anno_path)
    anno["repo"] = repo.name
    if issue is not None:
        anno["_issue_context"] = issue
    episode_index = int(anno["episode_index"])
    input_video = resolve_video_path(repo, info, episode_index, video_key)
    output_dir = output_root / ("issues" if issue is not None else task_slug(repo.name))
    output_video = output_dir / (output_name or f"{repo.name}_episode_{episode_index:06d}_review.mp4")

    if dry_run:
        print(f"review {input_video} -> {output_video}")
        return review_record(repo, anno, video_key, input_video, output_video, issue=issue)

    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video {input_video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or float(info.get("fps", 20.0))
    ok, first_frame = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError(f"cannot read first frame from {input_video}")

    canvas_width = max(int(review_width), int(first_frame.shape[1]) * max(1, review_scale))
    first_canvas = draw_review_frame(
        first_frame,
        anno,
        0,
        canvas_width,
        review_scale,
    )
    height, width = first_canvas.shape[:2]
    output_video.parent.mkdir(parents=True, exist_ok=True)
    tmp_video = output_video.with_suffix(".tmp.mp4")
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        f"{fps:.6f}",
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(tmp_video),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert proc.stdin is not None
    proc.stdin.write(np.ascontiguousarray(first_canvas).tobytes())
    frame_index = 1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        canvas = draw_review_frame(
            frame,
            anno,
            frame_index,
            canvas_width,
            review_scale
        )
        proc.stdin.write(np.ascontiguousarray(canvas).tobytes())
        frame_index += 1
    cap.release()
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed while writing {output_video}")
    tmp_video.replace(output_video)
    return review_record(repo, anno, video_key, input_video, output_video, issue=issue)


def task_slug(repo_name: str) -> str:
    marker = "-aloha-"
    if marker in repo_name:
        return repo_name.split(marker, 1)[0]
    return repo_name


def review_record(
    repo: Path,
    anno: dict[str, Any],
    video_key: str,
    input_video: Path,
    output_video: Path,
    issue: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record = {
        "repo": repo.name,
        "task_slug": task_slug(repo.name),
        "episode_index": int(anno["episode_index"]),
        "task_goal": anno.get("task_goal", ""),
        "video_key": video_key,
        "source_video": str(input_video),
        "review_video": str(output_video),
        "codec": "libx264",
        "pix_fmt": "yuv420p",
        "subtasks": [
            {
                "index": int(subtask["subtask_index"]),
                "start_frame": int(subtask["start_frame"]),
                "end_frame": int(subtask["end_frame"]),
                "boundary_source": subtask.get("boundary_source", ""),
                "goal": subtask["subtask_goal"],
            }
            for subtask in anno["subtasks"]
        ],
    }
    if issue is not None:
        record["issue"] = issue
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--mode", choices=["review", "clips", "contact-sheet"], default="review")
    parser.add_argument("--sample-count", type=int, default=2, help="Episodes to sample per repo; <=0 means all.")
    parser.add_argument(
        "--one-per-task",
        action="store_true",
        help="Export from one repo per task (preferring clean_50), instead of every clean/randomized repo.",
    )
    parser.add_argument("--selection", choices=["random", "first"], default="random")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--video-key", type=str, default=None, help="Full key or suffix, e.g. cam_main or cam_high.")
    parser.add_argument("--only", type=str, default=None, help="Only process repos whose names contain this string.")
    parser.add_argument("--review-scale", type=int, default=2, help="Scale factor for the middle video in review mode.")
    parser.add_argument("--review-width", type=int, default=960, help="Minimum canvas width in review mode.")
    parser.add_argument("--contact-window", type=int, default=3, help="Frames before/after each boundary for contact sheets.")
    parser.add_argument("--contact-cell-width", type=int, default=180, help="Thumbnail width for contact sheets.")
    parser.add_argument("--contact-scale", type=float, default=1.0, help="Thumbnail height scale for contact sheets.")
    parser.add_argument("--issues-log", type=Path, default=None, help="JSONL issue log to sample review videos from.")
    parser.add_argument("--issue-sample-count", type=int, default=10, help="Number of issue records to sample when --issues-log is set.")
    parser.add_argument("--crf", type=int, default=18, help="x264 CRF for exported videos.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.mode == "clips":
        output_root = args.output or (args.root / "_subtask_video_samples")
    else:
        output_root = args.output or (args.root / "_subtask_review_videos" / "subtask_split_review")
    all_records: list[dict[str, Any]] = []

    if args.issues_log is not None:
        issues = sample_issue_records(
            read_jsonl(args.issues_log),
            args.issue_sample_count,
            args.seed,
            args.selection,
        )
        for sample_index, raw_issue in enumerate(issues):
            issue = dict(raw_issue)
            issue["issue_sample_index"] = sample_index
            try:
                repo = args.root / str(issue["repo"])
                info = read_json(repo / "meta" / "info.json")
                video_key = select_video_key(info, args.video_key)
                episode_index = int(issue["episode_index"])
                anno_path = repo / "anno" / f"episode_{episode_index:06d}.json"
                all_records.append(
                    render_review_episode(
                        repo=repo,
                        anno_path=anno_path,
                        info=info,
                        video_key=video_key,
                        output_root=output_root,
                        dry_run=args.dry_run,
                        review_scale=max(1, args.review_scale),
                        review_width=args.review_width,
                        crf=args.crf,
                        issue=issue,
                        output_name=issue_output_name(issue, sample_index),
                    )
                )
            except Exception as exc:
                print(
                    f"[ERROR] issue_sample={sample_index} "
                    f"{issue.get('repo')} episode_{int(issue.get('episode_index', -1)):06d}: {exc}"
                )
        write_json(output_root / "manifest.json", all_records)
        action = "would export" if args.dry_run else "exported"
        print(f"{action} {len(all_records)} issue review videos to {output_root}")
        return

    repos = iter_repos(args.root, args.only)
    if args.one_per_task:
        repos = select_one_repo_per_task(repos)

    for repo in repos:
        info = read_json(repo / "meta" / "info.json")
        try:
            video_key = select_video_key(info, args.video_key)
            anno_paths = sample_annos(repo, args.sample_count, args.seed, args.selection)
            for anno_path in anno_paths:
                if args.mode == "clips":
                    all_records.extend(
                        export_episode(
                            repo=repo,
                            anno_path=anno_path,
                            info=info,
                            video_key=video_key,
                            output_root=output_root,
                            dry_run=args.dry_run,
                        )
                    )
                elif args.mode == "contact-sheet":
                    all_records.extend(
                        export_contact_sheets(
                            repo=repo,
                            anno_path=anno_path,
                            info=info,
                            video_key=video_key,
                            output_root=output_root,
                            dry_run=args.dry_run,
                            window=max(1, args.contact_window),
                            cell_width=max(64, args.contact_cell_width),
                            scale=max(0.1, args.contact_scale),
                        )
                    )
                else:
                    all_records.append(
                        render_review_episode(
                            repo=repo,
                            anno_path=anno_path,
                            info=info,
                            video_key=video_key,
                            output_root=output_root,
                            dry_run=args.dry_run,
                            review_scale=max(1, args.review_scale),
                            review_width=args.review_width,
                            crf=args.crf,
                        )
                    )
        except Exception as exc:
            print(f"[ERROR] {repo.name}: {exc}")

    write_json(output_root / "manifest.json", all_records)
    action = "would export" if args.dry_run else "exported"
    unit = "subtask clips" if args.mode == "clips" else "contact sheets" if args.mode == "contact-sheet" else "review videos"
    print(f"{action} {len(all_records)} {unit} to {output_root}")


if __name__ == "__main__":
    main()
