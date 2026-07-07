#!/usr/bin/env python3
"""Visualize RobotWin episode trajectories with progress curves."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

TOOLS_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TOOLS_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qwenvl.data.robotwin_processor import (  # noqa: E402
    _load_chunks_size,
    _load_observation_images,
    _view_hdf5_path,
    parse_robotwin_views,
)
from qwenvl.data.robotwin_progress import (  # noqa: E402
    GRIPPER_PROGRESS_MIN_TOTAL,
    ROTATION_PROGRESS_MIN_TOTAL,
    TRANSLATION_PROGRESS_MIN_TOTAL,
    _component_progress,
    build_subtask_progress_lookup,
    current_done_frame_indices,
    episode_parquet_path,
    load_episode_states,
    progress_for_subtask,
    time_progress_for_subtask,
)
from tools.utils.robotwin_video import (  # noqa: E402
    resize_letterbox,
    subtask_index_for_frame,
)


def _load_font(size: int = 14) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def build_progress_rows(
    anno_path: Path,
    *,
    views: Sequence[str] = ("main",),
    max_frames: Optional[int] = None,
    stride: int = 1,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Path]]:
    with open(anno_path, "r", encoding="utf-8") as f:
        anno = json.load(f)

    repo_dir = anno_path.parent.parent
    subtasks = [dict(item) for item in anno["subtasks"]]
    episode_index = int(anno["episode_index"])
    chunks_size = _load_chunks_size(repo_dir)
    image_hdf5_paths = {
        view: _view_hdf5_path(repo_dir, episode_index, chunks_size, view)
        for view in views
    }
    num_frames = int(anno["num_frames"])
    if max_frames is not None:
        num_frames = min(num_frames, max_frames)

    states = None
    progress_lookup = None
    state_path = episode_parquet_path(repo_dir, episode_index, chunks_size)
    if state_path.exists():
        states = load_episode_states(state_path)
        num_frames = min(num_frames, len(states))
        progress_lookup = build_subtask_progress_lookup(states, subtasks, anno)

    rows: List[Dict[str, Any]] = []
    for frame in range(0, num_frames, max(1, stride)):
        subtask_index = subtask_index_for_frame(subtasks, frame)
        current = subtasks[subtask_index]
        start = int(current["start_frame"])
        curve = progress_lookup.get(start) if progress_lookup is not None else None
        done_frames = set(
            current_done_frame_indices(
                current,
                num_frames,
                states=states,
                anno=anno,
                curve=curve,
            )
        )
        done = 1.0 if frame in done_frames else 0.0
        motion_progress = 1.0 if done else progress_for_subtask(
            current,
            frame,
            states=states,
            anno=anno,
            curve=curve,
        )
        time_progress = time_progress_for_subtask(current, frame)
        offset = max(0, min(frame - start, len(curve.trans) - 1)) if curve is not None else 0
        trans_progress = (
            _component_progress(float(curve.trans[offset]), curve.trans_total, TRANSLATION_PROGRESS_MIN_TOTAL)
            if curve is not None
            else None
        )
        rot_progress = (
            _component_progress(float(curve.rot[offset]), curve.rot_total, ROTATION_PROGRESS_MIN_TOTAL)
            if curve is not None
            else None
        )
        grip_progress = (
            _component_progress(float(curve.grip[offset]), curve.grip_total, GRIPPER_PROGRESS_MIN_TOTAL)
            if curve is not None
            else None
        )
        rows.append(
            {
                "frame_index": frame,
                "subtask_index": subtask_index,
                "subtask_goal": current["subtask_goal"],
                "done_label": done,
                "progress_motion": motion_progress,
                "progress_time": time_progress,
                "progress_trans": trans_progress if trans_progress is not None else float("nan"),
                "progress_rot": rot_progress if rot_progress is not None else float("nan"),
                "progress_grip": grip_progress if grip_progress is not None else float("nan"),
                "state_source": str(state_path) if state_path.exists() else "",
            }
        )
    return anno, rows, image_hdf5_paths


def draw_axes(draw: ImageDraw.ImageDraw, box: Tuple[int, int, int, int], title: str, font: ImageFont.ImageFont) -> None:
    x0, y0, x1, y1 = box
    draw.rectangle(box, outline=(60, 64, 72), width=1)
    draw.text((x0, y0 - 22), title, font=font, fill=(20, 24, 32))
    for value in (0.0, 0.5, 1.0):
        y = y1 - value * (y1 - y0)
        draw.line((x0, y, x1, y), fill=(228, 230, 235), width=1)
        draw.text((x0 - 34, y - 7), f"{value:.1f}", font=font, fill=(90, 96, 108))


def _point_for_frame(
    frame_index: int,
    value: float,
    box: Tuple[int, int, int, int],
    num_frames: int,
) -> Tuple[float, float]:
    x0, y0, x1, y1 = box
    x = x0 + frame_index / max(1, num_frames - 1) * (x1 - x0)
    clamped = max(0.0, min(1.0, value))
    y = y1 - clamped * (y1 - y0)
    return x, y


def draw_series(
    draw: ImageDraw.ImageDraw,
    rows: Sequence[Dict[str, Any]],
    key: str,
    box: Tuple[int, int, int, int],
    color: Tuple[int, int, int],
    num_frames: int,
    *,
    width: int = 2,
    dashed: bool = False,
) -> None:
    points: List[Tuple[float, float]] = []
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if numeric != numeric:
            continue
        points.append(_point_for_frame(int(row["frame_index"]), numeric, box, num_frames))
    if len(points) < 2:
        return
    if dashed:
        for start, end in zip(points[:-1], points[1:]):
            draw.line((start, end), fill=color, width=width)
    else:
        draw.line(points, fill=color, width=width, joint="curve")


def draw_subtask_boundaries(
    draw: ImageDraw.ImageDraw,
    subtasks: Sequence[Dict[str, Any]],
    box: Tuple[int, int, int, int],
    num_frames: int,
    font: ImageFont.ImageFont,
) -> None:
    x0, y0, x1, y1 = box
    for idx, subtask in enumerate(subtasks):
        start = int(subtask["start_frame"])
        if start < 0 or start >= num_frames:
            continue
        x = x0 + start / max(1, num_frames - 1) * (x1 - x0)
        draw.line((x, y0, x, y1), fill=(235, 210, 120), width=1)
        draw.text((x + 3, y0 + 4), f"s{idx}", font=font, fill=(130, 100, 20))


def draw_video_panel(
    row: Dict[str, Any],
    anno: Dict[str, Any],
    image_hdf5_paths: Dict[str, Path],
    views: Sequence[str],
    width: int,
    height: int,
    font: ImageFont.ImageFont,
) -> Image.Image:
    canvas = Image.new("RGB", (width, height), (242, 244, 247))
    draw = ImageDraw.Draw(canvas)
    pad = 20
    title_h = 72
    frame_index = int(row["frame_index"])
    images = _load_observation_images(image_hdf5_paths, frame_index, views)
    image_y = title_h + pad
    image_h = height - image_y - pad

    title = (
        f"{Path(anno.get('repo', 'repo')).name}  "
        f"episode_{int(anno['episode_index']):06d}  frame={frame_index}  "
        f"subtask={int(row['subtask_index'])}"
    )
    draw.text((pad, 16), title[:140], font=font, fill=(20, 24, 32))
    metric = (
        f"motion={float(row['progress_motion']):.3f}  "
        f"time={float(row['progress_time']):.3f}  "
        f"done={int(float(row['done_label']) >= 0.5)}"
    )
    draw.text((pad, 40), metric, font=font, fill=(50, 56, 68))
    goal = str(row.get("subtask_goal", ""))
    draw.text((pad, 58), goal[:120], font=font, fill=(80, 86, 96))

    if len(views) == 1:
        view = views[0]
        panel = resize_letterbox(images[view], (width - pad * 2, image_h))
        canvas.paste(panel, (pad, image_y))
        draw.rectangle((pad, image_y, pad + panel.width - 1, image_y + panel.height - 1), outline=(36, 42, 52), width=2)
        draw.text((pad + 8, image_y + 8), view, font=font, fill=(245, 247, 250))
    else:
        gap = 12
        main_w = int((width - pad * 2 - gap) * 0.66)
        side_w = width - pad * 2 - gap - main_w
        wrist_h = (image_h - gap) // 2
        boxes = (
            ("main", images["main"], (pad, image_y, main_w, image_h)),
            ("left wrist", images["left_wrist"], (pad + main_w + gap, image_y, side_w, wrist_h)),
            (
                "right wrist",
                images["right_wrist"],
                (pad + main_w + gap, image_y + wrist_h + gap, side_w, image_h - wrist_h - gap),
            ),
        )
        for label, image, (x, y, w, h) in boxes:
            panel = resize_letterbox(image, (w, h))
            canvas.paste(panel, (x, y))
            draw.rectangle((x, y, x + w - 1, y + h - 1), outline=(36, 42, 52), width=2)
            draw.text((x + 8, y + 8), label, font=font, fill=(245, 247, 250))
    return canvas


def draw_curve_panel(
    rows: Sequence[Dict[str, Any]],
    anno: Dict[str, Any],
    current_frame: int,
    width: int,
    height: int,
    font: ImageFont.ImageFont,
) -> Image.Image:
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    num_frames = max(int(row["frame_index"]) for row in rows) + 1
    progress_box = (72, 48, width - 32, height - 170)
    component_box = (72, height - 145, width - 32, height - 36)

    draw_axes(draw, progress_box, "Progress: blue=motion, gray=time", font)
    draw_series(draw, rows, "progress_time", progress_box, (170, 170, 170), num_frames, width=2)
    draw_series(draw, rows, "progress_motion", progress_box, (35, 105, 210), num_frames, width=3)
    draw_subtask_boundaries(draw, anno["subtasks"], progress_box, num_frames, font)

    draw_axes(draw, component_box, "Components: green=trans, orange=rot, purple=grip", font)
    draw_series(draw, rows, "progress_trans", component_box, (34, 139, 74), num_frames, width=2)
    draw_series(draw, rows, "progress_rot", component_box, (230, 126, 34), num_frames, width=2)
    draw_series(draw, rows, "progress_grip", component_box, (142, 68, 173), num_frames, width=2)

    cursor_x = progress_box[0] + current_frame / max(1, num_frames - 1) * (progress_box[2] - progress_box[0])
    draw.line(
        (cursor_x, progress_box[1], cursor_x, component_box[3]),
        fill=(20, 24, 32),
        width=2,
    )
    draw.text((24, height - 24), f"frame {current_frame}/{max(0, num_frames - 1)}", font=font, fill=(20, 24, 32))
    return canvas


def save_progress_video(
    anno: Dict[str, Any],
    rows: Sequence[Dict[str, Any]],
    image_hdf5_paths: Dict[str, Path],
    views: Sequence[str],
    output_path: Path,
    *,
    fps: float,
    width: int,
    top_height: int,
    curve_height: int,
) -> str:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to save the progress video.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    height = top_height + curve_height
    if width % 2:
        width += 1
    if height % 2:
        height += 1

    font = _load_font(14)
    small_font = _load_font(12)
    with tempfile.TemporaryDirectory(prefix="robotwin_progress_video_") as tmp:
        tmp_dir = Path(tmp)
        for idx, row in enumerate(rows):
            top = draw_video_panel(row, anno, image_hdf5_paths, views, width, top_height, font)
            curve = draw_curve_panel(rows, anno, int(row["frame_index"]), width, curve_height, small_font)
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


def resolve_anno_path(root: Path, repo: Optional[str], episode_index: Optional[int], anno_path: Optional[Path]) -> Path:
    if anno_path is not None:
        return anno_path
    if repo is None or episode_index is None:
        raise ValueError("Provide --anno-path or both --repo and --episode-index.")
    return root / repo / "anno" / f"episode_{episode_index:06d}.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/media/damoxing/datasets/VLN-CE/cogwam_data/20260629"),
    )
    parser.add_argument("--repo", default=None, help="Repo directory name under --root.")
    parser.add_argument("--episode-index", type=int, default=None)
    parser.add_argument("--anno-path", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--views", default="main", help="Comma-separated views, e.g. main or main,left_wrist,right_wrist")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--stride", type=int, default=2, help="Sample every N frames to keep video size reasonable.")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--top-height", type=int, default=720)
    parser.add_argument("--curve-height", type=int, default=360)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    anno_path = resolve_anno_path(args.root, args.repo, args.episode_index, args.anno_path)
    if not anno_path.exists():
        raise SystemExit(f"Annotation not found: {anno_path}")

    views = parse_robotwin_views(args.views)
    anno, rows, image_hdf5_paths = build_progress_rows(
        anno_path,
        views=views,
        max_frames=args.max_frames,
        stride=args.stride,
    )
    if not rows:
        raise SystemExit(f"No frames to visualize for {anno_path}")

    repo_name = anno_path.parent.parent.name
    episode_index = int(anno["episode_index"])
    output = args.output or (
        args.root
        / "_progress_videos"
        / f"{repo_name}_episode_{episode_index:06d}_progress.mp4"
    )
    saved = save_progress_video(
        anno,
        rows,
        image_hdf5_paths,
        views,
        output,
        fps=args.fps,
        width=args.width,
        top_height=args.top_height,
        curve_height=args.curve_height,
    )
    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8") as f:
        f.write(
            "frame_index,subtask_index,done_label,progress_motion,progress_time,"
            "progress_trans,progress_rot,progress_grip,state_source\n"
        )
        for row in rows:
            f.write(
                f"{row['frame_index']},{row['subtask_index']},{row['done_label']},"
                f"{row['progress_motion']:.6f},{row['progress_time']:.6f},"
                f"{row['progress_trans']},{row['progress_rot']},{row['progress_grip']},"
                f"{row['state_source']}\n"
            )
    print(f"Wrote video: {saved}")
    print(f"Wrote csv:   {csv_path}")
    print(f"Frames: {len(rows)}  state: {rows[0]['state_source']}")


if __name__ == "__main__":
    main()
