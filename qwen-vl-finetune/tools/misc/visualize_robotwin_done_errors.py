#!/usr/bin/env python
import argparse
import csv
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from qwenvl.data.robotwin_processor import build_robotwin_samples, _load_observation_images


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize RobotWin Q2 done FP/FN samples.")
    parser.add_argument("--predictions-csv", default="/media/damoxing/ckp/qwen_ft/robotwin_qwen3vl_2b/eval_q2_predictions.csv")
    parser.add_argument("--data-root", default="/media/damoxing/datasets/vae4d/lerobot-vae4d-org/robotwin_gt_depth")
    parser.add_argument("--output-dir", default="/media/damoxing/ckp/qwen_ft/robotwin_qwen3vl_2b/done_error_vis")
    parser.add_argument("--split", choices=("train", "test", "all"), default="test")
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--q2-frame-stride", type=int, default=8)
    parser.add_argument("--boundary-extra-frames", type=int, default=2)
    parser.add_argument("--num-per-type", type=int, default=8)
    parser.add_argument("--thumb-width", type=int, default=320)
    return parser.parse_args()


def load_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def wrap_text(text, max_chars):
    words = str(text).split()
    lines = []
    current = []
    current_len = 0
    for word in words:
        extra = 1 if current else 0
        if current and current_len + len(word) + extra > max_chars:
            lines.append(" ".join(current))
            current = [word]
            current_len = len(word)
        else:
            current.append(word)
            current_len += len(word) + extra
    if current:
        lines.append(" ".join(current))
    return lines


def draw_text_block(draw, xy, lines, font, fill=(20, 20, 20), line_h=16):
    x, y = xy
    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        y += line_h
    return y


def resize_keep(image, width):
    height = max(1, round(image.height * width / image.width))
    return image.resize((width, height), Image.Resampling.BICUBIC)


def make_panel(sample, row, error_type, thumb_width):
    font = ImageFont.load_default()
    images = _load_observation_images(sample.image_hdf5_paths, sample.frame_index)
    main = resize_keep(images["main"], thumb_width)
    left = resize_keep(images["left_wrist"], thumb_width // 2)
    right = resize_keep(images["right_wrist"], thumb_width // 2)

    image_gap = 8
    image_h = max(main.height, left.height, right.height)
    text_h = 180
    canvas_w = thumb_width + image_gap + thumb_width // 2 + image_gap + thumb_width // 2
    canvas_h = text_h + image_h + 34
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)

    color = (210, 45, 45) if error_type == "FN" else (220, 120, 20)
    draw.rectangle((0, 0, canvas_w - 1, canvas_h - 1), outline=color, width=5)
    current = sample.subtasks[sample.current_subtask_index]
    episode_name = Path(sample.image_hdf5_paths["main"]).stem
    header = (
        f"{error_type} sample={row['sample_index']} repo={sample.repo_dir.name} {episode_name} "
        f"frame={sample.frame_index} subtask={sample.current_subtask_index}"
    )
    metric = (
        f"done label={float(row['done_label']):.0f} pred={int(row['done_pred'])} "
        f"prob={float(row['done_prob']):.4f} | progress label={float(row['progress_label']):.3f} "
        f"pred={float(row['progress_pred']):.3f} abs_err={float(row['progress_abs_err']):.3f}"
    )
    y = 12
    y = draw_text_block(draw, (12, y), wrap_text(header, 110), font, fill=color)
    y = draw_text_block(draw, (12, y + 4), wrap_text(metric, 110), font)
    y = draw_text_block(draw, (12, y + 4), wrap_text(f"Task: {sample.task_goal}", 110), font)
    draw_text_block(draw, (12, y + 4), wrap_text(f"Prompt current subtask: {current['subtask_goal']}", 110), font)

    image_y = text_h
    x = 0
    for label, image in (("main", main), ("left wrist", left), ("right wrist", right)):
        canvas.paste(image, (x, image_y))
        draw.rectangle((x, image_y, x + image.width - 1, image_y + image.height - 1), outline=(30, 30, 30), width=1)
        draw.text((x + 6, image_y + image.height + 8), label, font=font, fill=(20, 20, 20))
        x += image.width + image_gap
    return canvas


def make_contact_sheet(images, cols=2, gap=12):
    if not images:
        return None
    w = max(image.width for image in images)
    h = max(image.height for image in images)
    rows = (len(images) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * w + (cols - 1) * gap, rows * h + (rows - 1) * gap), (245, 245, 245))
    for idx, image in enumerate(images):
        x = (idx % cols) * (w + gap)
        y = (idx // cols) * (h + gap)
        sheet.paste(image, (x, y))
    return sheet


def select_errors(rows, limit):
    false_pos = []
    false_neg = []
    for row in rows:
        label = int(float(row["done_label"]) >= 0.5)
        pred = int(row["done_pred"])
        if label == 0 and pred == 1:
            false_pos.append(row)
        elif label == 1 and pred == 0:
            false_neg.append(row)
    false_pos.sort(key=lambda r: float(r["done_prob"]), reverse=True)
    false_neg.sort(key=lambda r: float(r["done_prob"]))
    return false_pos[:limit], false_neg[:limit], len(false_pos), len(false_neg)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(args.predictions_csv)
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
    if len(rows) > len(samples):
        raise ValueError(f"CSV has {len(rows)} rows but only rebuilt {len(samples)} Q2 samples.")

    fp_rows, fn_rows, fp_total, fn_total = select_errors(rows, args.num_per_type)
    panels = []
    for error_type, selected in (("FP", fp_rows), ("FN", fn_rows)):
        for rank, row in enumerate(selected):
            sample = samples[int(row["sample_index"])]
            panel = make_panel(sample, row, error_type, args.thumb_width)
            out_path = output_dir / f"{error_type.lower()}_{rank:02d}_sample_{int(row['sample_index']):06d}.png"
            panel.save(out_path)
            panels.append(panel)
            print(out_path)

    sheet = make_contact_sheet(panels)
    if sheet is not None:
        sheet_path = output_dir / "done_errors_contact_sheet.png"
        sheet.save(sheet_path)
        print(sheet_path)
    print(f"total_fp={fp_total} total_fn={fn_total} saved_fp={len(fp_rows)} saved_fn={len(fn_rows)}")


if __name__ == "__main__":
    main()
