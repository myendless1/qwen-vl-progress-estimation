#!/usr/bin/env python
import csv
import json
import math
import os
import random
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from xml.sax.saxutils import escape
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from transformers import AutoProcessor, AutoTokenizer, Qwen3VLForConditionalGeneration

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qwenvl.data.data_processor import get_rope_index_3, update_processor_pixels
from qwenvl.data.robotwin_processor import (
    DONE_VOTING_QUERY_TOKENS,
    Q1_SYSTEM_PROMPT,
    QUERY_TOKENS,
    RobotWinDataCollator,
    _completed_subtasks,
    _future_subtasks,
    _load_observation_images,
    _messages_for_sample,
    _system_prompt,
    _user_content,
    build_robotwin_samples,
    parse_robotwin_views,
    preprocess_robotwin_sample,
    robotwin_special_tokens,
)
from qwenvl.train.robotwin_model import RobotWinQwenWrapper


def dtype_from_name(name: str):
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    return torch.float32


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def write_json(path: Path, value: Any, indent: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(value, f, ensure_ascii=False, indent=indent)


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)


def _xlsx_col_name(index: int) -> str:
    name = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        name = chr(65 + remainder) + name
    return name


def _xlsx_cell_value(value: Any) -> str:
    if isinstance(value, (dict, list)):
        text = json_dumps(value)
    elif value is None:
        text = ""
    else:
        text = str(value)
    return "".join(ch for ch in text if ch in "\t\n\r" or ord(ch) >= 32)


def write_xlsx(path: Path, rows: Sequence[Dict[str, Any]], sheet_name: str = "results") -> None:
    """Write a minimal XLSX workbook without requiring pandas/openpyxl."""
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = list(rows[0].keys()) if rows else []

    def cell_xml(row_index: int, col_index: int, value: Any) -> str:
        ref = f"{_xlsx_col_name(col_index)}{row_index}"
        text = escape(_xlsx_cell_value(value))
        return f'<c r="{ref}" t="inlineStr"><is><t xml:space="preserve">{text}</t></is></c>'

    sheet_rows = []
    if headers:
        sheet_rows.append(
            '<row r="1">' + "".join(cell_xml(1, col + 1, header) for col, header in enumerate(headers)) + "</row>"
        )
        for row_index, row in enumerate(rows, start=2):
            sheet_rows.append(
                f'<row r="{row_index}">'
                + "".join(cell_xml(row_index, col + 1, row.get(header, "")) for col, header in enumerate(headers))
                + "</row>"
            )

    safe_sheet_name = escape(sheet_name[:31] or "results")
    worksheet = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f'<sheetData>{"".join(sheet_rows)}</sheetData>'
        "</worksheet>"
    )
    workbook = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f'<sheets><sheet name="{safe_sheet_name}" sheetId="1" r:id="rId1"/></sheets>'
        "</workbook>"
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/worksheets/sheet1.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        "</Types>"
    )
    root_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/>'
        "</Relationships>"
    )
    workbook_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
        'Target="worksheets/sheet1.xml"/>'
        "</Relationships>"
    )

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", root_rels)
        zf.writestr("xl/workbook.xml", workbook)
        zf.writestr("xl/_rels/workbook.xml.rels", workbook_rels)
        zf.writestr("xl/worksheets/sheet1.xml", worksheet)


def read_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def batched(items: Sequence[Any], batch_size: int) -> Iterable[Tuple[int, Sequence[Any]]]:
    for start in range(0, len(items), batch_size):
        yield start, items[start : start + batch_size]


def move_to_device(batch: Dict[str, Any], device: str) -> Dict[str, Any]:
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


def load_robotwin_tokenizer(args):
    tokenizer_kwargs = {
        "model_max_length": args.model_max_length,
        "padding_side": "right",
        "use_fast": False,
    }
    done_vote_count = int(getattr(args, "done_vote_count", 5))
    special_tokens = robotwin_special_tokens(
        voting_done=bool(getattr(args, "voting_done", False)),
        done_vote_count=done_vote_count,
    )
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, **tokenizer_kwargs)
        tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        return tokenizer
    except Exception as exc:
        print(
            f"warning: failed to load tokenizer from checkpoint ({exc}); "
            "falling back to base tokenizer plus RobotWin query tokens.",
            file=sys.stderr,
        )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, **tokenizer_kwargs)
    config_path = Path(args.checkpoint) / "tokenizer_config.json"
    if config_path.exists():
        with open(config_path, "r") as f:
            tokenizer_config = json.load(f)
        special_tokens = tokenizer_config.get("extra_special_tokens", special_tokens)
        if getattr(args, "voting_done", False):
            for token in robotwin_special_tokens(voting_done=True, done_vote_count=done_vote_count):
                if token not in special_tokens:
                    special_tokens.append(token)
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    return tokenizer


def make_query_token_ids(tokenizer, voting_done: bool = False, done_vote_count: int = 5) -> Dict[str, int]:
    ids = {name: tokenizer.convert_tokens_to_ids(token) for name, token in QUERY_TOKENS.items()}
    if voting_done:
        for idx, token in enumerate(DONE_VOTING_QUERY_TOKENS[:done_vote_count]):
            ids[f"current_vote_{idx}"] = tokenizer.convert_tokens_to_ids(token)
    missing = [name for name, token_id in ids.items() if token_id is None or token_id < 0]
    if missing:
        raise ValueError(f"Missing RobotWin query tokens in tokenizer: {missing}")
    return ids


def load_robotwin_model(args, query_token_ids: Dict[str, int]):
    dtype = dtype_from_name(args.dtype)
    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.base_model,
        attn_implementation=args.attn_implementation,
        dtype=dtype if args.device != "cpu" else torch.float32,
    )
    base_model.resize_token_embeddings(max(query_token_ids.values()) + 1)
    model = RobotWinQwenWrapper(
        base_model,
        voting_done=bool(getattr(args, "voting_done", False)),
        done_vote_count=int(getattr(args, "done_vote_count", 5)),
    )

    state_path = Path(args.checkpoint) / "pytorch_model.bin"
    if not state_path.exists():
        state_path = Path(args.checkpoint)
    state_dict = torch.load(state_path, map_location="cpu")
    current_state = model.state_dict()
    compatible_state = {}
    skipped = []
    for key, value in state_dict.items():
        if key not in current_state or value.shape == current_state[key].shape:
            compatible_state[key] = value
        else:
            skipped.append((key, tuple(value.shape), tuple(current_state[key].shape)))
    if skipped:
        print(
            json.dumps(
                {
                    "shape_mismatch_warning": {
                        "skipped": skipped[:20],
                        "num_skipped": len(skipped),
                    }
                },
                ensure_ascii=False,
            )
        )
    state_dict = compatible_state
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


def make_data_args(args):
    return SimpleNamespace(
        robotwin_data_root=args.data_root,
        robotwin_test_ratio=args.test_ratio,
        robotwin_split_seed=args.split_seed,
        robotwin_q2_frame_stride=args.q2_frame_stride,
        robotwin_boundary_extra_frames=args.boundary_extra_frames,
        model_type="qwen3vl",
        max_pixels=getattr(args, "max_pixels", 28 * 28 * 576),
        min_pixels=getattr(args, "min_pixels", 28 * 28 * 16),
        video_max_pixels=getattr(args, "video_max_pixels", 28 * 28 * 576),
        video_min_pixels=getattr(args, "video_min_pixels", 28 * 28 * 144),
        video_max_frames=getattr(args, "video_max_frames", 8),
        video_min_frames=getattr(args, "video_min_frames", 4),
        video_fps=getattr(args, "video_fps", 2.0),
    )


def load_processor_tokenizer(args, prefer_checkpoint_processor: bool = False):
    processor_source = args.checkpoint if prefer_checkpoint_processor else args.base_model
    try:
        processor = AutoProcessor.from_pretrained(processor_source)
    except OSError as exc:
        if not prefer_checkpoint_processor or processor_source == args.base_model:
            raise
        print(
            f"warning: failed to load processor from checkpoint ({exc}); "
            "falling back to base model processor.",
            file=sys.stderr,
        )
        processor = AutoProcessor.from_pretrained(args.base_model)
    tokenizer = load_robotwin_tokenizer(args)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if hasattr(processor, "tokenizer"):
        processor.tokenizer = tokenizer
    processor = update_processor_pixels(processor, make_data_args(args))
    return processor, tokenizer


def load_eval_context(args, prefer_checkpoint_processor: bool = False) -> Dict[str, Any]:
    processor, tokenizer = load_processor_tokenizer(args, prefer_checkpoint_processor=prefer_checkpoint_processor)
    query_token_ids = make_query_token_ids(
        tokenizer,
        voting_done=bool(getattr(args, "voting_done", False)),
        done_vote_count=int(getattr(args, "done_vote_count", 5)),
    )
    collator = RobotWinDataCollator(
        tokenizer,
        query_token_ids=query_token_ids,
        voting_done=bool(getattr(args, "voting_done", False)),
    )
    model = load_robotwin_model(args, query_token_ids)
    return {
        "processor": processor,
        "tokenizer": tokenizer,
        "query_token_ids": query_token_ids,
        "merge_size": getattr(processor.image_processor, "merge_size", 2),
        "collator": collator,
        "model": model,
    }


def build_samples(args, kind: Optional[str] = None):
    views = parse_robotwin_views(getattr(args, "views", "main,left_wrist,right_wrist"))
    samples = build_robotwin_samples(
        args.data_root,
        q2_frame_stride=args.q2_frame_stride,
        boundary_extra_frames=args.boundary_extra_frames,
        max_episodes=args.max_episodes,
        split=args.split,
        test_ratio=args.test_ratio,
        split_seed=args.split_seed,
        anno_root=getattr(args, "anno_root", None),
        views=views,
        q2_progress_bucket_size=float(getattr(args, "q2_progress_bucket_size", 0.01)),
    )
    return [sample for sample in samples if kind is None or sample.kind == kind]


def prepare_q2_batch(samples, processor, merge_size: int, collator):
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


def _predict_robotwin_outputs(args, model, batch: Dict[str, Any], query_token_ids: Dict[str, int]):
    if not getattr(args, "voting_done", False):
        outputs = model(**batch)
        done_probs = torch.sigmoid(outputs.robotwin_logits["current_done"]).detach().float().cpu()
        progress_preds = outputs.robotwin_progress.detach().float().cpu()
        return done_probs, progress_preds

    vote_count = int(getattr(args, "done_vote_count", 5))
    vote_logits = []
    progress_preds = []
    current_token_id = int(query_token_ids["current"])
    for vote_idx in range(vote_count):
        vote_token_id = int(query_token_ids[f"current_vote_{vote_idx}"])
        vote_batch = dict(batch)
        input_ids = batch["input_ids"].clone()
        positions = batch["robotwin_current_query_pos"].to(input_ids.device)
        valid = positions.ge(0)
        if valid.any():
            rows = torch.arange(input_ids.shape[0], device=input_ids.device)[valid]
            cols = positions[valid]
            input_ids[rows, cols] = vote_token_id
        else:
            input_ids = input_ids.masked_fill(input_ids.eq(current_token_id), vote_token_id)
        vote_batch["input_ids"] = input_ids
        vote_batch["robotwin_done_vote_index"] = torch.full(
            (input_ids.shape[0],),
            vote_idx,
            dtype=torch.long,
            device=input_ids.device,
        )
        outputs = model(**vote_batch)
        vote_logits.append(outputs.robotwin_logits["current_done"].detach().float().cpu())
        progress_preds.append(outputs.robotwin_progress.detach().float().cpu())
    done_probs = torch.sigmoid(torch.stack(vote_logits, dim=1))
    progress_preds = torch.stack(progress_preds, dim=1).mean(dim=1)
    return done_probs, progress_preds


def run_q2_predictions(args, samples, context: Optional[Dict[str, Any]] = None, progress_prefix: str = "Q2"):
    if context is None:
        context = load_eval_context(args)
    rows = []
    model = context["model"]
    processor = context["processor"]
    collator = context["collator"]
    merge_size = context["merge_size"]
    query_token_ids = context["query_token_ids"]
    voting_done = bool(getattr(args, "voting_done", False))
    done_vote_threshold = int(getattr(args, "done_vote_threshold", 3))

    with torch.inference_mode():
        for batch_start, batch_samples in batched(samples, args.batch_size):
            batch = move_to_device(prepare_q2_batch(batch_samples, processor, merge_size, collator), args.device)
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            done_probs, progress_preds = _predict_robotwin_outputs(args, model, batch, query_token_ids)
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()

            done_labels = batch["robotwin_current_done"].detach().float().cpu()
            progress_labels = batch["robotwin_progress"].detach().float().cpu()
            if voting_done:
                done_votes = done_probs.ge(args.threshold).sum(dim=1)
                done_scalar_probs = done_probs.mean(dim=1)
                done_preds = done_votes.ge(done_vote_threshold)
            else:
                done_votes = done_probs.ge(args.threshold).long()
                done_scalar_probs = done_probs
                done_preds = done_scalar_probs.ge(args.threshold)
            done_true = done_labels.ge(args.threshold)
            errors = progress_preds - progress_labels

            for offset, sample in enumerate(batch_samples):
                episode_index = None
                anno_path = ""
                main_path = sample.image_hdf5_paths.get("main")
                if main_path is not None:
                    stem = Path(main_path).stem
                    if stem.startswith("episode_"):
                        episode_index = int(stem.split("_", 1)[1])
                        anno_path = str(sample.repo_dir / "anno" / f"episode_{episode_index:06d}.json")
                rows.append(
                    {
                        "sample_index": batch_start + offset,
                        "repo": sample.repo_dir.name,
                        "repo_dir": str(sample.repo_dir),
                        "anno_path": anno_path,
                        "episode_index": episode_index,
                        "frame_index": sample.frame_index,
                        "probe_kind": getattr(sample, "probe_kind", ""),
                        "current_subtask_index": sample.current_subtask_index,
                        "current_subtask_goal": sample.subtasks[sample.current_subtask_index]["subtask_goal"],
                        "q2_group": sample.q2_group,
                        "done_label": float(done_labels[offset].item()),
                        "done_prob": float(done_scalar_probs[offset].item()),
                        "done_pred": int(done_preds[offset].item()),
                        "done_pred_label": "done" if done_preds[offset].item() else "undone",
                        "done_correct": int(done_preds[offset].eq(done_true[offset]).item()),
                        "done_vote_count": int(done_votes[offset].item()),
                        "done_vote_threshold": done_vote_threshold if voting_done else "",
                        "done_vote_probs": json_dumps([float(x) for x in done_probs[offset].tolist()])
                        if voting_done
                        else "",
                        "progress_label": float(progress_labels[offset].item()),
                        "progress_pred": float(progress_preds[offset].item()),
                        "progress_abs_err": abs(float(errors[offset].item())),
                        "progress_sq_err": float(errors[offset].item()) ** 2,
                    }
                )
            if len(rows) == len(samples) or len(rows) % max(args.batch_size * 10, 20) == 0:
                print(f"{progress_prefix} evaluated {len(rows)}/{len(samples)}", flush=True)
    return rows


def safe_float(row: Dict[str, Any], key: str) -> float:
    return float(row[key])


def safe_int(row: Dict[str, Any], key: str) -> int:
    return int(float(row[key]))


def summarize_q2(rows: Sequence[Dict[str, Any]], threshold: float = 0.5) -> Dict[str, Any]:
    total = len(rows)
    confusion = Counter()
    by_group = defaultdict(Counter)
    sqerr = 0.0
    abserr = 0.0
    max_abs = 0.0

    for row in rows:
        label = int(safe_float(row, "done_label") >= threshold)
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
        by_group[row.get("q2_group") or "unknown"][key] += 1
        err = safe_float(row, "progress_pred") - safe_float(row, "progress_label")
        sqerr += err * err
        abserr += abs(err)
        max_abs = max(max_abs, abs(err))

    mse = sqerr / max(1, total)
    return {
        "num_q2_evaluated": total,
        "done_confusion_matrix": {
            "TP": confusion["TP"],
            "TN": confusion["TN"],
            "FP": confusion["FP"],
            "FN": confusion["FN"],
        },
        "done_confusion_matrix_2x2": {
            "actual_0": {
                "pred_0": confusion["TN"],
                "pred_1": confusion["FP"],
            },
            "actual_1": {
                "pred_0": confusion["FN"],
                "pred_1": confusion["TP"],
            },
        },
        "done_accuracy": (confusion["TP"] + confusion["TN"]) / max(1, total),
        "progress_mse": mse,
        "progress_rmse": math.sqrt(mse),
        "progress_mae": abserr / max(1, total),
        "progress_max_abs_err": max_abs,
        "by_group": {
            group: {
                "total": sum(counter.values()),
                "TP": counter["TP"],
                "TN": counter["TN"],
                "FP": counter["FP"],
                "FN": counter["FN"],
                "accuracy": (counter["TP"] + counter["TN"]) / max(1, sum(counter.values())),
            }
            for group, counter in sorted(by_group.items())
        },
    }


def normalize_generated_text(text: str) -> str:
    text = text.strip()
    try:
        return json_dumps(json.loads(text))
    except Exception:
        return " ".join(text.split())


def parse_json_or_none(text: str):
    try:
        return json.loads(text)
    except Exception:
        return None


def task_slug_from_repo_name(repo_name: str) -> str:
    marker = "-aloha-"
    if marker in repo_name:
        return repo_name.split(marker, 1)[0]
    return repo_name.split("-", 1)[0]


def prepare_q1_prompt(sample, processor):
    images = _load_observation_images(sample.image_hdf5_paths, sample.frame_index, sample.views)
    completed = _completed_subtasks(sample.subtasks, sample.current_subtask_index)
    user_content = _user_content(sample.task_goal, completed, images, sample.views)
    messages = _messages_for_sample(_system_prompt(Q1_SYSTEM_PROMPT, sample.views), user_content)
    return processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )


def q1_ground_truth(sample):
    return _future_subtasks(sample.subtasks, sample.current_subtask_index)


def select_q1_samples(samples, all_states: bool = False, current_subtask_index: int = 0):
    if all_states:
        return samples
    return [sample for sample in samples if sample.current_subtask_index == current_subtask_index]


def select_one_q1_sample_per_task(samples, max_tasks: Optional[int], seed: int, shuffle_tasks: bool = False):
    grouped = defaultdict(list)
    for sample in samples:
        grouped[task_slug_from_repo_name(sample.repo_dir.name)].append(sample)

    rng = random.Random(seed)
    selected = []
    for task_slug in sorted(grouped):
        choices = grouped[task_slug]
        selected.append(rng.choice(choices))

    if shuffle_tasks:
        rng.shuffle(selected)
    if max_tasks is not None:
        selected = selected[:max_tasks]
    return selected


def sample_items(items: Sequence[Any], max_items: Optional[int], seed: int, shuffle: bool = False):
    items = list(items)
    if shuffle:
        random.Random(seed).shuffle(items)
    if max_items is not None:
        items = items[:max_items]
    return items
