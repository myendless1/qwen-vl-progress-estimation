import json
import os
import pathlib
import sys
from typing import Any, Dict, Iterable, List, Optional


COMMON_MAPPINGS = {
    ("model", "model_name_or_path"): "base_model",
    ("data", "robotwin_data_root"): "data_root",
    ("data", "robotwin_anno_root"): "anno_root",
    ("data", "robotwin_views"): "views",
    ("data", "robotwin_memory_frames"): "memory_frames",
    ("data", "robotwin_memory_frame_stride"): "memory_frame_stride",
    ("data", "robotwin_test_ratio"): "test_ratio",
    ("data", "robotwin_q2_frame_stride"): "q2_frame_stride",
    ("data", "robotwin_boundary_extra_frames"): "boundary_extra_frames",
    ("training", "model_max_length"): "model_max_length",
    ("training", "voting_done"): "voting_done",
    ("training", "done_vote_count"): "done_vote_count",
}

TASK_MAPPINGS = {
    "q2": {
        ("data", "robotwin_q2_progress_bucket_size"): "q2_progress_bucket_size",
    },
    "q2_vis": {
        ("data", "robotwin_q2_progress_bucket_size"): "q2_progress_bucket_size",
    },
    "q2_bench": {
        ("data", "robotwin_q2_progress_bucket_size"): "q2_progress_bucket_size",
    },
}


def _pop_config_path(argv: List[str]) -> Optional[str]:
    for idx, arg in enumerate(list(argv)):
        if arg == "--config":
            if idx + 1 >= len(argv):
                raise ValueError("--config requires a path")
            config_path = argv[idx + 1]
            del argv[idx : idx + 2]
            return config_path
        if arg.startswith("--config="):
            config_path = arg.split("=", 1)[1]
            del argv[idx]
            return config_path
    return None


def _read_config(config_path: str) -> Dict[str, Any]:
    path = pathlib.Path(config_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Eval config does not exist: {path}")
    with open(path) as f:
        config = json.load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Eval config must be a JSON object: {path}")
    return config


def _expand_value(value: Any, context: Dict[str, Any]) -> Any:
    if not isinstance(value, str):
        return value
    expanded = os.path.expandvars(value)
    try:
        expanded = expanded.format(**context)
    except KeyError:
        pass
    return expanded


def _context(config: Dict[str, Any], task: str) -> Dict[str, Any]:
    context = {}
    for section in ("model", "data", "training"):
        values = config.get(section, {})
        if not isinstance(values, dict):
            raise ValueError(f"Config section '{section}' must be an object")
        context.update(values)
    return context


def _iter_mapped_values(config: Dict[str, Any], task: str) -> Iterable[tuple[str, Any]]:
    mappings = dict(COMMON_MAPPINGS)
    mappings.update(TASK_MAPPINGS.get(task, {}))
    for (section, key), arg_name in mappings.items():
        values = config.get(section, {})
        if isinstance(values, dict) and key in values:
            yield arg_name, values[key]


def _iter_eval_values(config: Dict[str, Any], task: str) -> Iterable[tuple[str, Any]]:
    eval_values = config.get("eval", {})
    if not isinstance(eval_values, dict):
        return
    for key, value in eval_values.items():
        if key in {"q1", "q2"} or isinstance(value, dict):
            continue
        yield key, value

    if task in {"q2_vis", "q2_bench"}:
        q2_values = eval_values.get("q2", {})
        if isinstance(q2_values, dict):
            yield from q2_values.items()

    task_values = eval_values.get(task, {})
    if isinstance(task_values, dict):
        yield from task_values.items()


def _append_arg(args: List[str], name: str, value: Any, context: Dict[str, Any]) -> None:
    if value is None and name != "output_xlsx":
        return
    value = _expand_value(value, context)
    flag = f"--{name.replace('_', '-')}"
    if isinstance(value, bool):
        if value and name != "output_xlsx":
            args.append(flag)
        elif name == "output_xlsx":
            args.extend([flag, str(value)])
        return
    if value is None:
        args.extend([flag, ""])
        return
    args.extend([flag, str(value)])


def config_to_args(config: Dict[str, Any], task: str) -> List[str]:
    context = _context(config, task)
    args: List[str] = []
    for name, value in _iter_mapped_values(config, task):
        _append_arg(args, name, value, context)
    for name, value in _iter_eval_values(config, task):
        _append_arg(args, name, value, context)
    return args


def apply_config_argv(task: str) -> None:
    argv = sys.argv[1:]
    config_path = _pop_config_path(argv)
    if not config_path:
        return
    config_args = config_to_args(_read_config(config_path), task)
    sys.argv = [sys.argv[0], *config_args, *argv]
