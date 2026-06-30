import io
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import h5py
import torch
import transformers
from PIL import Image
from torch.utils.data import Dataset

from .data_processor import IGNORE_INDEX, get_rope_index_2, get_rope_index_25, get_rope_index_3, pad_and_cat, update_processor_pixels
from .robotwin_progress import (
    build_subtask_progress_lookup,
    episode_parquet_path,
    load_episode_states,
    progress_for_subtask,
)


QUERY_TOKENS = {
    "current": "<current_query>",
    "plan": "<plan_query>",
    "incident": "<incident_query>",
    "value": "<value_query>",
}

ROBOTWIN_IGNORE_FLOAT = -100.0


def robotwin_special_tokens() -> List[str]:
    return list(QUERY_TOKENS.values())


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


VIEW_KEYS = {
    "main": "observation.images.cam_main",
    "left_wrist": "observation.images.cam_left_wrist",
    "right_wrist": "observation.images.cam_right_wrist",
}

VIEW_LABELS = {
    "main": "<main>",
    "left_wrist": "<left wrist>",
    "right_wrist": "<right wrist>",
}

DEFAULT_ROBOTWIN_VIEWS = ("main", "left_wrist", "right_wrist")


def parse_robotwin_views(raw: str) -> tuple[str, ...]:
    views = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not views:
        raise ValueError("robotwin_views must specify at least one view.")
    unknown = [view for view in views if view not in VIEW_KEYS]
    if unknown:
        raise ValueError(f"Unsupported robotwin views: {unknown}. Expected any of {list(VIEW_KEYS)}.")
    return views


def _observation_prompt(views: Sequence[str]) -> str:
    if len(views) == 1:
        return "image observation"
    return "multi-view image observations"


Q1_SYSTEM_PROMPT = (
    "You are a robot task planner. Given the global task, the completed subtasks, "
    "and {observation_prompt}, plan the remaining subtasks from the current state. "
    "Output a JSON array with only subtask_index and subtask_goal."
)

Q2_SYSTEM_PROMPT = (
    "You are a robot execution status estimator. Given the global task, the completed subtasks, "
    "the current subtask, and {observation_prompt}, predict the requested status values "
    "using the query tokens. Do not generate a natural-language answer."
)


def _system_prompt(template: str, views: Sequence[str]) -> str:
    return template.format(observation_prompt=_observation_prompt(views))


def _depth_png_to_rgb(encoded: bytes) -> Image.Image:
    import numpy as np

    depth = np.array(Image.open(io.BytesIO(encoded)), dtype=np.float32)
    lo, hi = np.percentile(depth, (2, 98))
    normalized = np.clip((depth - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    return Image.fromarray((normalized * 255).astype(np.uint8)).convert("RGB")


def _hdf5_num_frames(path: Path) -> int:
    with h5py.File(path, "r") as f:
        if "frames" in f:
            return len(f["frames"])
        if "depth_mm_png" in f:
            return len(f["depth_mm_png"])
    raise KeyError(f"{path} does not contain supported image datasets: frames or depth_mm_png")


def _decode_hdf5_frame(dataset, frame_index: int, *, dataset_key: str) -> Image.Image:
    if frame_index < 0 or frame_index >= len(dataset):
        raise IndexError(f"frame_index {frame_index} out of range for dataset with length {len(dataset)}")
    encoded = bytes(dataset[frame_index])
    if dataset_key == "depth_mm_png":
        return _depth_png_to_rgb(encoded)
    return Image.open(io.BytesIO(encoded)).convert("RGB")


def _read_hdf5_frame(path: Path, frame_index: int, resize: Optional[tuple[int, int]] = None) -> Image.Image:
    with h5py.File(path, "r") as f:
        if "frames" in f:
            image = _decode_hdf5_frame(f["frames"], frame_index, dataset_key="frames")
        elif "depth_mm_png" in f:
            image = _decode_hdf5_frame(f["depth_mm_png"], frame_index, dataset_key="depth_mm_png")
        else:
            raise KeyError(f"{path} does not contain supported image datasets: frames or depth_mm_png")
    if resize is not None:
        image = image.resize(resize, Image.Resampling.BICUBIC)
    return image


def _episode_chunk(episode_index: int, chunks_size: int) -> int:
    return episode_index // chunks_size


def _view_hdf5_path(image_repo_dir: Path, episode_index: int, chunks_size: int, view: str) -> Path:
    episode_name = f"episode_{episode_index:06d}.hdf5"
    rel = VIEW_KEYS[view]
    flat_path = image_repo_dir / rel / episode_name
    if flat_path.exists():
        return flat_path
    chunk = _episode_chunk(episode_index, chunks_size)
    return (
        image_repo_dir
        / "videos_240x320_240x320"
        / f"chunk-{chunk:03d}"
        / rel
        / episode_name
    )


def _task_resource_repo_dir(data_root: Path, anno_root: Optional[str], task_name: str) -> Path:
    if anno_root:
        candidate = Path(anno_root) / task_name
        if candidate.exists():
            return candidate
    return data_root / task_name


def _task_anno_dir(data_root: Path, anno_root: Optional[str], task_name: str) -> Path:
    return _task_resource_repo_dir(data_root, anno_root, task_name) / "anno"


def _load_chunks_size(repo_dir: Path) -> int:
    info_path = repo_dir / "meta" / "info.json"
    if not info_path.exists():
        return 1000
    with open(info_path, "r") as f:
        info = json.load(f)
    return int(info.get("chunks_size", 1000))


def load_robotwin_excluded_episodes(path: Optional[str]) -> set[tuple[str, int]]:
    if not path:
        return set()
    exclude_path = Path(path)
    if not exclude_path.exists():
        raise FileNotFoundError(f"RobotWin exclude list does not exist: {exclude_path}")

    excluded: set[tuple[str, int]] = set()
    if exclude_path.suffix == ".jsonl":
        with open(exclude_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                excluded.add((str(item["repo"]), int(item["episode_index"])))
        return excluded

    with open(exclude_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    episodes = payload.get("episodes", payload if isinstance(payload, list) else [])
    for item in episodes:
        excluded.add((str(item["repo"]), int(item["episode_index"])))
    return excluded


def _future_subtasks(subtasks: Sequence[Dict[str, Any]], current_index: int) -> List[Dict[str, Any]]:
    return [
        {
            "subtask_index": int(st["subtask_index"]),
            "subtask_goal": st["subtask_goal"],
        }
        for st in subtasks[current_index:]
    ]


def _completed_subtasks(subtasks: Sequence[Dict[str, Any]], current_index: int) -> List[Dict[str, Any]]:
    return [
        {
            "subtask_index": int(st["subtask_index"]),
            "subtask_goal": st["subtask_goal"],
        }
        for st in subtasks[:current_index]
    ]


def _load_observation_images(
    image_hdf5_paths: Dict[str, Path],
    frame_index: int,
    views: Sequence[str],
) -> Dict[str, Image.Image]:
    images: Dict[str, Image.Image] = {}
    main = None
    if "main" in views:
        main = _read_hdf5_frame(image_hdf5_paths["main"], frame_index)
        images["main"] = main
    wrist_size = None
    if main is not None:
        wrist_size = (max(1, main.width // 2), max(1, main.height // 2))
    for view in views:
        if view == "main":
            continue
        images[view] = _read_hdf5_frame(image_hdf5_paths[view], frame_index, resize=wrist_size)
    return images


def _user_content(
    task_goal: str,
    completed: Sequence[Dict[str, Any]],
    images: Dict[str, Image.Image],
    views: Sequence[str],
    current_goal: Optional[str] = None,
    include_query_tokens: bool = False,
) -> List[Dict[str, Any]]:
    content = [
        {
            "type": "text",
            "text": f"Global task: {task_goal}\nCompleted subtasks: {_json_dumps(list(completed))}\n",
        }
    ]
    if current_goal is not None:
        content.append({"type": "text", "text": f"Current subtask: {current_goal}\n"})
    observation_label = "Image observation" if len(views) == 1 else "Image observations"
    content.append({"type": "text", "text": f"{observation_label}:\n"})
    for view in views:
        content.append({"type": "text", "text": f"{VIEW_LABELS[view]} "})
        content.append({"type": "image", "image": images[view]})
        content.append({"type": "text", "text": "\n"})
    if include_query_tokens:
        content.append(
            {
                "type": "text",
                "text": f"{QUERY_TOKENS['current']}{QUERY_TOKENS['plan']}{QUERY_TOKENS['incident']}{QUERY_TOKENS['value']}",
            }
        )
    return content


def _messages_for_sample(
    system_prompt: str,
    user_content: List[Dict[str, Any]],
    assistant: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": user_content,
        }
    ]
    if assistant is not None:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": _json_dumps(assistant)}]})
    return messages


@dataclass
class RobotWinSample:
    kind: str
    repo_dir: Path
    image_hdf5_paths: Dict[str, Path]
    frame_index: int
    frame_start: int
    frame_end: int
    task_goal: str
    subtasks: List[Dict[str, Any]]
    current_subtask_index: int
    views: tuple[str, ...] = DEFAULT_ROBOTWIN_VIEWS
    image_repo_dir: Optional[Path] = None
    current_done: float = ROBOTWIN_IGNORE_FLOAT
    need_replan: float = ROBOTWIN_IGNORE_FLOAT
    incident: float = ROBOTWIN_IGNORE_FLOAT
    progress: float = ROBOTWIN_IGNORE_FLOAT
    q2_group: str = ""


def _robotwin_repo_dirs(
    data_root: str,
    split: Optional[str] = None,
    test_ratio: float = 0.05,
    split_seed: int = 0,
    anno_root: Optional[str] = None,
) -> List[Path]:
    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(f"RobotWin root does not exist: {root}")

    repo_dirs = sorted(
        image_repo_dir
        for image_repo_dir in root.iterdir()
        if image_repo_dir.is_dir()
        and _task_anno_dir(root, anno_root, image_repo_dir.name).exists()
    )
    if split is None or split == "all":
        return repo_dirs

    split = split.lower()
    if split not in {"train", "test", "eval"}:
        raise ValueError(f"Unsupported RobotWin split: {split}")

    if len(repo_dirs) < 2 or test_ratio <= 0:
        return repo_dirs if split == "train" else []

    if test_ratio >= 1:
        raise ValueError("robotwin_test_ratio must be less than 1.0 so the train split is non-empty.")

    shuffled = repo_dirs[:]
    random.Random(split_seed).shuffle(shuffled)
    test_count = max(1, int(round(len(shuffled) * test_ratio)))
    test_count = min(test_count, len(shuffled) - 1)
    test_names = {repo.name for repo in shuffled[:test_count]}

    if split in {"test", "eval"}:
        return [repo for repo in repo_dirs if repo.name in test_names]
    return [repo for repo in repo_dirs if repo.name not in test_names]


def build_robotwin_split_manifest(
    data_root: str,
    test_ratio: float = 0.05,
    split_seed: int = 0,
    anno_root: Optional[str] = None,
) -> Dict[str, Any]:
    root = Path(data_root)
    all_dirs = _robotwin_repo_dirs(
        data_root,
        split="all",
        test_ratio=test_ratio,
        split_seed=split_seed,
        anno_root=anno_root,
    )
    train_dirs = _robotwin_repo_dirs(
        data_root,
        split="train",
        test_ratio=test_ratio,
        split_seed=split_seed,
        anno_root=anno_root,
    )
    test_dirs = _robotwin_repo_dirs(
        data_root,
        split="test",
        test_ratio=test_ratio,
        split_seed=split_seed,
        anno_root=anno_root,
    )
    return {
        "data_root": str(root.resolve()),
        "anno_root": str(Path(anno_root).resolve()) if anno_root else str(root.resolve()),
        "test_ratio": test_ratio,
        "split_seed": split_seed,
        "num_tasks": len(all_dirs),
        "num_train_tasks": len(train_dirs),
        "num_test_tasks": len(test_dirs),
        "all_tasks": [repo.name for repo in all_dirs],
        "train_tasks": [repo.name for repo in train_dirs],
        "test_tasks": [repo.name for repo in test_dirs],
        "train_paths": [str(repo.resolve()) for repo in train_dirs],
        "test_paths": [str(repo.resolve()) for repo in test_dirs],
    }


def save_robotwin_split_manifest(
    data_root: str,
    output_dir: str,
    test_ratio: float = 0.05,
    split_seed: int = 0,
    anno_root: Optional[str] = None,
) -> Path:
    manifest = build_robotwin_split_manifest(
        data_root,
        test_ratio=test_ratio,
        split_seed=split_seed,
        anno_root=anno_root,
    )
    output_path = Path(output_dir) / "robotwin_split.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return output_path


def build_robotwin_samples(
    data_root: str,
    q2_frame_stride: int,
    boundary_extra_frames: int,
    max_episodes: Optional[int] = None,
    split: Optional[str] = None,
    test_ratio: float = 0.05,
    split_seed: int = 0,
    anno_root: Optional[str] = None,
    views: Sequence[str] = DEFAULT_ROBOTWIN_VIEWS,
    exclude_episodes: Optional[set[tuple[str, int]]] = None,
) -> List[RobotWinSample]:
    active_views = tuple(views)
    def make_q2_sample(
        repo_dir: Path,
        image_repo_dir: Path,
        image_hdf5_paths: Dict[str, Path],
        frame: int,
        task_goal: str,
        subtasks: List[Dict[str, Any]],
        current_subtask_index: int,
        current_done: float,
        progress: float,
        q2_group: str,
    ) -> RobotWinSample:
        return RobotWinSample(
            kind="q2",
            repo_dir=repo_dir,
            image_hdf5_paths=image_hdf5_paths,
            frame_index=frame,
            frame_start=frame,
            frame_end=frame,
            task_goal=task_goal,
            subtasks=subtasks,
            current_subtask_index=current_subtask_index,
            views=active_views,
            image_repo_dir=image_repo_dir,
            current_done=current_done,
            need_replan=ROBOTWIN_IGNORE_FLOAT,
            incident=ROBOTWIN_IGNORE_FLOAT,
            progress=progress,
            q2_group=q2_group,
        )

    def clipped_frames(frames: Sequence[int], num_frames: int) -> List[int]:
        return sorted({frame for frame in frames if 0 <= frame < num_frames})

    samples: List[RobotWinSample] = []
    episode_count = 0
    excluded_episode_count = 0
    data_root_path = Path(data_root)
    excluded = exclude_episodes or set()
    for image_repo_dir in _robotwin_repo_dirs(
        data_root,
        split=split,
        test_ratio=test_ratio,
        split_seed=split_seed,
        anno_root=anno_root,
    ):
        resource_repo_dir = _task_resource_repo_dir(data_root_path, anno_root, image_repo_dir.name)
        anno_dir = resource_repo_dir / "anno"
        if not anno_dir.exists():
            continue
        chunks_size = _load_chunks_size(resource_repo_dir)
        for anno_path in sorted(anno_dir.glob("episode_*.json")):
            if max_episodes is not None and episode_count >= max_episodes:
                return samples
            with open(anno_path, "r") as f:
                anno = json.load(f)
            subtasks = anno.get("subtasks", [])
            if not subtasks:
                continue
            episode_index = int(anno["episode_index"])
            if (image_repo_dir.name, episode_index) in excluded:
                excluded_episode_count += 1
                continue
            image_hdf5_paths = {
                view: _view_hdf5_path(image_repo_dir, episode_index, chunks_size, view)
                for view in active_views
            }
            if not all(path.exists() for path in image_hdf5_paths.values()):
                continue
            try:
                available_frames = min(_hdf5_num_frames(path) for path in image_hdf5_paths.values())
            except (KeyError, OSError):
                continue
            num_frames = min(int(anno["num_frames"]), available_frames)
            if num_frames <= 0:
                continue
            state_parquet_path = episode_parquet_path(resource_repo_dir, episode_index, chunks_size)
            states = None
            progress_lookup = None
            if state_parquet_path.exists():
                try:
                    states = load_episode_states(state_parquet_path)
                    num_frames = min(num_frames, len(states))
                    progress_lookup = build_subtask_progress_lookup(states, subtasks, anno)
                except Exception:
                    states = None
                    progress_lookup = None
            task_goal = anno["task_goal"]

            for idx, st in enumerate(subtasks):
                start = int(st["start_frame"])
                end = int(st["end_frame"])
                if start >= num_frames:
                    continue
                end = min(end, num_frames - 1)
                samples.append(
                    RobotWinSample(
                        kind="q1",
                        repo_dir=resource_repo_dir,
                        image_hdf5_paths=image_hdf5_paths,
                        frame_index=max(0, min(start, num_frames - 1)),
                        frame_start=max(0, min(start, num_frames - 1)),
                        frame_end=max(0, min(end, num_frames - 1)),
                        task_goal=task_goal,
                        subtasks=subtasks,
                        current_subtask_index=idx,
                        views=active_views,
                        image_repo_dir=image_repo_dir,
                    )
                )

            for idx, st in enumerate(subtasks):
                start = int(st["start_frame"])
                end = int(st["end_frame"])
                if start >= num_frames:
                    continue
                end = min(end, num_frames - 1)
                curve = progress_lookup.get(start) if progress_lookup is not None else None
                current_done_frames = clipped_frames(range(max(start, end - 2), end + 1), num_frames)
                not_done_end = min(end, min(current_done_frames) - 1) if current_done_frames else end
                for frame in range(start, not_done_end + 1, max(1, q2_frame_stride)):
                    progress = progress_for_subtask(
                        st,
                        frame,
                        states=states,
                        anno=anno,
                        curve=curve,
                    )
                    samples.append(
                        make_q2_sample(
                            resource_repo_dir,
                            image_repo_dir,
                            image_hdf5_paths,
                            frame,
                            task_goal,
                            subtasks,
                            idx,
                            current_done=0.0,
                            progress=progress,
                            q2_group="undone",
                        )
                    )

                for frame in current_done_frames:
                    samples.append(
                        make_q2_sample(
                            resource_repo_dir,
                            image_repo_dir,
                            image_hdf5_paths,
                            frame,
                            task_goal,
                            subtasks,
                            idx,
                            current_done=1.0,
                            progress=1.0,
                            q2_group="current_done",
                        )
                    )

                if idx > 0:
                    for frame in clipped_frames(range(start, start + 3), num_frames):
                        samples.append(
                            make_q2_sample(
                                resource_repo_dir,
                                image_repo_dir,
                                image_hdf5_paths,
                                frame,
                                task_goal,
                                subtasks,
                                idx - 1,
                                current_done=1.0,
                                progress=1.0,
                                q2_group="prev_done",
                            )
                        )
            episode_count += 1
    if excluded and excluded_episode_count:
        print(
            f"Skipped {excluded_episode_count} RobotWin episodes from exclude list "
            f"({len(excluded)} entries)"
        )
    return samples


def _sample_frame_index(sample: RobotWinSample) -> int:
    if sample.kind != "q1":
        return sample.frame_index
    start = min(sample.frame_start, sample.frame_end)
    end = max(sample.frame_start, sample.frame_end)
    if end - start >= 2:
        start += 1
        end -= 1
    return random.randint(start, end)


def preprocess_robotwin_sample(sample: RobotWinSample, processor) -> Dict[str, torch.Tensor]:
    frame_index = _sample_frame_index(sample)
    views = sample.views
    images = _load_observation_images(sample.image_hdf5_paths, frame_index, views)
    future = _future_subtasks(sample.subtasks, sample.current_subtask_index)
    completed = _completed_subtasks(sample.subtasks, sample.current_subtask_index)
    if sample.kind == "q1":
        user_content = _user_content(sample.task_goal, completed, images, views)
        messages = _messages_for_sample(_system_prompt(Q1_SYSTEM_PROMPT, views), user_content, assistant=future)
        prompt_messages = _messages_for_sample(_system_prompt(Q1_SYSTEM_PROMPT, views), user_content)
        full_result = processor.apply_chat_template(messages, tokenize=True, return_dict=True, return_tensors="pt")
        prompt_result = processor.apply_chat_template(
            prompt_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        input_ids = full_result["input_ids"]
        labels = torch.full_like(input_ids, IGNORE_INDEX)
        prompt_len = prompt_result["input_ids"].shape[1]
        labels[:, prompt_len:] = input_ids[:, prompt_len:]
    else:
        current = sample.subtasks[sample.current_subtask_index]
        user_content = _user_content(
            sample.task_goal,
            completed,
            images,
            views,
            current_goal=current["subtask_goal"],
            include_query_tokens=True,
        )
        messages = _messages_for_sample(_system_prompt(Q2_SYSTEM_PROMPT, views), user_content)
        full_result = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        input_ids = full_result["input_ids"]
        labels = torch.full_like(input_ids, IGNORE_INDEX)

    full_result["labels"] = labels
    full_result["input_ids"] = input_ids
    full_result["robotwin_frame_index"] = torch.tensor(frame_index, dtype=torch.long)
    full_result["robotwin_current_done"] = torch.tensor(sample.current_done, dtype=torch.float32)
    full_result["robotwin_need_replan"] = torch.tensor(sample.need_replan, dtype=torch.float32)
    full_result["robotwin_incident"] = torch.tensor(sample.incident, dtype=torch.float32)
    full_result["robotwin_progress"] = torch.tensor(sample.progress, dtype=torch.float32)
    return full_result


class RobotWinDataset(Dataset):
    def __init__(self, processor, data_args, split: str = "train"):
        super().__init__()
        self.processor = update_processor_pixels(processor, data_args)
        self.tokenizer = processor.tokenizer
        self.data_args = data_args
        self.split = split
        self.merge_size = getattr(processor.image_processor, "merge_size", 2)
        self.model_type = data_args.model_type
        if data_args.model_type == "qwen3vl":
            self.get_rope_index = get_rope_index_3
        elif data_args.model_type == "qwen2.5vl":
            self.get_rope_index = get_rope_index_25
        elif data_args.model_type == "qwen2vl":
            self.get_rope_index = get_rope_index_2
        else:
            raise ValueError(f"model_type: {data_args.model_type} not supported")
        self.samples = build_robotwin_samples(
            data_args.robotwin_data_root,
            q2_frame_stride=data_args.robotwin_q2_frame_stride,
            boundary_extra_frames=data_args.robotwin_boundary_extra_frames,
            split=split,
            test_ratio=data_args.robotwin_test_ratio,
            split_seed=data_args.robotwin_split_seed,
            anno_root=data_args.robotwin_anno_root,
            views=parse_robotwin_views(data_args.robotwin_views),
            exclude_episodes=load_robotwin_excluded_episodes(
                getattr(data_args, "robotwin_exclude_episodes", None)
            ),
        )
        self.q1_samples = [sample for sample in self.samples if sample.kind == "q1"]
        self.q2_samples = [sample for sample in self.samples if sample.kind == "q2"]
        self.undone_samples = [
            sample for sample in self.q2_samples if sample.q2_group == "undone"
        ]
        self.done_samples = [
            sample for sample in self.q2_samples if sample.q2_group in {"current_done", "prev_done"}
        ]
        self.done_sample_prob = max(
            0.0,
            min(1.0, float(getattr(data_args, "robotwin_done_sample_prob", 0.4))),
        )
        if split == "train":
            random.shuffle(self.samples)
            random.shuffle(self.undone_samples)
            random.shuffle(self.done_samples)
        print(
            f"Loaded RobotWin {split} samples: {len(self.samples)} "
            f"(q1={len(self.q1_samples)}, q2={len(self.q2_samples)}, "
            f"undone={len(self.undone_samples)}, done={len(self.done_samples)}, "
            f"done_prob={self.done_sample_prob})"
        )

    def __len__(self):
        if self.split == "train" and self.undone_samples and self.done_samples:
            return len(self.undone_samples) + len(self.done_samples)
        return len(self.samples)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if self.split == "train" and self.undone_samples and self.done_samples:
            if random.random() < self.done_sample_prob:
                sample = random.choice(self.done_samples)
            else:
                sample = random.choice(self.undone_samples)
        else:
            sample = self.samples[i]
        data_dict = preprocess_robotwin_sample(sample, self.processor)
        seq_len = data_dict["input_ids"][0].size(0)

        grid_thw = data_dict.get("image_grid_thw")
        if grid_thw is not None and not isinstance(grid_thw, Sequence):
            grid_thw = [grid_thw]

        position_ids, _ = self.get_rope_index(
            self.merge_size,
            data_dict["input_ids"],
            image_grid_thw=torch.cat(grid_thw, dim=0) if grid_thw else None,
        )
        data_dict["position_ids"] = position_ids
        data_dict["attention_mask"] = [seq_len]
        return data_dict


@dataclass
class RobotWinDataCollator:
    tokenizer: transformers.PreTrainedTokenizer
    query_token_ids: Dict[str, int]

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, position_ids = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "position_ids")
        )
        input_ids = [ids.squeeze(0) for ids in input_ids]
        labels = [ids.squeeze(0) for ids in labels]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        position_ids = pad_and_cat(position_ids)
        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]
        position_ids = position_ids[:, :, : self.tokenizer.model_max_length]

        batch = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": input_ids.ne(self.tokenizer.pad_token_id),
            "position_ids": position_ids,
        }

        images = [instance["pixel_values"] for instance in instances if "pixel_values" in instance]
        if images:
            batch["pixel_values"] = torch.cat(images, dim=0)
            batch["image_grid_thw"] = torch.cat(
                [instance["image_grid_thw"] for instance in instances if "image_grid_thw" in instance],
                dim=0,
            )
        else:
            batch["pixel_values"] = None
            batch["image_grid_thw"] = None
        batch["pixel_values_videos"] = None
        batch["video_grid_thw"] = None

        for name, token_id in self.query_token_ids.items():
            positions = torch.full((input_ids.shape[0],), -1, dtype=torch.long)
            matches = input_ids.eq(token_id)
            for row in range(input_ids.shape[0]):
                found = torch.nonzero(matches[row], as_tuple=False).flatten()
                if found.numel() > 0:
                    positions[row] = found[-1]
            batch[f"robotwin_{name}_query_pos"] = positions

        batch["robotwin_current_done"] = torch.stack([i["robotwin_current_done"] for i in instances])
        batch["robotwin_need_replan"] = torch.stack([i["robotwin_need_replan"] for i in instances])
        batch["robotwin_incident"] = torch.stack([i["robotwin_incident"] for i in instances])
        batch["robotwin_progress"] = torch.stack([i["robotwin_progress"] for i in instances])
        return batch


def make_robotwin_data_module(processor, data_args, query_token_ids: Dict[str, int]) -> Dict:
    train_dataset = RobotWinDataset(processor, data_args=data_args, split="train")
    eval_dataset = RobotWinDataset(processor, data_args=data_args, split="test")
    data_collator = RobotWinDataCollator(processor.tokenizer, query_token_ids=query_token_ids)
    return {
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset if len(eval_dataset) > 0 else None,
        "data_collator": data_collator,
    }
