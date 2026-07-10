#!/usr/bin/env python3
"""Generate VLM-style subtask annotations for RoboTwin LeRobot repos.

The task-specific rules and frame-alignment logic live in the reusable
``robotwin_vlm`` package. This module remains the compatible CLI and
episode-level orchestration entrypoint.
"""

from __future__ import annotations

import argparse
import importlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from robotwin_vlm.alignment import (
    assign_spans,
    detect_gripper_events,
    flip_spatial_lr_mentions,
    merge_stack_arm_switches,
    merge_tiny_post_gripper_motion,
    publish_actual_arm_labels,
    relabel_dual_container_first_place,
)
from robotwin_vlm.alignment_coarse import assign_coarse_spans
from robotwin_vlm.models import GripperEvent, StepSpec, TaskContext


DEFAULT_ROOT = Path("/media/damoxing/datasets/vae4d/lerobot-vae4d-org/robotwin_gt_depth")
DEFAULT_RAW_ROOT = Path("/media/damoxing/datasets/robotwin-depth-f1")
ANNOTATION_VERSIONS = {
    "fine": "robotwin_vlm_fine_atomic_delayed_v3",
    "coarse": "robotwin_vlm_coarse_v1",
}


def load_task_rules(granularity: str):
    if granularity not in ANNOTATION_VERSIONS:
        raise ValueError(f"unsupported annotation granularity: {granularity}")
    return importlib.import_module(f"robotwin_vlm.task_rules_{granularity}")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def task_slug_from_repo(repo: Path) -> str:
    name = repo.name
    marker = "-aloha-agilex_"
    if marker in name:
        return name.split(marker, 1)[0]
    return name.split("-", 1)[0]


def task_dir_from_repo(repo: Path) -> str:
    match = re.search(r"aloha-agilex_(?:clean_50|randomized_500)$", repo.name)
    if match:
        return match.group(0)
    return repo.name.split("-", 1)[-1]


def raw_config_dir_from_repo(repo: Path) -> str:
    if repo.name.endswith("-aloha-agilex_randomized_500"):
        return "demo_randomized"
    if repo.name.endswith("-aloha-agilex_clean_50"):
        return "demo_clean"
    return task_dir_from_repo(repo)


def episode_parquet_path(repo: Path, episode_index: int, info: dict[str, Any]) -> Path:
    chunk_size = int(info.get("chunks_size", 1000))
    chunk = episode_index // chunk_size
    return repo / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet"


def load_states(parquet_path: Path) -> np.ndarray:
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Reading RoboTwin episode data requires pyarrow. "
            "Install pyarrow in the active environment before generating annotations."
        ) from exc
    table = pq.read_table(parquet_path, columns=["observation.state"])
    values = table.column("observation.state").to_pylist()
    return np.asarray(values, dtype=np.float32)


def raw_scene_info(
    repo: Path,
    raw_root: Path,
    slug: str,
    task_dir: str,
    episode_index: int,
) -> dict[str, str]:
    path = raw_root / slug / task_dir / "scene_info.json"
    if not path.exists():
        return {}
    try:
        scene = read_json(path)
    except Exception:
        return {}
    key = f"episode_{episode_index}"
    source_meta_path = repo / "source_meta" / f"episode_{episode_index:06d}.json"
    if source_meta_path.exists():
        try:
            source_meta = read_json(source_meta_path)
            episode_path = str(
                source_meta.get("source_meta", {})
                .get("raw_record_payload", {})
                .get("episode_path", "")
            )
            match = re.search(r"episode(\d+)\.hdf5$", episode_path)
            if match:
                key = f"episode_{int(match.group(1))}"
        except Exception:
            pass
    info = scene.get(key, {}).get("info", {})
    if isinstance(info, dict):
        return {str(k): str(v) for k, v in info.items()}
    return {}


def annotate_episode(
    repo: Path,
    episode: dict[str, Any],
    info_json: dict[str, Any],
    raw_root: Path,
    gripper_threshold: float,
    granularity: str,
    task_rules,
) -> dict[str, Any]:
    episode_index = int(episode["episode_index"])
    slug = task_slug_from_repo(repo)
    task_goal = episode.get("tasks", [""])[0] or ""
    if not task_goal:
        task_goal = episode.get("task", "")
    task_goal = task_rules.canonical_task_goal(slug, task_goal)
    states = load_states(episode_parquet_path(repo, episode_index, info_json))
    events = detect_gripper_events(states, threshold=gripper_threshold)
    scene_info = raw_scene_info(repo, raw_root, slug, raw_config_dir_from_repo(repo), episode_index)
    steps = task_rules.build_steps(slug, task_goal, scene_info, events)
    if granularity == "coarse":
        subtasks = assign_coarse_spans(
            steps,
            events,
            int(states.shape[0]),
            states=states,
            merge_open_move_text=slug not in {"place_burger_fries", "place_container_plate"},
        )
        alignment_policy = "coarse motion segments with at most one terminal gripper event"
    else:
        subtasks = assign_spans(
            steps,
            events,
            int(states.shape[0]),
            states=states,
            prefer_specified_arm=slug not in task_rules.CHRONOLOGICAL_ARM_TASKS,
        )
        subtasks = merge_stack_arm_switches(subtasks, slug)
        subtasks = merge_tiny_post_gripper_motion(subtasks, states)
        subtasks = relabel_dual_container_first_place(subtasks, states, slug)
        alignment_policy = "atomic actions with simultaneous delayed termination candidates"
    anno = {
        "episode_index": episode_index,
        "repo": repo.name,
        "task_slug": slug,
        "task_goal": task_goal,
        "num_frames": int(states.shape[0]),
        "subtasks": subtasks,
        "metadata": {
            "annotation_version": ANNOTATION_VERSIONS[granularity],
            "annotation_granularity": granularity,
            "gripper_threshold": gripper_threshold,
            "alignment_policy": alignment_policy,
            "detected_gripper_events": [
                {
                    "frame": event.frame,
                    "arm": event.arm,
                    "kind": event.kind,
                    "start_frame": event.start_frame,
                    "end_frame": event.end_frame,
                }
                for event in events
            ],
            "scene_info": scene_info,
        },
    }
    # Do not flip left/right arm labels at publish time. The RoboTwin state
    # indices are already consumed consistently by the rule/alignment code.
    # return publish_actual_arm_labels(anno)
    if slug not in {"place_a2b_left", "place_a2b_right"}:
        anno["task_goal"] = flip_spatial_lr_mentions(str(anno.get("task_goal", "")))
        for subtask in anno.get("subtasks", []):
            subtask["subtask_goal"] = flip_spatial_lr_mentions(str(subtask.get("subtask_goal", "")))
    return anno


def iter_repos(root: Path, only: str | None) -> list[Path]:
    repos = []
    for repo in sorted(root.iterdir()):
        if not repo.is_dir():
            continue
        if only and only not in repo.name:
            continue
        if (repo / "meta" / "episodes.jsonl").exists() and (repo / "meta" / "info.json").exists():
            repos.append(repo)
    return repos


def print_rules(root: Path, granularity: str, task_rules) -> None:
    task_rules.validate_task_registry()
    discovered = {task_slug_from_repo(repo) for repo in iter_repos(root, None)}
    missing = discovered - task_rules.TASK_BUILDERS.keys()
    if missing:
        raise ValueError(f"dataset contains unregistered RoboTwin tasks: {sorted(missing)}")
    for slug in sorted(task_rules.EXPECTED_TASK_SLUGS):
        print(f"{slug}: {len(task_rules.build_steps(slug, '', {}))} {granularity} base rule steps")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--only", type=str, default=None, help="Only process repos whose names contain this string.")
    parser.add_argument("--limit", type=int, default=None, help="Limit episodes per repo.")
    parser.add_argument("--dry-run", action="store_true", help="Print examples without writing anno files.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing annotation files.")
    parser.add_argument("--gripper-threshold", type=float, default=0.5)
    parser.add_argument(
        "--granularity",
        choices=tuple(ANNOTATION_VERSIONS),
        default="fine",
        help="Annotation granularity. fine writes anno_fine/; coarse writes anno_coarse/.",
    )
    parser.add_argument("--print-rules", action="store_true", help="Print task slug to subtask-count mapping and exit.")
    args = parser.parse_args()
    task_rules = load_task_rules(args.granularity)
    if args.print_rules:
        print_rules(args.root, args.granularity, task_rules)
        return
    anno_dir_name = f"anno_{args.granularity}"
    total = 0
    skipped = 0
    for repo in iter_repos(args.root, args.only):
        info_json = read_json(repo / "meta" / "info.json")
        episodes = read_jsonl(repo / "meta" / "episodes.jsonl")
        if args.limit is not None:
            episodes = episodes[: args.limit]
        for episode in episodes:
            episode_index = int(episode["episode_index"])
            out_path = repo / anno_dir_name / f"episode_{episode_index:06d}.json"
            if out_path.exists() and not args.overwrite and not args.dry_run:
                skipped += 1
                continue
            try:
                anno = annotate_episode(
                    repo=repo,
                    episode=episode,
                    info_json=info_json,
                    raw_root=args.raw_root,
                    gripper_threshold=args.gripper_threshold,
                    granularity=args.granularity,
                    task_rules=task_rules,
                )
            except Exception as exc:
                print(f"[ERROR] {repo.name} episode_{episode_index:06d}: {exc}")
                skipped += 1
                continue
            if args.dry_run:
                print(json.dumps(anno, ensure_ascii=False, indent=2))
            else:
                write_json(out_path, anno)
            total += 1
    action = "would write" if args.dry_run else "wrote"
    print(f"{action} {total} annotation files; skipped {skipped}.")


if __name__ == "__main__":
    main()
