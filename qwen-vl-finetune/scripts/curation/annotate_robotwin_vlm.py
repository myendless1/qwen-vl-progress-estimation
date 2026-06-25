#!/usr/bin/env python3
"""Generate VLM-style subtask annotations for RoboTwin LeRobot repos.

The task-specific rules and frame-alignment logic live in the reusable
``robotwin_vlm`` package. This module remains the compatible CLI and
episode-level orchestration entrypoint.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from robotwin_vlm.alignment import (
    GRASP_CLOSE_THRESHOLD,
    RELEASE_OPEN_THRESHOLD,
    assign_spans,
    detect_gripper_events,
    insert_retreat_subtasks,
    merge_stack_arm_switches,
    publish_actual_arm_labels,
)
from robotwin_vlm.models import GripperEvent, StepSpec, TaskContext
from robotwin_vlm.task_rules import (
    CHRONOLOGICAL_ARM_TASKS,
    EXPECTED_TASK_SLUGS,
    NO_RETREAT_TASKS,
    TASK_BUILDERS,
    build_steps,
    canonical_task_goal,
    validate_task_registry,
)


DEFAULT_ROOT = Path("/media/damoxing/datasets/vae4d/lerobot-vae4d-org/robotwin")
DEFAULT_RAW_ROOT = Path("/media/damoxing/datasets/RoboTwin2_0/dataset")


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
    raw_root: Path,
    slug: str,
    task_dir: str,
    episode_index: int,
) -> dict[str, str]:
    path = raw_root / slug / task_dir / "scene_info.json"
    key = f"episode_{episode_index}"
    if not path.exists():
        return {}
    try:
        scene = read_json(path)
    except Exception:
        return {}
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
    grasp_close_threshold: float = GRASP_CLOSE_THRESHOLD,
    release_open_threshold: float = RELEASE_OPEN_THRESHOLD,
    insert_retreat: bool = True,
) -> dict[str, Any]:
    episode_index = int(episode["episode_index"])
    slug = task_slug_from_repo(repo)
    task_goal = episode.get("tasks", [""])[0] or ""
    if not task_goal:
        task_goal = episode.get("task", "")
    task_goal = canonical_task_goal(slug, task_goal)
    states = load_states(episode_parquet_path(repo, episode_index, info_json))
    events = detect_gripper_events(states, threshold=gripper_threshold)
    scene_info = raw_scene_info(raw_root, slug, task_dir_from_repo(repo), episode_index)
    steps = build_steps(slug, task_goal, scene_info, events)
    subtasks = assign_spans(
        steps,
        events,
        int(states.shape[0]),
        states=states,
        grasp_close_threshold=grasp_close_threshold,
        release_open_threshold=release_open_threshold,
        prefer_specified_arm=slug not in CHRONOLOGICAL_ARM_TASKS,
    )
    subtasks = merge_stack_arm_switches(subtasks, slug)
    effective_insert_retreat = insert_retreat and slug not in NO_RETREAT_TASKS
    if effective_insert_retreat:
        subtasks = insert_retreat_subtasks(subtasks, states)
    anno = {
        "episode_index": episode_index,
        "repo": repo.name,
        "task_slug": slug,
        "task_goal": task_goal,
        "num_frames": int(states.shape[0]),
        "subtasks": subtasks,
        "metadata": {
            "annotation_version": "robotwin_vlm_subtask_v2",
            "gripper_threshold": gripper_threshold,
            "grasp_close_threshold": grasp_close_threshold,
            "release_open_threshold": release_open_threshold,
            "retreat_subtasks_enabled": effective_insert_retreat,
            "detected_gripper_events": [
                {"frame": event.frame, "arm": event.arm, "kind": event.kind}
                for event in events
            ],
            "scene_info": scene_info,
        },
    }
    return publish_actual_arm_labels(anno)


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


def print_rules(root: Path) -> None:
    validate_task_registry()
    discovered = {task_slug_from_repo(repo) for repo in iter_repos(root, None)}
    missing = discovered - TASK_BUILDERS.keys()
    if missing:
        raise ValueError(f"dataset contains unregistered RoboTwin tasks: {sorted(missing)}")
    for slug in sorted(EXPECTED_TASK_SLUGS):
        print(f"{slug}: {len(build_steps(slug, '', {}))} subtasks")


def report_retreat_candidates(
    root: Path,
    raw_root: Path,
    only: str | None,
    limit: int | None,
    gripper_threshold: float,
    grasp_close_threshold: float,
    release_open_threshold: float,
) -> None:
    summary: dict[str, dict[str, Any]] = {}
    for repo in iter_repos(root, only):
        info_json = read_json(repo / "meta" / "info.json")
        episodes = read_jsonl(repo / "meta" / "episodes.jsonl")
        if limit is not None:
            episodes = episodes[:limit]
        slug = task_slug_from_repo(repo)
        entry = summary.setdefault(
            slug,
            {"repos": set(), "episodes": 0, "episodes_with_retreat": 0, "retreat_subtasks": 0},
        )
        entry["repos"].add(repo.name)
        for episode in episodes:
            try:
                anno = annotate_episode(
                    repo=repo,
                    episode=episode,
                    info_json=info_json,
                    raw_root=raw_root,
                    gripper_threshold=gripper_threshold,
                    grasp_close_threshold=grasp_close_threshold,
                    release_open_threshold=release_open_threshold,
                    insert_retreat=True,
                )
            except Exception as exc:
                print(f"[ERROR] {repo.name} episode_{int(episode['episode_index']):06d}: {exc}")
                continue
            retreat_count = sum(
                1
                for subtask in anno["subtasks"]
                if str(subtask.get("boundary_source", "")).startswith("eef_")
            )
            entry["episodes"] += 1
            entry["retreat_subtasks"] += retreat_count
            if retreat_count:
                entry["episodes_with_retreat"] += 1
    for slug, entry in sorted(summary.items()):
        if entry["retreat_subtasks"] == 0:
            continue
        print(
            f"{slug}: repos={len(entry['repos'])}, "
            f"episodes_with_retreat={entry['episodes_with_retreat']}/{entry['episodes']}, "
            f"retreat_subtasks={entry['retreat_subtasks']}"
        )


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
        "--grasp-close-threshold",
        type=float,
        default=GRASP_CLOSE_THRESHOLD,
        help="Stricter gripper value used to end Grasp subtasks after the initial close event.",
    )
    parser.add_argument(
        "--release-open-threshold",
        type=float,
        default=RELEASE_OPEN_THRESHOLD,
        help="Stricter gripper value used to end Place/Release subtasks after the initial open event.",
    )
    parser.add_argument(
        "--no-retreat-subtasks",
        action="store_true",
        help="Do not insert EEF-velocity-based arm return/retreat subtasks after release.",
    )
    parser.add_argument("--print-rules", action="store_true", help="Print task slug to subtask-count mapping and exit.")
    parser.add_argument(
        "--report-retreat-candidates",
        action="store_true",
        help="Report task slugs where EEF velocity creates return/retreat subtasks.",
    )
    args = parser.parse_args()
    if args.print_rules:
        print_rules(args.root)
        return
    if args.report_retreat_candidates:
        report_retreat_candidates(
            root=args.root,
            raw_root=args.raw_root,
            only=args.only,
            limit=args.limit,
            gripper_threshold=args.gripper_threshold,
            grasp_close_threshold=args.grasp_close_threshold,
            release_open_threshold=args.release_open_threshold,
        )
        return

    total = 0
    skipped = 0
    for repo in iter_repos(args.root, args.only):
        info_json = read_json(repo / "meta" / "info.json")
        episodes = read_jsonl(repo / "meta" / "episodes.jsonl")
        if args.limit is not None:
            episodes = episodes[: args.limit]
        for episode in episodes:
            episode_index = int(episode["episode_index"])
            out_path = repo / "anno" / f"episode_{episode_index:06d}.json"
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
                    grasp_close_threshold=args.grasp_close_threshold,
                    release_open_threshold=args.release_open_threshold,
                    insert_retreat=not args.no_retreat_subtasks,
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
