#!/usr/bin/env python3
"""Evaluate RoboTwin subtask annotations against state-based split rules."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_ROOT = Path("/media/damoxing/datasets/vae4d/lerobot-vae4d-org/robotwin_gt_depth")
DEFAULT_LOG_DIR = Path("/media/damoxing/fileset/Qwen3-VL/qwen-vl-finetune/scripts/curation/tests")

LEFT_GRIPPER_DIM = 7
RIGHT_GRIPPER_DIM = 15
ARM_XYZ_DIMS = {
    "left": slice(0, 3),
    "right": slice(8, 11),
}
GRIPPER_DIMS = {
    "left": LEFT_GRIPPER_DIM,
    "right": RIGHT_GRIPPER_DIM,
}
ARM_FLIP = {
    "left": "right",
    "right": "left",
}
TASK_OTHER_ARM_MOTION_THRESHOLD: dict[str, float] = {
    "handover_mic": 0.05,
    # Holding arm may shift slightly when the other arm approaches to place bread.
    "place_bread_basket": 0.04,
    "place_bread_skillet": 0.04,
    # Cabinet door / holding arm may shift when the placing arm moves in.
    "put_object_cabinet": 0.04,
}


def other_arm_motion_threshold(slug: str, default: float) -> float:
    return TASK_OTHER_ARM_MOTION_THRESHOLD.get(slug, default)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def episode_parquet_path(repo: Path, episode_index: int, info: dict[str, Any]) -> Path:
    chunk_size = int(info.get("chunks_size", 1000))
    chunk = episode_index // chunk_size
    return repo / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet"


def load_states(path: Path) -> np.ndarray:
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("pyarrow is required to evaluate RoboTwin annotations") from exc
    table = pq.read_table(path, columns=["observation.state"])
    return np.asarray(table.column("observation.state").to_pylist(), dtype=np.float32)


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


def task_slug(repo_name: str) -> str:
    marker = "-aloha-"
    if marker in repo_name:
        return repo_name.split(marker, 1)[0]
    return repo_name.split("-", 1)[0]


def annotation_arm_to_state_arm(anno: dict[str, Any], arm: str) -> str:
    mapping = str(anno.get("metadata", {}).get("arm_label_mapping", ""))
    if "flipped" in mapping:
        return ARM_FLIP.get(arm, arm)
    return arm


def arm_mentions(text: str) -> set[str]:
    lowered = text.lower()
    arms: set[str] = set()
    if re.search(r"\bleft arm\b", lowered):
        arms.add("left")
    if re.search(r"\bright arm\b", lowered):
        arms.add("right")
    if re.search(r"\b(?:both arms|dual arms|both grippers|grippers of both arms|both objects)\b", lowered):
        arms.update({"left", "right"})
    return arms


def action_kind(text: str) -> str:
    lowered = text.lower()
    if re.search(r"\bopen the gripper|open the grippers|release\b", lowered):
        return "open"
    if re.search(r"\bclose the gripper|close the grippers|partially close\b", lowered):
        return "close"
    if re.search(r"\b(move|lift|place|return|pull|scan|rotate|shake|press|operate|click|hit|pour|open the .*door|open the .*lid)\b", lowered):
        return "move"
    return "unknown"


def subtask_type(subtask: dict[str, Any]) -> str:
    value = str(subtask.get("subtask_type", "")).strip()
    if value:
        return value
    return action_kind(str(subtask.get("subtask_goal", "")))


def subtask_truncation_rule(subtask: dict[str, Any]) -> str:
    value = str(subtask.get("truncation_rule", "")).strip()
    if value:
        return value
    source = str(subtask.get("boundary_source", "")).strip()
    if "gripper" in source and "open" in source:
        return "gripper_open"
    if "gripper" in source and "close" in source:
        return "gripper_close"
    if "eef" in source and "motion" in source:
        return "eef_motion"
    if source:
        return source
    return "unknown"


def is_episode_end_tail_subtask(
    subtask: dict[str, Any],
    *,
    position: int,
    num_subtasks: int,
    num_frames: int,
) -> bool:
    """Return True when the final subtask is only short because the episode ended."""
    if position != num_subtasks - 1:
        return False
    end = int(subtask["end_frame"])
    if end != num_frames - 1:
        return False
    source = str(subtask.get("boundary_source", "")).strip()
    if source == "episode_end" or source.startswith("episode_end_after_"):
        return True
    return subtask_truncation_rule(subtask) == "episode_end"


def compact_subtask_context(subtask: dict[str, Any] | None) -> dict[str, Any] | None:
    if subtask is None:
        return None
    return {
        "subtask_index": int(subtask.get("subtask_index", -1)),
        "subtask_type": subtask_type(subtask),
        "truncation_rule": subtask_truncation_rule(subtask),
        "boundary_source": str(subtask.get("boundary_source", "")),
    }


def span_slice(states: np.ndarray, start: int, end: int) -> np.ndarray:
    start = max(0, min(start, len(states) - 1))
    end = max(start, min(end, len(states) - 1))
    return states[start : end + 1]


def arm_path_length(states: np.ndarray, arm: str, start: int, end: int) -> float:
    if arm not in ARM_XYZ_DIMS or end <= start:
        return 0.0
    xyz = span_slice(states, start, end)[:, ARM_XYZ_DIMS[arm]]
    if len(xyz) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum())


def arm_max_step(states: np.ndarray, arm: str, start: int, end: int) -> float:
    if arm not in ARM_XYZ_DIMS or end <= start:
        return 0.0
    xyz = span_slice(states, start, end)[:, ARM_XYZ_DIMS[arm]]
    if len(xyz) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).max(initial=0.0))


def gripper_stats(
    states: np.ndarray,
    arm: str,
    start: int,
    end: int,
    *,
    open_threshold: float,
) -> dict[str, float | bool]:
    dim = GRIPPER_DIMS[arm]
    if states.shape[1] <= dim:
        return {"range": 0.0, "delta": 0.0, "crossed": False}
    values = span_slice(states, start, end)[:, dim]
    if len(values) == 0:
        return {"range": 0.0, "delta": 0.0, "crossed": False}
    is_open = values > open_threshold
    return {
        "range": float(values.max() - values.min()),
        "delta": float(values[-1] - values[0]),
        "crossed": bool(np.any(is_open[1:] != is_open[:-1])) if len(values) > 1 else False,
    }


def gripper_changed(stats: dict[str, float | bool], threshold: float) -> bool:
    return bool(stats["crossed"]) or abs(float(stats["range"])) >= threshold


def gripper_motion_end_frame(
    states: np.ndarray,
    arm: str,
    start: int,
    end: int,
    *,
    eps: float,
) -> int | None:
    """Return the last frame where the gripper is still changing in a span."""
    dim = GRIPPER_DIMS[arm]
    if states.shape[1] <= dim or end <= start:
        return None
    values = span_slice(states, start, end)[:, dim]
    if len(values) < 2:
        return None
    active = np.flatnonzero(np.abs(np.diff(values)) > eps)
    if len(active) == 0:
        return None
    return int(start + active[-1] + 1)


def state_changed(
    arm: str,
    motion: dict[str, dict[str, float]],
    grippers: dict[str, dict[str, float | bool]],
    *,
    motion_threshold: float,
    gripper_change_threshold: float,
) -> bool:
    return (
        motion[arm]["path"] >= motion_threshold
        or gripper_changed(grippers[arm], gripper_change_threshold)
    )


def expected_event_for_boundary(source: str) -> tuple[str, str] | None:
    match = re.match(r"^(?:episode_end_after_)?before_gripper_(left|right)_(open|close)$", source)
    if not match:
        return None
    return match.group(1), match.group(2)


def boundary_motion_arm(source: str) -> str | None:
    match = re.match(r"^(?:episode_end_after_)?before_eef_(left|right)_motion$", source)
    return match.group(1) if match else None


def event_matches_boundary(
    anno: dict[str, Any],
    source: str,
    next_start: int | None,
) -> bool:
    expected = expected_event_for_boundary(source)
    if expected is None or next_start is None:
        return True
    arm, kind = expected
    for event in anno.get("metadata", {}).get("detected_gripper_events", []):
        if event.get("arm") != arm or event.get("kind") != kind:
            continue
        event_start = int(event.get("start_frame", event.get("frame", -1)))
        if event_start == next_start:
            return True
    return False


def add_issue(
    issues: list[dict[str, Any]],
    *,
    repo: Path,
    anno_path: Path,
    anno: dict[str, Any],
    subtask: dict[str, Any] | None,
    rule: str,
    message: str,
    severity: str = "error",
    details: dict[str, Any] | None = None,
) -> None:
    item: dict[str, Any] = {
        "severity": severity,
        "rule": rule,
        "repo": repo.name,
        "episode_index": int(anno.get("episode_index", -1)),
    }
    del anno_path, message, details
    if subtask is not None:
        current = compact_subtask_context(subtask)
        left = compact_subtask_context(subtask.get("_left_subtask"))
        item["subtask_index"] = current["subtask_index"] if current is not None else -1
        item["left_subtask"] = left
        item["current_subtask"] = current
    issues.append(item)


def evaluate_annotation(
    repo: Path,
    anno_path: Path,
    info: dict[str, Any],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    anno = read_json(anno_path)
    episode_index = int(anno["episode_index"])
    states = load_states(episode_parquet_path(repo, episode_index, info))
    subtasks = list(anno.get("subtasks", []))
    num_frames = int(anno.get("num_frames", len(states)))
    slug = str(anno.get("task_slug", task_slug(repo.name)))
    issues: list[dict[str, Any]] = []

    if num_frames != len(states):
        add_issue(
            issues,
            repo=repo,
            anno_path=anno_path,
            anno=anno,
            subtask=None,
            rule="num_frames_matches_state",
            message=f"annotation num_frames={num_frames} but state frames={len(states)}",
        )
    if not subtasks:
        add_issue(
            issues,
            repo=repo,
            anno_path=anno_path,
            anno=anno,
            subtask=None,
            rule="has_subtasks",
            message="annotation has no subtasks",
        )
        return issues

    previous_end = -1
    for position, subtask in enumerate(subtasks):
        subtask["_left_subtask"] = subtasks[position - 1] if position > 0 else None
        start = int(subtask["start_frame"])
        end = int(subtask["end_frame"])
        length = end - start + 1
        text = str(subtask.get("subtask_goal", ""))
        source = str(subtask.get("boundary_source", ""))
        stype = subtask_type(subtask)
        mentions = arm_mentions(text)
        if stype.startswith("dual_"):
            mentions.update({"left", "right"})
        kind = stype.removeprefix("dual_") if stype.startswith("dual_") else action_kind(text)
        next_subtask = subtasks[position + 1] if position + 1 < len(subtasks) else None
        next_start = int(next_subtask["start_frame"]) if next_subtask is not None else None

        if int(subtask.get("subtask_index", position)) != position:
            add_issue(
                issues,
                repo=repo,
                anno_path=anno_path,
                anno=anno,
                subtask=subtask,
                rule="sequential_subtask_index",
                message=f"subtask_index should be {position}",
            )
        if start != previous_end + 1:
            add_issue(
                issues,
                repo=repo,
                anno_path=anno_path,
                anno=anno,
                subtask=subtask,
                rule="continuous_spans",
                message=f"start_frame={start} should equal previous_end+1={previous_end + 1}",
            )
        if start < 0 or end < start or end >= len(states):
            add_issue(
                issues,
                repo=repo,
                anno_path=anno_path,
                anno=anno,
                subtask=subtask,
                rule="valid_frame_range",
                message=f"invalid frame span {start}-{end} for {len(states)} state frames",
            )
            previous_end = end
            continue
        if length < args.min_frames and not is_episode_end_tail_subtask(
            subtask,
            position=position,
            num_subtasks=len(subtasks),
            num_frames=len(states),
        ):
            add_issue(
                issues,
                repo=repo,
                anno_path=anno_path,
                anno=anno,
                subtask=subtask,
                rule="min_frames",
                message=f"subtask has {length} frames, expected at least {args.min_frames}",
            )

        state_arms = {
            arm: annotation_arm_to_state_arm(anno, arm)
            for arm in ("left", "right")
        }
        motion = {
            arm: {
                "path": arm_path_length(states, state_arm, start, end),
                "max_step": arm_max_step(states, state_arm, start, end),
            }
            for arm, state_arm in state_arms.items()
        }
        grippers = {
            arm: gripper_stats(
                states,
                state_arm,
                start,
                end,
                open_threshold=args.gripper_open_threshold,
            )
            for arm, state_arm in state_arms.items()
        }
        changed = {
            arm: state_changed(
                arm,
                motion,
                grippers,
                motion_threshold=args.motion_threshold,
                gripper_change_threshold=args.gripper_change_threshold,
            )
            for arm in ("left", "right")
        }
        details = {
            "kind": kind,
            "mentions": sorted(mentions),
            "motion": motion,
            "grippers": grippers,
            "changed": changed,
        }

        if mentions in ({"left"}, {"right"}):
            mentioned_arm = next(iter(mentions))
            other = ARM_FLIP[mentioned_arm]
            other_motion_threshold = other_arm_motion_threshold(slug, args.motion_threshold)
            other_changed = (
                motion[other]["path"] >= other_motion_threshold
                or gripper_changed(grippers[other], args.gripper_change_threshold)
            )
            if other_changed:
                other_path = motion[other]["path"]
                mentioned_path = motion[mentioned_arm]["path"]
                if other_path > mentioned_path + args.motion_threshold:
                    add_issue(
                        issues,
                        repo=repo,
                        anno_path=anno_path,
                        anno=anno,
                        subtask=subtask,
                        rule="single_arm_label_maybe_flipped",
                        message=(
                            f"text mentions only {mentioned_arm} arm, but {other} arm moved more; "
                            "arm label may be swapped"
                        ),
                        details={
                            **details,
                            "mentioned_arm": mentioned_arm,
                            "dominant_motion_arm": other,
                        },
                    )
                else:
                    add_issue(
                        issues,
                        repo=repo,
                        anno_path=anno_path,
                        anno=anno,
                        subtask=subtask,
                        rule="single_arm_other_arm_static",
                        message=f"text mentions only {mentioned_arm} arm, but {other} arm state changed",
                        details=details,
                    )
        skip_dual_both_changed = (
            slug in {"place_dual_shoes", "place_cans_plasticbox"}
            and stype == "dual_move"
            and re.match(r"^Move the (left|right) arm to the place pose", text)
            and not re.search(r"\bwhile returning\b", text, flags=re.IGNORECASE)
            and changed["left"] != changed["right"]
        )
        if (
            mentions == {"left", "right"}
            and (not changed["left"] or not changed["right"])
            and not skip_dual_both_changed
        ):
            add_issue(
                issues,
                repo=repo,
                anno_path=anno_path,
                anno=anno,
                subtask=subtask,
                rule="dual_arm_both_changed",
                message="text describes a dual-arm action, but not both arms changed",
                details=details,
            )

        if kind in {"open", "close"}:
            post_gripper_motion: dict[str, dict[str, float | int | None]] = {}
            moving_arms = []
            for arm in ("left", "right"):
                state_arm = state_arms[arm]
                gripper_end = gripper_motion_end_frame(
                    states,
                    state_arm,
                    start,
                    end,
                    eps=args.gripper_motion_eps,
                )
                check_start = gripper_end if gripper_end is not None else start
                post_path = arm_path_length(states, state_arm, check_start, end)
                post_max_step = arm_max_step(states, state_arm, check_start, end)
                post_gripper_motion[arm] = {
                    "gripper_motion_end_frame": gripper_end,
                    "path_after_gripper_motion": post_path,
                    "max_step_after_gripper_motion": post_max_step,
                }
                if post_path >= args.open_close_motion_threshold:
                    moving_arms.append(arm)
            if moving_arms:
                add_issue(
                    issues,
                    repo=repo,
                    anno_path=anno_path,
                    anno=anno,
                    subtask=subtask,
                    rule="open_close_no_displacement",
                    message=f"{kind} action has EEF displacement after gripper motion stops in arms: {moving_arms}",
                    details={**details, "post_gripper_motion": post_gripper_motion},
                )

        if kind == "move":
            changed_gripper_arms = [
                arm for arm in ("left", "right")
                if gripper_changed(grippers[arm], args.gripper_change_threshold)
            ]
            if changed_gripper_arms:
                add_issue(
                    issues,
                    repo=repo,
                    anno_path=anno_path,
                    anno=anno,
                    subtask=subtask,
                    rule="move_no_gripper_change",
                    message=f"move action includes gripper open/close change in arms: {changed_gripper_arms}",
                    details=details,
                )

        expected_event = expected_event_for_boundary(source)
        if expected_event is not None:
            expected_arm, expected_kind = expected_event
            # A dual-arm motion subtask moves both arms, so the segmentation
            # boundary may land on either arm's gripper event. The following
            # subtask can then be a gripper action on either arm, so the
            # next-arm consistency check is skipped for such subtasks.
            is_dual_arm_motion = changed["left"] and changed["right"]
            if next_subtask is not None:
                next_text = str(next_subtask.get("subtask_goal", ""))
                next_kind = action_kind(next_text)
                next_mentions = arm_mentions(next_text)
                if next_kind != expected_kind:
                    add_issue(
                        issues,
                        repo=repo,
                        anno_path=anno_path,
                        anno=anno,
                        subtask=subtask,
                        rule="gripper_boundary_next_action",
                        message=f"{source} should be followed by {expected_kind}, got {next_kind}",
                        details={"next_subtask_goal": next_text},
                    )
                if next_mentions and expected_arm not in next_mentions and not is_dual_arm_motion:
                    add_issue(
                        issues,
                        repo=repo,
                        anno_path=anno_path,
                        anno=anno,
                        subtask=subtask,
                        rule="gripper_boundary_next_arm",
                        message=f"{source} should be followed by an action mentioning {expected_arm} arm",
                        details={"next_subtask_goal": next_text, "next_mentions": sorted(next_mentions)},
                    )
            if not event_matches_boundary(anno, source, next_start):
                add_issue(
                    issues,
                    repo=repo,
                    anno_path=anno_path,
                    anno=anno,
                    subtask=subtask,
                    rule="gripper_boundary_matches_detected_event",
                    message=f"{source} does not align to a detected gripper event at next_start={next_start}",
                )

        motion_arm = boundary_motion_arm(source)
        if motion_arm is not None and kind == "unknown":
            add_issue(
                issues,
                repo=repo,
                anno_path=anno_path,
                anno=anno,
                subtask=subtask,
                rule="eef_boundary_current_action",
                message=f"{source} should terminate a recognized current action, got {kind}",
            )

        previous_end = end

    last = subtasks[-1]
    if int(last["end_frame"]) != len(states) - 1:
        add_issue(
            issues,
            repo=repo,
            anno_path=anno_path,
            anno=anno,
            subtask=last,
            rule="last_subtask_reaches_episode_end",
            message=f"last end_frame={last['end_frame']} should be {len(states) - 1}",
        )
    return issues


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--only", type=str, default=None, help="Only process repos whose names contain this string.")
    parser.add_argument("--limit-repos", type=int, default=None)
    parser.add_argument("--limit-annos", type=int, default=None, help="Limit annotations per repo.")
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--log-name", type=str, default=None, help="Base log filename without extension.")
    parser.add_argument("--min-frames", type=int, default=7)
    parser.add_argument("--motion-threshold", type=float, default=0.015, help="Path length treated as arm state change.")
    parser.add_argument(
        "--open-close-motion-threshold",
        type=float,
        default=0.015,
        help="EEF path after gripper motion stops that is treated as a new subtask.",
    )
    parser.add_argument("--gripper-open-threshold", type=float, default=0.5)
    parser.add_argument("--gripper-change-threshold", type=float, default=0.25)
    parser.add_argument("--gripper-motion-eps", type=float, default=1e-4)
    parser.add_argument("--fail-on-issues", action="store_true")
    args = parser.parse_args()

    repos = iter_repos(args.root, args.only)
    if args.limit_repos is not None:
        repos = repos[: args.limit_repos]

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = args.log_name or f"robotwin_anno_eval_{stamp}"
    jsonl_path = args.log_dir / f"{base}.jsonl"
    summary_path = args.log_dir / f"{base}_summary.json"
    text_summary_path = args.log_dir / f"{base}_summary.txt"
    args.log_dir.mkdir(parents=True, exist_ok=True)

    total_annos = 0
    total_subtasks = 0
    issue_count = 0
    issue_rules: Counter[str] = Counter()
    issue_repos: Counter[str] = Counter()

    with jsonl_path.open("w", encoding="utf-8") as log_f:
        for repo in repos:
            info = read_json(repo / "meta" / "info.json")
            anno_paths = sorted((repo / "anno").glob("episode_*.json"))
            if args.limit_annos is not None:
                anno_paths = anno_paths[: args.limit_annos]
            for anno_path in anno_paths:
                total_annos += 1
                try:
                    anno = read_json(anno_path)
                    total_subtasks += len(anno.get("subtasks", []))
                    issues = evaluate_annotation(repo, anno_path, info, args)
                except Exception as exc:
                    issues = [
                        {
                            "severity": "error",
                            "rule": "evaluation_exception",
                            "repo": repo.name,
                            "episode_index": int(re.search(r"episode_(\d+)", anno_path.stem).group(1))
                            if re.search(r"episode_(\d+)", anno_path.stem)
                            else -1,
                            "message": repr(exc),
                        }
                    ]
                for issue in issues[:1]:
                    issue_count += 1
                    issue_rules[str(issue["rule"])] += 1
                    issue_repos[str(issue["repo"])] += 1
                    log_f.write(json.dumps(issue, ensure_ascii=False, sort_keys=True) + "\n")

    summary = {
        "root": str(args.root),
        "only": args.only,
        "repos": len(repos),
        "annotations": total_annos,
        "subtasks": total_subtasks,
        "issues": issue_count,
        "issue_rules": dict(issue_rules.most_common()),
        "issue_repos": dict(issue_repos.most_common(50)),
        "jsonl_log": str(jsonl_path),
    }
    write_json(summary_path, summary)
    with text_summary_path.open("w", encoding="utf-8") as f:
        f.write(f"RoboTwin annotation evaluation\n")
        f.write(f"root: {args.root}\n")
        f.write(f"repos: {len(repos)}\n")
        f.write(f"annotations: {total_annos}\n")
        f.write(f"subtasks: {total_subtasks}\n")
        f.write(f"issues: {issue_count}\n")
        f.write(f"jsonl_log: {jsonl_path}\n")
        f.write("\nissues_by_rule:\n")
        for rule, count in issue_rules.most_common():
            f.write(f"  {rule}: {count}\n")

    print(f"evaluated {total_annos} annotations / {total_subtasks} subtasks")
    print(f"found {issue_count} issues")
    print(f"wrote {jsonl_path}")
    print(f"wrote {summary_path}")
    if args.fail_on_issues and issue_count:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
