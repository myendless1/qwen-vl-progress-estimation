"""Motion-based subtask progress from robot state trajectories."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

LEFT_GRIPPER_DIM = 7
RIGHT_GRIPPER_DIM = 15
ARM_XYZ_DIMS = {
    "left": slice(0, 3),
    "right": slice(8, 11),
}
ARM_ORIENT_DIMS = {
    "left": slice(3, 7),
    "right": slice(11, 15),
}
GRIPPER_DIMS = {
    "left": LEFT_GRIPPER_DIM,
    "right": RIGHT_GRIPPER_DIM,
}
STATE_PROMPT_DIMS = {
    "left_gripper": LEFT_GRIPPER_DIM,
    "right_gripper": RIGHT_GRIPPER_DIM,
    "left_z": 2,
    "right_z": 10,
}
ANNOTATION_ARM_FLIP = {
    "left": "right",
    "right": "left",
}
MOTION_EPS = 1e-12
DONE_PROGRESS_THRESHOLD = 0.995
TRANSLATION_PROGRESS_MIN_TOTAL = 0.01
ROTATION_PROGRESS_MIN_TOTAL = float(np.deg2rad(2.0))
GRIPPER_PROGRESS_MIN_TOTAL = 0.05


def episode_parquet_path(repo_dir: Path, episode_index: int, chunks_size: int) -> Path:
    chunk = episode_index // chunks_size
    rel = Path(f"chunk-{chunk:03d}") / f"episode_{episode_index:06d}.parquet"
    for data_dir in ("data-lerobot", "data"):
        candidate = repo_dir / data_dir / rel
        if candidate.exists():
            return candidate
    return repo_dir / "data" / rel


def load_episode_states(path: Path) -> np.ndarray:
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=["observation.state"])
    return np.asarray(table.column("observation.state").to_pylist(), dtype=np.float32)


def state_prompt_values(states: np.ndarray | None, frame: int) -> Dict[str, float] | None:
    if states is None or len(states) == 0:
        return None
    frame = max(0, min(int(frame), len(states) - 1))
    state = states[frame]
    if state.shape[0] <= max(STATE_PROMPT_DIMS.values()):
        return None
    return {name: float(state[dim]) for name, dim in STATE_PROMPT_DIMS.items()}


def annotation_arm_to_state_arm(anno: Mapping[str, Any], arm: str) -> str:
    mapping = str(anno.get("metadata", {}).get("arm_label_mapping", ""))
    if "flipped" in mapping:
        return ANNOTATION_ARM_FLIP.get(arm, arm)
    return arm


def arm_mentions(text: str) -> set[str]:
    lowered = text.lower()
    arms: set[str] = set()
    if re.search(r"\bleft arm\b", lowered):
        arms.add("left")
    if re.search(r"\bright arm\b", lowered):
        arms.add("right")
    if re.search(
        r"\b(?:both arms|dual arms|both grippers|grippers of both arms|both objects)\b",
        lowered,
    ):
        arms.update({"left", "right"})
    return arms


def active_annotation_arms(subtask: Mapping[str, Any]) -> set[str]:
    subtask_type = str(subtask.get("subtask_type", "")).strip()
    if subtask_type.startswith("dual_"):
        return {"left", "right"}
    mentions = arm_mentions(str(subtask.get("subtask_goal", "")))
    if mentions:
        return mentions
    if subtask_type in {"move", "press", "final"}:
        return {"left", "right"}
    return {"left", "right"}


def active_state_arms(subtask: Mapping[str, Any], anno: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                annotation_arm_to_state_arm(anno, arm)
                for arm in active_annotation_arms(subtask)
            }
        )
    )


def _normalize_quaternion(quaternion: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(quaternion))
    if norm <= MOTION_EPS:
        return quaternion
    return quaternion / norm


def _quaternion_geodesic_step(quaternion_prev: np.ndarray, quaternion_curr: np.ndarray) -> float:
    """Geodesic rotation angle between consecutive unit quaternions."""
    prev = _normalize_quaternion(quaternion_prev)
    curr = _normalize_quaternion(quaternion_curr)
    dot = abs(float(np.dot(prev, curr)))
    dot = min(1.0, max(-1.0, dot))
    return float(2.0 * np.arccos(dot))


def _step_translation(states: np.ndarray, frame: int, state_arms: Sequence[str]) -> float:
    if frame <= 0 or frame >= len(states):
        return 0.0
    distance = 0.0
    for arm in state_arms:
        xyz_prev = states[frame - 1, ARM_XYZ_DIMS[arm]]
        xyz_curr = states[frame, ARM_XYZ_DIMS[arm]]
        distance += float(np.linalg.norm(xyz_curr - xyz_prev))
    return distance


def _step_rotation(states: np.ndarray, frame: int, state_arms: Sequence[str]) -> float:
    if frame <= 0 or frame >= len(states):
        return 0.0
    angle = 0.0
    for arm in state_arms:
        orient_prev = states[frame - 1, ARM_ORIENT_DIMS[arm]]
        orient_curr = states[frame, ARM_ORIENT_DIMS[arm]]
        angle += _quaternion_geodesic_step(orient_prev, orient_curr)
    return angle


def _step_gripper(states: np.ndarray, frame: int, state_arms: Sequence[str]) -> float:
    if frame <= 0 or frame >= len(states):
        return 0.0
    delta = 0.0
    for arm in state_arms:
        dim = GRIPPER_DIMS[arm]
        if states.shape[1] <= dim:
            continue
        delta += abs(float(states[frame, dim] - states[frame - 1, dim]))
    return delta


@dataclass(frozen=True)
class SubtaskProgressCurve:
    """Per-frame cumulative motion for each normalized progress component."""

    trans: np.ndarray
    rot: np.ndarray
    grip: np.ndarray

    @property
    def trans_total(self) -> float:
        return float(self.trans[-1])

    @property
    def rot_total(self) -> float:
        return float(self.rot[-1])

    @property
    def grip_total(self) -> float:
        return float(self.grip[-1])

    def has_motion(self) -> bool:
        return (
            self.trans_total >= TRANSLATION_PROGRESS_MIN_TOTAL
            or self.rot_total >= ROTATION_PROGRESS_MIN_TOTAL
            or self.grip_total >= GRIPPER_PROGRESS_MIN_TOTAL
        )


def build_subtask_progress_curve(
    states: np.ndarray,
    start: int,
    end: int,
    state_arms: Sequence[str],
) -> SubtaskProgressCurve:
    start = max(0, min(start, len(states) - 1))
    end = max(start, min(end, len(states) - 1))
    length = end - start + 1
    trans = np.zeros(length, dtype=np.float64)
    rot = np.zeros(length, dtype=np.float64)
    grip = np.zeros(length, dtype=np.float64)
    trans_total = 0.0
    rot_total = 0.0
    grip_total = 0.0
    for offset in range(1, length):
        frame = start + offset
        trans_total += _step_translation(states, frame, state_arms)
        rot_total += _step_rotation(states, frame, state_arms)
        grip_total += _step_gripper(states, frame, state_arms)
        trans[offset] = trans_total
        rot[offset] = rot_total
        grip[offset] = grip_total
    return SubtaskProgressCurve(trans=trans, rot=rot, grip=grip)


def _component_progress(value: float, total: float, min_total: float = MOTION_EPS) -> float | None:
    if total <= MOTION_EPS or total < min_total:
        return None
    return max(0.0, min(1.0, value / total))


def progress_from_curve(curve: SubtaskProgressCurve, offset: int) -> float | None:
    offset = max(0, min(offset, len(curve.trans) - 1))
    parts = [
        _component_progress(
            float(curve.trans[offset]),
            curve.trans_total,
            TRANSLATION_PROGRESS_MIN_TOTAL,
        ),
        _component_progress(
            float(curve.rot[offset]),
            curve.rot_total,
            ROTATION_PROGRESS_MIN_TOTAL,
        ),
        _component_progress(
            float(curve.grip[offset]),
            curve.grip_total,
            GRIPPER_PROGRESS_MIN_TOTAL,
        ),
    ]
    active = [part for part in parts if part is not None]
    if not active:
        return None
    return max(0.0, min(1.0, float(sum(active) / len(active))))


def time_progress_for_subtask(subtask: Mapping[str, Any], frame: int) -> float:
    start = int(subtask["start_frame"])
    end = int(subtask["end_frame"])
    denom = max(1, end - start)
    if frame <= start:
        return 0.0
    if frame >= end:
        return 1.0
    return max(0.0, min(1.0, (frame - start) / denom))


def progress_for_subtask(
    subtask: Mapping[str, Any],
    frame: int,
    *,
    states: np.ndarray | None = None,
    anno: Mapping[str, Any] | None = None,
    curve: SubtaskProgressCurve | None = None,
) -> float:
    start = int(subtask["start_frame"])
    end = int(subtask["end_frame"])
    if frame <= start:
        return 0.0
    if frame >= end:
        return 1.0

    if curve is None:
        if states is None or anno is None:
            return time_progress_for_subtask(subtask, frame)
        curve = build_subtask_progress_curve(states, start, end, active_state_arms(subtask, anno))

    if not curve.has_motion():
        return time_progress_for_subtask(subtask, frame)

    progress = progress_from_curve(curve, frame - start)
    if progress is None:
        return time_progress_for_subtask(subtask, frame)
    return progress


def motion_progress_for_subtask(
    subtask: Mapping[str, Any],
    frame: int,
    *,
    states: np.ndarray | None = None,
    anno: Mapping[str, Any] | None = None,
    curve: SubtaskProgressCurve | None = None,
) -> float:
    """Motion-based progress without forcing 1.0 at frame >= end."""
    start = int(subtask["start_frame"])
    end = int(subtask["end_frame"])
    if frame <= start:
        return 0.0

    if curve is None:
        if states is None or anno is None:
            return time_progress_for_subtask(subtask, frame)
        curve = build_subtask_progress_curve(states, start, end, active_state_arms(subtask, anno))

    if not curve.has_motion():
        return time_progress_for_subtask(subtask, frame)

    offset = max(0, min(frame - start, len(curve.trans) - 1))
    progress = progress_from_curve(curve, offset)
    if progress is None:
        return time_progress_for_subtask(subtask, frame)
    return progress


def current_done_frame_indices(
    subtask: Mapping[str, Any],
    num_frames: int,
    *,
    states: np.ndarray | None = None,
    anno: Mapping[str, Any] | None = None,
    curve: SubtaskProgressCurve | None = None,
    threshold: float = DONE_PROGRESS_THRESHOLD,
) -> list[int]:
    """Frames marked done by scanning backward from end until progress < threshold."""
    start = int(subtask["start_frame"])
    end = min(int(subtask["end_frame"]), num_frames - 1)
    if start >= num_frames or end < start:
        return []

    done_frames: list[int] = []
    for frame in range(end, start - 1, -1):
        progress = motion_progress_for_subtask(
            subtask,
            frame,
            states=states,
            anno=anno,
            curve=curve,
        )
        if progress + 1e-9 >= threshold:
            done_frames.append(frame)
        else:
            break

    if not done_frames:
        done_frames = [end]
    done_frames.reverse()
    return [frame for frame in done_frames if 0 <= frame < num_frames]


def q1_plan_start_index(
    current_subtask_index: int,
    frame_index: int,
    num_subtasks: int,
    done_start_frame: int | None,
) -> int:
    """Return the first subtask Q1 should include in the remaining plan."""
    if done_start_frame is not None and frame_index >= done_start_frame:
        return min(current_subtask_index + 1, num_subtasks)
    return current_subtask_index


def select_frames_by_progress_bucket(
    frame_progress: Sequence[tuple[int, float]],
    bucket_size: float = 0.01,
) -> List[int]:
    """Pick one frame per progress bucket.

    For each bucket [k * bucket_size, (k + 1) * bucket_size), keep the frame whose
    progress is closest to the bucket center. Ties prefer the later frame.
    """
    if bucket_size <= 0:
        return [frame for frame, _ in frame_progress]
    if not frame_progress:
        return []

    buckets: Dict[int, List[tuple[int, float]]] = defaultdict(list)
    max_bucket = max(0, int(1.0 / bucket_size) - 1)
    for frame, progress in frame_progress:
        if progress >= 1.0:
            continue
        bucket = min(max_bucket, int(progress / bucket_size))
        buckets[bucket].append((frame, progress))

    selected: List[int] = []
    for bucket in sorted(buckets):
        center = (bucket + 0.5) * bucket_size
        frame, _ = min(buckets[bucket], key=lambda item: (abs(item[1] - center), -item[0]))
        selected.append(frame)
    return selected


def select_undone_frame_indices(
    start: int,
    not_done_end: int,
    *,
    subtask: Mapping[str, Any],
    states: np.ndarray | None = None,
    anno: Mapping[str, Any] | None = None,
    curve: SubtaskProgressCurve | None = None,
    q2_frame_stride: int = 1,
    q2_progress_bucket_size: float = 0.0,
) -> List[int]:
    """Select undone Q2 frame indices via progress buckets or frame stride."""
    if not_done_end < start:
        return []

    all_frames = list(range(start, not_done_end + 1))
    if q2_progress_bucket_size > 0:
        frame_progress = [
            (
                frame,
                progress_for_subtask(subtask, frame, states=states, anno=anno, curve=curve),
            )
            for frame in all_frames
        ]
        return select_frames_by_progress_bucket(frame_progress, q2_progress_bucket_size)
    return all_frames[:: max(1, q2_frame_stride)]


def build_subtask_progress_lookup(
    states: np.ndarray,
    subtasks: Sequence[Mapping[str, Any]],
    anno: Mapping[str, Any],
) -> Dict[int, SubtaskProgressCurve]:
    lookup: Dict[int, SubtaskProgressCurve] = {}
    for subtask in subtasks:
        start = int(subtask["start_frame"])
        end = int(subtask["end_frame"])
        lookup[start] = build_subtask_progress_curve(
            states,
            start,
            end,
            active_state_arms(subtask, anno),
        )
    return lookup
