"""Motion-based subtask progress from robot state trajectories."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

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
ANNOTATION_ARM_FLIP = {
    "left": "right",
    "right": "left",
}
MOTION_EPS = 1e-12


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
            self.trans_total > MOTION_EPS
            or self.rot_total > MOTION_EPS
            or self.grip_total > MOTION_EPS
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


def _component_progress(value: float, total: float) -> float | None:
    if total <= MOTION_EPS:
        return None
    return max(0.0, min(1.0, value / total))


def progress_from_curve(curve: SubtaskProgressCurve, offset: int) -> float | None:
    offset = max(0, min(offset, len(curve.trans) - 1))
    parts = [
        _component_progress(float(curve.trans[offset]), curve.trans_total),
        _component_progress(float(curve.rot[offset]), curve.rot_total),
        _component_progress(float(curve.grip[offset]), curve.grip_total),
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
