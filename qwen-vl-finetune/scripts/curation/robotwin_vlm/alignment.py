"""Gripper-event detection, frame alignment, and annotation post-processing."""

from __future__ import annotations

import math
import re
from typing import Any

import numpy as np

from .models import GripperEvent, StepSpec
from .task_rules import RETREAT_MERGE_TASKS


LEFT_GRIPPER_DIM = 7
RIGHT_GRIPPER_DIM = 15
ARM_XYZ_DIMS = {
    "left": slice(0, 3),
    "right": slice(8, 11),
}
BOTH_FULL_CLOSE_THRESHOLD = 0.01
GRASP_CLOSE_THRESHOLD = 0.1
RELEASE_OPEN_THRESHOLD = 0.9
ARM_LABEL_FLIP = {
    "left": "right",
    "right": "left",
}


def detect_gripper_events(
    states: np.ndarray,
    threshold: float = 0.5,
    min_gap: int = 3,
) -> list[GripperEvent]:
    events: list[GripperEvent] = []
    for dim, arm in ((LEFT_GRIPPER_DIM, "left"), (RIGHT_GRIPPER_DIM, "right")):
        if states.shape[1] <= dim:
            continue
        gripper = states[:, dim]
        is_open = gripper > threshold
        transitions = np.flatnonzero(is_open[1:] != is_open[:-1]) + 1
        last_frame = -10**9
        for frame in transitions.tolist():
            if frame - last_frame < min_gap:
                continue
            kind = "open" if bool(is_open[frame]) else "close"
            events.append(GripperEvent(frame=int(frame), arm=arm, kind=kind))
            last_frame = int(frame)
    return sorted(events, key=lambda event: (event.frame, event.arm, event.kind))


def pick_event(
    events: list[GripperEvent],
    event_kind: str,
    start_after: int,
    arm: str | None,
    used: set[int],
    prefer_specified_arm: bool = True,
) -> GripperEvent | None:
    if not prefer_specified_arm:
        for idx, event in enumerate(events):
            if idx in used or event.frame <= start_after or event.kind != event_kind:
                continue
            used.add(idx)
            return event
    for idx, event in enumerate(events):
        if idx in used or event.frame <= start_after or event.kind != event_kind:
            continue
        if arm is not None and event.arm != arm:
            continue
        used.add(idx)
        return event
    if arm is not None:
        for idx, event in enumerate(events):
            if idx in used or event.frame <= start_after or event.kind != event_kind:
                continue
            used.add(idx)
            return event
    return None


def peek_next_event(
    events: list[GripperEvent],
    start_after: int,
    used: set[int],
    event_kind: str | None = None,
) -> GripperEvent | None:
    for idx, event in enumerate(events):
        if idx in used or event.frame <= start_after:
            continue
        if event_kind is not None and event.kind != event_kind:
            continue
        return event
    return None


def find_both_full_close_frame(
    states: np.ndarray | None,
    start_after: int,
    threshold: float = BOTH_FULL_CLOSE_THRESHOLD,
) -> int | None:
    if states is None or states.shape[1] <= RIGHT_GRIPPER_DIM:
        return None
    left = states[:, LEFT_GRIPPER_DIM]
    right = states[:, RIGHT_GRIPPER_DIM]
    candidates = np.flatnonzero((left <= threshold) & (right <= threshold))
    candidates = candidates[candidates > start_after]
    if len(candidates) == 0:
        return None
    return int(candidates[0])


def find_strict_grasp_close_frame(
    states: np.ndarray | None,
    events: list[GripperEvent],
    matched_event: GripperEvent,
    *,
    threshold: float,
    both_arms: bool,
) -> int | None:
    if states is None or states.shape[1] <= RIGHT_GRIPPER_DIM:
        return None
    arms = ["left", "right"] if both_arms else [matched_event.arm]
    close_frames: list[int] = []
    for arm in arms:
        dim = LEFT_GRIPPER_DIM if arm == "left" else RIGHT_GRIPPER_DIM
        start = matched_event.frame
        later_same_arm_events = [event.frame for event in events if event.arm == arm and event.frame > start]
        stop = min(later_same_arm_events) if later_same_arm_events else len(states) - 1
        if stop <= start:
            return None
        gripper = states[start : stop + 1, dim]
        candidates = np.flatnonzero(gripper <= threshold)
        if len(candidates) == 0:
            return None
        close_frames.append(start + int(candidates[0]))
    return max(close_frames) if close_frames else None


def find_strict_release_open_frame(
    states: np.ndarray | None,
    events: list[GripperEvent],
    matched_event: GripperEvent,
    *,
    threshold: float,
    both_arms: bool,
) -> int | None:
    if states is None or states.shape[1] <= RIGHT_GRIPPER_DIM:
        return None
    arms = ["left", "right"] if both_arms else [matched_event.arm]
    open_frames: list[int] = []
    for arm in arms:
        dim = LEFT_GRIPPER_DIM if arm == "left" else RIGHT_GRIPPER_DIM
        start = matched_event.frame
        later_same_arm_events = [event.frame for event in events if event.arm == arm and event.frame > start]
        stop = min(later_same_arm_events) if later_same_arm_events else len(states) - 1
        if stop <= start:
            return None
        gripper = states[start : stop + 1, dim]
        candidates = np.flatnonzero(gripper >= threshold)
        if len(candidates) == 0:
            return None
        open_frames.append(start + int(candidates[0]))
    return max(open_frames) if open_frames else None


def arm_context_from_text(text: str) -> str | None:
    has_left = bool(re.search(r"\bleft arm\b", text))
    has_right = bool(re.search(r"\bright arm\b", text))
    if has_left and has_right:
        return "both"
    if has_left:
        return "left"
    if has_right:
        return "right"
    return None


def arm_context_from_boundary(boundary_source: str) -> str | None:
    match = re.search(r"_(left|right)_", boundary_source)
    return match.group(1) if match else None


def append_arm_to_text(text: str, arm: str | None) -> str:
    if arm not in {"left", "right"} or arm_context_from_text(text) is not None:
        return text
    if text.endswith("."):
        return f"{text[:-1]} with the {arm} arm."
    return f"{text} with the {arm} arm"


def align_text_with_event_arm(text: str, event: GripperEvent | None) -> str:
    if event is None:
        return text
    if arm_context_from_text(text) == "both":
        return text
    if re.search(r"\bfirst arm\b", text) and re.search(r"\breceiving arm\b", text):
        return text
    text = re.sub(r"\bthe (first|receiving) arm\b", f"the {event.arm} arm", text)
    text = re.sub(r"\b(first|receiving) arm\b", f"{event.arm} arm", text)
    text = re.sub(r"\b(left|right) arm\b", f"{event.arm} arm", text)
    text = re.sub(r"\bthe other arm\b", f"the {event.arm} arm", text)
    return append_arm_to_text(text, event.arm)


def ensure_arm_mentions(spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    last_context: str | None = None
    transfer_first_arm: str | None = None
    transfer_receiving_arm: str | None = None
    output: list[dict[str, Any]] = []
    for span in spans:
        span = dict(span)
        text = str(span.get("subtask_goal", ""))
        boundary_context = arm_context_from_boundary(str(span.get("boundary_source", "")))
        if "first arm" in text and transfer_first_arm:
            text = re.sub(r"\bthe first arm\b", f"the {transfer_first_arm} arm", text)
            text = re.sub(r"\bfirst arm\b", f"{transfer_first_arm} arm", text)
        if "receiving arm" in text and transfer_receiving_arm:
            text = re.sub(r"\bthe receiving arm\b", f"the {transfer_receiving_arm} arm", text)
            text = re.sub(r"\breceiving arm\b", f"{transfer_receiving_arm} arm", text)
        text_context = arm_context_from_text(text)
        context = text_context or boundary_context or last_context
        if text_context is None and context in {"left", "right"}:
            text = append_arm_to_text(text, context)
            span["subtask_goal"] = text
            text_context = context
        else:
            span["subtask_goal"] = text
        lowered = text.lower()
        if boundary_context in {"left", "right"}:
            if (lowered.startswith("pick up") or lowered.startswith("grasp")) and transfer_first_arm is None:
                transfer_first_arm = boundary_context
            elif lowered.startswith("grasp") and transfer_first_arm and boundary_context != transfer_first_arm:
                transfer_receiving_arm = boundary_context
        if text_context in {"left", "right", "both"}:
            last_context = text_context
        elif boundary_context in {"left", "right"}:
            last_context = boundary_context
        output.append(span)
    return output


def assign_spans(
    steps: list[StepSpec],
    events: list[GripperEvent],
    n_frames: int,
    states: np.ndarray | None = None,
    grasp_close_threshold: float = GRASP_CLOSE_THRESHOLD,
    release_open_threshold: float = RELEASE_OPEN_THRESHOLD,
    prefer_specified_arm: bool = True,
) -> list[dict[str, Any]]:
    if n_frames <= 0:
        raise ValueError("episode has no frames")
    spans: list[dict[str, Any]] = []
    used: set[int] = set()
    prev_end = -1
    for i, step in enumerate(steps):
        start = 0 if i == 0 else min(prev_end + 1, n_frames - 1)
        is_last = i == len(steps) - 1
        end: int | None = None
        matched_event: GripperEvent | None = None
        boundary_source: str | None = None
        if step.event_kind in {"close", "open"}:
            matched_event = pick_event(
                events,
                step.event_kind,
                prev_end,
                step.arm,
                used,
                prefer_specified_arm=prefer_specified_arm,
            )
            if matched_event is not None and not is_last:
                end = matched_event.frame
                if step.event_kind == "close":
                    strict_close_frame = find_strict_grasp_close_frame(
                        states=states,
                        events=events,
                        matched_event=matched_event,
                        threshold=grasp_close_threshold,
                        both_arms=arm_context_from_text(step.text) == "both",
                    )
                    if strict_close_frame is not None:
                        end = strict_close_frame
                        boundary_source = (
                            "gripper_both_strict_close"
                            if arm_context_from_text(step.text) == "both"
                            else f"gripper_{matched_event.arm}_strict_close"
                        )
                else:
                    strict_open_frame = find_strict_release_open_frame(
                        states=states,
                        events=events,
                        matched_event=matched_event,
                        threshold=release_open_threshold,
                        both_arms=arm_context_from_text(step.text) == "both",
                    )
                    if strict_open_frame is not None:
                        end = strict_open_frame
                        boundary_source = (
                            "gripper_both_strict_open"
                            if arm_context_from_text(step.text) == "both"
                            else f"gripper_{matched_event.arm}_strict_open"
                        )
        elif step.event_kind == "both_full_close":
            full_close_frame = find_both_full_close_frame(states, prev_end)
            if full_close_frame is not None and not is_last:
                end = full_close_frame
                boundary_source = "gripper_both_full_close"
        elif step.event_kind == "midpoint":
            next_event = peek_next_event(events, prev_end, used)
            if next_event is not None and not is_last:
                end = max(start, (prev_end + next_event.frame) // 2)
                boundary_source = f"midpoint_before_gripper_{next_event.arm}_{next_event.kind}"
        if end is None:
            if is_last or step.event_kind == "final":
                end = n_frames - 1
            else:
                remaining_steps = len(steps) - i
                remaining_frames = max(1, n_frames - start)
                end = min(n_frames - 1, start + max(1, math.floor(remaining_frames / remaining_steps)) - 1)
        if end < start:
            end = start
        if is_last:
            end = n_frames - 1
        spans.append(
            {
                "subtask_index": i,
                "subtask_goal": align_text_with_event_arm(step.text, matched_event),
                "start_frame": int(start),
                "end_frame": int(end),
                "boundary_source": (
                    boundary_source
                    if boundary_source is not None
                    else f"gripper_{matched_event.arm}_{matched_event.kind}"
                    if matched_event is not None
                    else ("episode_end" if is_last or step.event_kind == "final" else "uniform_fallback")
                ),
            }
        )
        prev_end = end
    return ensure_arm_mentions(spans)


def estimate_retreat_end_frame(
    states: np.ndarray,
    arm: str,
    start_frame: int,
    end_before: int,
    min_gap: int,
    min_displacement: float,
    still_threshold: float,
    still_window: int,
) -> int | None:
    if arm not in ARM_XYZ_DIMS:
        return None
    start = start_frame + 1
    stop = end_before - 1
    if stop - start + 1 < min_gap:
        return None
    xyz = states[:, ARM_XYZ_DIMS[arm]]
    speed = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    release_xyz = xyz[start_frame]
    interval_xyz = xyz[start : stop + 1]
    if len(interval_xyz) == 0:
        return None
    displacement = np.linalg.norm(interval_xyz - release_xyz, axis=1)
    if float(displacement.max(initial=0.0)) < min_displacement:
        return None
    segment_speed = speed[start:stop]
    if len(segment_speed) == 0:
        return None
    moving_threshold = max(still_threshold * 2.5, float(np.percentile(segment_speed, 90)) * 0.2)
    moving_offsets = np.flatnonzero(segment_speed > moving_threshold)
    if len(moving_offsets) == 0:
        return None
    last_motion_frame = start + int(moving_offsets[-1])
    search_start = max(last_motion_frame + 1, start + still_window)
    for frame in range(search_start, stop + 1):
        lo = max(0, frame - still_window)
        hi = min(len(speed), frame)
        if hi > lo and float(speed[lo:hi].mean()) <= still_threshold:
            return frame
    fallback = end_before - max(2, still_window // 2)
    if fallback > start:
        return fallback
    return None


def insert_retreat_subtasks(
    spans: list[dict[str, Any]],
    states: np.ndarray,
    *,
    min_gap: int = 20,
    min_displacement: float = 0.05,
    still_threshold: float = 0.002,
    still_window: int = 6,
) -> list[dict[str, Any]]:
    if len(spans) < 2:
        return spans
    output: list[dict[str, Any]] = []
    inserted = 0
    for idx, span in enumerate(spans):
        span = dict(span)
        output.append(span)
        if idx + 1 >= len(spans):
            continue
        source = str(span.get("boundary_source", ""))
        match = re.fullmatch(r"gripper_(left|right)_(?:strict_)?open", source)
        if not match:
            continue
        next_span = spans[idx + 1]
        next_source = str(next_span.get("boundary_source", ""))
        if not re.fullmatch(r"gripper_(left|right)_(?:strict_)?close", next_source):
            continue
        arm = match.group(1)
        release_frame = int(span["end_frame"])
        next_close_frame = int(next_span["end_frame"])
        retreat_end = estimate_retreat_end_frame(
            states=states,
            arm=arm,
            start_frame=release_frame,
            end_before=next_close_frame,
            min_gap=min_gap,
            min_displacement=min_displacement,
            still_threshold=still_threshold,
            still_window=still_window,
        )
        if retreat_end is None:
            continue
        retreat_start = release_frame + 1
        if retreat_end < retreat_start or retreat_end >= next_close_frame:
            continue
        output.append(
            {
                "subtask_index": -1,
                "subtask_goal": f"Return the {arm} arm to a neutral pose after releasing the object.",
                "start_frame": int(retreat_start),
                "end_frame": int(retreat_end),
                "boundary_source": f"eef_{arm}_retreat_velocity",
            }
        )
        inserted += 1
        spans[idx + 1] = dict(next_span)
        spans[idx + 1]["start_frame"] = int(retreat_end + 1)
    if inserted == 0:
        return spans
    for new_index, span in enumerate(output):
        span["subtask_index"] = new_index
    return output


def merge_stack_arm_switches(spans: list[dict[str, Any]], slug: str) -> list[dict[str, Any]]:
    if slug not in RETREAT_MERGE_TASKS or len(spans) < 2:
        return spans
    output = [dict(span) for span in spans]
    for idx in range(len(output) - 1):
        source = str(output[idx].get("boundary_source", ""))
        next_source = str(output[idx + 1].get("boundary_source", ""))
        release = re.fullmatch(r"gripper_(left|right)_(?:strict_)?open", source)
        pickup = re.fullmatch(r"gripper_(left|right)_(?:strict_)?close", next_source)
        if not release or not pickup:
            continue
        released_arm = release.group(1)
        pickup_arm = pickup.group(1)
        if released_arm == pickup_arm:
            continue
        text = str(output[idx + 1].get("subtask_goal", ""))
        if re.search(r"\breturn\b.+\bneutral pose\b", text, flags=re.IGNORECASE):
            continue
        if text.endswith("."):
            text = text[:-1]
        concurrent_text = re.sub(r"^Grasp\b", "grasping", text)
        concurrent_text = re.sub(r"^Pick up\b", "picking up", concurrent_text)
        if concurrent_text == text and concurrent_text:
            concurrent_text = concurrent_text[0].lower() + concurrent_text[1:]
        output[idx + 1]["subtask_goal"] = (
            f"Return the {released_arm} arm to a neutral pose while {concurrent_text}."
            if concurrent_text
            else f"Return the {released_arm} arm to a neutral pose while the {pickup_arm} arm grasps the next object."
        )
    return output


def flip_arm_label(arm: str | None) -> str | None:
    if arm is None:
        return None
    return ARM_LABEL_FLIP.get(arm, arm)


def flip_arm_mentions(text: str) -> str:
    placeholders = {"left arm": "__LEFT_ARM__", "right arm": "__RIGHT_ARM__"}
    for old, placeholder in placeholders.items():
        text = re.sub(rf"\b{old}\b", placeholder, text, flags=re.IGNORECASE)
    return text.replace("__LEFT_ARM__", "right arm").replace("__RIGHT_ARM__", "left arm")


def flip_boundary_arm_label(source: str) -> str:
    source = re.sub(r"(?<=_)left(?=_)", "__LEFT__", source)
    source = re.sub(r"(?<=_)right(?=_)", "__RIGHT__", source)
    return source.replace("__LEFT__", "right").replace("__RIGHT__", "left")


def publish_actual_arm_labels(anno: dict[str, Any]) -> dict[str, Any]:
    anno["task_goal"] = flip_arm_mentions(str(anno.get("task_goal", "")))
    for subtask in anno.get("subtasks", []):
        subtask["subtask_goal"] = flip_arm_mentions(str(subtask.get("subtask_goal", "")))
        subtask["boundary_source"] = flip_boundary_arm_label(str(subtask.get("boundary_source", "")))
    metadata = anno.get("metadata", {})
    for event in metadata.get("detected_gripper_events", []):
        if "arm" in event:
            event["arm"] = flip_arm_label(str(event["arm"]))
    scene_info = metadata.get("scene_info")
    if isinstance(scene_info, dict):
        for key, value in list(scene_info.items()):
            if value in ARM_LABEL_FLIP:
                scene_info[key] = ARM_LABEL_FLIP[value]
    metadata["arm_label_mapping"] = (
        "annotation arm labels are flipped from state indices: "
        "state-left is actual right, state-right is actual left"
    )
    return anno
