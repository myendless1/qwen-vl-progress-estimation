"""Coarse RoboTwin alignment from state motion and gripper events."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from .alignment import (
    ARM_ORIENT_DIMS,
    ARM_XYZ_DIMS,
    ORIENTATION_MOTION_START_THRESHOLD,
    append_arm_to_text,
    arm_context_from_text,
    eef_step_orientation_displacement,
    step_uses_both_arms,
)
from .models import GripperEvent, StepSpec
from .primitives import height_above_table_cm


COARSE_MOTION_THRESHOLD = 0.0015
COARSE_MOTION_MIN_RUN = 2
COARSE_GAP_MERGE_FRAMES = 2
COARSE_DUAL_EVENT_MAX_GAP = 3


@dataclass(frozen=True)
class MotionSegment:
    start: int
    end: int
    arms: frozenset[str]


@dataclass(frozen=True)
class GripperGroup:
    start: int
    end: int
    frame: int
    kind: str
    arms: frozenset[str]


def _step_moving(states: np.ndarray, arm: str, frame: int) -> bool:
    if frame <= 0 or frame >= len(states):
        return False
    xyz = states[:, ARM_XYZ_DIMS[arm]]
    displacement = float(np.linalg.norm(xyz[frame] - xyz[frame - 1]))
    if displacement > COARSE_MOTION_THRESHOLD:
        return True
    orientation = eef_step_orientation_displacement(states, arm, frame)
    return orientation is not None and orientation > ORIENTATION_MOTION_START_THRESHOLD


def _raw_active_sets(states: np.ndarray) -> list[frozenset[str]]:
    active: list[frozenset[str]] = [frozenset()]
    for frame in range(1, len(states)):
        arms = frozenset(arm for arm in ("left", "right") if _step_moving(states, arm, frame))
        active.append(arms)
    return active


def _sustain_active_sets(active: list[frozenset[str]]) -> list[frozenset[str]]:
    output = list(active)
    for frame, arms in enumerate(active):
        if not arms:
            continue
        stop = min(len(active), frame + COARSE_MOTION_MIN_RUN)
        if any(active[item] & arms for item in range(frame, stop)):
            for item in range(frame, stop):
                output[item] = output[item] | arms
    return output


def motion_segments(states: np.ndarray) -> list[MotionSegment]:
    if len(states) <= 1:
        return []
    active = _sustain_active_sets(_raw_active_sets(states))
    segments: list[MotionSegment] = []
    start: int | None = None
    current = frozenset()
    last_active = -1
    for frame, arms in enumerate(active[1:], start=1):
        if not arms:
            if start is not None and frame - last_active > COARSE_GAP_MERGE_FRAMES:
                segments.append(MotionSegment(max(0, start - 1), last_active, current))
                start = None
                current = frozenset()
            continue
        if start is None:
            start = frame
            current = arms
        elif arms != current:
            segments.append(MotionSegment(max(0, start - 1), frame - 1, current))
            start = frame
            current = arms
        last_active = frame
    if start is not None:
        segments.append(MotionSegment(max(0, start - 1), last_active, current))
    return _merge_short_gaps(segments)


def _merge_short_gaps(segments: list[MotionSegment]) -> list[MotionSegment]:
    if not segments:
        return []
    output = [segments[0]]
    for seg in segments[1:]:
        prev = output[-1]
        if seg.arms == prev.arms and seg.start - prev.end <= COARSE_GAP_MERGE_FRAMES + 1:
            output[-1] = MotionSegment(prev.start, seg.end, prev.arms)
        else:
            output.append(seg)
    return output


def gripper_groups(events: list[GripperEvent]) -> list[GripperGroup]:
    groups: list[GripperGroup] = []
    used: set[int] = set()
    for idx, event in enumerate(events):
        if idx in used:
            continue
        used.add(idx)
        paired = [event]
        for other_idx, other in enumerate(events):
            if other_idx in used:
                continue
            if other.kind != event.kind or other.arm == event.arm:
                continue
            if abs(other.frame - event.frame) > COARSE_DUAL_EVENT_MAX_GAP:
                continue
            used.add(other_idx)
            paired.append(other)
        groups.append(
            GripperGroup(
                start=min(item.start_frame if item.start_frame is not None else item.frame for item in paired),
                end=max(item.end_frame if item.end_frame is not None else item.frame for item in paired),
                frame=min(item.frame for item in paired),
                kind=event.kind,
                arms=frozenset(item.arm for item in paired),
            )
        )
    return sorted(groups, key=lambda item: (item.start, item.frame, sorted(item.arms)))


def _coalesce_motion_segments(
    segments: list[MotionSegment],
    groups: list[GripperGroup],
) -> list[MotionSegment]:
    if not segments:
        return []
    output: list[MotionSegment] = []
    idx = 0
    while idx < len(segments):
        current = segments[idx]
        while idx + 1 < len(segments) and segments[idx + 1].arms == current.arms:
            next_seg = segments[idx + 1]
            has_gripper_between = any(
                group.start > current.end and group.start < next_seg.start
                for group in groups
            )
            if has_gripper_between:
                break
            current = MotionSegment(current.start, next_seg.end, current.arms)
            idx += 1
        output.append(current)
        idx += 1
    return output


def _split_segments_at_grippers(
    segments: list[MotionSegment],
    groups: list[GripperGroup],
) -> list[MotionSegment]:
    output: list[MotionSegment] = []
    for segment in segments:
        cursor = segment.start
        cuts = sorted(
            {
                group.end
                for group in groups
                if segment.start <= group.end < segment.end
                and group.start <= segment.end
                and _compatible(segment, group)
            }
        )
        for cut in cuts:
            if cursor <= cut:
                output.append(MotionSegment(cursor, cut, segment.arms))
            cursor = cut + 1
        if cursor <= segment.end:
            output.append(MotionSegment(cursor, segment.end, segment.arms))
    return output


def _compatible(segment: MotionSegment, group: GripperGroup) -> bool:
    return bool(segment.arms & group.arms) or (len(segment.arms) == 2 and len(group.arms) == 1)


def _subtask_type(segment: MotionSegment, group: GripperGroup | None) -> str:
    prefix = "dual" if len(segment.arms) == 2 else "single"
    if group is None:
        return f"{prefix}_move"
    return f"{prefix}_{'grasp' if group.kind == 'close' else 'place'}"


def _span_type(arms: frozenset[str], group: GripperGroup | None) -> str:
    type_arms = group.arms if group is not None and len(group.arms) == 1 else arms
    prefix = "dual" if len(type_arms) == 2 else "single"
    if group is None:
        return f"{prefix}_move"
    return f"{prefix}_{'grasp' if group.kind == 'close' else 'place'}"


def _context_from_step(step: StepSpec) -> frozenset[str] | None:
    if arm_context_from_text(step.text) == "both" or step_uses_both_arms(step):
        return frozenset(("left", "right"))
    if step.arm in {"left", "right"}:
        return frozenset((step.arm,))
    context = arm_context_from_text(step.text)
    if context in {"left", "right"}:
        return frozenset((context,))
    return None


def _action_from_step(step: StepSpec) -> str:
    if step.event_kind == "handover":
        return "handover"
    if step.event_kind == "open_move":
        return "open_move"
    if step.event_kind in {"close"}:
        return "grasp"
    if step.event_kind in {"open"}:
        return "place"
    return "move"


def _hint_matches(step: StepSpec, subtask_type: str, arms: frozenset[str]) -> bool:
    action = _action_from_step(step)
    if action == "handover" and subtask_type.endswith("place"):
        context = _context_from_step(step)
        return context is None or bool(context & arms)
    if action not in subtask_type:
        return False
    context = _context_from_step(step)
    return context is None or context == arms


def _pick_hint(steps: list[StepSpec], start_idx: int, subtask_type: str, arms: frozenset[str]) -> tuple[str, int, str | None]:
    if subtask_type.endswith("move"):
        if start_idx < len(steps) and _hint_matches(steps[start_idx], subtask_type, arms):
            return steps[start_idx].text, start_idx + 1, steps[start_idx].event_kind
        return _generic_goal(subtask_type, arms), start_idx, None
    for idx in range(start_idx, len(steps)):
        if _hint_matches(steps[idx], subtask_type, arms):
            return steps[idx].text, idx + 1, steps[idx].event_kind
    if start_idx < len(steps):
        return steps[start_idx].text, start_idx + 1, steps[start_idx].event_kind
    return _generic_goal(subtask_type, arms), start_idx, None


def _generic_goal(subtask_type: str, arms: frozenset[str]) -> str:
    if len(arms) == 2:
        actor = "both arms"
    else:
        actor = f"the {next(iter(arms), 'active')} arm"
    if subtask_type.endswith("grasp"):
        return f"Grasp the target with {actor}."
    if subtask_type.endswith("place"):
        return f"Place the target with {actor}."
    if subtask_type == "open_move":
        return f"Open and retreat {actor}."
    return f"Move {actor}."


def _format_goal(text: str, arms: frozenset[str]) -> str:
    if len(arms) == 1:
        return append_arm_to_text(text, next(iter(arms)))
    return text


def _with_actual_retract_height(
    text: str,
    arms: frozenset[str],
    states: np.ndarray,
    frame: int,
) -> str:
    if len(arms) != 1 or "above the table" not in text or not text.lower().startswith("retract"):
        return text
    arm = next(iter(arms))
    if arm not in ARM_XYZ_DIMS:
        return text
    z_dim = ARM_XYZ_DIMS[arm].start + 2
    if frame < 0 or frame >= len(states):
        return text
    height_cm = max(0, height_above_table_cm(float(states[frame, z_dim])))
    return re.sub(
        r"\bto (?:at least|about) \d+ cm above the table\b",
        f"to about {height_cm} cm above the table",
        text,
        flags=re.IGNORECASE,
    )


def _single_arm(arms: frozenset[str]) -> str | None:
    if len(arms) == 1:
        arm = next(iter(arms))
        if arm in {"left", "right"}:
            return arm
    return None


def _terminal_group_for_segment(
    segment: MotionSegment,
    groups: list[GripperGroup],
    used_groups: set[int],
) -> tuple[int | None, GripperGroup | None]:
    for idx, group in enumerate(groups):
        if idx in used_groups:
            continue
        if group.end != segment.end:
            continue
        if _compatible(segment, group):
            return idx, group
    return None, None


def _terminal_group_before_next_motion(
    span_start: int,
    next_start: int,
    arms: frozenset[str],
    groups: list[GripperGroup],
    used_groups: set[int],
) -> tuple[int | None, GripperGroup | None]:
    probe = MotionSegment(span_start, max(span_start, next_start - 1), arms)
    for idx, group in enumerate(groups):
        if idx in used_groups:
            continue
        if group.start < span_start or group.start >= next_start:
            continue
        if _compatible(probe, group):
            return idx, group
    return None, None


def _pending_open_group_for_segment(
    segment: MotionSegment,
    groups: list[GripperGroup],
    used_groups: set[int],
    last_emitted_end: int,
) -> tuple[int | None, GripperGroup | None]:
    if len(segment.arms) != 1:
        return None, None
    for idx, group in enumerate(groups):
        if idx in used_groups:
            continue
        if group.kind != "open":
            continue
        if group.end <= last_emitted_end or group.start > segment.start:
            continue
        if group.arms == segment.arms:
            return idx, group
    return None, None


def assign_coarse_spans(
    steps: list[StepSpec],
    events: list[GripperEvent],
    n_frames: int,
    states: np.ndarray,
    *,
    merge_open_move_text: bool = True,
) -> list[dict[str, Any]]:
    if n_frames <= 0:
        raise ValueError("episode has no frames")
    groups = gripper_groups(events)
    segments = _split_segments_at_grippers(
        _coalesce_motion_segments(motion_segments(states), groups),
        groups,
    )
    used_groups: set[int] = set()
    spans: list[dict[str, Any]] = []
    hint_idx = 0
    span_start: int | None = None
    span_end = -1
    span_arms = frozenset()
    current_single: str | None = None
    last_emitted_end = -1
    last_terminal_kind: str | None = None

    def emit(end: int, terminal: GripperGroup | None, arms: frozenset[str]) -> None:
        nonlocal hint_idx, span_start, span_end, span_arms, current_single, last_emitted_end, last_terminal_kind
        if span_start is None:
            return
        type_arms = terminal.arms if terminal is not None and len(terminal.arms) == 1 else (arms or span_arms)
        pending_indices = [
            idx
            for idx, group in enumerate(groups)
            if idx not in used_groups
            and last_emitted_end < group.start <= span_start
            and bool(type_arms & group.arms)
        ]
        start = min([span_start] + [groups[idx].start for idx in pending_indices])
        subtask_type = _span_type(type_arms or span_arms, terminal)
        goal, hint_idx, hint_kind = _pick_hint(steps, hint_idx, subtask_type, type_arms or span_arms)
        if hint_kind == "handover":
            subtask_type = "handover"
        goal = _with_actual_retract_height(
            goal,
            type_arms or span_arms,
            states,
            int(min(n_frames - 1, max(end, start))),
        )
        for idx in pending_indices:
            used_groups.add(idx)
        spans.append(
            {
                "subtask_index": len(spans),
                "subtask_goal": _format_goal(goal, type_arms or span_arms),
                "subtask_type": subtask_type,
                "start_frame": int(max(last_emitted_end + 1, start)),
                "end_frame": int(min(n_frames - 1, max(end, start))),
                "boundary_source": (
                    f"after_gripper_{'_'.join(sorted(terminal.arms))}_{terminal.kind}"
                    if terminal is not None
                    else "single_arm_switch"
                ),
                "truncation_rule": (
                    f"gripper_{terminal.kind}"
                    if terminal is not None
                    else "single_arm_switch"
                ),
            }
        )
        last_emitted_end = spans[-1]["end_frame"]
        last_terminal_kind = terminal.kind if terminal is not None else None
        span_start = None
        span_end = -1
        span_arms = frozenset()
        current_single = None

    for seg_idx, segment in enumerate(segments):
        next_start = segments[seg_idx + 1].start if seg_idx + 1 < len(segments) else n_frames
        single = _single_arm(segment.arms)
        pending_open_idx, pending_open = _pending_open_group_for_segment(
            segment,
            groups,
            used_groups,
            last_emitted_end,
        )
        next_segment = segments[seg_idx + 1] if seg_idx + 1 < len(segments) else None
        if (
            span_start is None
            and pending_open is not None
            and next_segment is not None
            and bool(next_segment.arms - segment.arms)
        ):
            subtask_type = "open_move"
            goal, hint_idx, _hint_kind = _pick_hint(steps, hint_idx, subtask_type, segment.arms)
            used_groups.add(pending_open_idx)  # type: ignore[arg-type]
            spans.append(
                {
                    "subtask_index": len(spans),
                    "subtask_goal": _format_goal(goal, segment.arms),
                    "subtask_type": subtask_type,
                    "start_frame": int(max(last_emitted_end + 1, pending_open.start)),
                    "end_frame": int(min(n_frames - 1, max(next_segment.start - 1, pending_open.end))),
                    "boundary_source": "before_other_arm_motion",
                    "truncation_rule": "other_arm_motion",
                }
            )
            last_emitted_end = spans[-1]["end_frame"]
            continue
        if span_start is None:
            span_start = segment.start
            span_end = segment.end
            span_arms = segment.arms
            current_single = single
        else:
            prev_segment = segments[seg_idx - 1] if seg_idx > 0 else None
            prev_single = _single_arm(prev_segment.arms) if prev_segment is not None else None
            direct_single_switch = (
                single is not None
                and prev_single is not None
                and single != prev_single
                and current_single == prev_single
                and last_terminal_kind != "open"
            )
            if direct_single_switch:
                emit(segment.start - 1, None, frozenset((prev_single,)))
                span_start = segment.start
                span_end = segment.end
                span_arms = segment.arms
                current_single = single
            else:
                span_end = segment.end
                span_arms = span_arms | segment.arms
                if single is not None:
                    current_single = current_single or single

        terminal_idx, terminal = _terminal_group_before_next_motion(
            span_start if span_start is not None else segment.start,
            next_start,
            span_arms or segment.arms,
            groups,
            used_groups,
        )
        if terminal is not None:
            used_groups.add(terminal_idx)  # type: ignore[arg-type]
            emit(terminal.end, terminal, terminal.arms if len(terminal.arms) == 1 else span_arms)

    if span_start is not None:
        emit(span_end, None, span_arms)
    if not spans:
        spans.append(
            {
                "subtask_index": 0,
                "subtask_goal": steps[0].text if steps else "Complete the task.",
                "subtask_type": "single_move",
                "start_frame": 0,
                "end_frame": n_frames - 1,
                "boundary_source": "episode_end",
                "truncation_rule": "episode_end",
            }
        )
    if spans and int(spans[0]["start_frame"]) > 0:
        spans[0]["start_frame"] = 0
        spans[0]["gap_fill_policy"] = "extended_to_episode_start"
    for idx in range(len(spans) - 1):
        next_start = int(spans[idx + 1]["start_frame"])
        if int(spans[idx]["end_frame"]) < next_start - 1:
            spans[idx]["end_frame"] = next_start - 1
            spans[idx]["gap_fill_policy"] = "extended_to_next_subtask_start"
    if spans[-1]["truncation_rule"] == "single_arm_switch":
        spans[-1]["end_frame"] = n_frames - 1
        spans[-1]["boundary_source"] = "episode_end"
        spans[-1]["truncation_rule"] = "episode_end"
    spans = merge_open_move_handover_starts(spans)
    spans = merge_handover_followups(spans)
    spans = merge_open_state_moves_into_previous(spans, merge_text=merge_open_move_text)
    return merge_open_move_handover_starts(spans)


def _lower_initial(text: str) -> str:
    if not text:
        return text
    return text[0].lower() + text[1:]


def _append_while(prev: dict[str, Any], text: str) -> None:
    addon = text.rstrip(".")
    if not addon:
        return
    prev_text = str(prev.get("subtask_goal", ""))
    prev["subtask_goal"] = f"{prev_text.rstrip('.')} while {_lower_initial(addon)}."


def _same_single_arm_context(left_text: str, right_text: str) -> bool:
    left_arm = arm_context_from_text(left_text)
    right_arm = arm_context_from_text(right_text)
    return left_arm in {"left", "right"} and left_arm == right_arm


def _as_concurrent_clause(text: str) -> str:
    text = text.strip().rstrip(".")
    lowered = text.lower()
    replacements = (
        ("use ", "using "),
        ("move ", "moving "),
        ("place ", "placing "),
        ("put ", "putting "),
        ("drop ", "dropping "),
    )
    for prefix, replacement in replacements:
        if lowered.startswith(prefix):
            return replacement + _lower_initial(text[len(prefix):])
    return _lower_initial(text)


def _merge_handover_text(open_text: str, handover_text: str) -> str:
    open_text = open_text.strip().rstrip(".")
    if not open_text:
        return handover_text
    handover_clause = _as_concurrent_clause(handover_text)
    if not handover_clause:
        return open_text + "."
    return f"{open_text} while {handover_clause}."


def merge_open_move_handover_starts(spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(spans) < 2:
        return spans
    output: list[dict[str, Any]] = []
    idx = 0
    while idx < len(spans):
        span = dict(spans[idx])
        if (
            str(span.get("subtask_type", "")) == "open_move"
            and idx + 1 < len(spans)
            and str(spans[idx + 1].get("subtask_type", "")) == "handover"
        ):
            follow = dict(spans[idx + 1])
            follow["subtask_goal"] = _merge_handover_text(
                str(span.get("subtask_goal", "")),
                str(follow.get("subtask_goal", "")),
            )
            follow["start_frame"] = span.get("start_frame", follow.get("start_frame"))
            output.append(follow)
            idx += 2
            continue
        output.append(span)
        idx += 1
    for new_idx, item in enumerate(output):
        item["subtask_index"] = new_idx
    return output


def merge_handover_followups(spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(spans) < 2:
        return spans
    output: list[dict[str, Any]] = []
    idx = 0
    mergeable_types = {"single_place", "dual_place", "single_move", "dual_move", "open_move"}
    while idx < len(spans):
        span = dict(spans[idx])
        if (
            str(span.get("subtask_type", "")) == "handover"
            and idx + 1 < len(spans)
            and str(spans[idx + 1].get("subtask_type", "")) in mergeable_types
        ):
            follow = spans[idx + 1]
            if str(follow.get("subtask_type", "")) != "open_move":
                _append_while(span, str(follow.get("subtask_goal", "")))
            span["end_frame"] = follow.get("end_frame", span.get("end_frame"))
            span["boundary_source"] = follow.get("boundary_source", span.get("boundary_source"))
            span["truncation_rule"] = follow.get("truncation_rule", span.get("truncation_rule"))
            output.append(span)
            idx += 2
            continue
        output.append(span)
        idx += 1
    for new_idx, item in enumerate(output):
        item["subtask_index"] = new_idx
    return output


def merge_open_state_moves_into_previous(
    spans: list[dict[str, Any]],
    *,
    merge_text: bool = True,
) -> list[dict[str, Any]]:
    if len(spans) < 2:
        return spans
    output: list[dict[str, Any]] = []
    for span in spans:
        span = dict(span)
        if (
            output
            and str(span.get("subtask_type", "")) in {"single_move", "dual_move", "open_move"}
            and str(output[-1].get("subtask_type", "")) in {"single_place", "dual_place", "open_move", "handover"}
        ):
            text = str(span.get("subtask_goal", ""))
            prev = output[-1]
            if merge_text and not _same_single_arm_context(str(prev.get("subtask_goal", "")), text):
                _append_while(prev, text)
            prev["end_frame"] = span.get("end_frame", prev.get("end_frame"))
            prev["boundary_source"] = span.get("boundary_source", prev.get("boundary_source"))
            prev["truncation_rule"] = span.get("truncation_rule", prev.get("truncation_rule"))
            continue
        output.append(span)
    for idx, item in enumerate(output):
        item["subtask_index"] = idx
    return output
