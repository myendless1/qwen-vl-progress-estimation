"""Gripper-event detection, frame alignment, and annotation post-processing."""

from __future__ import annotations

import re
from dataclasses import dataclass
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
ARM_ORIENT_DIMS = {
    "left": slice(3, 7),
    "right": slice(11, 15),
}
PLACE_MOTION_START_THRESHOLD = 0.0005
ORIENTATION_MOTION_START_THRESHOLD = 0.01
OTHER_ARM_MOTION_START_THRESHOLD = 0.002
TINY_POST_GRIPPER_MOTION_PATH = 0.01
POST_GRIPPER_RETREAT_TEXT = re.compile(
    r"\b(?:lift|return)\b.+\bafter (?:releasing|pressing)\b",
    flags=re.IGNORECASE,
)
OPEN_CLOSE_MOTION_START_THRESHOLD = 0.0005
OPEN_CLOSE_MOTION_START_MIN_RUN = 3
MOVE_MOTION_BEFORE_GRIPPER_MARGIN = 4
ARM_LABEL_FLIP = {
    "left": "right",
    "right": "left",
}
ATOMIC_ACTIONS = {"move", "open", "close", "press", "final"}
MOVE_ALIASES = {
    "move_to_close",
    "move_to_open",
    "move_to_open_late",
    "move_to_both_full_close",
    "midpoint",
    "handover_lift",
    "handover_retreat",
    "move_until_next_motion",
    "eef_lift",
    "eef_retreat",
    "eef_shake",
    "eef_shake_xy",
    "press_lift",
}
CLOSE_ALIASES = {
    "close",
    "close_until_motion",
    "handover_close_until_release",
    "both_partial_close",
    "both_full_close",
}
OPEN_ALIASES = {"open", "handover_release_open"}
PRESS_ALIASES = {"press", "press_down"}


@dataclass(frozen=True)
class BoundaryCandidate:
    frame: int
    source: str
    rule: str
    matched_event: GripperEvent | None = None


def default_termination_flags(action: str) -> tuple[str, ...]:
    if action == "move":
        return ("gripper_open", "gripper_close", "other_arm_motion")
    if action in {"open", "close"}:
        return ("gripper_open", "gripper_close", "current_arm_motion", "other_arm_motion")
    if action == "press":
        return ("gripper_open", "gripper_close", "other_arm_motion")
    return ("episode_end",)


def termination_flags(step: StepSpec, action: str) -> set[str]:
    return set(step.terminates_on or default_termination_flags(action))


def detect_gripper_events(
    states: np.ndarray,
    threshold: float = 0.5,
    min_gap: int = 3,
    min_delta: float = 0.25,
) -> list[GripperEvent]:
    events: list[GripperEvent] = []
    for dim, arm in ((LEFT_GRIPPER_DIM, "left"), (RIGHT_GRIPPER_DIM, "right")):
        if states.shape[1] <= dim:
            continue
        gripper = states[:, dim]
        deltas = np.diff(gripper)
        active = np.abs(deltas) > 1e-4
        last_frame = -10**9
        idx = 0
        while idx < len(active):
            if not active[idx]:
                idx += 1
                continue
            sign = 1.0 if float(deltas[idx]) > 0 else -1.0
            start = idx
            idx += 1
            while idx < len(active) and active[idx] and sign * float(deltas[idx]) > 0:
                idx += 1
            end = idx
            start_frame = start
            end_frame = end
            frame = (start_frame + end_frame) // 2
            if frame - last_frame < min_gap:
                continue
            delta = float(gripper[end_frame] - gripper[start_frame])
            if abs(delta) < min_delta:
                continue
            kind = "open" if delta > 0 else "close"
            events.append(
                GripperEvent(
                    frame=frame,
                    arm=arm,
                    kind=kind,
                    start_frame=start_frame,
                    end_frame=end_frame,
                )
            )
            last_frame = int(frame)
    return sorted(events, key=lambda event: (event.frame, event.arm, event.kind))


def gripper_actuation_span(
    gripper: np.ndarray,
    crossing_frame: int,
    kind: str,
    eps: float = 1e-4,
) -> tuple[int, int]:
    """Return the full monotonic gripper-motion span around a threshold crossing."""
    if len(gripper) == 0:
        return crossing_frame, crossing_frame
    sign = 1.0 if kind == "open" else -1.0
    diffs = np.diff(gripper)
    start = max(0, min(crossing_frame, len(gripper) - 1))
    while start > 0 and sign * float(diffs[start - 1]) > eps:
        start -= 1
    end = max(0, min(crossing_frame, len(gripper) - 1))
    while end < len(gripper) - 1 and sign * float(diffs[end]) > eps:
        end += 1
    return int(start), int(end)


def event_start_frame(event: GripperEvent) -> int:
    return event.start_frame if event.start_frame is not None else event.frame


def event_end_frame(event: GripperEvent) -> int:
    return event.end_frame if event.end_frame is not None else event.frame


def eef_step_displacement(
    states: np.ndarray | None,
    arm: str | None,
    frame: int,
) -> float | None:
    if states is None or arm not in ARM_XYZ_DIMS or frame <= 0 or frame >= len(states):
        return None
    xyz = states[:, ARM_XYZ_DIMS[arm]]
    delta = xyz[frame] - xyz[frame - 1]
    return float(np.linalg.norm(delta))


def eef_step_orientation_displacement(
    states: np.ndarray | None,
    arm: str | None,
    frame: int,
) -> float | None:
    if states is None or arm not in ARM_ORIENT_DIMS or frame <= 0 or frame >= len(states):
        return None
    orient = states[:, ARM_ORIENT_DIMS[arm]]
    delta = orient[frame] - orient[frame - 1]
    return float(np.linalg.norm(delta))


def arm_step_is_moving(
    states: np.ndarray | None,
    arm: str | None,
    frame: int,
    *,
    displacement_threshold: float,
    orientation_threshold: float = ORIENTATION_MOTION_START_THRESHOLD,
) -> bool:
    displacement = eef_step_displacement(states, arm, frame)
    if displacement is not None and displacement > displacement_threshold:
        return True
    orientation = eef_step_orientation_displacement(states, arm, frame)
    return orientation is not None and orientation > orientation_threshold


def find_sustained_motion_start_frame(
    states: np.ndarray | None,
    arm: str | None,
    start_frame: int,
    end_frame: int,
    *,
    threshold: float = PLACE_MOTION_START_THRESHOLD,
    min_run: int = 3,
    orientation_threshold: float = ORIENTATION_MOTION_START_THRESHOLD,
) -> int | None:
    if states is None or arm not in ARM_XYZ_DIMS:
        return None
    start = max(1, min(start_frame, len(states) - 1))
    stop = max(start, min(end_frame, len(states) - 1))
    for frame in range(start, stop + 1):
        last = min(stop, frame + min_run - 1)
        if last - frame + 1 < min_run:
            break
        if all(
            arm_step_is_moving(
                states,
                arm,
                item,
                displacement_threshold=threshold,
                orientation_threshold=orientation_threshold,
            )
            for item in range(frame, last + 1)
        ):
            return frame
    return None


def find_both_arms_settle_frame(
    states: np.ndarray | None,
    start_frame: int,
    end_frame: int,
    *,
    threshold: float = 0.002,
    min_run: int = 3,
) -> int | None:
    """First frame at which both arms come to rest after a sustained motion.

    Returns the frame index where both arms have stayed below ``threshold`` for
    ``min_run`` consecutive steps, searching only after an initial sustained-motion
    period (so a still prefix does not match). Suitable for terminating a dual-arm
    lift step when the coordinated motion ends.
    """
    if states is None or states.shape[1] < 11:
        return None
    n = len(states)
    left = np.linalg.norm(np.diff(states[:, ARM_XYZ_DIMS["left"]], axis=0), axis=1)
    right = np.linalg.norm(np.diff(states[:, ARM_XYZ_DIMS["right"]], axis=0), axis=1)
    start = max(0, min(start_frame, n - 2))
    stop = max(start, min(end_frame, n - 2))
    motion_start = None
    for frame in range(start, stop + 1):
        if frame + min_run - 1 > stop:
            break
        if any(left[frame + k] > threshold or right[frame + k] > threshold for k in range(min_run)):
            motion_start = frame
            break
    if motion_start is None:
        return None
    for frame in range(motion_start, stop + 1):
        if frame + min_run - 1 > stop:
            break
        if all(left[frame + k] < threshold and right[frame + k] < threshold for k in range(min_run)):
            return frame
    return None


def both_arms_settle_candidate(
    states: np.ndarray | None,
    start_frame: int,
    end_frame: int,
) -> BoundaryCandidate | None:
    frame = find_both_arms_settle_frame(states, start_frame, end_frame)
    if frame is None:
        return None
    return BoundaryCandidate(frame=frame, source="both_arms_settle", rule="both_arms_settle")


DUAL_MOVE_BREAK_THRESHOLD = 0.002
DUAL_MOVE_BREAK_MIN_RUN = 3
DUAL_MOVE_BREAK_PROTECTION_FRAMES = 3


def find_dual_move_break_frame(
    states: np.ndarray | None,
    next_arm: str | None,
    start_frame: int,
    end_frame: int,
    *,
    threshold: float = DUAL_MOVE_BREAK_THRESHOLD,
    min_run: int = DUAL_MOVE_BREAK_MIN_RUN,
    protection_frames: int = DUAL_MOVE_BREAK_PROTECTION_FRAMES,
) -> int | None:
    """First frame at which a coordinated dual-arm motion breaks apart.

    The first ``protection_frames`` after ``start_frame`` are skipped so the
    detector never fires on the ramp-up of the dual motion itself. Afterwards
    we look for the first sustained run where the next action's arm is moving
    while the other arm has already stopped -- i.e. the coordination has ended
    and the next arm has begun its independent motion. Returns ``None`` while
    both arms keep moving together (the dual motion is still ongoing).
    """
    if states is None or next_arm not in ARM_XYZ_DIMS:
        return None
    other = other_arm(next_arm)
    if other is None or states.shape[1] < 11:
        return None
    n = len(states)
    left_disp = np.linalg.norm(np.diff(states[:, ARM_XYZ_DIMS["left"]], axis=0), axis=1)
    right_disp = np.linalg.norm(np.diff(states[:, ARM_XYZ_DIMS["right"]], axis=0), axis=1)
    disp = {"left": left_disp, "right": right_disp}
    scan_start = max(1, min(start_frame + protection_frames, n - 1))
    stop = max(scan_start, min(end_frame, n - 2))
    for frame in range(scan_start, stop + 1):
        if frame + min_run - 1 > stop:
            break
        if all(
            disp[next_arm][frame + k - 1] > threshold
            and disp[other][frame + k - 1] <= threshold
            for k in range(min_run)
        ):
            return frame
    return None


def dual_move_break_candidate(
    states: np.ndarray | None,
    next_arm: str | None,
    start_frame: int,
    end_frame: int,
) -> BoundaryCandidate | None:
    frame = find_dual_move_break_frame(states, next_arm, start_frame, end_frame)
    if frame is None:
        return None
    return BoundaryCandidate(
        frame=frame - 1,
        source=f"before_eef_{next_arm}_motion",
        rule="other_arm_motion",
    )


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


def step_uses_both_arms(step: StepSpec) -> bool:
    text = step.text.lower()
    if "both arms" in text or "dual arms" in text or "corresponding arms" in text:
        return True
    if "both grippers" in text or "grippers of both arms" in text:
        return True
    return bool(
        re.search(
            r"\bleft arm\b.+\bwhile moving the right arm\b|\bright arm\b.+\bwhile moving the left arm\b",
            text,
        )
    )

def arm_context_from_text(text: str) -> str | None:
    if re.search(
        r"\b(?:both arms|dual arms|both objects|both grippers|grippers of both arms|corresponding arms)\b",
        text,
        flags=re.IGNORECASE,
    ):
        return "both"
    has_left = bool(re.search(r"\bleft arm\b", text, flags=re.IGNORECASE))
    has_right = bool(re.search(r"\bright arm\b", text, flags=re.IGNORECASE))
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
    if text == "Open the gripper.":
        return f"Open the gripper of the {arm} arm."
    if text == "Close the gripper.":
        return f"Close the gripper of the {arm} arm."
    match = re.match(r"^Move to (.+)$", text.rstrip("."), flags=re.IGNORECASE)
    if match:
        return f"Move the {arm} arm to {match.group(1)}."
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


def atomic_action(step: StepSpec) -> str:
    kind = step.event_kind
    if kind in ATOMIC_ACTIONS:
        return kind
    if kind in MOVE_ALIASES:
        return "move"
    if kind in CLOSE_ALIASES:
        return "close"
    if kind in OPEN_ALIASES:
        return "open"
    if kind in PRESS_ALIASES:
        return "press"
    return "final"


def action_actor(step: StepSpec, last_event_arm: str | None) -> str | None:
    context = arm_context_from_text(step.text)
    if context == "both" or step_uses_both_arms(step):
        return None
    if context in {"left", "right"}:
        return context
    if step.arm in {"left", "right"}:
        return step.arm
    return last_event_arm


def subtask_type_for_step(step: StepSpec, action: str) -> str:
    if arm_context_from_text(step.text) == "both" or step_uses_both_arms(step):
        return f"dual_{action}"
    return action


def next_step_arm(steps: list[StepSpec], index: int) -> str | None:
    if index + 1 >= len(steps):
        return None
    step = steps[index + 1]
    context = arm_context_from_text(step.text)
    if context in {"left", "right"}:
        return context
    return step.arm


def other_arm(arm: str | None) -> str | None:
    if arm == "left":
        return "right"
    if arm == "right":
        return "left"
    return None


def describes_concurrent_return(text: str) -> bool:
    return bool(
        re.search(
            r"\bwhile returning\b.+\b(?:neutral|default) pose\b",
            text,
            flags=re.IGNORECASE,
        )
    )


def gripper_event_start_candidate(
    event: GripperEvent,
    states: np.ndarray | None,
    start: int,
) -> BoundaryCandidate:
    del states
    event_start = max(start, event_start_frame(event))
    frame = event_start - 1
    return BoundaryCandidate(
        frame=frame,
        source=f"before_gripper_{event.arm}_{event.kind}",
        rule=f"gripper_{event.kind}",
        matched_event=event,
    )


def consume_paired_gripper_events(
    events: list[GripperEvent],
    matched_event: GripperEvent,
    used: set[int],
    *,
    max_gap: int = 3,
) -> list[GripperEvent]:
    paired = [matched_event]
    for idx, event in enumerate(events):
        if idx in used or event == matched_event:
            continue
        if event.kind != matched_event.kind:
            continue
        if event.arm == matched_event.arm:
            continue
        if abs(event.frame - matched_event.frame) > max_gap:
            continue
        used.add(idx)
        paired.append(event)
    return paired


def upcoming_gripper_candidates(
    events: list[GripperEvent],
    start_after: int,
    used: set[int],
    states: np.ndarray | None,
    start: int,
    *,
    kinds: set[str] | None = None,
    arms: set[str] | None = None,
) -> list[BoundaryCandidate]:
    candidates: list[BoundaryCandidate] = []
    for event in events:
        if event.frame <= start_after:
            continue
        if kinds is not None and event.kind not in kinds:
            continue
        if arms is not None and event.arm not in arms:
            continue
        candidates.append(gripper_event_start_candidate(event, states, start))
    return candidates


def motion_candidate(
    states: np.ndarray | None,
    arm: str | None,
    start_frame: int,
    end_frame: int,
    *,
    threshold: float = PLACE_MOTION_START_THRESHOLD,
    min_run: int = 3,
    orientation_threshold: float = ORIENTATION_MOTION_START_THRESHOLD,
    rule: str = "motion",
) -> BoundaryCandidate | None:
    frame = find_sustained_motion_start_frame(
        states,
        arm,
        start_frame,
        end_frame,
        threshold=threshold,
        min_run=min_run,
        orientation_threshold=orientation_threshold,
    )
    if frame is None:
        return None
    return BoundaryCandidate(frame=frame - 1, source=f"before_eef_{arm or 'unknown'}_motion", rule=rule)


def collect_motion_candidates(
    states: np.ndarray | None,
    arms: list[str | None],
    start_frame: int,
    end_frame: int,
    *,
    threshold: float = PLACE_MOTION_START_THRESHOLD,
    min_run: int = 3,
    orientation_threshold: float = ORIENTATION_MOTION_START_THRESHOLD,
    rule: str = "motion",
) -> list[BoundaryCandidate]:
    output: list[BoundaryCandidate] = []
    seen: set[str] = set()
    for arm in arms:
        if arm not in ARM_XYZ_DIMS or arm in seen:
            continue
        seen.add(arm)
        candidate = motion_candidate(
            states,
            arm,
            start_frame,
            end_frame,
            threshold=threshold,
            min_run=min_run,
            orientation_threshold=orientation_threshold,
            rule=rule,
        )
        if candidate is not None:
            output.append(candidate)
    return output


def choose_boundary(candidates: list[BoundaryCandidate]) -> BoundaryCandidate | None:
    valid = [candidate for candidate in candidates if candidate.frame is not None]
    if not valid:
        return None
    return min(valid, key=lambda candidate: candidate.frame)


def choose_move_boundary(candidates: list[BoundaryCandidate]) -> BoundaryCandidate | None:
    valid = [candidate for candidate in candidates if candidate.frame is not None]
    if not valid:
        return None
    gripper = [candidate for candidate in valid if candidate.matched_event is not None]
    motion = [candidate for candidate in valid if candidate.matched_event is None]
    first_gripper = min(gripper, key=lambda candidate: candidate.frame) if gripper else None
    if first_gripper is None:
        return min(motion, key=lambda candidate: candidate.frame) if motion else None
    early_motion = [
        candidate
        for candidate in motion
        if candidate.frame + MOVE_MOTION_BEFORE_GRIPPER_MARGIN < first_gripper.frame
    ]
    if early_motion:
        return min(early_motion, key=lambda candidate: candidate.frame)
    return first_gripper


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
        source = str(span.get("boundary_source", ""))
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
        elif context == "both":
            last_context = context
        elif boundary_context in {"left", "right"}:
            last_context = boundary_context
        output.append(span)
    return output


def assign_spans(
    steps: list[StepSpec],
    events: list[GripperEvent],
    n_frames: int,
    states: np.ndarray | None = None,
    prefer_specified_arm: bool = True,
) -> list[dict[str, Any]]:
    if n_frames <= 0:
        raise ValueError("episode has no frames")
    spans: list[dict[str, Any]] = []
    used: set[int] = set()
    prev_end = -1
    last_event_arm: str | None = None
    for i, step in enumerate(steps):
        if prev_end >= n_frames - 1:
            break
        start = 0 if i == 0 else min(prev_end + 1, n_frames - 1)
        is_last = i == len(steps) - 1
        matched_event: GripperEvent | None = None
        action = atomic_action(step)
        subtask_type = subtask_type_for_step(step, action)
        actor = action_actor(step, last_event_arm)
        flags = termination_flags(step, action)
        boundary: BoundaryCandidate | None = None

        if action in {"open", "close"}:
            matched_event = pick_event(
                events,
                action,
                prev_end,
                actor if prefer_specified_arm else None,
                used,
                prefer_specified_arm=prefer_specified_arm,
            )
            if matched_event is not None:
                matched_events = (
                    consume_paired_gripper_events(events, matched_event, used)
                    if actor is None or step_uses_both_arms(step)
                    else [matched_event]
                )
                candidates: list[BoundaryCandidate] = []
                search_start = max(prev_end, *(event_end_frame(event) for event in matched_events))
                gripper_kinds = {
                    kind
                    for flag, kind in (("gripper_open", "open"), ("gripper_close", "close"))
                    if flag in flags
                }
                if gripper_kinds:
                    candidates.extend(
                        upcoming_gripper_candidates(
                            events,
                            search_start,
                            used,
                            states,
                            start,
                            kinds=gripper_kinds,
                        )
                    )
                motion_arms = []
                if "current_arm_motion" in flags:
                    candidates.extend(
                        collect_motion_candidates(
                            states,
                            [event.arm for event in matched_events],
                            max(event_end_frame(event) for event in matched_events) + 1,
                            n_frames - 1,
                            threshold=OPEN_CLOSE_MOTION_START_THRESHOLD,
                            min_run=OPEN_CLOSE_MOTION_START_MIN_RUN,
                            rule="current_arm_motion",
                        )
                    )
                if "other_arm_motion" in flags:
                    if len(matched_events) == 1:
                        motion_arms.append(other_arm(matched_event.arm))
                    motion_arms.append(next_step_arm(steps, i))
                if motion_arms:
                    candidates.extend(
                        collect_motion_candidates(
                            states,
                            motion_arms,
                            max(event_end_frame(event) for event in matched_events) + 1,
                            n_frames - 1,
                            threshold=OPEN_CLOSE_MOTION_START_THRESHOLD,
                            min_run=OPEN_CLOSE_MOTION_START_MIN_RUN,
                            rule="other_arm_motion",
                        )
                    )
                boundary = choose_boundary(candidates)

        elif action == "move":
            next_arm = next_step_arm(steps, i)
            candidates: list[BoundaryCandidate] = []
            gripper_kinds = {
                kind
                for flag, kind in (("gripper_open", "open"), ("gripper_close", "close"))
                if flag in flags
            }
            if gripper_kinds:
                candidates.extend(
                    upcoming_gripper_candidates(
                        events,
                        prev_end,
                        used,
                        states,
                        start,
                        kinds=gripper_kinds,
                    )
                )
            motion_arms = (
                []
                if describes_concurrent_return(step.text)
                else [next_arm] if subtask_type == "dual_move" and next_arm
                else [next_arm if next_arm != actor else None]
            )
            if "other_arm_motion" in flags:
                if subtask_type == "dual_move" and next_arm:
                    break_candidate = dual_move_break_candidate(
                        states, next_arm, start, n_frames - 1
                    )
                    if break_candidate is not None:
                        candidates.append(break_candidate)
                else:
                    candidates.extend(
                        collect_motion_candidates(
                            states,
                            motion_arms,
                            start + 1,
                            n_frames - 1,
                            threshold=OTHER_ARM_MOTION_START_THRESHOLD,
                            rule="other_arm_motion",
                        )
                    )
            if "both_arms_settle" in flags:
                settle = both_arms_settle_candidate(states, start + 1, n_frames - 1)
                if settle is not None:
                    candidates.append(settle)
            boundary = choose_move_boundary(candidates)

        elif action == "press":
            candidates = []
            gripper_kinds = {
                kind
                for flag, kind in (("gripper_open", "open"), ("gripper_close", "close"))
                if flag in flags
            }
            if gripper_kinds:
                candidates.extend(
                    upcoming_gripper_candidates(
                        events,
                        prev_end,
                        used,
                        states,
                        start,
                        kinds=gripper_kinds,
                    )
                )
            next_arm = next_step_arm(steps, i)
            motion_arms = [next_arm if next_arm != actor else None]
            if actor in {"left", "right"}:
                motion_arms.append(other_arm(actor))
            if "other_arm_motion" in flags:
                candidates.extend(
                    collect_motion_candidates(
                        states,
                        motion_arms,
                        start + 1,
                        n_frames - 1,
                        threshold=OTHER_ARM_MOTION_START_THRESHOLD,
                        rule="other_arm_motion",
                    )
                )
            boundary = choose_boundary(candidates)

        if boundary is not None:
            end = min(n_frames - 1, boundary.frame)
            boundary_source = boundary.source
            truncation_rule = boundary.rule
            if matched_event is None and boundary.matched_event is not None and action != "move":
                matched_event = boundary.matched_event
        else:
            end = n_frames - 1
            boundary_source = "episode_end"
            truncation_rule = "episode_end"

        if end < start and not is_last:
            continue
        if end < start:
            end = start
        if is_last:
            end = n_frames - 1
            boundary_source = "episode_end" if boundary is None else f"episode_end_after_{boundary_source}"
            truncation_rule = "episode_end"
        spans.append(
            {
                "subtask_index": i,
                "subtask_goal": align_text_with_event_arm(step.text, matched_event),
                "subtask_type": subtask_type,
                "start_frame": int(start),
                "end_frame": int(end),
                "boundary_source": boundary_source,
                "truncation_rule": truncation_rule,
            }
        )
        if matched_event is not None:
            last_event_arm = matched_event.arm
        else:
            context_arm = arm_context_from_text(step.text) or step.arm
            if context_arm in {"left", "right"}:
                last_event_arm = context_arm
            elif boundary is not None and boundary.matched_event is not None:
                # The move step's boundary was set by a gripper event (e.g.
                # ``before_gripper_<arm>_<kind>``); the next close/open step
                # must match that same event, so propagate its arm instead of
                # keeping a stale ``last_event_arm`` from an earlier action.
                last_event_arm = boundary.matched_event.arm
        prev_end = end
    spans = describe_secondary_arm_motion(spans, states)
    spans = ensure_arm_mentions(spans)
    for new_index, span in enumerate(spans):
        span["subtask_index"] = new_index
    return spans


def arm_path_length(
    states: np.ndarray | None,
    arm: str,
    start_frame: int,
    end_frame: int,
) -> float:
    if states is None or arm not in ARM_XYZ_DIMS or end_frame <= start_frame:
        return 0.0
    xyz = states[start_frame : end_frame + 1, ARM_XYZ_DIMS[arm]]
    if len(xyz) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum())


def describe_secondary_arm_motion(
    spans: list[dict[str, Any]],
    states: np.ndarray | None,
    *,
    min_secondary_path: float = 0.05,
    relative_path_ratio: float = 0.55,
) -> list[dict[str, Any]]:
    if states is None:
        return spans
    output: list[dict[str, Any]] = []
    for span in spans:
        span = dict(span)
        text = str(span.get("subtask_goal", ""))
        lowered = text.lower()
        source = str(span.get("boundary_source", ""))
        original_text_context = arm_context_from_text(text)
        boundary_context = arm_context_from_boundary(source)
        text_context = original_text_context or boundary_context
        is_grasp_close_boundary = bool(
            re.search(r"^(?:episode_end_after_)?(?:before_)?gripper_(?:left|right)_close", source)
            and re.search(r"\bgrasp pose\b", lowered)
        )
        if (
            text_context not in {"left", "right"}
            or re.search(r"\bwhile\b|both arms|dual arms|other arm|both grippers", lowered)
            or (re.search(r"^(?:episode_end_after_)?(?:before_)?gripper_", source) and not is_grasp_close_boundary)
            or (
                re.search(r"^(?:episode_end_after_)?(?:before_)?eef_(?:left|right)_", source)
                and boundary_context in {"left", "right"}
                and boundary_context != text_context
            )
            or not re.search(
                r"\b(move|lift|return|place|pull|scan|rotate|shake)\b",
                lowered,
            )
        ):
            output.append(span)
            continue
        other = "right" if text_context == "left" else "left"
        start = int(span["start_frame"])
        end = int(span["end_frame"])
        primary_path = arm_path_length(states, text_context, start, end)
        secondary_path = arm_path_length(states, other, start, end)
        if secondary_path < min_secondary_path or secondary_path < primary_path * relative_path_ratio:
            output.append(span)
            continue
        if original_text_context is None:
            text = append_arm_to_text(text, text_context)
        phrase = (
            f" while returning the {other} arm to a neutral pose"
            if re.search(r"\bgrasp pose\b", lowered)
            else f" while moving the {other} arm"
        )
        if text.endswith("."):
            text = f"{text[:-1]}{phrase}."
        else:
            text = f"{text}{phrase}"
        span["subtask_goal"] = text
        output.append(span)
    return output


def merge_tiny_post_gripper_motion(
    spans: list[dict[str, Any]],
    states: np.ndarray | None,
) -> list[dict[str, Any]]:
    """Merge a tiny same-arm retreat at episode end into the preceding open/close step."""
    if states is None or len(spans) < 2:
        return spans
    output = [dict(span) for span in spans]
    prev = output[-2]
    curr = output[-1]
    if str(prev.get("subtask_type", "")) not in {"open", "close", "dual_open", "dual_close"}:
        return spans
    if str(curr.get("subtask_type", "")) != "move":
        return spans
    source = str(curr.get("boundary_source", ""))
    if source != "episode_end" and not source.startswith("episode_end_after_"):
        return spans
    prev_arm = arm_context_from_text(str(prev.get("subtask_goal", "")))
    curr_arm = arm_context_from_text(str(curr.get("subtask_goal", "")))
    if prev_arm not in {"left", "right"} or prev_arm != curr_arm:
        return spans
    if not POST_GRIPPER_RETREAT_TEXT.search(str(curr.get("subtask_goal", ""))):
        return spans
    start = int(curr.get("start_frame", 0))
    end = int(curr.get("end_frame", start))
    if arm_path_length(states, curr_arm, start, end) >= TINY_POST_GRIPPER_MOTION_PATH:
        return spans
    prev["end_frame"] = end
    prev["boundary_source"] = "episode_end"
    prev["truncation_rule"] = "episode_end"
    output.pop()
    for index, span in enumerate(output):
        span["subtask_index"] = index
    return output


def merge_stack_arm_switches(spans: list[dict[str, Any]], slug: str) -> list[dict[str, Any]]:
    if slug not in RETREAT_MERGE_TASKS or len(spans) < 2:
        return spans
    output = [dict(span) for span in spans]
    for idx in range(len(output) - 1):
        source = str(output[idx].get("boundary_source", ""))
        next_source = str(output[idx + 1].get("boundary_source", ""))
        release = re.fullmatch(r"(?:before_)?gripper_(left|right)_(?:strict_)?open", source)
        pickup = re.fullmatch(r"(?:before_)?gripper_(left|right)_(?:strict_)?close", next_source)
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


DUAL_CONTAINER_FIRST_PLACE_ASIDE_THRESHOLDS: dict[str, float] = {
    "place_dual_shoes": 0.015,
    # Cans episodes only show tiny holding-arm jitter during the first place.
    "place_cans_plasticbox": 0.02,
}


def relabel_dual_container_first_place(
    spans: list[dict[str, Any]],
    states: np.ndarray | None,
    slug: str,
) -> list[dict[str, Any]]:
    """Finalize dual-container first place steps as task-specific dual_move.

    Supported tasks emit a single-arm place move that terminates only on gripper
    open/close (see ``first_place_terminates_on`` in task rules). This pass
    always relabels that step ``dual_move`` without using generic dual_move
    break logic. Aside motion is described in the goal only when the non-placing
    arm path exceeds a task-specific threshold.
    """
    aside_threshold = DUAL_CONTAINER_FIRST_PLACE_ASIDE_THRESHOLDS.get(slug)
    if aside_threshold is None or states is None:
        return spans
    output = [dict(span) for span in spans]
    for span in output:
        text = str(span.get("subtask_goal", ""))
        if not re.match(r"^Move the (left|right) arm to the place pose", text):
            continue
        if re.search(r"\bwhile (returning|moving)\b", text, flags=re.IGNORECASE):
            break
        placing_arm = arm_context_from_text(text)
        if placing_arm not in {"left", "right"}:
            break
        other_arm = ARM_LABEL_FLIP[placing_arm]
        start = int(span.get("start_frame", 0))
        end = int(span.get("end_frame", start))
        other_path = arm_path_length(states, other_arm, start, end)
        span["subtask_type"] = "dual_move"
        if other_path >= aside_threshold:
            if text.endswith("."):
                text = text[:-1]
            span["subtask_goal"] = f"{text} while moving the {other_arm} arm aside."
        break
    return output


def relabel_dual_shoes_first_place(
    spans: list[dict[str, Any]],
    states: np.ndarray | None,
    slug: str,
) -> list[dict[str, Any]]:
    return relabel_dual_container_first_place(spans, states, slug)


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
