"""Atomic prompt text and reusable subtask compositions."""

from __future__ import annotations

from .models import StepSpec


def pick_text(obj: str, arm: str | None = None) -> str:
    if arm:
        return f"Grasp the {obj} with the {arm} arm."
    return f"Grasp the {obj}."


def place_text(obj: str, dst: str, arm: str | None = None) -> str:
    if arm:
        return f"Place the {obj} {dst} with the {arm} arm."
    return f"Place the {obj} {dst}."


def lift_text(obj: str, arm: str | None = None) -> str:
    if arm:
        return f"Lift the {obj} with the {arm} arm."
    return f"Lift the {obj}."


def hold_text(obj: str, arm: str | None = None) -> str:
    if arm:
        return f"Hold the {obj} with the {arm} arm."
    return f"Hold the {obj}."


def release_text(obj: str, arm: str | None = None) -> str:
    if arm:
        return f"Release the {obj} from the {arm} arm."
    return f"Release the {obj}."


def return_to_neutral_text(arm: str) -> str:
    return f"Return the {arm} arm to a neutral pose."


def rotate_text(obj: str, result: str, arm: str | None = None) -> str:
    suffix = f" with the {arm} arm" if arm else ""
    return f"Rotate the {obj} {result}{suffix}."


def press_text(obj: str, target: str | None = None, arm: str | None = None) -> str:
    suffix = f" with the {arm} arm" if arm else ""
    if target:
        return f"Press the {obj} onto the {target}{suffix}."
    return f"Press the {obj}{suffix}."


def pair_steps(obj: str, dst: str, arm: str | None = None) -> list[StepSpec]:
    return [
        StepSpec(pick_text(obj, arm), "close", arm),
        StepSpec(place_text(obj, dst, arm), "open", arm),
    ]


def distinct_pair(first: str, second: str) -> tuple[str, str]:
    if first == second:
        return f"right {first}", f"left {second}"
    return first, second


def dual_pair_steps(first: str, second: str, dst: str) -> list[StepSpec]:
    first, second = distinct_pair(first, second)
    return [
        StepSpec(f"Grasp the {first} with the left arm while grasping the {second} with the right arm.", "close"),
        StepSpec(f"Place the {first} {dst} with the left arm while placing the {second} {dst} with the right arm.", "open"),
    ]


def dual_pick_separate_place(first: str, second: str, dst: str) -> list[StepSpec]:
    first, second = distinct_pair(first, second)
    return [
        StepSpec(f"Grasp the {first} with the left arm while grasping the {second} with the right arm.", "close"),
        StepSpec(f"Place the {first} {dst} with the left arm.", "open", "left"),
        StepSpec(f"Place the {second} {dst} with the right arm.", "open", "right"),
    ]


def dual_pick_place_then_return(first: str, second: str, dst: str) -> list[StepSpec]:
    first, second = distinct_pair(first, second)
    return [
        StepSpec(f"Grasp the {first} with the left arm while grasping the {second} with the right arm.", "close"),
        StepSpec(f"Place the {first} {dst} with the left arm.", "open", "left"),
        StepSpec(
            f"Return the left arm to a neutral pose while placing the {second} {dst} with the right arm.",
            "open",
            "right",
        ),
    ]


def grasp_then_final(
    obj: str,
    final_text: str,
    arm: str | None = None,
    *,
    final_event_kind: str = "final",
) -> list[StepSpec]:
    return [
        StepSpec(pick_text(obj, arm), "close", arm),
        StepSpec(final_text, final_event_kind),
    ]


def handover_steps(obj: str, final_text: str, final_event_kind: str = "final") -> list[StepSpec]:
    return [
        StepSpec(f"Grasp the {obj} with the first arm.", "close"),
        StepSpec(f"Grasp the {obj} with the receiving arm.", "close"),
        StepSpec(f"Release the {obj} from the first arm.", "open"),
        StepSpec(final_text, final_event_kind),
    ]
