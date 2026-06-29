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


def move_to_grasp_text(obj: str, arm: str | None = None) -> str:
    if arm:
        return f"Move the {arm} arm to the grasp pose of the {obj}."
    return f"Move to the grasp pose of the {obj}."


def close_gripper_text(arm: str | None = None) -> str:
    if arm:
        return f"Close the gripper of the {arm} arm."
    return "Close the gripper."


def move_to_place_text(obj: str, dst: str, arm: str | None = None) -> str:
    if arm:
        return f"Move the {arm} arm to the place pose of the {obj} {dst}."
    return f"Move to the place pose of the {obj} {dst}."


def open_gripper_text(arm: str | None = None) -> str:
    if arm:
        return f"Open the gripper of the {arm} arm."
    return "Open the gripper."


def grasp_steps(obj: str, arm: str | None = None) -> list[StepSpec]:
    return [
        StepSpec(move_to_grasp_text(obj, arm), "move", arm),
        StepSpec(close_gripper_text(arm), "close", arm),
    ]


def place_steps(obj: str, dst: str, arm: str | None = None) -> list[StepSpec]:
    return [
        StepSpec(move_to_place_text(obj, dst, arm), "move", arm),
        StepSpec(open_gripper_text(arm), "open", arm),
    ]


def pair_steps(obj: str, dst: str, arm: str | None = None) -> list[StepSpec]:
    return grasp_steps(obj, arm) + place_steps(obj, dst, arm)


def distinct_pair(first: str, second: str) -> tuple[str, str]:
    if first == second:
        return f"right {first}", f"left {second}"
    return first, second


def dual_pair_steps(first: str, second: str, dst: str) -> list[StepSpec]:
    first, second = distinct_pair(first, second)
    return [
        StepSpec(
            f"Move the left arm to the grasp pose of the {first} while moving the right arm to the grasp pose of the {second}.",
            "move",
        ),
        StepSpec("Close the grippers of both arms.", "close"),
        StepSpec(
            f"Move the left arm to the place pose of the {first} {dst} while moving the right arm to the place pose of the {second} {dst}.",
            "move",
        ),
        StepSpec("Open the grippers of both arms.", "open"),
    ]


def dual_pick_separate_place(first: str, second: str, dst: str) -> list[StepSpec]:
    first, second = distinct_pair(first, second)
    return [
        StepSpec(
            f"Move the left arm to the grasp pose of the {first} while moving the right arm to the grasp pose of the {second}.",
            "move",
        ),
        StepSpec("Close the grippers of both arms.", "close"),
        StepSpec(move_to_place_text(first, dst, "left"), "move", "left"),
        StepSpec(open_gripper_text("left"), "open", "left"),
        StepSpec(move_to_place_text(second, dst, "right"), "move", "right"),
        StepSpec(open_gripper_text("right"), "open", "right"),
    ]


def dual_pick_place_then_return(
    first: str,
    second: str,
    dst: str,
    *,
    lift_between_releases: bool = True,
    lift_after_final_release: bool = True,
    first_place_terminates_on: tuple[str, ...] = (),
) -> list[StepSpec]:
    first, second = distinct_pair(first, second)
    steps: list[StepSpec] = [
        StepSpec(
            f"Move the left arm to the grasp pose of the {first} while moving the right arm to the grasp pose of the {second}.",
            "move",
        ),
        StepSpec("Close the grippers of both arms.", "close"),
        StepSpec("Lift both objects to the middle position.", "move"),
        StepSpec(
            move_to_place_text(first, dst, "left"),
            "move",
            "left",
            terminates_on=first_place_terminates_on,
        ),
        StepSpec(open_gripper_text("left"), "open", "left"),
    ]
    if lift_between_releases:
        steps.append(StepSpec("Lift the left arm after releasing the object.", "move", "left"))
    steps.append(
        StepSpec(
            f"Move the right arm to the place pose of the {second} {dst} while returning the left arm to a neutral pose.",
            "move",
            "right",
            terminates_on=("gripper_open",),
        ),
    )
    steps.append(StepSpec(open_gripper_text("right"), "open", "right"))
    if lift_after_final_release:
        steps.append(StepSpec("Lift the right arm after releasing the object.", "move", "right"))
    return steps


def grasp_then_final(
    obj: str,
    final_text: str,
    arm: str | None = None,
    *,
    final_event_kind: str = "final",
) -> list[StepSpec]:
    return grasp_steps(obj, arm) + [StepSpec(final_text, final_event_kind)]


def handover_steps(obj: str, final_text: str, final_event_kind: str = "final") -> list[StepSpec]:
    return [
        StepSpec(f"Move the left arm to the grasp pose of the {obj}.", "move", "left"),
        StepSpec("Close the gripper of the left arm.", "close", "left"),
        StepSpec(f"Lift the {obj} to the middle position with the left arm.", "move", "left"),
        StepSpec(f"Move the right arm to the grasp pose of the {obj}.", "move", "right"),
        StepSpec("Close the gripper of the right arm.", "close", "right"),
        StepSpec(f"Release the {obj} from the left arm.", "open", "left"),
        StepSpec("Separate the two arms after the handover.", "move"),
        StepSpec(final_text, final_event_kind),
    ]
