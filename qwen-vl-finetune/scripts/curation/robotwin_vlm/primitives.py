"""Atomic prompt text and reusable subtask compositions."""

from __future__ import annotations

from .models import StepSpec

# Nominal tabletop height in the RoboTwin world frame (before per-episode table_z_bias).
TABLE_SURFACE_Z = 0.74


def height_above_table_cm(world_z: float) -> int:
    return round((world_z - TABLE_SURFACE_Z) * 100)


def normalize_place_destination(dst: str) -> str:
    """Append bottom-contact wording for in-container placements."""
    lowered = dst.strip().lower().rstrip(".")
    if lowered.startswith(("into the ", "inside the ")):
        if "resting on the bottom" not in lowered:
            return f"{dst.strip().rstrip('.')}, resting on the bottom"
    return dst


def pick_text(obj: str, arm: str | None = None) -> str:
    if arm:
        return f"Grasp the {obj} with the {arm} arm."
    return f"Grasp the {obj}."


def place_text(obj: str, dst: str, arm: str | None = None) -> str:
    dst = normalize_place_destination(dst)
    if arm:
        return f"Place the {obj} {dst} with the {arm} arm."
    return f"Place the {obj} {dst}."


def lift_above_table_text(
    obj: str,
    *,
    cm: int | None = None,
    min_cm: int | None = None,
    arm: str | None = None,
) -> str:
    if min_cm is not None:
        height = f"to at least {min_cm} cm above the table"
    elif cm is not None:
        height = f"to about {cm} cm above the table"
    else:
        height = "above the table"
    if arm:
        return f"Lift the {obj} with the {arm} arm {height}."
    return f"Lift the {obj} {height}."


def lift_text(obj: str, arm: str | None = None) -> str:
    return lift_above_table_text(obj, cm=10, arm=arm)


def dual_lift_above_table_text(*, cm: int = 10, min_cm: int | None = None) -> str:
    if min_cm is not None:
        return f"Lift both objects to at least {min_cm} cm above the table with both arms."
    return f"Lift both objects to about {cm} cm above the table with both arms."


def dual_lift_named_objects_text(
    first: str,
    second: str,
    *,
    cm: int = 10,
    min_cm: int | None = None,
) -> str:
    if min_cm is not None:
        height = f"to at least {min_cm} cm above the table"
    else:
        height = f"to about {cm} cm above the table"
    return f"Lift the {first} and the {second} {height} with both arms."


def retract_above_table_text(
    arm: str | None = None,
    *,
    obj: str | None = None,
    min_cm: int = 5,
) -> str:
    if arm and obj:
        return (
            f"Retract the {arm} arm to at least {min_cm} cm above the table "
            f"after releasing the {obj}."
        )
    if arm:
        return f"Retract the {arm} arm to at least {min_cm} cm above the table."
    return f"Retract to at least {min_cm} cm above the table."


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


def partial_close_gripper_text(arm: str | None = None, *, percent: int) -> str:
    if arm:
        return f"Partially close the gripper of the {arm} arm to about {percent}%."
    return f"Partially close the grippers of both arms to about {percent}%."


def move_to_place_text(obj: str, dst: str, arm: str | None = None) -> str:
    dst = normalize_place_destination(dst)
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
    dst = normalize_place_destination(dst)
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
    dst = normalize_place_destination(dst)
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
    lift_cm: int = 10,
    retract_min_cm: int = 5,
    first_place_terminates_on: tuple[str, ...] = (),
) -> list[StepSpec]:
    first, second = distinct_pair(first, second)
    dst = normalize_place_destination(dst)
    steps: list[StepSpec] = [
        StepSpec(
            f"Move the left arm to the grasp pose of the {first} while moving the right arm to the grasp pose of the {second}.",
            "move",
        ),
        StepSpec("Close the grippers of both arms.", "close"),
        StepSpec(dual_lift_above_table_text(cm=lift_cm), "move"),
        StepSpec(
            move_to_place_text(first, dst, "left"),
            "move",
            "left",
            terminates_on=first_place_terminates_on,
        ),
        StepSpec(open_gripper_text("left"), "open", "left"),
    ]
    if lift_between_releases:
        steps.append(
            StepSpec(retract_above_table_text("left", min_cm=retract_min_cm), "move", "left")
        )
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
        steps.append(
            StepSpec(retract_above_table_text("right", min_cm=retract_min_cm), "move", "right")
        )
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
