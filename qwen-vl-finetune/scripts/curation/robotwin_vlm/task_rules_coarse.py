"""Coarse RoboTwin task compositions.

Coarse rules keep the task-specific language from the fine rules, but compress
the expected action sequence to one motion plus at most one following gripper
event. Frame boundaries are produced independently by ``alignment_coarse``.
"""

from __future__ import annotations

import re

from .models import GripperEvent, StepSpec
from .task_rules_fine import (
    CHRONOLOGICAL_ARM_TASKS,
    EXPECTED_TASK_SLUGS,
    NO_RETREAT_TASKS,
    RETREAT_MERGE_TASKS,
    TASK_BUILDERS,
    build_steps as build_fine_steps,
    canonical_task_goal,
    prompt_bread_basket_objects,
    prompt_bread_basket_target,
    validate_task_registry,
)


def _action(step: StepSpec) -> str:
    if step.event_kind == "handover":
        return "handover"
    if step.event_kind == "open_move":
        return "open_move"
    if step.event_kind in {"close", "close_until_motion", "handover_close_until_release", "both_partial_close", "both_full_close"}:
        return "close"
    if step.event_kind in {"open", "handover_release_open"}:
        return "open"
    return "move"


def _context(text: str, arm: str | None) -> str | None:
    lowered = text.lower()
    if (
        "both arms" in lowered
        or "both grippers" in lowered
        or "grippers of both arms" in lowered
        or ("left arm" in lowered and "right arm" in lowered)
    ):
        return None
    if re.search(r"\bleft arm\b", lowered):
        return "left"
    if re.search(r"\bright arm\b", lowered):
        return "right"
    return arm


def _is_dual_motion_text(text: str) -> bool:
    lowered = text.lower()
    return (
        "both arms" in lowered
        or "dual arms" in lowered
        or ("left arm" in lowered and "right arm" in lowered)
    )


def _append_arm(text: str, arm: str | None) -> str:
    if arm not in {"left", "right"}:
        return text
    if re.search(r"\b(left|right) arm\b", text, flags=re.IGNORECASE):
        return text
    if text.endswith("."):
        return f"{text[:-1]} with the {arm} arm."
    return f"{text} with the {arm} arm"


def _object_from_grasp_text(text: str) -> str | None:
    match = re.search(r"\bgrasp pose of the (.+?)\.?$", text, flags=re.IGNORECASE)
    return match.group(1) if match else None


def _place_from_text(text: str) -> tuple[str, str] | None:
    match = re.search(
        r"\b(?:place pose of|place) the (.+?) (?:onto|on|into|inside|at|to|away from|next to) (.+?)\.?$",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    obj = match.group(1)
    dst = text[text.lower().find(obj.lower()) + len(obj) :].strip().rstrip(".")
    return obj, dst


def _dual_grasp_text(move_text: str) -> str | None:
    match = re.search(
        r"Move the (left|right) arm to the grasp pose of the (.+?) while moving the (left|right) arm to the grasp pose of the (.+?)\.?$",
        move_text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    first_arm, first_obj, second_arm, second_obj = match.groups()
    return f"Grasp the {first_obj} with the {first_arm} arm while grasping the {second_obj} with the {second_arm} arm."


def _dual_place_text(move_text: str) -> str | None:
    match = re.search(
        r"Move the (left|right) arm to the place pose of the (.+?) (.+?) while moving the (left|right) arm to the place pose of the (.+?) (.+?)\.?$",
        move_text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    first_arm, first_obj, first_dst, second_arm, second_obj, second_dst = match.groups()
    if first_dst == second_dst:
        return f"Place the {first_obj} with the {first_arm} arm while placing the {second_obj} with the {second_arm} arm {first_dst}."
    return f"Place the {first_obj} {first_dst} with the {first_arm} arm while placing the {second_obj} {second_dst} with the {second_arm} arm."


def _bread_skillet_bread(task_goal: str) -> str:
    match = re.search(
        r"\b(?:put|place|move|set)\s+(?:the\s+)?(.+?)\s+into\s+(?:the\s+)?(?:metal\s+)?skillet\b",
        task_goal,
        flags=re.IGNORECASE,
    )
    bread = match.group(1).strip() if match else "bread"
    return "bread" if bread.lower() in {"it", "this", "that"} else bread


def _other_arm(arm: str | None) -> str:
    return "left" if arm == "right" else "right"


def _first_grasp_object(steps: list[StepSpec], fallback: str) -> str:
    for step in steps:
        obj = _object_from_grasp_text(step.text)
        if obj:
            return obj
    return fallback


def _stamp_target(task_goal: str) -> str:
    match = re.search(r"\bover\s+(.+?)\s+and\s+stamp\b", task_goal, flags=re.IGNORECASE)
    if match:
        target = match.group(1).strip()
        return f"{target} area" if not target.lower().endswith("area") else target
    return "target area"


def _dual_text_for_gripper(move: StepSpec, gripper: StepSpec) -> str:
    move_text = move.text.rstrip(".")
    if _action(gripper) == "close":
        return _dual_grasp_text(move_text) or "Grasp the targets with both arms."
    return _dual_place_text(move_text) or "Place the targets with both arms."


def _single_text_for_gripper(move: StepSpec, gripper: StepSpec) -> str:
    arm = _context(move.text, move.arm) or _context(gripper.text, gripper.arm)
    move_text = _append_arm(move.text, arm).rstrip(".")
    if _action(gripper) == "close":
        obj = _object_from_grasp_text(move_text)
        if obj:
            return _append_arm(f"Grasp the {obj}.", arm)
        return _append_arm("Grasp the target.", arm)
    place = _place_from_text(move_text)
    if place:
        obj, dst = place
        return _append_arm(f"Place the {obj} {dst}.", arm)
    return _append_arm("Place the target.", arm)


def _coarse_pair_text(move: StepSpec, gripper: StepSpec) -> str:
    if _is_dual_motion_text(move.text):
        return _dual_text_for_gripper(move, gripper)
    return _single_text_for_gripper(move, gripper)


def build_steps(
    slug: str,
    task_goal: str,
    info: dict[str, str],
    events: list[GripperEvent] | None = None,
) -> list[StepSpec]:
    fine_steps = build_fine_steps(slug, task_goal, info, events)
    if slug in {"click_alarmclock", "click_bell"}:
        close_events = [event for event in events or [] if event.kind == "close"]
        arm = close_events[0].arm if close_events else None
        action = "Click the alarm clock button." if slug == "click_alarmclock" else "Press the top of the bell."
        target = "alarm clock button" if slug == "click_alarmclock" else "bell"
        close_text = (
            f"Move the {arm} arm above the {target} and close the gripper."
            if arm in {"left", "right"}
            else f"Move above the {target} and close the gripper."
        )
        return [
            StepSpec(close_text, "close", arm),
            StepSpec(action, "move", arm),
        ]
    if slug == "press_stapler":
        close_events = [event for event in events or [] if event.kind == "close"]
        arm = close_events[0].arm if close_events else None
        close_text = (
            f"Move the {arm} arm above the stapler and close the gripper."
            if arm in {"left", "right"}
            else "Move above the stapler and close the gripper."
        )
        return [
            StepSpec(close_text, "close", arm),
            StepSpec("Press down the stapler.", "move", arm),
        ]
    if slug == "place_burger_fries":
        close_events = [event for event in events or [] if event.kind == "close"]
        open_events = [event for event in events or [] if event.kind == "open"]
        first_arm = close_events[0].arm if close_events else "left"
        second_arm = close_events[1].arm if len(close_events) > 1 else _other_arm(first_arm)
        first_open = open_events[0].arm if open_events else first_arm
        second_open = open_events[1].arm if len(open_events) > 1 else second_arm
        return [
            StepSpec(
                f"Grasp the hamburger with the {first_arm} arm while grasping the french fries with the {second_arm} arm.",
                "close",
            ),
            StepSpec("Place the hamburger onto the tray.", "open", first_open),
            StepSpec("Place the french fries onto the tray.", "open", second_open),
        ]
    if slug == "place_cans_plasticbox":
        close_events = [event for event in events or [] if event.kind == "close"]
        open_events = [event for event in events or [] if event.kind == "open"]
        first_arm = close_events[0].arm if close_events else "left"
        second_arm = close_events[1].arm if len(close_events) > 1 else _other_arm(first_arm)
        first_open = open_events[0].arm if open_events else first_arm
        second_open = open_events[1].arm if len(open_events) > 1 else second_arm
        return [
            StepSpec(
                f"Grasp the left can with the {first_arm} arm while grasping the right can with the {second_arm} arm.",
                "close",
            ),
            StepSpec("Place the left can into the plastic box.", "open", first_open),
            StepSpec("Place the right can into the plastic box.", "open", second_open),
            StepSpec(f"Lift up the {second_open} arm.", "move", second_open),
        ]
    if slug == "place_dual_shoes":
        close_events = [event for event in events or [] if event.kind == "close"]
        open_events = [event for event in events or [] if event.kind == "open"]
        first_arm = close_events[0].arm if close_events else "left"
        second_arm = close_events[1].arm if len(close_events) > 1 else _other_arm(first_arm)
        first_open = open_events[0].arm if open_events else first_arm
        second_open = open_events[1].arm if len(open_events) > 1 else second_arm
        return [
            StepSpec(
                f"Grasp the left shoe with the {first_arm} arm while grasping the right shoe with the {second_arm} arm.",
                "close",
            ),
            StepSpec("Place the left shoe into the shoe box.", "open", first_open),
            StepSpec("Place the right shoe into the shoe box.", "open", second_open),
        ]
    if slug in {"place_can_basket", "place_object_basket"}:
        close_events = [event for event in events or [] if event.kind == "close"]
        can_arm = close_events[0].arm if close_events else "left"
        basket_arm = close_events[1].arm if len(close_events) > 1 else _other_arm(can_arm)
        obj = _first_grasp_object(fine_steps, "can" if slug == "place_can_basket" else "object")
        return [
            StepSpec(f"Grasp the {obj}.", "close", can_arm),
            StepSpec(f"Drop the {obj} into the basket.", "open", can_arm),
            StepSpec("Grasp the basket.", "close", basket_arm),
            StepSpec("Lift the basket.", "move", basket_arm),
        ]
    if slug == "put_bottles_dustbin":
        close_events = [event for event in events or [] if event.kind == "close"]
        first_arm = close_events[0].arm if close_events else "left"
        second_carrier = close_events[1].arm if len(close_events) > 1 else _other_arm(first_arm)
        second_receiver = close_events[2].arm if len(close_events) > 2 else _other_arm(second_carrier)
        third_carrier = close_events[3].arm if len(close_events) > 3 else second_carrier
        third_receiver = close_events[4].arm if len(close_events) > 4 else _other_arm(third_carrier)
        return [
            StepSpec("Grasp the first bottle.", "close", first_arm),
            StepSpec("Drop the first bottle into the dustbin.", "open", first_arm),
            StepSpec("Grasp the second bottle.", "close", second_carrier),
            StepSpec("Lift and move the second bottle to the center of the table.", "move", second_carrier),
            StepSpec("Grasp the second bottle.", "close", second_receiver),
            StepSpec(
                "Use the left arm to put the second bottle into the dustbin.",
                "handover",
                "left",
            ),
            StepSpec(
                f"Move the {second_receiver} arm to drop the second bottle into the dustbin.",
                "open",
                second_receiver,
            ),
            StepSpec("Grasp the third bottle.", "close", third_carrier),
            StepSpec("Lift and move the third bottle to the center of the table.", "move", third_carrier),
            StepSpec("Grasp the third bottle.", "close", third_receiver),
            StepSpec(
                "Use the left arm to put the third bottle into the dustbin.",
                "handover",
                "left",
            ),
            StepSpec(
                f"Move the {third_receiver} arm to drop the third bottle into the dustbin.",
                "open",
                third_receiver,
            ),
        ]
    if slug == "put_object_cabinet":
        close_events = [event for event in events or [] if event.kind == "close"]
        object_arm = close_events[0].arm if close_events else "left"
        handle_arm = close_events[1].arm if len(close_events) > 1 else _other_arm(object_arm)
        obj = _first_grasp_object(fine_steps, "object")
        return [
            StepSpec(f"Grasp the {obj}.", "close", object_arm),
            StepSpec("Grasp the cabinet handle.", "close", handle_arm),
            StepSpec("Pull open the cabinet.", "move", handle_arm),
            StepSpec(f"Drop the {obj} into the cabinet.", "open", object_arm),
        ]
    if slug == "rotate_qrcode":
        close_events = [event for event in events or [] if event.kind == "close"]
        arm = close_events[0].arm if close_events else None
        obj = _first_grasp_object(fine_steps, "payment sign")
        return [
            StepSpec(f"Grasp the {obj}.", "close", arm),
            StepSpec(f"Rotate the {obj} until the QR code faces the robot and put down the QR code.", "open", arm),
        ]
    if slug == "scan_object":
        close_events = [event for event in events or [] if event.kind == "close"]
        first_arm = close_events[0].arm if close_events else "left"
        second_arm = close_events[1].arm if len(close_events) > 1 else _other_arm(first_arm)
        grasp_text = _dual_text_for_gripper(fine_steps[0], StepSpec("", "close")) if fine_steps else "Grasp the scanner and the object with both arms."
        match = re.search(r"grasping the (.+?) with the (left|right) arm", grasp_text, flags=re.IGNORECASE)
        scanner = match.group(1) if match else "scanner"
        scanner_arm = fine_steps[-1].arm if fine_steps else None
        scanner_arm = scanner_arm or (match.group(2).lower() if match else second_arm)
        held_obj = "object"
        object_arm = fine_steps[-2].arm if len(fine_steps) >= 2 else None
        first = re.search(r"Grasp the (.+?) with the (left|right) arm while", grasp_text, flags=re.IGNORECASE)
        if first:
            held_obj = first.group(1) if first.group(1) != scanner else held_obj
            object_arm = object_arm or first.group(2).lower()
        object_arm = object_arm or first_arm
        return [
            StepSpec(grasp_text, "close"),
            StepSpec(
                f"Move the {object_arm} arm holding the {held_obj} to the scan pose.",
                "move",
                object_arm,
            ),
            StepSpec(
                f"Move the {scanner_arm} arm holding the {scanner} to scan the {held_obj}.",
                "move",
                scanner_arm,
            ),
        ]
    if slug == "shake_bottle":
        close_events = [event for event in events or [] if event.kind == "close"]
        arm = close_events[0].arm if close_events else None
        obj = _first_grasp_object(fine_steps, "bottle")
        return [
            StepSpec(f"Grasp the {obj}.", "close", arm),
            StepSpec(f"Lift and shake the {obj}.", "move", arm),
        ]
    if slug == "shake_bottle_horizontally":
        close_events = [event for event in events or [] if event.kind == "close"]
        arm = close_events[0].arm if close_events else None
        obj = _first_grasp_object(fine_steps, "bottle")
        return [
            StepSpec(f"Grasp the {obj}.", "close", arm),
            StepSpec(f"Lift and shake the {obj} horizontally.", "move", arm),
        ]
    if slug == "stack_bowls_two":
        close_events = [event for event in events or [] if event.kind == "close"]
        first_arm = close_events[0].arm if close_events else "left"
        second_arm = close_events[1].arm if len(close_events) > 1 else first_arm
        return [
            StepSpec("Grasp the base bowl.", "close", first_arm),
            StepSpec("Place the base bowl at the base position.", "open", first_arm),
            StepSpec("Grasp the top bowl.", "close", second_arm),
            StepSpec("Place the top bowl inside the base bowl.", "open", second_arm),
        ]
    if slug == "stack_bowls_three":
        close_events = [event for event in events or [] if event.kind == "close"]
        first_arm = close_events[0].arm if close_events else "left"
        second_arm = close_events[1].arm if len(close_events) > 1 else first_arm
        third_arm = close_events[2].arm if len(close_events) > 2 else _other_arm(second_arm)
        return [
            StepSpec("Grasp the base bowl.", "close", first_arm),
            StepSpec("Place the base bowl at the base position.", "open", first_arm),
            StepSpec("Grasp the middle bowl.", "close", second_arm),
            StepSpec("Place the middle bowl inside the base bowl.", "open", second_arm),
            StepSpec("Grasp the top bowl.", "close", third_arm),
            StepSpec("Place the top bowl inside the stacked bowls.", "open", third_arm),
        ]
    if slug == "stamp_seal":
        close_events = [event for event in events or [] if event.kind == "close"]
        arm = close_events[0].arm if close_events else None
        target = _stamp_target(task_goal)
        return [
            StepSpec("Grasp the seal.", "close", arm),
            StepSpec(f"Place the seal on the {target}.", "open", arm),
        ]
    if slug == "place_bread_basket":
        close_events = [event for event in events or [] if event.kind == "close"]
        open_events = [event for event in events or [] if event.kind == "open"]
        target = prompt_bread_basket_target(task_goal)
        objects = prompt_bread_basket_objects(task_goal) or ["bread"]
        while len(objects) < 2:
            objects.append("bread")
        first, second = objects[:2]
        first_arm = close_events[0].arm if close_events else (open_events[0].arm if open_events else "left")
        second_arm = (
            close_events[1].arm
            if len(close_events) > 1
            else open_events[1].arm
            if len(open_events) > 1
            else ("right" if first_arm == "left" else "left")
        )
        return [
            StepSpec(
                f"Grasp the {first} with the {first_arm} arm while grasping the {second} with the {second_arm} arm.",
                "close",
            ),
            StepSpec(f"Place the {first} into the {target}, resting on the bottom.", "open", first_arm),
            StepSpec(f"Place the {second} into the {target}, resting on the bottom.", "open", second_arm),
        ]
    if slug == "place_bread_skillet":
        close_events = [event for event in events or [] if event.kind == "close"]
        open_events = [event for event in events or [] if event.kind == "open"]
        bread_arm = open_events[0].arm if open_events else (close_events[0].arm if close_events else "right")
        skillet_arm = "left" if bread_arm == "right" else "right"
        bread = _bread_skillet_bread(task_goal)
        skillet = "skillet"
        return [
            StepSpec(
                f"Grasp the {bread} with the {bread_arm} arm while grasping the {skillet} with the {skillet_arm} arm.",
                "close",
            ),
            StepSpec(f"Move the {skillet} to the center of the table.", "move", skillet_arm),
            StepSpec(f"Drop the {bread} into the {skillet}, resting on the bottom.", "open", bread_arm),
        ]
    if slug == "handover_block":
        close_events = [event for event in events or [] if event.kind == "close"]
        first_arm = close_events[0].arm if close_events else "left"
        receiving_arm = close_events[1].arm if len(close_events) > 1 else ("right" if first_arm == "left" else "left")
        return [
            StepSpec("Grasp the red block.", "close", first_arm),
            StepSpec("Move the red block to the handover position.", "move", first_arm),
            StepSpec("Grasp the red block.", "close", receiving_arm),
            StepSpec(
                "Use the right arm to place the red block on the blue pad.",
                "handover",
                "right",
            ),
            StepSpec("Use the right arm to place the red block on the blue pad.", "open", "right"),
        ]
    if slug == "handover_mic":
        close_events = [event for event in events or [] if event.kind == "close"]
        first_arm = close_events[0].arm if close_events else "left"
        receiving_arm = close_events[1].arm if len(close_events) > 1 else ("right" if first_arm == "left" else "left")
        return [
            StepSpec("Grasp the microphone.", "close", first_arm),
            StepSpec("Move the microphone to the handover position.", "move", first_arm),
            StepSpec("Grasp the microphone.", "close", receiving_arm),
            StepSpec("Separate the two arms.", "move"),
        ]
    if slug == "dump_bin_bigbin":
        close_events = [event for event in events or [] if event.kind == "close"]
        first_arm = close_events[0].arm if close_events else "left"
        second_arm = close_events[1].arm if len(close_events) > 1 else ("right" if first_arm == "left" else "left")
        return [
            StepSpec("Grasp the trash bin.", "close", first_arm),
            StepSpec("Move the trash bin to the center of the table.", "open", first_arm),
            StepSpec("Grasp the trash bin.", "close", second_arm),
            StepSpec("Pour the trash bin contents into the big bin.", "move", second_arm),
        ]
    if slug == "hanging_mug":
        close_events = [event for event in events or [] if event.kind == "close"]
        first_arm = close_events[0].arm if close_events else "right"
        second_arm = close_events[1].arm if len(close_events) > 1 else ("left" if first_arm == "right" else "right")
        return [
            StepSpec("Grasp the mug.", "close", first_arm),
            StepSpec("Place the mug at the center of the table and rotate it to a suitable angle.", "open", first_arm),
            StepSpec("Grasp the mug.", "close", second_arm),
            StepSpec("Place the mug on the rack.", "open", second_arm),
            StepSpec(f"Return the {second_arm} arm to a neutral pose after releasing the mug.", "move", second_arm),
        ]
    if slug == "lift_pot":
        return [
            StepSpec("Grasp the handles of the kitchen pot with both arms.", "close"),
            StepSpec("Lift the pot with both arms.", "move"),
        ]
    if slug in {"pick_diverse_bottles", "pick_dual_bottles"}:
        return [
            StepSpec("Grasp the two bottles with both arms.", "close"),
            StepSpec("Move the two bottles above the center of the table.", "move"),
        ]
    coarse_steps: list[StepSpec] = []
    idx = 0
    while idx < len(fine_steps):
        step = fine_steps[idx]
        action = _action(step)
        if action in {"close", "open"}:
            idx += 1
            continue
        next_step = fine_steps[idx + 1] if idx + 1 < len(fine_steps) else None
        if next_step is not None and _action(next_step) in {"close", "open"}:
            kind = _action(next_step)
            coarse_steps.append(
                StepSpec(
                    _coarse_pair_text(step, next_step),
                    event_kind=kind,
                    arm=_context(step.text, step.arm),
                )
            )
            idx += 2
        else:
            coarse_steps.append(
                StepSpec(
                    _append_arm(step.text, _context(step.text, step.arm)),
                    event_kind="move",
                    arm=_context(step.text, step.arm),
                )
            )
            idx += 1
    return coarse_steps or [StepSpec(task_goal, "move")]
