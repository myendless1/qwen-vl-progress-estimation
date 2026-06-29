"""Task-specific subtask composition built from reusable primitives."""

from __future__ import annotations

import re
from collections.abc import Callable

from .models import GripperEvent, StepSpec, TaskBuilder, TaskContext
from .primitives import (
    close_gripper_text,
    distinct_pair,
    dual_pick_place_then_return,
    grasp_steps,
    handover_steps,
    move_to_grasp_text,
    move_to_place_text,
    open_gripper_text,
    pair_steps,
    place_steps,
    pick_text,
    place_text,
)
from .prompts import (
    info_arm,
    info_obj,
    is_color_name,
    parse_a2b_objects_from_prompt,
    parse_arm_sequence_from_prompt,
    prompt_arm_near_object,
    prompt_basket_object,
    prompt_cabinet_object,
    prompt_fan_target,
    prompt_mat_target,
    prompt_mouse_object,
    prompt_object_before_target,
    prompt_scan_arms,
)


NO_RETREAT_TASKS = {
    "blocks_ranking_rgb",
    "blocks_ranking_size",
    "place_object_basket",
    "place_can_basket",
    "put_bottles_dustbin",
    "hanging_mug",
    "stack_blocks_three",
    "stack_blocks_two",
    "stack_bowls_three",
    "stack_bowls_two",
}
STACK_TASKS = {
    "stack_blocks_three",
    "stack_blocks_two",
    "stack_bowls_three",
    "stack_bowls_two",
}
RETREAT_MERGE_TASKS = STACK_TASKS | {"hanging_mug"}
CHRONOLOGICAL_ARM_TASKS = {
    "place_object_basket",
    "stack_blocks_two",
    "stack_bowls_two",
}

EXPECTED_TASK_SLUGS = {
    "adjust_bottle", "beat_block_hammer", "blocks_ranking_rgb",
    "blocks_ranking_size", "click_alarmclock", "click_bell",
    "dump_bin_bigbin", "grab_roller", "handover_block", "handover_mic",
    "hanging_mug", "lift_pot", "move_can_pot", "move_pillbottle_pad",
    "move_playingcard_away", "move_stapler_pad", "open_laptop",
    "open_microwave", "pick_diverse_bottles", "pick_dual_bottles",
    "place_a2b_left", "place_a2b_right", "place_bread_basket",
    "place_bread_skillet", "place_burger_fries", "place_can_basket",
    "place_cans_plasticbox", "place_container_plate", "place_dual_shoes",
    "place_empty_cup", "place_fan", "place_mouse_pad",
    "place_object_basket", "place_object_scale", "place_object_stand",
    "place_phone_stand", "place_shoe", "press_stapler",
    "put_bottles_dustbin", "put_object_cabinet", "rotate_qrcode",
    "scan_object", "shake_bottle", "shake_bottle_horizontally",
    "stack_blocks_three", "stack_blocks_two", "stack_bowls_three",
    "stack_bowls_two", "stamp_seal", "turn_switch",
}

TASK_BUILDERS: dict[str, TaskBuilder] = {}


def register(*slugs: str) -> Callable[[TaskBuilder], TaskBuilder]:
    def decorator(builder: TaskBuilder) -> TaskBuilder:
        for slug in slugs:
            if slug in TASK_BUILDERS:
                raise ValueError(f"duplicate RoboTwin task builder: {slug}")
            TASK_BUILDERS[slug] = builder
        return builder
    return decorator


def build_place_can_basket_steps(
    events: tuple[GripperEvent, ...] | list[GripperEvent] | None,
    can: str,
    basket: str,
) -> list[StepSpec]:
    close_events = [event for event in events or [] if event.kind == "close"]
    can_arm = close_events[0].arm if close_events else None
    basket_arm = close_events[1].arm if len(close_events) > 1 else None
    if basket_arm is None and can_arm in {"left", "right"}:
        basket_arm = "left" if can_arm == "right" else "right"
    return (
        pair_steps(can, f"into the {basket}", can_arm)
        + [
            StepSpec(
                f"Move the {basket_arm} arm to the grasp pose of the {basket} while returning the {can_arm} arm to a neutral pose."
                if can_arm and basket_arm
                else f"Move to the grasp pose of the {basket}.",
                "move",
                basket_arm,
            ),
            StepSpec(close_gripper_text(basket_arm), "close", basket_arm),
            StepSpec(f"Lift the {basket} to a stable carrying height.", "move", basket_arm),
        ]
    )


def ordinal_object(index: int, base: str = "bottle") -> str:
    names = ["first", "second", "third", "fourth", "fifth"]
    if index < len(names):
        return f"{names[index]} {base}"
    return f"{base} {index + 1}"


def build_put_bottles_dustbin_steps(
    events: tuple[GripperEvent, ...] | list[GripperEvent] | None,
) -> list[StepSpec]:
    fallback = [
        step
        for obj in ["first bottle", "second bottle", "third bottle"]
        for step in pair_steps(obj, "into the dustbin")
    ]
    if not events:
        return fallback

    steps: list[StepSpec] = []
    i = 0
    obj_index = 0
    previous_place_arm: str | None = None
    usable = [event for event in events if event.kind in {"close", "open"}]
    while i < len(usable) and obj_index < 5:
        obj = ordinal_object(obj_index)
        if usable[i].kind != "close":
            i += 1
            continue
        first_close = usable[i]
        next_close = next((j for j in range(i + 1, len(usable)) if usable[j].kind == "close"), None)
        next_open = next((j for j in range(i + 1, len(usable)) if usable[j].kind == "open"), None)
        if next_open is None:
            steps.extend(grasp_steps(obj, first_close.arm))
            break

        if next_close is not None and next_close < next_open:
            carrier = first_close.arm
            receiver = usable[next_close].arm
            receiver_open = next(
                (j for j in range(next_close + 1, len(usable)) if usable[j].kind == "open" and usable[j].arm == receiver),
                None,
            )
            carrier_open = next(
                (j for j in range(next_close + 1, len(usable)) if usable[j].kind == "open" and usable[j].arm == carrier),
                None,
            )
            if receiver_open is None:
                receiver_open = next_open
            if previous_place_arm and previous_place_arm != carrier:
                steps.extend(
                    [
                        StepSpec(
                            f"Move the {carrier} arm to the grasp pose of the {obj} while returning the {previous_place_arm} arm to a neutral pose.",
                            "move",
                            carrier,
                        ),
                        StepSpec(close_gripper_text(carrier), "close", carrier),
                    ]
                )
            else:
                steps.extend(grasp_steps(obj, carrier))
            steps.append(StepSpec(f"Move the {obj} to the middle with the {carrier} arm.", "move"))
            steps.extend(grasp_steps(obj, receiver))
            if carrier_open is not None:
                steps.append(StepSpec(open_gripper_text(carrier), "open", carrier))
            steps.extend(
                [
                    StepSpec(
                        f"Move the {receiver} arm to place the {obj} into the dustbin while returning the {carrier} arm to a neutral pose.",
                        "move",
                        receiver,
                    ),
                    StepSpec(open_gripper_text(receiver), "open", receiver),
                ]
            )
            previous_place_arm = receiver
            i = receiver_open + 1
        else:
            arm = first_close.arm
            if previous_place_arm and previous_place_arm != arm:
                steps.extend(
                    [
                        StepSpec(
                            f"Move the {arm} arm to the grasp pose of the {obj} while returning the {previous_place_arm} arm to a neutral pose.",
                            "move",
                            arm,
                        ),
                        StepSpec(close_gripper_text(arm), "close", arm),
                    ]
                )
            else:
                steps.extend(grasp_steps(obj, arm))
            steps.extend(place_steps(obj, "into the dustbin", arm))
            previous_place_arm = arm
            i = next_open + 1
        obj_index += 1
    return steps or fallback


def build_put_object_cabinet_steps(ctx: TaskContext) -> list[StepSpec]:
    obj = prompt_cabinet_object(ctx.task_goal, ctx.object_a)
    close_events = [event for event in ctx.events if event.kind == "close"]
    open_events = [event for event in ctx.events if event.kind == "open"]
    object_arm = close_events[0].arm if close_events else None
    handle_arm = close_events[1].arm if len(close_events) > 1 else None
    pull_arm = handle_arm or (open_events[0].arm if open_events else None)
    if pull_arm:
        pull = f"Pull open the cabinet with the {pull_arm} arm."
    else:
        pull = "Pull open the cabinet."
    return [
        *grasp_steps(obj, object_arm),
        StepSpec(
            f"Move the {handle_arm} arm to the cabinet handle."
            if handle_arm
            else "Move to the cabinet handle.",
            "move",
            handle_arm,
        ),
        StepSpec(close_gripper_text(handle_arm), "close", handle_arm),
        StepSpec(pull, "move", pull_arm),
        StepSpec(move_to_place_text(obj, "inside the cabinet", object_arm), "move", object_arm),
        StepSpec(open_gripper_text(object_arm), "open", object_arm),
    ]


@register("blocks_ranking_rgb")
def blocks_ranking_rgb(ctx: TaskContext) -> list[StepSpec]:
    arms = parse_arm_sequence_from_prompt(ctx.task_goal, ["red block", "green block", "blue block"])
    items = [
        (info_obj(ctx.info, "{A}", "red block"), "at the right position", arms[0]),
        (info_obj(ctx.info, "{B}", "green block"), "at the middle position", arms[1]),
        (info_obj(ctx.info, "{C}", "blue block"), "at the left position", arms[2]),
    ]
    return [step for obj, dst, arm in items for step in pair_steps(obj, dst, arm)] + [
        StepSpec("Lift the arm after releasing the last block.", "move", arms[2])
    ]


@register("blocks_ranking_size")
def blocks_ranking_size(ctx: TaskContext) -> list[StepSpec]:
    arms = parse_arm_sequence_from_prompt(ctx.task_goal, ["small block", "medium block", "large block"])
    items = [
        (info_obj(ctx.info, "{C}", "small block"), "at the left position", arms[0]),
        (info_obj(ctx.info, "{B}", "medium block"), "at the middle position", arms[1]),
        (info_obj(ctx.info, "{A}", "large block"), "at the right position", arms[2]),
    ]
    return [step for obj, dst, arm in items for step in pair_steps(obj, dst, arm)] + [
        StepSpec("Lift the arm after releasing the last block.", "move", arms[2])
    ]


@register("stack_blocks_three")
def stack_blocks_three(ctx: TaskContext) -> list[StepSpec]:
    return (
        pair_steps("red block", "at the center as the base", ctx.arm_a)
        + pair_steps("green block", "on top of the red block", ctx.arm_b)
        + pair_steps("blue block", "on top of the green block", ctx.arm_c)
        + [StepSpec("Lift the arm after releasing the last block.", "move", ctx.arm_c)]
    )


@register("stack_blocks_two")
def stack_blocks_two(ctx: TaskContext) -> list[StepSpec]:
    return (
        pair_steps("red block", "at the center as the base", ctx.arm_a)
        + pair_steps("green block", "on top of the red block", ctx.arm_b)
        + [StepSpec("Lift the arm after releasing the last block.", "move", ctx.arm_b)]
    )


@register("stack_bowls_three")
def stack_bowls_three(ctx: TaskContext) -> list[StepSpec]:
    a, b, c = ctx.object_a, ctx.object_b, ctx.object_c
    return (
        pair_steps(a if a != b else "base bowl", "at the base position", ctx.arm_a)
        + pair_steps(b if a != b else "middle bowl", "inside the base bowl", ctx.arm_b)
        + pair_steps(c if c != "target object" else "top bowl", "inside the stacked bowls", ctx.arm_c)
        + [StepSpec("Lift the arm after releasing the last bowl.", "move", ctx.arm_c)]
    )


@register("stack_bowls_two")
def stack_bowls_two(ctx: TaskContext) -> list[StepSpec]:
    a, b = ctx.object_a, ctx.object_b
    return (
        pair_steps(a if a != b else "base bowl", "at the base position", ctx.arm_a)
        + pair_steps(b if a != b else "top bowl", "inside the base bowl", ctx.arm_b)
        + [StepSpec("Lift the arm after releasing the last bowl.", "move", ctx.arm_b)]
    )


def simple_move_builder(destination: str | Callable[[TaskContext], str]) -> TaskBuilder:
    def builder(ctx: TaskContext) -> list[StepSpec]:
        dst = destination(ctx) if callable(destination) else destination
        return pair_steps(ctx.object_a, dst, ctx.arm_a)
    return builder


TASK_BUILDERS["move_pillbottle_pad"] = simple_move_builder("onto the pad")
TASK_BUILDERS["move_playingcard_away"] = simple_move_builder("away from its initial position")
TASK_BUILDERS["move_stapler_pad"] = simple_move_builder(
    lambda ctx: f"onto the {prompt_mat_target(ctx.task_goal, f'{ctx.object_b} mat' if ctx.object_b != 'target object' else 'mat')}"
)
TASK_BUILDERS["place_shoe"] = simple_move_builder("onto the mat")


def avoid_arm_object_phrase(value: str, fallback: str) -> str:
    if re.fullmatch(r"(?:the\s+)?(?:left|right)\s+arm", value.strip(), flags=re.IGNORECASE):
        return fallback
    return value


def a2b_builder(side: str) -> TaskBuilder:
    def builder(ctx: TaskContext) -> list[StepSpec]:
        parsed = parse_a2b_objects_from_prompt(ctx.task_goal, side)
        moved, reference = parsed if parsed is not None else (ctx.object_a, ctx.object_b)
        moved = avoid_arm_object_phrase(moved, "object")
        reference = avoid_arm_object_phrase(reference, "reference object")
        target_side = "right" if side == "left" else "left"
        return pair_steps(moved, f"to the {target_side} of the {reference}", ctx.arm_a)
    return builder


TASK_BUILDERS["place_a2b_left"] = a2b_builder("left")
TASK_BUILDERS["place_a2b_right"] = a2b_builder("right")


@register("move_can_pot")
def move_can_pot(ctx: TaskContext) -> list[StepSpec]:
    return pair_steps(ctx.object_b, f"next to the {ctx.object_a}", ctx.arm_a)


@register("place_bread_skillet")
def place_bread_skillet(ctx: TaskContext) -> list[StepSpec]:
    bread = ctx.object_b if ctx.object_b != "target object" else "bread"
    skillet = ctx.object_a if ctx.object_a != "target object" else "skillet"
    bread_arm = next(
        (event.arm for event in ctx.events if event.kind == "open"),
        prompt_arm_near_object(ctx.task_goal, r"bread") or ctx.arm_a,
    )
    skillet_arm = "left" if bread_arm == "right" else "right" if bread_arm == "left" else None
    return [
        StepSpec(
            f"Move the {bread_arm} arm to the grasp pose of the {bread} while moving the {skillet_arm} arm to the grasp pose of the {skillet}."
            if bread_arm and skillet_arm
            else f"Move to the grasp pose of the {bread} and the {skillet}.",
            "move",
        ),
        StepSpec(
            "Close the grippers of both arms.",
            "close",
        ),
        StepSpec("Lift both objects to the middle position.", "move"),
        StepSpec(
            f"Move the {skillet_arm} arm to bring the {skillet} to the placement position."
            if bread_arm and skillet_arm
            else f"Move the {skillet} to the placement position.",
            "move",
            skillet_arm,
        ),
        StepSpec(
            f"Move the {bread_arm} arm above the {skillet}."
            if bread_arm and skillet_arm
            else f"Move the {bread} above the {skillet}.",
            "move",
            bread_arm,
        ),
        StepSpec(
            f"Move the {bread_arm} arm to place the {bread} into the {skillet}."
            if bread_arm and skillet_arm
            else f"Move to place the {bread} into the {skillet}.",
            "move",
            bread_arm,
        ),
        StepSpec(
            open_gripper_text(bread_arm),
            "open",
            bread_arm,
        ),
    ]


@register("place_container_plate")
def place_container_plate(ctx: TaskContext) -> list[StepSpec]:
    obj = ctx.object_b if ctx.object_b != "target object" else "container"
    return pair_steps(ctx.object_b, f"onto the {ctx.object_a}", ctx.arm_a) + [
        StepSpec(
            f"Lift the {ctx.arm_a} arm after releasing the {obj}." if ctx.arm_a else f"Lift after releasing the {obj}.",
            "move",
            ctx.arm_a,
        ),
    ]


@register("place_empty_cup")
def place_empty_cup(ctx: TaskContext) -> list[StepSpec]:
    close_events = [event for event in ctx.events if event.kind == "close"]
    arm = close_events[0].arm if close_events else ctx.arm_a
    return [
        StepSpec(
            close_gripper_text(arm),
            "close",
            arm,
            terminates_on=("current_arm_motion", "other_arm_motion"),
        ),
        StepSpec(move_to_grasp_text("cup", arm), "move", arm),
        StepSpec(close_gripper_text(arm), "close", arm),
        StepSpec(move_to_place_text("cup", "onto the coaster", arm), "move", arm),
        StepSpec(open_gripper_text(arm), "open", arm),
        StepSpec(
            f"Lift the {arm} arm after releasing the cup." if arm else "Lift after releasing the cup.",
            "move",
            arm,
        ),
    ]


@register("place_mouse_pad")
def place_mouse_pad(ctx: TaskContext) -> list[StepSpec]:
    obj = prompt_mouse_object(ctx.task_goal, ctx.object_a if ctx.object_a != "target object" else "mouse")
    return pair_steps(obj, "onto the mat", ctx.arm_a)


@register("place_object_scale")
def place_object_scale(ctx: TaskContext) -> list[StepSpec]:
    obj = prompt_object_before_target(ctx.task_goal, r"(?:electronicscale|electronic scale|scale)", ctx.object_b)
    return pair_steps(obj, "onto the electronic scale", ctx.arm_a)


@register("place_object_stand")
def place_object_stand(ctx: TaskContext) -> list[StepSpec]:
    obj = prompt_object_before_target(ctx.task_goal, r"(?:displaystand|display stand|stand)", ctx.object_a)
    return pair_steps(obj, "onto the display stand", ctx.arm_a)


@register("place_phone_stand")
def place_phone_stand(ctx: TaskContext) -> list[StepSpec]:
    obj = ctx.object_a if ctx.object_a != "target object" else "phone"
    target = ctx.object_b if ctx.object_b != "target object" else "phone stand"
    return pair_steps(obj, f"onto the {target}", ctx.arm_a)


@register("place_fan")
def place_fan(ctx: TaskContext) -> list[StepSpec]:
    fan = ctx.object_a if ctx.object_a != "target object" else "fan"
    dst = prompt_fan_target(
        ctx.task_goal,
        f"{ctx.object_b} mat" if ctx.object_b != "target object" else "mat",
    )
    return grasp_steps(fan, ctx.arm_a) + [
        StepSpec(
            f"Move the {ctx.arm_a} arm to place the {fan} onto the {dst} and face it toward the robot."
            if ctx.arm_a
            else f"Move to place the {fan} onto the {dst} and face it toward the robot.",
            "move",
            ctx.arm_a,
        ),
        StepSpec(open_gripper_text(ctx.arm_a), "open", ctx.arm_a),
    ]


@register("place_burger_fries")
def place_burger_fries(ctx: TaskContext) -> list[StepSpec]:
    burger = ctx.object_a if ctx.object_a != "target object" else "hamburger"
    fries = ctx.object_c if ctx.object_c != "target object" else "french fries"
    tray = ctx.object_b if ctx.object_b != "target object" else "tray"
    return dual_pick_place_then_return(burger, fries, f"onto the {tray}")


def clean_bread_prompt_text(value: str) -> str:
    value = value.replace("bread basket", "breadbasket")
    value = clean_duplicate_articles(value)
    return " ".join(value.strip().strip(" .;,").split())


def clean_duplicate_articles(value: str) -> str:
    previous = None
    while previous != value:
        previous = value
        value = re.sub(r"\b(the|a|an)\s+\1\b", r"\1", value, flags=re.IGNORECASE)
    return " ".join(value.split())


def breadbasket_preposition_pattern() -> re.Pattern[str]:
    actions = (
        r"place|put|set|drop|move|transfer|grab|grasp|pick(?:\s+up)?|"
        r"lift|take|shift"
    )
    return re.compile(
        r"\b(?:into|inside|onto|in|to|for)\s+(?:(?:the|a|an)\s+){0,3}"
        rf"(?!(?:{actions})\b)"
        r"(?P<target>[^.,;]*?\b(?:breadbasket|basket)\b[^.,;]*)",
        flags=re.IGNORECASE,
    )


def contains_bread_object(value: str) -> bool:
    return bool(re.search(r"\b(?:bread|loaf)\b", value, flags=re.IGNORECASE))


def strip_bread_action_prefix(value: str) -> str:
    patterns = [
        r"^(?:simultaneously\s+)?use\s+(?:the\s+)?(?:left|right|dual)\s+arm\s+to\s+(?:grab|grasp|drop|move|place|set|pick(?:\s+up)?|lift)\s+",
        r"^use\s+(?:two|both)\s+arms?\s+(?:and\s+)?(?:to\s+)?(?:grab|grasp|drop|move|place|set|pick(?:\s+up)?|lift)\s+",
        r"^(?:simultaneously\s+)?(?:grab|grasp|pick(?:\s+up)?|take|lift|drop|shift|move|put|place|set)\s+(?:both\s+)?(?:two\s+breads?\s+)?",
    ]
    previous = None
    while previous != value:
        previous = value
        for pattern in patterns:
            value = re.sub(pattern, "", value, count=1, flags=re.IGNORECASE)
    return value


def normalize_bread_object(value: str) -> str:
    value = clean_bread_prompt_text(value)
    value = re.split(
        r",?\s*(?:then|and)\s+(?:place|put|set|drop|move|transfer)\b",
        value,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    value = re.sub(
        r"\s+(?:using|with)\s+(?:the\s+)?(?:left|right|dual|both|one|two)\s+arms?\b.*$",
        "",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(r"\s+with\s+an?\s+arm\b.*$", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\s+\b(?:at once|together|quickly)\b", "", value, flags=re.IGNORECASE)
    value = strip_bread_action_prefix(value)
    value = re.sub(r"^(?:both\s+)?(?:the|a|an)\s+", "", value, flags=re.IGNORECASE)
    return clean_bread_prompt_text(value)


def split_bread_objects(segment: str) -> list[str]:
    segment = normalize_bread_object(segment)
    comma_parts = [
        normalize_bread_object(part)
        for part in re.split(r"\s*,\s*", segment)
        if normalize_bread_object(part)
    ]
    if len(comma_parts) >= 2 and all(contains_bread_object(part) for part in comma_parts[:2]):
        return comma_parts[:2]

    candidates: list[tuple[int, int, str, str]] = []
    for match in re.finditer(r"\s+and\s+", segment, flags=re.IGNORECASE):
        left = segment[: match.start()]
        right = segment[match.end() :]
        if not contains_bread_object(left) or not contains_bread_object(right):
            continue
        right_starts_with_article = bool(
            re.match(r"(?:both\s+)?(?:the|a|an)\s+", right, flags=re.IGNORECASE)
        )
        candidates.append((0 if right_starts_with_article else 1, match.start(), left, right))
    if candidates:
        _, _, left, right = sorted(candidates)[0]
        return [normalize_bread_object(left), normalize_bread_object(right)]
    return [segment] if segment else []


def prompt_bread_basket_target(task_goal: str, fallback: str = "breadbasket") -> str:
    text = clean_bread_prompt_text(task_goal)
    matches = list(breadbasket_preposition_pattern().finditer(text))
    if not matches:
        return fallback
    target = matches[-1].group("target")
    target = re.split(
        r"\b(?:after|before|using\s+(?:the\s+)?(?:left|right|dual|both|one|two)\s+arms?|with\s+(?:the\s+)?(?:left|right|dual|both|one|two)\s+arms?|with\s+an?\s+arm)\b",
        target,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    target = re.sub(r"^(?:the|a|an)\s+", "", target, flags=re.IGNORECASE)
    return clean_bread_prompt_text(target) or fallback


def prompt_bread_basket_objects(task_goal: str) -> list[str]:
    text = clean_bread_prompt_text(task_goal)
    target_match = None
    matches = list(breadbasket_preposition_pattern().finditer(text))
    if matches:
        target_match = matches[-1]
    segment = text[: target_match.start()] if target_match else text
    segment = re.split(
        r",\s*(?:then\s+)?(?:put|place|drop|set|move|transfer)\b",
        segment,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    segment = re.split(
        r"\b(?:then|and)\s+(?:place|put|set|drop|move|transfer)\b",
        segment,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    objects = [item for item in split_bread_objects(segment) if contains_bread_object(item)]
    return objects[:2]


def distinct_bread_objects(first: str, second: str) -> tuple[str, str]:
    if first == second:
        return f"first {first}", f"second {second}"
    return first, second


def bread_pick_place_steps(obj: str, dst: str, arm: str | None) -> list[StepSpec]:
    lift_text = f"Lift the {arm} arm after releasing the {obj}." if arm else f"Lift after releasing the {obj}."
    return [
        StepSpec(move_to_grasp_text(obj, arm), "move", arm),
        StepSpec(close_gripper_text(arm), "close", arm),
        StepSpec(move_to_place_text(obj, dst, arm), "move", arm),
        StepSpec(open_gripper_text(arm), "open", arm),
        StepSpec(lift_text, "move", arm),
    ]


@register("place_bread_basket")
def place_bread_basket(ctx: TaskContext) -> list[StepSpec]:
    close_events = [event for event in ctx.events if event.kind == "close"]
    target = prompt_bread_basket_target(ctx.task_goal)
    dst = f"into the {target}"
    objects = prompt_bread_basket_objects(ctx.task_goal)
    if not objects:
        objects = ["bread"]

    simultaneous_dual = (
        len(close_events) >= 2
        and close_events[0].arm != close_events[1].arm
        and abs(close_events[0].frame - close_events[1].frame) <= 3
    )
    if simultaneous_dual:
        while len(objects) < 2:
            objects.append("bread")
        first, second = distinct_bread_objects(objects[0], objects[1])
        first_arm = close_events[0].arm
        second_arm = close_events[1].arm
        return [
            StepSpec(
                f"Move the {first_arm} arm to the grasp pose of the {first} while moving the {second_arm} arm to the grasp pose of the {second}.",
                "move",
            ),
            StepSpec("Close the grippers of both arms.", "close"),
            StepSpec(
                f"Lift the {first} and the {second} to the middle position with the corresponding arms.",
                "move",
            ),
            StepSpec(move_to_place_text(first, dst, first_arm), "move", first_arm),
            StepSpec(open_gripper_text(first_arm), "open", first_arm),
            StepSpec(f"Lift the {first_arm} arm after releasing the {first}.", "move", first_arm),
            StepSpec(
                f"Move the {second_arm} arm to the place pose of the {second} {dst} "
                f"while returning the {first_arm} arm to a neutral pose.",
                "move",
                second_arm,
                terminates_on=("gripper_open",),
            ),
            StepSpec(open_gripper_text(second_arm), "open", second_arm),
            StepSpec(f"Lift the {second_arm} arm after releasing the {second}.", "move", second_arm),
        ]

    usable_closes = close_events[:2] if close_events else []
    if not usable_closes:
        return bread_pick_place_steps(objects[0], dst, ctx.arm_a)
    steps: list[StepSpec] = []
    for index, close_event in enumerate(usable_closes):
        obj = objects[index] if index < len(objects) else f"bread {index + 1}"
        steps.extend(bread_pick_place_steps(obj, dst, close_event.arm))
    return steps


def dual_container_builder(
    first: str,
    second: str,
    fallback_target: str,
    target_attr: str,
    *,
    lift_between_releases: bool = True,
    lift_after_final_release: bool = True,
    first_place_terminates_on: tuple[str, ...] = (),
) -> TaskBuilder:
    def builder(ctx: TaskContext) -> list[StepSpec]:
        target = getattr(ctx, target_attr)
        if target == "target object":
            target = fallback_target
        return dual_pick_place_then_return(
            first,
            second,
            f"into the {target}",
            lift_between_releases=lift_between_releases,
            lift_after_final_release=lift_after_final_release,
            first_place_terminates_on=first_place_terminates_on,
        )
    return builder


TASK_BUILDERS["place_cans_plasticbox"] = dual_container_builder(
    "right can", "left can", "plastic box", "object_b",
    lift_between_releases=False,
    first_place_terminates_on=("gripper_open", "gripper_close"),
)
TASK_BUILDERS["place_dual_shoes"] = dual_container_builder(
    "right shoe", "left shoe", "shoe box", "object_b",
    lift_between_releases=False, lift_after_final_release=False,
    first_place_terminates_on=("gripper_open", "gripper_close"),
)


@register("place_can_basket")
def place_can_basket(ctx: TaskContext) -> list[StepSpec]:
    can = ctx.object_a if ctx.object_a != "target object" else "can"
    basket = ctx.object_b if ctx.object_b != "target object" else "basket"
    return build_place_can_basket_steps(ctx.events, can, basket)


@register("place_object_basket")
def place_object_basket(ctx: TaskContext) -> list[StepSpec]:
    obj = prompt_basket_object(ctx.task_goal, ctx.object_a)
    basket = ctx.object_b if ctx.object_b != "target object" else "basket"
    return build_place_can_basket_steps(ctx.events, obj, basket)


@register("put_bottles_dustbin")
def put_bottles_dustbin(ctx: TaskContext) -> list[StepSpec]:
    return build_put_bottles_dustbin_steps(ctx.events)


TASK_BUILDERS["put_object_cabinet"] = build_put_object_cabinet_steps


@register("handover_block")
def handover_block(ctx: TaskContext) -> list[StepSpec]:
    close_events = [event for event in ctx.events if event.kind == "close"]
    open_events = [event for event in ctx.events if event.kind == "open"]
    first_arm = close_events[0].arm if close_events else ctx.arm_a or "left"
    receiving_arm = (
        close_events[1].arm
        if len(close_events) > 1
        else ("right" if first_arm == "left" else "left")
    )
    first_open_arm = open_events[0].arm if open_events else first_arm
    receiving_open_arm = open_events[1].arm if len(open_events) > 1 else receiving_arm
    return [
        StepSpec(move_to_grasp_text("red block", first_arm), "move", first_arm),
        StepSpec(close_gripper_text(first_arm), "close", first_arm),
        StepSpec(f"Move the red block to the handover position with the {first_arm} arm.", "move", first_arm),
        StepSpec(move_to_grasp_text("red block", receiving_arm), "move", receiving_arm),
        StepSpec(close_gripper_text(receiving_arm), "close", receiving_arm),
        StepSpec(open_gripper_text(first_open_arm), "open", first_open_arm),
        StepSpec(
            f"Return the {first_open_arm} arm to a neutral pose.",
            "move",
            first_open_arm,
        ),
        StepSpec(
            f"Return the {first_open_arm} arm to a neutral pose while moving the {receiving_arm} arm to place the red block on the target pad.",
            "move",
            receiving_arm,
            terminates_on=("gripper_open",),
        ),
        StepSpec(open_gripper_text(receiving_open_arm), "open", receiving_open_arm),
    ]


@register("handover_mic")
def handover_mic(ctx: TaskContext) -> list[StepSpec]:
    obj = ctx.object_a if ctx.object_a != "target object" else "microphone"
    close_events = [event for event in ctx.events if event.kind == "close"]
    open_events = [event for event in ctx.events if event.kind == "open"]
    first_arm = close_events[0].arm if close_events else ctx.arm_a or "left"
    receiving_arm = (
        close_events[1].arm
        if len(close_events) > 1
        else ("right" if first_arm == "left" else "left")
    )
    release_arm = open_events[0].arm if open_events else first_arm
    return [
        StepSpec(move_to_grasp_text(obj, first_arm), "move", first_arm),
        StepSpec(close_gripper_text(first_arm), "close", first_arm),
        StepSpec(f"Move the {obj} to the handover position with the {first_arm} arm.", "move", first_arm),
        StepSpec(move_to_grasp_text(obj, receiving_arm), "move", receiving_arm),
        StepSpec(close_gripper_text(receiving_arm), "close", receiving_arm),
        StepSpec(open_gripper_text(release_arm), "open", release_arm),
        StepSpec(
            f"Return the {release_arm} arm to a neutral pose after releasing the {obj}.",
            "final",
            release_arm,
        ),
    ]


@register("hanging_mug")
def hanging_mug(ctx: TaskContext) -> list[StepSpec]:
    mug = ctx.object_a if ctx.object_a != "target object" else "mug"
    return (
        grasp_steps(mug, ctx.arm_a)
        + [
            StepSpec(
                f"Move the {ctx.arm_a} arm with the {mug} to the middle position."
                if ctx.arm_a
                else f"Move the {mug} to the middle position.",
                "move",
                ctx.arm_a,
            ),
            StepSpec(
                f"Move the {ctx.arm_a} arm to place the {mug} down, rotating it if necessary."
                if ctx.arm_a
                else f"Move to place the {mug} down, rotating it if necessary.",
                "move",
                ctx.arm_a,
            ),
            StepSpec(open_gripper_text(ctx.arm_a), "open", ctx.arm_a),
        ]
        + grasp_steps(mug, ctx.arm_b)
        + place_steps(mug, "on the rack", ctx.arm_b)
        + [
            StepSpec(
                f"Return the {ctx.arm_b} arm to a neutral pose after releasing the {mug}."
                if ctx.arm_b
                else f"Return to a neutral pose after releasing the {mug}.",
                "move",
                ctx.arm_b,
            ),
        ]
    )


@register("scan_object")
def scan_object(ctx: TaskContext) -> list[StepSpec]:
    obj_arm, scanner_arm = prompt_scan_arms(ctx.task_goal, ctx.arm_a, ctx.arm_b)
    return [
        StepSpec(
            f"Move the {obj_arm} arm to the grasp pose of the {ctx.object_a} while moving the {scanner_arm} arm to the grasp pose of the {ctx.object_b}.",
            "move",
        ),
        StepSpec(
            "Close the grippers of both arms.",
            "close",
        ),
        StepSpec(
            f"Lift both arms to raise the {ctx.object_a} and the {ctx.object_b} to the scan position.",
            "move",
            terminates_on=("both_arms_settle",),
        ),
        StepSpec(
            f"Lift the {ctx.object_a} to the scan position and angle with the {obj_arm} arm.",
            "move",
            obj_arm,
        ),
        StepSpec(
            f"Move the {ctx.object_b} with the {scanner_arm} arm to scan the held {ctx.object_a}.",
            "move",
            scanner_arm,
        ),
    ]


@register("pick_diverse_bottles", "pick_dual_bottles")
def pick_dual_objects(ctx: TaskContext) -> list[StepSpec]:
    first, second = distinct_pair(ctx.object_a, ctx.object_b)
    return [
        StepSpec(
            f"Move the left arm to the grasp pose of the {first} while moving the right arm to the grasp pose of the {second}.",
            "move",
        ),
        StepSpec("Close the grippers of both arms.", "close"),
        StepSpec(
            f"Lift the {first} and the {second} to the middle position with the corresponding arms.",
            "move",
        ),
    ]


def object_action_builder(
    fallback: str,
    action: Callable[[str, TaskContext], str],
    *,
    final_event_kind: str = "final",
) -> TaskBuilder:
    def builder(ctx: TaskContext) -> list[StepSpec]:
        obj = ctx.object_a if ctx.object_a != "target object" else fallback
        return grasp_steps(obj, ctx.arm_a) + [StepSpec(action(obj, ctx), final_event_kind, ctx.arm_a)]
    return builder


TASK_BUILDERS["adjust_bottle"] = object_action_builder("bottle", lambda obj, ctx: f"Lift the {obj} upright.", final_event_kind="move")
TASK_BUILDERS["beat_block_hammer"] = object_action_builder("hammer", lambda obj, ctx: f"Use the {obj} to hit the block.")


@register("rotate_qrcode")
def rotate_qrcode(ctx: TaskContext) -> list[StepSpec]:
    obj = ctx.object_a if ctx.object_a != "target object" else "payment sign"
    return grasp_steps(obj, ctx.arm_a) + [
        StepSpec(f"Rotate the {obj} until the QR code faces the robot.", "move", ctx.arm_a),
        StepSpec(open_gripper_text(ctx.arm_a), "open", ctx.arm_a),
    ]


TASK_BUILDERS["shake_bottle"] = object_action_builder("bottle", lambda obj, ctx: f"Shake the {obj}.", final_event_kind="move")
TASK_BUILDERS["shake_bottle_horizontally"] = object_action_builder(
    "bottle",
    lambda obj, ctx: f"Shake the {obj} horizontally.",
    final_event_kind="move",
)


@register("dump_bin_bigbin")
def dump_bin_bigbin(ctx: TaskContext) -> list[StepSpec]:
    obj = ctx.object_a if ctx.object_a != "target object" else "trash bin"
    gripper_events = [event for event in ctx.events if event.kind in {"close", "open"}]
    close_events = [event for event in gripper_events if event.kind == "close"]
    first_arm = (close_events[0].arm if close_events else None) or ctx.arm_a or "left"
    second_arm = (
        close_events[1].arm
        if len(close_events) > 1
        else ctx.arm_b or ("right" if first_arm == "left" else "left")
    )
    if ctx.events and len(close_events) < 2:
        return grasp_steps(obj, first_arm) + [
            StepSpec(f"Lift the {obj} and pour its contents into the big bin.", "final", first_arm)
        ]
    return [
        StepSpec(move_to_grasp_text(obj, first_arm), "move", first_arm),
        StepSpec(close_gripper_text(first_arm), "close", first_arm),
        StepSpec(f"Move the {first_arm} arm with the {obj} to the middle position.", "move", first_arm),
        StepSpec(open_gripper_text(first_arm), "open", first_arm),
        StepSpec(
            f"Move the {second_arm} arm to the middle position near the {obj} while returning the {first_arm} arm to the default pose.",
            "move",
            second_arm,
        ),
        StepSpec(close_gripper_text(second_arm), "close", second_arm),
        StepSpec(f"Lift the {obj} with the {second_arm} arm and pour its contents into the big bin.", "move", second_arm),
    ]


@register("grab_roller")
def grab_roller(ctx: TaskContext) -> list[StepSpec]:
    obj = ctx.object_a if ctx.object_a != "target object" else "roller"
    return [
        StepSpec(
            f"Move both arms to the grasp pose of the {obj}.",
            "move",
        ),
        StepSpec("Close the grippers of both arms.", "close"),
        StepSpec(f"Lift the {obj} off the table with both arms.", "move"),
    ]


@register("lift_pot")
def lift_pot(ctx: TaskContext) -> list[StepSpec]:
    obj = ctx.object_a if ctx.object_a != "target object" else "pot"
    return [
        StepSpec(
            "Partially close the grippers of both arms.",
            "close",
        ),
        StepSpec(
            f"Move both arms to the grasp pose of the {obj}.",
            "move",
        ),
        StepSpec(
            "Close the grippers of both arms.",
            "close",
        ),
        StepSpec(
            f"Lift the {obj} with both arms.",
            "move",
        ),
    ]


def open_builder(fallback: str, part: str) -> TaskBuilder:
    def builder(ctx: TaskContext) -> list[StepSpec]:
        obj = ctx.object_a if ctx.object_a != "target object" else fallback
        return grasp_steps(f"{obj} {part}", ctx.arm_a) + [
            StepSpec(f"Open the {obj} {part}.", "final"),
        ]
    return builder


TASK_BUILDERS["open_laptop"] = open_builder("laptop", "lid")


@register("open_microwave")
def open_microwave(ctx: TaskContext) -> list[StepSpec]:
    obj = ctx.object_a if ctx.object_a != "target object" else "microwave"
    part = "door"
    arm = ctx.arm_a
    close_events = [
        event for event in ctx.events
        if event.kind == "close" and (arm is None or event.arm == arm)
    ]
    steps = grasp_steps(f"{obj} {part}", arm)
    # RoboTwin open_microwave episodes often release the door handle and
    # re-grasp it partway through (close -> open -> close). The default
    # open_builder folds the whole pull into a single final span that only
    # terminates at episode_end, burying the re-grasp gripper events. Split
    # each extra close into its own pull / release / re-grasp subtasks so the
    # gripper events become first-class segment boundaries.
    for _ in close_events[1:]:
        steps.extend([
            StepSpec(
                f"Pull the {obj} {part} open partway.",
                "move",
                arm,
                terminates_on=("gripper_open",),
            ),
            StepSpec(open_gripper_text(arm), "open", arm),
            StepSpec(
                move_to_grasp_text(f"{obj} {part}", arm),
                "move",
                arm,
                terminates_on=("gripper_close",),
            ),
            StepSpec(close_gripper_text(arm), "close", arm),
        ])
    final_text = (
        f"Open the {obj} {part}."
        if len(close_events) <= 1
        else f"Continue opening the {obj} {part}."
    )
    steps.append(StepSpec(final_text, "final"))
    return steps


@register("stamp_seal")
def stamp_seal(ctx: TaskContext) -> list[StepSpec]:
    seal = ctx.object_a if ctx.object_a != "target object" else "seal"
    target = ctx.object_b if ctx.object_b != "target object" and not is_color_name(ctx.object_b) else "target area"
    return grasp_steps(seal, ctx.arm_a) + [
        StepSpec(f"Move the {ctx.arm_a} arm above the {target}." if ctx.arm_a else f"Move above the {target}.", "move", ctx.arm_a),
        StepSpec(
            f"Open the gripper of the {ctx.arm_a} arm to release the {seal} onto the {target}."
            if ctx.arm_a
            else f"Open the gripper to release the {seal} onto the {target}.",
            "open",
            ctx.arm_a,
        ),
    ]


def single_press_builder(text: str | Callable[[TaskContext], str]) -> TaskBuilder:
    def builder(ctx: TaskContext) -> list[StepSpec]:
        value = text(ctx) if callable(text) else text
        return [
            StepSpec(f"Move the {ctx.arm_a} arm to the operation pose." if ctx.arm_a else "Move to the operation pose.", "move", ctx.arm_a),
            StepSpec(close_gripper_text(ctx.arm_a), "close", ctx.arm_a),
            StepSpec(value, "press", ctx.arm_a),
            StepSpec(f"Lift the {ctx.arm_a} arm after pressing." if ctx.arm_a else "Lift after pressing.", "move", ctx.arm_a),
        ]
    return builder


TASK_BUILDERS["click_alarmclock"] = single_press_builder("Click the alarm clock button.")
TASK_BUILDERS["click_bell"] = single_press_builder("Press the top of the bell.")


@register("press_stapler")
def press_stapler(ctx: TaskContext) -> list[StepSpec]:
    return [
        StepSpec(
            f"Move the {ctx.arm_a} arm to the stapler pressing pose."
            if ctx.arm_a
            else "Move to the stapler pressing pose.",
            "move",
            ctx.arm_a,
        ),
        StepSpec(close_gripper_text(ctx.arm_a), "close", ctx.arm_a),
        StepSpec("Press down the stapler.", "press", ctx.arm_a),
    ]


@register("turn_switch")
def turn_switch(ctx: TaskContext) -> list[StepSpec]:
    switch = ctx.object_a if ctx.object_a != "target object" else "switch"
    return [
        StepSpec(close_gripper_text(ctx.arm_a), "close", ctx.arm_a),
        StepSpec(f"Operate the {switch}.", "press", ctx.arm_a),
    ]


def build_steps(
    slug: str,
    task_goal: str,
    info: dict[str, str],
    events: list[GripperEvent] | None = None,
) -> list[StepSpec]:
    ctx = TaskContext.create(
        slug,
        task_goal,
        info,
        events,
        object_parser=info_obj,
        arm_parser=info_arm,
    )
    builder = TASK_BUILDERS.get(slug)
    if builder is None:
        return [StepSpec(task_goal, "final")]
    return builder(ctx)


def canonical_task_goal(slug: str, task_goal: str) -> str:
    if slug == "blocks_ranking_rgb":
        return "Arrange the blue block, green block, and red block from left to right."
    if slug == "blocks_ranking_size":
        return "Arrange the small block, medium block, and large block from left to right."
    if slug == "place_bread_basket":
        return clean_duplicate_articles(task_goal)
    return task_goal


def validate_task_registry() -> None:
    missing = EXPECTED_TASK_SLUGS - TASK_BUILDERS.keys()
    extra = TASK_BUILDERS.keys() - EXPECTED_TASK_SLUGS
    if missing or extra:
        raise ValueError(
            f"invalid RoboTwin task registry: missing={sorted(missing)}, extra={sorted(extra)}"
        )


validate_task_registry()
