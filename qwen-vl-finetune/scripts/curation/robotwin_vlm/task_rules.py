"""Task-specific subtask composition built from reusable primitives."""

from __future__ import annotations

from collections.abc import Callable

from .models import GripperEvent, StepSpec, TaskBuilder, TaskContext
from .primitives import (
    distinct_pair,
    dual_pick_place_then_return,
    handover_steps,
    pair_steps,
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
    return [
        StepSpec(pick_text(can, can_arm), "close", can_arm),
        StepSpec(place_text(can, f"into the {basket}", can_arm), "open", can_arm),
        StepSpec(
            f"Return the {can_arm} arm to a neutral pose while the {basket_arm} arm grasps the {basket}."
            if can_arm and basket_arm
            else f"Prepare to grasp the {basket}.",
            "close",
            basket_arm,
        ),
        StepSpec(f"Lift the {basket}.", "final", basket_arm),
    ]


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
            steps.append(StepSpec(pick_text(obj, first_close.arm), "close", first_close.arm))
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
                steps.append(
                    StepSpec(
                        f"Return the {previous_place_arm} arm to a neutral pose while grasping the {obj} with the {carrier} arm.",
                        "close",
                        carrier,
                    )
                )
            else:
                steps.append(StepSpec(pick_text(obj, carrier), "close", carrier))
            steps.append(StepSpec(f"Move the {obj} to the middle with the {carrier} arm.", "midpoint"))
            steps.append(StepSpec(f"Grasp the {obj} with the {receiver} arm.", "close", receiver))
            release_arm = carrier if carrier_open is not None else carrier
            steps.append(
                StepSpec(
                    f"Open and return the {release_arm} arm to a neutral pose while placing the {obj} into the dustbin with the {receiver} arm.",
                    "open",
                    receiver,
                )
            )
            previous_place_arm = receiver
            i = receiver_open + 1
        else:
            arm = first_close.arm
            if previous_place_arm and previous_place_arm != arm:
                steps.append(
                    StepSpec(
                        f"Return the {previous_place_arm} arm to a neutral pose while grasping the {obj} with the {arm} arm.",
                        "close",
                        arm,
                    )
                )
            else:
                steps.append(StepSpec(pick_text(obj, arm), "close", arm))
            steps.append(StepSpec(place_text(obj, "into the dustbin", arm), "open", arm))
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
    if pull_arm and object_arm and pull_arm != object_arm:
        pull = f"Pull open the cabinet with the {pull_arm} arm while holding the {obj} with the {object_arm} arm."
    elif pull_arm:
        pull = f"Pull open the cabinet with the {pull_arm} arm."
    else:
        pull = "Pull open the cabinet."
    return [
        StepSpec(pick_text(obj, object_arm), "close", object_arm),
        StepSpec(f"Grasp the cabinet handle with the {handle_arm} arm." if handle_arm else "Grasp the cabinet handle.", "close", handle_arm),
        StepSpec(pull, "open"),
        StepSpec(f"Place the {obj} inside the cabinet with the {object_arm} arm." if object_arm else f"Place the {obj} inside the cabinet.", "final", object_arm),
    ]


@register("blocks_ranking_rgb")
def blocks_ranking_rgb(ctx: TaskContext) -> list[StepSpec]:
    arms = parse_arm_sequence_from_prompt(ctx.task_goal, ["red block", "green block", "blue block"])
    items = [
        (info_obj(ctx.info, "{A}", "red block"), "at the right position", arms[0]),
        (info_obj(ctx.info, "{B}", "green block"), "at the middle position", arms[1]),
        (info_obj(ctx.info, "{C}", "blue block"), "at the left position", arms[2]),
    ]
    return [step for obj, dst, arm in items for step in pair_steps(obj, dst, arm)]


@register("blocks_ranking_size")
def blocks_ranking_size(ctx: TaskContext) -> list[StepSpec]:
    arms = parse_arm_sequence_from_prompt(ctx.task_goal, ["small block", "medium block", "large block"])
    items = [
        (info_obj(ctx.info, "{C}", "small block"), "at the left position", arms[0]),
        (info_obj(ctx.info, "{B}", "medium block"), "at the middle position", arms[1]),
        (info_obj(ctx.info, "{A}", "large block"), "at the right position", arms[2]),
    ]
    return [step for obj, dst, arm in items for step in pair_steps(obj, dst, arm)]


@register("stack_blocks_three")
def stack_blocks_three(ctx: TaskContext) -> list[StepSpec]:
    return (
        pair_steps("red block", "at the center as the base", ctx.arm_a)
        + pair_steps("green block", "on top of the red block", ctx.arm_b)
        + pair_steps("blue block", "on top of the green block", ctx.arm_c)
    )


@register("stack_blocks_two")
def stack_blocks_two(ctx: TaskContext) -> list[StepSpec]:
    return pair_steps("red block", "at the center as the base", ctx.arm_a) + pair_steps(
        "green block", "on top of the red block", ctx.arm_b
    )


@register("stack_bowls_three")
def stack_bowls_three(ctx: TaskContext) -> list[StepSpec]:
    a, b, c = ctx.object_a, ctx.object_b, ctx.object_c
    return (
        pair_steps(a if a != b else "base bowl", "at the base position", ctx.arm_a)
        + pair_steps(b if a != b else "middle bowl", "inside the base bowl", ctx.arm_b)
        + pair_steps(c if c != "target object" else "top bowl", "inside the stacked bowls", ctx.arm_c)
    )


@register("stack_bowls_two")
def stack_bowls_two(ctx: TaskContext) -> list[StepSpec]:
    a, b = ctx.object_a, ctx.object_b
    return pair_steps(a if a != b else "base bowl", "at the base position", ctx.arm_a) + pair_steps(
        b if a != b else "top bowl", "inside the base bowl", ctx.arm_b
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


def a2b_builder(side: str) -> TaskBuilder:
    def builder(ctx: TaskContext) -> list[StepSpec]:
        parsed = parse_a2b_objects_from_prompt(ctx.task_goal, side)
        moved, reference = parsed if parsed is not None else (ctx.object_a, ctx.object_b)
        return pair_steps(moved, f"to the {side} of the {reference}", ctx.arm_a)
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
    bread_arm = prompt_arm_near_object(ctx.task_goal, r"bread") or next(
        (event.arm for event in ctx.events if event.kind == "open"),
        ctx.arm_a,
    )
    skillet_arm = "left" if bread_arm == "right" else "right" if bread_arm == "left" else None
    return [
        StepSpec(
            f"Grasp the {bread} with the {bread_arm} arm while grasping the {skillet} with the {skillet_arm} arm."
            if bread_arm and skillet_arm
            else f"Grasp the {bread} while grasping the {skillet}.",
            "close",
        ),
        StepSpec(
            f"Lift the {skillet} with the {skillet_arm} arm, then place the {bread} into the {skillet} with the {bread_arm} arm."
            if bread_arm and skillet_arm
            else f"Lift the {skillet}, then place the {bread} into the {skillet}.",
            "open",
            bread_arm,
        ),
    ]


@register("place_container_plate")
def place_container_plate(ctx: TaskContext) -> list[StepSpec]:
    return pair_steps(ctx.object_b, f"onto the {ctx.object_a}", ctx.arm_a)


@register("place_empty_cup")
def place_empty_cup(ctx: TaskContext) -> list[StepSpec]:
    return pair_steps("cup", "onto the coaster", ctx.arm_a)


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
    return [
        StepSpec(pick_text(fan, ctx.arm_a), "close", ctx.arm_a),
        StepSpec(f"Place the {fan} onto the {dst} and face it toward the robot.", "open", ctx.arm_a),
    ]


@register("place_burger_fries")
def place_burger_fries(ctx: TaskContext) -> list[StepSpec]:
    burger = ctx.object_a if ctx.object_a != "target object" else "hamburger"
    fries = ctx.object_c if ctx.object_c != "target object" else "french fries"
    tray = ctx.object_b if ctx.object_b != "target object" else "tray"
    return dual_pick_place_then_return(burger, fries, f"onto the {tray}")


def dual_container_builder(first: str, second: str, fallback_target: str, target_attr: str) -> TaskBuilder:
    def builder(ctx: TaskContext) -> list[StepSpec]:
        target = getattr(ctx, target_attr)
        if target == "target object":
            target = fallback_target
        return dual_pick_place_then_return(first, second, f"into the {target}")
    return builder


TASK_BUILDERS["place_bread_basket"] = dual_container_builder(
    "right bread", "left bread", "bread basket", "object_a"
)
TASK_BUILDERS["place_cans_plasticbox"] = dual_container_builder(
    "right can", "left can", "plastic box", "object_b"
)
TASK_BUILDERS["place_dual_shoes"] = dual_container_builder(
    "right shoe", "left shoe", "shoe box", "object_b"
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
    return handover_steps(
        "red block",
        "Place the red block on the target pad with the receiving arm while returning the first arm to a neutral pose.",
        "open",
    )


@register("handover_mic")
def handover_mic(ctx: TaskContext) -> list[StepSpec]:
    return handover_steps(ctx.object_a, f"Hold the {ctx.object_a} securely with the receiving arm.")


@register("hanging_mug")
def hanging_mug(ctx: TaskContext) -> list[StepSpec]:
    mug = ctx.object_a if ctx.object_a != "target object" else "mug"
    return [
        StepSpec(pick_text(mug, ctx.arm_a), "close", ctx.arm_a),
        StepSpec(f"Place the {mug} down with the first arm, rotating it if necessary.", "open", ctx.arm_a),
        StepSpec(f"Grasp the {mug} with the other arm.", "close", ctx.arm_b),
        StepSpec(f"Hang the {mug} on the rack.", "open", ctx.arm_b),
    ]


@register("scan_object")
def scan_object(ctx: TaskContext) -> list[StepSpec]:
    obj_arm, scanner_arm = prompt_scan_arms(ctx.task_goal, ctx.arm_a, ctx.arm_b)
    return [
        StepSpec(
            f"Grasp the {ctx.object_a} with the {obj_arm} arm while grasping the {ctx.object_b} with the {scanner_arm} arm.",
            "close",
        ),
        StepSpec(
            f"Scan the {ctx.object_a} with the {ctx.object_b}, holding the {ctx.object_a} with the {obj_arm} arm and the {ctx.object_b} with the {scanner_arm} arm.",
            "final",
        ),
    ]


@register("pick_diverse_bottles", "pick_dual_bottles")
def pick_dual_objects(ctx: TaskContext) -> list[StepSpec]:
    first, second = distinct_pair(ctx.object_a, ctx.object_b)
    return [
        StepSpec(f"Grasp the {first} with the left arm while grasping the {second} with the right arm.", "close"),
        StepSpec(f"Lift and hold the {first} with the left arm while holding the {second} with the right arm.", "final"),
    ]


def object_action_builder(
    fallback: str,
    action: Callable[[str, TaskContext], str],
) -> TaskBuilder:
    def builder(ctx: TaskContext) -> list[StepSpec]:
        obj = ctx.object_a if ctx.object_a != "target object" else fallback
        return [
            StepSpec(pick_text(obj, ctx.arm_a), "close", ctx.arm_a),
            StepSpec(action(obj, ctx), "final"),
        ]
    return builder


TASK_BUILDERS["adjust_bottle"] = object_action_builder("bottle", lambda obj, ctx: f"Keep the {obj} upright.")
TASK_BUILDERS["beat_block_hammer"] = object_action_builder("hammer", lambda obj, ctx: f"Use the {obj} to hit the block.")
TASK_BUILDERS["rotate_qrcode"] = object_action_builder(
    "payment sign",
    lambda obj, ctx: f"Rotate the {obj} until the QR code faces the robot.",
)
TASK_BUILDERS["shake_bottle"] = object_action_builder("bottle", lambda obj, ctx: f"Shake the {obj}.")
TASK_BUILDERS["shake_bottle_horizontally"] = object_action_builder(
    "bottle",
    lambda obj, ctx: f"Shake the {obj} horizontally.",
)


@register("dump_bin_bigbin")
def dump_bin_bigbin(ctx: TaskContext) -> list[StepSpec]:
    obj = ctx.object_a if ctx.object_a != "target object" else "trash bin"
    return [
        StepSpec(pick_text(obj), "close"),
        StepSpec(f"Lift the {obj} above the big bin.", "midpoint"),
        StepSpec(f"Pour the contents of the {obj} into the big bin.", "midpoint"),
        StepSpec(f"Finish emptying the {obj} and hold it steady.", "final"),
    ]


@register("grab_roller")
def grab_roller(ctx: TaskContext) -> list[StepSpec]:
    obj = ctx.object_a if ctx.object_a != "target object" else "roller"
    return [
        StepSpec(f"Grasp the {obj}.", "close"),
        StepSpec(f"Lift the {obj} off the table.", "final"),
    ]


@register("lift_pot")
def lift_pot(ctx: TaskContext) -> list[StepSpec]:
    obj = ctx.object_a if ctx.object_a != "target object" else "pot"
    return [
        StepSpec(
            f"Grasp the {obj} with the left arm while grasping it with the right arm.",
            "both_full_close",
        ),
        StepSpec(
            f"Lift the {obj} off the table with the left arm while lifting it with the right arm.",
            "final",
        ),
    ]


def open_builder(fallback: str, part: str) -> TaskBuilder:
    def builder(ctx: TaskContext) -> list[StepSpec]:
        obj = ctx.object_a if ctx.object_a != "target object" else fallback
        return [
            StepSpec(f"Grasp the {obj} {part}.", "close", ctx.arm_a),
            StepSpec(f"Open the {obj} {part}.", "final"),
        ]
    return builder


TASK_BUILDERS["open_laptop"] = open_builder("laptop", "lid")
TASK_BUILDERS["open_microwave"] = open_builder("microwave", "door")


@register("stamp_seal")
def stamp_seal(ctx: TaskContext) -> list[StepSpec]:
    seal = ctx.object_a if ctx.object_a != "target object" else "seal"
    target = ctx.object_b if ctx.object_b != "target object" and not is_color_name(ctx.object_b) else "target area"
    return [
        StepSpec(pick_text(seal, ctx.arm_a), "close", ctx.arm_a),
        StepSpec(f"Press the {seal} onto the {target}.", "final"),
    ]


def single_press_builder(text: str | Callable[[TaskContext], str]) -> TaskBuilder:
    def builder(ctx: TaskContext) -> list[StepSpec]:
        value = text(ctx) if callable(text) else text
        return [StepSpec(value, "close", ctx.arm_a)]
    return builder


TASK_BUILDERS["click_alarmclock"] = single_press_builder("Click the alarm clock button.")
TASK_BUILDERS["click_bell"] = single_press_builder("Press the top of the bell.")
TASK_BUILDERS["press_stapler"] = single_press_builder("Press down the stapler.")
TASK_BUILDERS["turn_switch"] = single_press_builder(
    lambda ctx: f"Operate the {ctx.object_a if ctx.object_a != 'target object' else 'switch'}."
)


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
    return task_goal


def validate_task_registry() -> None:
    missing = EXPECTED_TASK_SLUGS - TASK_BUILDERS.keys()
    extra = TASK_BUILDERS.keys() - EXPECTED_TASK_SLUGS
    if missing or extra:
        raise ValueError(
            f"invalid RoboTwin task registry: missing={sorted(missing)}, extra={sorted(extra)}"
        )


validate_task_registry()
