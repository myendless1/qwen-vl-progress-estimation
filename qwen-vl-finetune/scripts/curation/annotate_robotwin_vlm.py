#!/usr/bin/env python3
"""Generate VLM-style subtask annotations for RoboTwin LeRobot repos.

The annotations are written as:

    <repo>/anno/episode_000000.json

Each file contains the episode-level task goal copied from the existing
metadata, plus a list of atomic-ish subtasks with inclusive frame boundaries.
Boundaries are matched to sampled LeRobot frames, not raw hdf5 frames.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq


DEFAULT_ROOT = Path("/media/damoxing/datasets/vae4d/lerobot-vae4d-org/robotwin")
DEFAULT_RAW_ROOT = Path("/media/damoxing/datasets/RoboTwin2_0/dataset")

LEFT_GRIPPER_DIM = 7
RIGHT_GRIPPER_DIM = 15
ARM_XYZ_DIMS = {
    "left": slice(0, 3),
    "right": slice(8, 11),
}
BOTH_FULL_CLOSE_THRESHOLD = 0.01
GRASP_CLOSE_THRESHOLD = 0.1
RELEASE_OPEN_THRESHOLD = 0.9
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
ARM_LABEL_FLIP = {
    "left": "right",
    "right": "left",
}


def flip_arm_label(arm: str | None) -> str | None:
    if arm is None:
        return None
    return ARM_LABEL_FLIP.get(arm, arm)


def flip_arm_mentions(text: str) -> str:
    placeholders = {
        "left arm": "__LEFT_ARM__",
        "right arm": "__RIGHT_ARM__",
    }
    for old, placeholder in placeholders.items():
        text = re.sub(rf"\b{old}\b", placeholder, text, flags=re.IGNORECASE)
    text = text.replace("__LEFT_ARM__", "right arm")
    text = text.replace("__RIGHT_ARM__", "left arm")
    return text


def flip_boundary_arm_label(source: str) -> str:
    source = re.sub(r"(?<=_)left(?=_)", "__LEFT__", source)
    source = re.sub(r"(?<=_)right(?=_)", "__RIGHT__", source)
    source = source.replace("__LEFT__", "right")
    source = source.replace("__RIGHT__", "left")
    return source


@dataclass(frozen=True)
class GripperEvent:
    frame: int
    arm: str
    kind: str


@dataclass(frozen=True)
class StepSpec:
    text: str
    event_kind: str = "final"
    arm: str | None = None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def task_slug_from_repo(repo: Path) -> str:
    name = repo.name
    marker = "-aloha-agilex_"
    if marker in name:
        return name.split(marker, 1)[0]
    return name.split("-", 1)[0]


def task_dir_from_repo(repo: Path) -> str:
    match = re.search(r"aloha-agilex_(?:clean_50|randomized_500)$", repo.name)
    if match:
        return match.group(0)
    return repo.name.split("-", 1)[-1]


def episode_parquet_path(repo: Path, episode_index: int, info: dict[str, Any]) -> Path:
    chunk_size = int(info.get("chunks_size", 1000))
    chunk = episode_index // chunk_size
    return repo / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet"


def load_states(parquet_path: Path) -> np.ndarray:
    table = pq.read_table(parquet_path, columns=["observation.state"])
    values = table.column("observation.state").to_pylist()
    return np.asarray(values, dtype=np.float32)


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
    return sorted(events, key=lambda e: (e.frame, e.arm, e.kind))


def raw_scene_info(
    raw_root: Path,
    slug: str,
    task_dir: str,
    episode_index: int,
) -> dict[str, str]:
    # Do not fall back across clean/randomized splits: episode IDs are reused,
    # and stale scene_info is worse than no scene_info for object captions.
    candidates = [raw_root / slug / task_dir / "scene_info.json"]
    key = f"episode_{episode_index}"
    for path in candidates:
        if not path.exists():
            continue
        try:
            scene = read_json(path)
        except Exception:
            continue
        info = scene.get(key, {}).get("info", {})
        if isinstance(info, dict):
            return {str(k): str(v) for k, v in info.items()}
    return {}


def clean_object_name(value: str | None, fallback: str) -> str:
    if not value:
        return fallback
    value = value.split("/", 1)[0]
    value = re.sub(r"^\d+_", "", value)
    value = value.replace("_", " ").replace("-", " ")
    value = re.sub(r"\bbase\d+\b", "", value)
    replacements = {
        "hamburg": "hamburger",
        "plasticbox": "plastic box",
        "kitchenpot": "kitchen pot",
        "tabletrashbin": "trash bin",
        "playingcards": "playing card box",
        "pillbottle": "pill bottle",
        "toycar": "toy car",
        "rubikscube": "rubik's cube",
        "electronicscale": "electronic scale",
        "remotecontrol": "remote control",
        "displaystand": "display stand",
        "shoebox": "shoe box",
        "phonestand": "phone stand",
        "paymentsign": "payment sign",
        "qrcode": "QR code",
        "drinkbox": "drink box",
        "breadbasket": "bread basket",
    }
    words = " ".join(value.split())
    return replacements.get(words, words or fallback)


def info_obj(info: dict[str, str], key: str, fallback: str) -> str:
    return clean_object_name(info.get(key), fallback)


def info_arm(info: dict[str, str], key: str, fallback: str | None = None) -> str | None:
    value = info.get(key)
    if value in {"left", "right"}:
        return value
    if value == "dual":
        return None
    return fallback


def parse_arm_sequence_from_prompt(task_goal: str, object_names: list[str]) -> list[str | None]:
    """Infer per-object arm order from the episode prompt when it is stated.

    Some converted repos carry stale raw scene_info, while the prompt and the
    actual gripper events are episode-specific. For ranking tasks, object order
    is fixed by the final arrangement, but the arm assignment varies by episode.
    """
    text = task_goal.lower()
    arms: list[str | None] = []
    for obj in object_names:
        pattern = rf"{re.escape(obj.lower())}[^.]*?\b(left|right) arm\b"
        match = re.search(pattern, text)
        arms.append(match.group(1) if match else None)
    if all(arm is not None for arm in arms):
        return arms

    arm_mentions = re.findall(r"\b(left|right) arm\b", text)
    if len(arm_mentions) >= len(object_names):
        return [arm_mentions[i] for i in range(len(object_names))]

    return [None] * len(object_names)


def normalize_prompt_object_phrase(text: str) -> str:
    text = text.lower()
    if re.search(r"\b(coffee|coffee-box)\b", text):
        return "coffee box"
    if re.search(r"\b(tea|sachet|sachets)\b", text):
        return "tea box"
    if re.search(r"\b(playing\s*cards?|playingcards|cards?\s+inside|box\s+with\s+cards?)\b", text):
        return "playing card box"
    text = re.sub(r"\b(using|with|via) the (left|right) arm\b", "", text)
    text = re.sub(r"\b(the|a|an)\b", " ", text)
    text = re.sub(
        r"\b(carefully|precisely|directly|exactly|smoothly|firmly|neatly|gently|slightly|position|place|put|set|move|shift|stick|bring|grab|grasp|drop|ensure|use|using|with|rest|made|for|to|on|at|in|side|position)\b",
        " ",
        text,
    )
    adjectives = {
        "angled",
        "rectangular",
        "light",
        "brown",
        "dark",
        "ergonomic",
        "compact",
        "matte",
        "gray",
        "smooth",
        "plastic",
        "golden",
        "polished",
        "wooden",
        "green",
        "round",
        "rounded",
        "kids",
        "bright",
        "blue",
        "colorful",
        "hand",
        "sized",
        "rolling",
        "soft",
        "printed",
        "back",
        "small",
        "white",
        "black",
        "bottom",
        "shiny",
        "yellow",
        "multi",
        "colored",
        "miniature",
        "pink",
        "sleek",
        "vivid",
        "orange",
        "red",
        "wireless",
        "teal",
        "solid",
        "cleaning",
        "top",
        "medium",
        "fluffy",
        "baked",
        "curved",
        "two",
        "buttons",
        "wheel",
        "flat",
        "touchscreen",
        "silver",
        "mechanism",
        "rectangle",
        "shaped",
        "cards",
        "inside",
        "storage",
        "sachets",
        "material",
        "texture",
        "design",
        "logo",
        "grain",
        "matching",
        "block",
        "triangle",
    }
    words = [w for w in re.findall(r"[a-z0-9]+", text) if w not in adjectives]
    phrase = " ".join(words)
    replacements = {
        "coffee box": "coffee box",
        "coffee": "coffee box",
        "box playingcards": "playing card box",
        "playingcards box": "playing card box",
        "playingcards case": "playing card box",
        "box playing cards": "playing card box",
        "box cards": "playing card box",
        "cards box": "playing card box",
        "bread loaf": "bread",
        "loaf": "bread",
        "bread": "bread",
        "toycar": "toy car",
        "car": "toy car",
        "toy": "toy car",
        "rubikscube": "rubik's cube",
        "cube": "rubik's cube",
        "remotecontrol": "remote control",
        "remote control": "remote control",
        "woodenblock": "wooden block",
        "building": "wooden block",
        "bell": "bell",
        "mouse": "mouse",
        "soap bar": "soap",
        "soap": "soap",
        "phone": "phone",
        "stapler": "stapler",
        "tea box": "tea box",
        "box tea": "tea box",
        "displaystand": "display stand",
        "electronicscale": "electronic scale",
        "hamburg": "hamburger",
        "hamburger": "hamburger",
        "french fries": "french fries",
        "fries": "french fries",
        "scanner": "scanner",
        "cabinet": "cabinet",
        "drawer": "cabinet",
    }
    for key, value in sorted(replacements.items(), key=lambda item: -len(item[0])):
        if key in phrase:
            return value
    return phrase or "target object"


def prompt_arm_near_object(task_goal: str, obj_pattern: str) -> str | None:
    text = task_goal.lower()
    patterns = [
        rf"\b(left|right) arm\b[^.,;]*?\b{obj_pattern}\b",
        rf"\b{obj_pattern}\b[^.,;]*?\b(?:with|using|by)\s+the\s+(left|right)\s+arm\b",
        rf"\b{obj_pattern}\b[^.,;]*?\b(left|right)\s+arm\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    return None


def prompt_object_before_target(task_goal: str, target_pattern: str, fallback: str) -> str:
    text = task_goal.strip().rstrip(".")
    match = re.search(rf"(.+?)\b(?:on|onto|in|inside|into)\b[^.]*?\b{target_pattern}\b", text, flags=re.IGNORECASE)
    if match:
        obj = normalize_prompt_object_phrase(match.group(1))
        if obj != "target object":
            return obj

    obj = normalize_prompt_object_phrase(text)
    return obj if obj != "target object" else fallback


def prompt_basket_object(task_goal: str, fallback: str) -> str:
    object_aliases = [
        ("playing card box", r"\b(?:card box|cards box|cards container|box\s+for\s+(?:playingcards|card storage)|box\s+for\s+playingcards|playing\s*cards?|playingcards|cards?\s+case|box\s+with\s+cards?)\b"),
        ("toy car", r"\b(?:toycar|toy car|small plastic toycar|mini car toy|small car toy|pink car)\b"),
        ("rubik's cube", r"\b(?:rubikscube|rubik'?s?\s+cube|cube)\b"),
        ("remote control", r"\b(?:remotecontrol|remote control)\b"),
        ("stapler", r"\bstapler\b"),
        ("mouse", r"\bmouse\b"),
        ("soap", r"\bsoap\b"),
        ("phone", r"\bphone\b"),
        ("bread", r"\bbread\b"),
    ]
    for name, pattern in object_aliases:
        if re.search(pattern, task_goal, flags=re.IGNORECASE):
            return name

    patterns = [
        r"\b(?:grab|grasp|pick up|pick|lift|take)\s+(.+?)(?:,|\s+and\s+place|\s+then\s+place|\s+place\s+it)",
        r"\b(?:place|put|set|move)\s+(.+?)\s+\b(?:in|into|inside)\b[^.]*?\bbasket\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, task_goal, flags=re.IGNORECASE)
        if not match:
            continue
        obj = normalize_prompt_object_phrase(match.group(1))
        if obj != "target object" and obj != "basket":
            return obj
    return fallback


def prompt_fan_target(task_goal: str, fallback: str) -> str:
    match = re.search(r"\b(?:on|onto)\s+(?:the\s+)?([A-Za-z -]*?\bmat\b)", task_goal, flags=re.IGNORECASE)
    if not match:
        match = re.search(r"\b(?:on|onto)\s+([A-Za-z]+)\s+mat\b", task_goal, flags=re.IGNORECASE)
    if match:
        phrase = " ".join(match.group(1).replace("-", " ").split())
        return phrase if phrase.lower().endswith("mat") else f"{phrase} mat"
    return fallback


def prompt_mat_target(task_goal: str, fallback: str) -> str:
    match = re.search(
        r"\b([A-Za-z]+)(?:\s+colored)?\s+mat\b",
        task_goal,
        flags=re.IGNORECASE,
    )
    if match:
        return f"{match.group(1)} mat"
    return prompt_fan_target(task_goal, fallback)


def is_color_name(value: str) -> bool:
    return value.lower() in {
        "gray",
        "grey",
        "magenta",
        "blue",
        "yellow",
        "orange",
        "silver",
        "cyan",
        "beige",
        "brown",
        "turquoise",
        "indigo",
        "red",
        "green",
        "purple",
        "pink",
        "black",
        "white",
    }


def prompt_cabinet_object(task_goal: str, fallback: str) -> str:
    object_aliases = [
        ("playing card box", r"\b(?:card box|cards box|cards container|box\s+for\s+card storage|box\s+with\s+cards?|playing\s*cards?|playingcards|cards?\s+case)\b"),
        ("black box", r"\b(?:black box|box with rectangular shape|small black box)\b"),
        ("box", r"\b(?:light beige box|embossed gold leaf)\b"),
        ("coffee box", r"\bcoffee[\s-]?box\b"),
        ("tea box", r"\btea[\s-]?box\b"),
        ("rubik's cube", r"\b(?:rubikscube|rubik'?s?\s+cube|multi[- ]?colored cube|cube)\b"),
        ("remote control", r"\b(?:remotecontrol|remote control)\b"),
        ("toy car", r"\b(?:toycar|toy car|mini car toy|small car toy|pink car)\b"),
        ("stapler", r"\bstapler\b"),
        ("mouse", r"\bmouse\b"),
        ("soap", r"\bsoap\b"),
        ("phone", r"\b(?:phone|smartphone)\b"),
        ("bread", r"\b(?:bread|loaf)\b"),
    ]
    for name, pattern in object_aliases:
        if re.search(pattern, task_goal, flags=re.IGNORECASE):
            return name
    obj = normalize_prompt_object_phrase(task_goal)
    blocked = {"cabinet", "target object"}
    return obj if obj not in blocked else fallback


def prompt_mouse_object(task_goal: str, fallback: str) -> str:
    if re.search(r"\bmouse\b", task_goal, flags=re.IGNORECASE):
        return "mouse"
    return fallback


def build_place_can_basket_steps(events: list[GripperEvent] | None, can: str, basket: str) -> list[StepSpec]:
    close_events = [event for event in events or [] if event.kind == "close"]
    open_events = [event for event in events or [] if event.kind == "open"]
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


def prompt_scan_arms(task_goal: str, fallback_obj_arm: str | None, fallback_scanner_arm: str | None) -> tuple[str, str]:
    scanner_arm = None
    obj_arm = None
    scanner_match = re.search(r"\b(left|right)\s+arm\b[^.]*?\bscanner\b", task_goal, flags=re.IGNORECASE)
    if scanner_match:
        scanner_arm = scanner_match.group(1).lower()
    else:
        scanner_match = re.search(r"\bscanner\b[^.]*?\b(?:with|using)\s+the\s+(left|right)\s+arm\b", task_goal, flags=re.IGNORECASE)
        if scanner_match:
            scanner_arm = scanner_match.group(1).lower()

    obj_match = re.search(
        r"\b(?:grab|grasp|pick|hold)\b[^.]*?\b(?:tea[\s-]?box|object)\b[^.]*?\b(?:with|using)\s+the\s+(left|right)\s+arm\b",
        task_goal,
        flags=re.IGNORECASE,
    )
    if obj_match:
        obj_arm = obj_match.group(1).lower()

    scanner_arm = scanner_arm or fallback_scanner_arm or "right"
    obj_arm = obj_arm or fallback_obj_arm or ("left" if scanner_arm == "right" else "right")
    if obj_arm == scanner_arm:
        obj_arm = "left" if scanner_arm == "right" else "right"
    return obj_arm, scanner_arm


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


def ordinal_object(index: int, base: str = "bottle") -> str:
    names = ["first", "second", "third", "fourth", "fifth"]
    if index < len(names):
        return f"{names[index]} {base}"
    return f"{base} {index + 1}"


def build_put_bottles_dustbin_steps(events: list[GripperEvent] | None) -> list[StepSpec]:
    if not events:
        return [step for obj in ["first bottle", "second bottle", "third bottle"] for step in pair_steps(obj, "into the dustbin")]

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

    return steps or [step for obj in ["first bottle", "second bottle", "third bottle"] for step in pair_steps(obj, "into the dustbin")]


def build_put_object_cabinet_steps(task_goal: str, events: list[GripperEvent] | None, fallback_obj: str) -> list[StepSpec]:
    obj = prompt_cabinet_object(task_goal, fallback_obj)
    close_events = [event for event in events or [] if event.kind == "close"]
    open_events = [event for event in events or [] if event.kind == "open"]
    object_arm = close_events[0].arm if close_events else None
    handle_arm = close_events[1].arm if len(close_events) > 1 else None
    pull_arm = handle_arm or (open_events[0].arm if open_events else None)
    if pull_arm and object_arm and pull_arm != object_arm:
        pull_text = f"Pull open the cabinet with the {pull_arm} arm while holding the {obj} with the {object_arm} arm."
    elif pull_arm:
        pull_text = f"Pull open the cabinet with the {pull_arm} arm."
    else:
        pull_text = "Pull open the cabinet."
    return [
        StepSpec(pick_text(obj, object_arm), "close", object_arm),
        StepSpec(f"Grasp the cabinet handle with the {handle_arm} arm." if handle_arm else "Grasp the cabinet handle.", "close", handle_arm),
        StepSpec(pull_text, "open"),
        StepSpec(f"Place the {obj} inside the cabinet with the {object_arm} arm." if object_arm else f"Place the {obj} inside the cabinet.", "final", object_arm),
    ]


def parse_a2b_objects_from_prompt(task_goal: str, side: str) -> tuple[str, str] | None:
    text = task_goal.strip().rstrip(".")
    side_alt = "left" if side == "left" else "right"
    patterns = [
        rf"(.+?)\s+to\s+the\s+{side_alt}\s+of\s+(.+)",
        rf"(.+?)\s+{side_alt}\s+of\s+(.+)",
        rf"(.+?)\s+to\s+the\s+{side_alt}\s+side\s+of\s+(.+)",
        rf"(.+?)\s+to\s+(.+?)'s\s+{side_alt}\b",
        rf"(.+?)\s+on\s+(.+?)'s\s+{side_alt}\s+side\b",
        rf"(.+?)\s+at\s+the\s+{side_alt}\s+of\s+(.+)",
        rf"(.+?)\s+{side_alt}\s+position\s+of\s+(.+)",
        rf"(.+?)\s+on\s+the\s+{side_alt}\s+of\s+(.+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        moved = normalize_prompt_object_phrase(match.group(1))
        reference = normalize_prompt_object_phrase(match.group(2))
        if moved != "target object" and reference != "target object":
            return moved, reference
    return None


def pick_text(obj: str, arm: str | None = None) -> str:
    if arm:
        return f"Grasp the {obj} with the {arm} arm."
    return f"Grasp the {obj}."


def place_text(obj: str, dst: str, arm: str | None = None) -> str:
    if arm:
        return f"Place the {obj} {dst} with the {arm} arm."
    return f"Place the {obj} {dst}."


def pair_steps(obj: str, dst: str, arm: str | None = None) -> list[StepSpec]:
    return [
        StepSpec(pick_text(obj, arm), "close", arm),
        StepSpec(place_text(obj, dst, arm), "open", arm),
    ]


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


def distinct_pair(first: str, second: str) -> tuple[str, str]:
    if first == second:
        return f"right {first}", f"left {second}"
    return first, second


def build_steps(
    slug: str,
    task_goal: str,
    info: dict[str, str],
    events: list[GripperEvent] | None = None,
) -> list[StepSpec]:
    a = info_obj(info, "{A}", "target object")
    b = info_obj(info, "{B}", "target object")
    c = info_obj(info, "{C}", "target object")
    arm_a = info_arm(info, "{a}")
    arm_b = info_arm(info, "{b}")
    arm_c = info_arm(info, "{c}")

    if slug == "blocks_ranking_rgb":
        arms = parse_arm_sequence_from_prompt(task_goal, ["red block", "green block", "blue block"])
        items = [
            (info_obj(info, "{A}", "red block"), "at the right position", arms[0]),
            (info_obj(info, "{B}", "green block"), "at the middle position", arms[1]),
            (info_obj(info, "{C}", "blue block"), "at the left position", arms[2]),
        ]
        return [step for obj, dst, arm in items for step in pair_steps(obj, dst, arm)]

    if slug == "blocks_ranking_size":
        arms = parse_arm_sequence_from_prompt(task_goal, ["small block", "medium block", "large block"])
        items = [
            (info_obj(info, "{C}", "small block"), "at the left position", arms[0]),
            (info_obj(info, "{B}", "medium block"), "at the middle position", arms[1]),
            (info_obj(info, "{A}", "large block"), "at the right position", arms[2]),
        ]
        return [step for obj, dst, arm in items for step in pair_steps(obj, dst, arm)]

    if slug == "stack_blocks_three":
        return (
            pair_steps("red block", "at the center as the base", arm_a)
            + pair_steps("green block", "on top of the red block", arm_b)
            + pair_steps("blue block", "on top of the green block", arm_c)
        )

    if slug == "stack_blocks_two":
        return pair_steps("red block", "at the center as the base", arm_a) + pair_steps(
            "green block", "on top of the red block", arm_b
        )

    if slug == "stack_bowls_three":
        return (
            pair_steps(a if a != b else "base bowl", "at the base position", arm_a)
            + pair_steps(b if a != b else "middle bowl", "inside the base bowl", arm_b)
            + pair_steps(c if c != "target object" else "top bowl", "inside the stacked bowls", arm_c)
        )

    if slug == "stack_bowls_two":
        return pair_steps(a if a != b else "base bowl", "at the base position", arm_a) + pair_steps(
            b if a != b else "top bowl", "inside the base bowl", arm_b
        )

    if slug in {"move_pillbottle_pad", "move_playingcard_away", "move_stapler_pad", "place_shoe"}:
        dst = "to the target area"
        if slug == "move_stapler_pad":
            dst = f"onto the {prompt_mat_target(task_goal, f'{b} mat' if b != 'target object' else 'mat')}"
        elif slug == "move_pillbottle_pad":
            dst = "onto the pad"
        elif slug == "move_playingcard_away":
            dst = "away from its initial position"
        elif slug == "place_shoe":
            dst = "onto the mat"
        return pair_steps(a, dst, arm_a)

    if slug in {"place_a2b_left", "place_a2b_right"}:
        side = "to the left of" if slug.endswith("left") else "to the right of"
        parsed = parse_a2b_objects_from_prompt(task_goal, "left" if slug.endswith("left") else "right")
        moved_obj, reference_obj = parsed if parsed is not None else (a, b)
        return pair_steps(moved_obj, f"{side} the {reference_obj}", arm_a)

    if slug == "move_can_pot":
        return pair_steps(b, f"next to the {a}", arm_a)

    if slug == "place_bread_skillet":
        bread = b if b != "target object" else "bread"
        skillet = a if a != "target object" else "skillet"
        bread_arm = prompt_arm_near_object(task_goal, r"bread") or next((event.arm for event in events or [] if event.kind == "open"), arm_a)
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

    if slug == "place_container_plate":
        return pair_steps(b, f"onto the {a}", arm_a)

    if slug in {"place_empty_cup", "place_mouse_pad", "place_object_scale", "place_object_stand", "place_phone_stand"}:
        if slug == "place_object_scale":
            obj = prompt_object_before_target(task_goal, r"(?:electronicscale|electronic scale|scale)", b)
            return pair_steps(obj, "onto the electronic scale", arm_a)
        if slug == "place_object_stand":
            obj = prompt_object_before_target(task_goal, r"(?:displaystand|display stand|stand)", a)
            return pair_steps(obj, "onto the display stand", arm_a)
        targets = {
            "place_empty_cup": "onto the coaster",
            "place_mouse_pad": "onto the mat",
            "place_phone_stand": f"onto the {b if b != 'target object' else 'phone stand'}",
        }
        obj = "cup" if slug == "place_empty_cup" else a
        if slug == "place_mouse_pad":
            obj = prompt_mouse_object(task_goal, a if a != "target object" else "mouse")
        if slug == "place_phone_stand":
            obj = a if a != "target object" else "phone"
        return pair_steps(obj, targets[slug], arm_a)

    if slug == "place_fan":
        fan = a if a != "target object" else "fan"
        dst = prompt_fan_target(task_goal, f"{b} mat" if b != "target object" else "mat")
        return [
            StepSpec(pick_text(fan, arm_a), "close", arm_a),
            StepSpec(f"Place the {fan} onto the {dst} and face it toward the robot.", "open", arm_a),
        ]

    if slug == "place_burger_fries":
        burger = a if a != "target object" else "hamburger"
        fries = c if c != "target object" else "french fries"
        tray = b if b != "target object" else "tray"
        return dual_pick_place_then_return(burger, fries, f"onto the {tray}")

    if slug in {"place_bread_basket", "place_cans_plasticbox", "place_dual_shoes"}:
        if slug == "place_bread_basket":
            return dual_pick_place_then_return("right bread", "left bread", f"into the {a if a != 'target object' else 'bread basket'}")
        if slug == "place_cans_plasticbox":
            return dual_pick_place_then_return("right can", "left can", f"into the {b if b != 'target object' else 'plastic box'}")
        return dual_pick_place_then_return("right shoe", "left shoe", f"into the {b if b != 'target object' else 'shoe box'}")

    if slug == "place_can_basket":
        can = a if a != "target object" else "can"
        basket = b if b != "target object" else "basket"
        return build_place_can_basket_steps(events, can, basket)

    if slug == "place_object_basket":
        obj = prompt_basket_object(task_goal, a)
        basket = b if b != "target object" else "basket"
        return build_place_can_basket_steps(events, obj, basket)

    if slug == "put_bottles_dustbin":
        return build_put_bottles_dustbin_steps(events)

    if slug == "put_object_cabinet":
        return build_put_object_cabinet_steps(task_goal, events, a)

    if slug == "handover_block":
        return [
            StepSpec("Grasp the red block with the first arm.", "close"),
            StepSpec("Grasp the red block with the receiving arm.", "close"),
            StepSpec("Release the red block from the first arm.", "open"),
            StepSpec(
                "Place the red block on the target pad with the receiving arm while returning the first arm to a neutral pose.",
                "open",
            ),
        ]

    if slug == "handover_mic":
        return [
            StepSpec(f"Grasp the {a} with the first arm.", "close"),
            StepSpec(f"Grasp the {a} with the receiving arm.", "close"),
            StepSpec(f"Release the {a} from the first arm.", "open"),
            StepSpec(f"Hold the {a} securely with the receiving arm.", "final"),
        ]

    if slug == "hanging_mug":
        mug = a if a != "target object" else "mug"
        return [
            StepSpec(pick_text(mug, arm_a), "close", arm_a),
            StepSpec(f"Place the {mug} down with the first arm, rotating it if necessary.", "open", arm_a),
            StepSpec(f"Grasp the {mug} with the other arm.", "close", arm_b),
            StepSpec(f"Hang the {mug} on the rack.", "open", arm_b),
        ]

    if slug == "scan_object":
        obj_arm, scanner_arm = prompt_scan_arms(task_goal, arm_a, arm_b)
        return [
            StepSpec(
                f"Grasp the {a} with the {obj_arm} arm while grasping the {b} with the {scanner_arm} arm.",
                "close",
            ),
            StepSpec(f"Scan the {a} with the {b}, holding the {a} with the {obj_arm} arm and the {b} with the {scanner_arm} arm.", "final"),
        ]

    if slug in {"pick_diverse_bottles", "pick_dual_bottles"}:
        first, second = distinct_pair(a, b)
        return [
            StepSpec(f"Grasp the {first} with the left arm while grasping the {second} with the right arm.", "close"),
            StepSpec(f"Lift and hold the {first} with the left arm while holding the {second} with the right arm.", "final"),
        ]

    if slug == "adjust_bottle":
        bottle = a if a != "target object" else "bottle"
        return [
            StepSpec(pick_text(bottle, arm_a), "close", arm_a),
            StepSpec(f"Keep the {bottle} upright.", "final"),
        ]

    if slug == "beat_block_hammer":
        hammer = a if a != "target object" else "hammer"
        return [
            StepSpec(pick_text(hammer, arm_a), "close", arm_a),
            StepSpec(f"Use the {hammer} to hit the block.", "final"),
        ]

    if slug == "dump_bin_bigbin":
        bin_name = a if a != "target object" else "trash bin"
        return [
            StepSpec(pick_text(bin_name), "close"),
            StepSpec(f"Lift the {bin_name} above the big bin.", "midpoint"),
            StepSpec(f"Pour the contents of the {bin_name} into the big bin.", "midpoint"),
            StepSpec(f"Finish emptying the {bin_name} and hold it steady.", "final"),
        ]

    if slug == "grab_roller":
        obj = a if a != "target object" else "roller"
        return [
            StepSpec(f"Grasp the {obj}.", "close"),
            StepSpec(f"Lift the {obj} off the table.", "final"),
        ]

    if slug == "lift_pot":
        obj = a if a != "target object" else ("roller" if slug == "grab_roller" else "pot")
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

    if slug in {"open_laptop", "open_microwave"}:
        obj = a if a != "target object" else ("laptop" if slug == "open_laptop" else "microwave")
        part = "lid" if slug == "open_laptop" else "door"
        return [
            StepSpec(f"Grasp the {obj} {part}.", "close", arm_a),
            StepSpec(f"Open the {obj} {part}.", "final"),
        ]

    if slug == "rotate_qrcode":
        sign = a if a != "target object" else "payment sign"
        return [
            StepSpec(pick_text(sign, arm_a), "close", arm_a),
            StepSpec(f"Rotate the {sign} until the QR code faces the robot.", "final"),
        ]

    if slug in {"shake_bottle", "shake_bottle_horizontally"}:
        bottle = a if a != "target object" else "bottle"
        direction = " horizontally" if slug.endswith("horizontally") else ""
        return [
            StepSpec(pick_text(bottle, arm_a), "close", arm_a),
            StepSpec(f"Shake the {bottle}{direction}.", "final"),
        ]

    if slug == "stamp_seal":
        seal = a if a != "target object" else "seal"
        target = b if b != "target object" and not is_color_name(b) else "target area"
        return [
            StepSpec(pick_text(seal, arm_a), "close", arm_a),
            StepSpec(f"Press the {seal} onto the {target}.", "final"),
        ]

    if slug in {"click_alarmclock", "click_bell", "press_stapler", "turn_switch"}:
        verbs = {
            "click_alarmclock": "Click the alarm clock button.",
            "click_bell": "Press the top of the bell.",
            "press_stapler": "Press down the stapler.",
            "turn_switch": f"Operate the {a if a != 'target object' else 'switch'}.",
        }
        return [StepSpec(verbs[slug], "close", arm_a)]

    # Conservative fallback: keep the original prompt but still produce a valid
    # single span. This makes the script robust if new RoboTwin tasks are added.
    return [StepSpec(task_goal, "final")]


def canonical_task_goal(slug: str, task_goal: str) -> str:
    if slug == "blocks_ranking_rgb":
        return "Arrange the blue block, green block, and red block from left to right."
    if slug == "blocks_ranking_size":
        return "Arrange the small block, medium block, and large block from left to right."
    return task_goal


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
            if idx in used:
                continue
            if event.frame <= start_after:
                continue
            if event.kind != event_kind:
                continue
            used.add(idx)
            return event

    for idx, event in enumerate(events):
        if idx in used:
            continue
        if event.frame <= start_after:
            continue
        if event.kind != event_kind:
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
        if idx in used:
            continue
        if event.frame <= start_after:
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
                elif step.event_kind == "open":
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
                    else
                    f"gripper_{matched_event.arm}_{matched_event.kind}"
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
    """Find when an opened gripper's arm finishes moving away.

    The candidate interval is (start_frame, end_before). We require the arm to
    move a meaningful distance after release, then look for a short low-speed
    window after that movement. The returned frame is inclusive.
    """
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
    max_disp = float(displacement.max(initial=0.0))
    if max_disp < min_displacement:
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
        # speed[k] is the motion from frame k to frame k+1. A still window
        # ending at `frame` therefore covers recent motions before/at frame.
        lo = max(0, frame - still_window)
        hi = min(len(speed), frame)
        if hi <= lo:
            continue
        if float(speed[lo:hi].mean()) <= still_threshold:
            return frame

    # If it moved away but did not settle before the next close, keep a small
    # pick-up interval for the next subtask.
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

        # The following pick subtask starts after the retreat stage. The loop
        # will append its own copy later, so mutate the source list entry.
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
        concurrent_text = text
        concurrent_text = re.sub(r"^Grasp\b", "grasping", concurrent_text)
        concurrent_text = re.sub(r"^Pick up\b", "picking up", concurrent_text)
        if concurrent_text == text and concurrent_text:
            concurrent_text = concurrent_text[0].lower() + concurrent_text[1:]
        output[idx + 1]["subtask_goal"] = (
            f"Return the {released_arm} arm to a neutral pose while {concurrent_text}."
            if concurrent_text
            else f"Return the {released_arm} arm to a neutral pose while the {pickup_arm} arm grasps the next object."
        )

    return output


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

    metadata["arm_label_mapping"] = "annotation arm labels are flipped from state indices: state-left is actual right, state-right is actual left"
    return anno


def annotate_episode(
    repo: Path,
    episode: dict[str, Any],
    info_json: dict[str, Any],
    raw_root: Path,
    gripper_threshold: float,
    grasp_close_threshold: float = GRASP_CLOSE_THRESHOLD,
    release_open_threshold: float = RELEASE_OPEN_THRESHOLD,
    insert_retreat: bool = True,
) -> dict[str, Any]:
    episode_index = int(episode["episode_index"])
    slug = task_slug_from_repo(repo)
    task_goal = episode.get("tasks", [""])[0] or ""
    if not task_goal:
        task_goal = episode.get("task", "")
    task_goal = canonical_task_goal(slug, task_goal)
    parquet_path = episode_parquet_path(repo, episode_index, info_json)
    states = load_states(parquet_path)
    events = detect_gripper_events(states, threshold=gripper_threshold)
    scene_info = raw_scene_info(raw_root, slug, task_dir_from_repo(repo), episode_index)
    steps = build_steps(slug, task_goal, scene_info, events)
    subtasks = assign_spans(
        steps,
        events,
        int(states.shape[0]),
        states=states,
        grasp_close_threshold=grasp_close_threshold,
        release_open_threshold=release_open_threshold,
        prefer_specified_arm=slug not in CHRONOLOGICAL_ARM_TASKS,
    )
    subtasks = merge_stack_arm_switches(subtasks, slug)
    effective_insert_retreat = insert_retreat and slug not in NO_RETREAT_TASKS
    if effective_insert_retreat:
        subtasks = insert_retreat_subtasks(subtasks, states)

    anno = {
        "episode_index": episode_index,
        "repo": repo.name,
        "task_slug": slug,
        "task_goal": task_goal,
        "num_frames": int(states.shape[0]),
        "subtasks": subtasks,
        "metadata": {
            "annotation_version": "robotwin_vlm_subtask_v2",
            "gripper_threshold": gripper_threshold,
            "grasp_close_threshold": grasp_close_threshold,
            "release_open_threshold": release_open_threshold,
            "retreat_subtasks_enabled": effective_insert_retreat,
            "detected_gripper_events": [
                {"frame": event.frame, "arm": event.arm, "kind": event.kind} for event in events
            ],
            "scene_info": scene_info,
        },
    }
    return publish_actual_arm_labels(anno)


def iter_repos(root: Path, only: str | None) -> list[Path]:
    repos = []
    for repo in sorted(root.iterdir()):
        if not repo.is_dir():
            continue
        if only and only not in repo.name:
            continue
        if (repo / "meta" / "episodes.jsonl").exists() and (repo / "meta" / "info.json").exists():
            repos.append(repo)
    return repos


def print_rules(root: Path) -> None:
    seen: dict[str, int] = {}
    for repo in iter_repos(root, None):
        slug = task_slug_from_repo(repo)
        if slug in seen:
            continue
        steps = build_steps(slug, "", {})
        seen[slug] = len(steps)
    for slug, count in sorted(seen.items()):
        print(f"{slug}: {count} subtasks")


def report_retreat_candidates(
    root: Path,
    raw_root: Path,
    only: str | None,
    limit: int | None,
    gripper_threshold: float,
    grasp_close_threshold: float,
    release_open_threshold: float,
) -> None:
    summary: dict[str, dict[str, Any]] = {}
    for repo in iter_repos(root, only):
        info_json = read_json(repo / "meta" / "info.json")
        episodes = read_jsonl(repo / "meta" / "episodes.jsonl")
        if limit is not None:
            episodes = episodes[:limit]
        slug = task_slug_from_repo(repo)
        entry = summary.setdefault(
            slug,
            {"repos": set(), "episodes": 0, "episodes_with_retreat": 0, "retreat_subtasks": 0},
        )
        entry["repos"].add(repo.name)
        for episode in episodes:
            try:
                anno = annotate_episode(
                    repo=repo,
                    episode=episode,
                    info_json=info_json,
                    raw_root=raw_root,
                    gripper_threshold=gripper_threshold,
                    grasp_close_threshold=grasp_close_threshold,
                    release_open_threshold=release_open_threshold,
                    insert_retreat=True,
                )
            except Exception as exc:
                print(f"[ERROR] {repo.name} episode_{int(episode['episode_index']):06d}: {exc}")
                continue
            retreat_count = sum(
                1 for st in anno["subtasks"] if str(st.get("boundary_source", "")).startswith("eef_")
            )
            entry["episodes"] += 1
            entry["retreat_subtasks"] += retreat_count
            if retreat_count:
                entry["episodes_with_retreat"] += 1

    for slug, entry in sorted(summary.items()):
        if entry["retreat_subtasks"] == 0:
            continue
        repo_count = len(entry["repos"])
        print(
            f"{slug}: repos={repo_count}, "
            f"episodes_with_retreat={entry['episodes_with_retreat']}/{entry['episodes']}, "
            f"retreat_subtasks={entry['retreat_subtasks']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--only", type=str, default=None, help="Only process repos whose names contain this string.")
    parser.add_argument("--limit", type=int, default=None, help="Limit episodes per repo.")
    parser.add_argument("--dry-run", action="store_true", help="Print examples without writing anno files.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing annotation files.")
    parser.add_argument("--gripper-threshold", type=float, default=0.5)
    parser.add_argument(
        "--grasp-close-threshold",
        type=float,
        default=GRASP_CLOSE_THRESHOLD,
        help="Stricter gripper value used to end Grasp subtasks after the initial close event.",
    )
    parser.add_argument(
        "--release-open-threshold",
        type=float,
        default=RELEASE_OPEN_THRESHOLD,
        help="Stricter gripper value used to end Place/Release subtasks after the initial open event.",
    )
    parser.add_argument(
        "--no-retreat-subtasks",
        action="store_true",
        help="Do not insert EEF-velocity-based arm return/retreat subtasks after release.",
    )
    parser.add_argument("--print-rules", action="store_true", help="Print task slug to subtask-count mapping and exit.")
    parser.add_argument(
        "--report-retreat-candidates",
        action="store_true",
        help="Report task slugs where EEF velocity creates return/retreat subtasks.",
    )
    args = parser.parse_args()

    if args.print_rules:
        print_rules(args.root)
        return
    if args.report_retreat_candidates:
        report_retreat_candidates(
            root=args.root,
            raw_root=args.raw_root,
            only=args.only,
            limit=args.limit,
            gripper_threshold=args.gripper_threshold,
            grasp_close_threshold=args.grasp_close_threshold,
            release_open_threshold=args.release_open_threshold,
        )
        return

    total = 0
    skipped = 0
    for repo in iter_repos(args.root, args.only):
        info_json = read_json(repo / "meta" / "info.json")
        episodes = read_jsonl(repo / "meta" / "episodes.jsonl")
        if args.limit is not None:
            episodes = episodes[: args.limit]

        for episode in episodes:
            episode_index = int(episode["episode_index"])
            out_path = repo / "anno" / f"episode_{episode_index:06d}.json"
            if out_path.exists() and not args.overwrite and not args.dry_run:
                skipped += 1
                continue
            try:
                anno = annotate_episode(
                    repo=repo,
                    episode=episode,
                    info_json=info_json,
                    raw_root=args.raw_root,
                    gripper_threshold=args.gripper_threshold,
                    grasp_close_threshold=args.grasp_close_threshold,
                    release_open_threshold=args.release_open_threshold,
                    insert_retreat=not args.no_retreat_subtasks,
                )
            except Exception as exc:
                print(f"[ERROR] {repo.name} episode_{episode_index:06d}: {exc}")
                skipped += 1
                continue

            if args.dry_run:
                print(json.dumps(anno, ensure_ascii=False, indent=2))
            else:
                write_json(out_path, anno)
            total += 1

    action = "would write" if args.dry_run else "wrote"
    print(f"{action} {total} annotation files; skipped {skipped}.")


if __name__ == "__main__":
    main()
