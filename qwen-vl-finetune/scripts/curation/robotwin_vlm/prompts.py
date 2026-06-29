"""Prompt and scene-info parsing helpers for RoboTwin task rules."""

from __future__ import annotations

import re


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
    """Infer per-object arm order from an episode prompt when stated."""
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
        "angled", "rectangular", "light", "brown", "dark", "ergonomic",
        "compact", "matte", "gray", "smooth", "plastic", "golden",
        "polished", "wooden", "green", "round", "rounded", "kids",
        "bright", "blue", "colorful", "hand", "sized", "rolling", "soft",
        "printed", "back", "small", "white", "black", "bottom", "shiny",
        "yellow", "multi", "colored", "miniature", "pink", "sleek",
        "vivid", "orange", "red", "wireless", "teal", "solid", "cleaning",
        "top", "medium", "fluffy", "baked", "curved", "two", "buttons",
        "wheel", "flat", "touchscreen", "silver", "mechanism", "rectangle",
        "shaped", "cards", "inside", "storage", "sachets", "material",
        "texture", "design", "logo", "grain", "matching", "block", "triangle",
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
    match = re.search(
        rf"(.+?)\b(?:on|onto|in|inside|into)\b[^.]*?\b{target_pattern}\b",
        text,
        flags=re.IGNORECASE,
    )
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
        if obj not in {"target object", "basket"}:
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
    match = re.search(r"\b([A-Za-z]+)(?:\s+colored)?\s+mat\b", task_goal, flags=re.IGNORECASE)
    if match:
        return f"{match.group(1)} mat"
    return prompt_fan_target(task_goal, fallback)


def is_color_name(value: str) -> bool:
    return value.lower() in {
        "gray", "grey", "magenta", "blue", "yellow", "orange", "silver",
        "cyan", "beige", "brown", "turquoise", "indigo", "red", "green",
        "purple", "pink", "black", "white",
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
    return obj if obj not in {"cabinet", "target object"} else fallback


def prompt_mouse_object(task_goal: str, fallback: str) -> str:
    if re.search(r"\bmouse\b", task_goal, flags=re.IGNORECASE):
        return "mouse"
    return fallback


def _clause_arm_for_object(task_goal: str, object_regex: str, grab_verbs: str) -> str | None:
    """Find the arm that grabs/picks/holds ``object_regex`` by scanning clauses.

    Clauses are delimited by commas or periods so that an arm mentioned in one
    clause is never wrongly attributed to an object mentioned in another clause
    (e.g. "the left arm picks the tea box, the right arm grabs the scanner" must
    not bind "left arm" to "scanner"). Within a clause we first look for the
    object followed by its own arm ("... object ... with/using/in the X arm"),
    which correctly handles "and"-conjoined clauses such as "Hold the scanner
    with the right arm and the tea box with the left arm"; otherwise we look for
    an arm that grabs/picks the object ("X arm ... grab ... object").
    """
    obj_re = object_regex
    for clause in re.split(r"[,.]", task_goal):
        if not re.search(obj_re, clause, flags=re.IGNORECASE):
            continue
        if not re.search(r"\b(left|right)\s+arm\b", clause, flags=re.IGNORECASE):
            continue
        # Object then its arm.
        m = re.search(
            obj_re + r"[^.,]*?\b(?:with|using|in)\s+the\s+(left|right)\s+arm\b",
            clause,
            flags=re.IGNORECASE,
        )
        if m:
            return m.group(1).lower()
        # Arm then a grab verb then the object.
        m = re.search(
            r"\b(left|right)\s+arm\b[^.,]*?" + grab_verbs + r"[^.,]*?" + obj_re,
            clause,
            flags=re.IGNORECASE,
        )
        if m:
            return m.group(1).lower()
    return None


def prompt_scan_arms(
    task_goal: str,
    fallback_obj_arm: str | None,
    fallback_scanner_arm: str | None,
) -> tuple[str, str]:
    scanner_arm = None
    obj_arm = None
    grab_verbs = r"\b(?:grab|grabs|grabbing|grasp|grasps|hold|holds|holding|pick|picks|picking|take|takes|taking|use|uses|using)\b"
    # Clause-level: arm that grabs/holds/picks the scanner.
    scanner_arm = _clause_arm_for_object(task_goal, r"\bscanner\b", grab_verbs)
    if scanner_arm is None:
        # "scanner ... with/using the X arm" within a single clause.
        for clause in re.split(r"[,.]", task_goal):
            m = re.search(
                r"\bscanner\b[^.,]*?\b(?:with|using)\s+the\s+(left|right)\s+arm\b",
                clause,
                flags=re.IGNORECASE,
            )
            if m:
                scanner_arm = m.group(1).lower()
                break
    # Clause-level: arm that grabs/picks the tea box / object.
    obj_arm = _clause_arm_for_object(task_goal, r"\b(?:tea[\s-]?box|object)\b", grab_verbs)
    scanner_arm = scanner_arm or fallback_scanner_arm or "right"
    obj_arm = obj_arm or fallback_obj_arm or ("left" if scanner_arm == "right" else "right")
    if obj_arm == scanner_arm:
        obj_arm = "left" if scanner_arm == "right" else "right"
    return obj_arm, scanner_arm


def parse_a2b_objects_from_prompt(task_goal: str, side: str) -> tuple[str, str] | None:
    text = task_goal.strip().rstrip(".")
    patterns = [
        rf"(.+?)\s+to\s+the\s+{side}\s+of\s+(.+)",
        rf"(.+?)\s+{side}\s+of\s+(.+)",
        rf"(.+?)\s+to\s+the\s+{side}\s+side\s+of\s+(.+)",
        rf"(.+?)\s+to\s+(.+?)'s\s+{side}\b",
        rf"(.+?)\s+on\s+(.+?)'s\s+{side}\s+side\b",
        rf"(.+?)\s+at\s+the\s+{side}\s+of\s+(.+)",
        rf"(.+?)\s+{side}\s+position\s+of\s+(.+)",
        rf"(.+?)\s+on\s+the\s+{side}\s+of\s+(.+)",
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
