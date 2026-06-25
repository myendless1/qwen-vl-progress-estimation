"""Shared data models for RoboTwin subtask generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


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


@dataclass(frozen=True)
class TaskContext:
    """All episode-specific inputs used to compose a task's subtask rules."""

    slug: str
    task_goal: str
    info: dict[str, str]
    events: tuple[GripperEvent, ...]
    object_a: str
    object_b: str
    object_c: str
    arm_a: str | None
    arm_b: str | None
    arm_c: str | None

    @classmethod
    def create(
        cls,
        slug: str,
        task_goal: str,
        info: dict[str, str],
        events: list[GripperEvent] | None,
        *,
        object_parser: Callable[[dict[str, str], str, str], str],
        arm_parser: Callable[[dict[str, str], str, str | None], str | None],
    ) -> "TaskContext":
        return cls(
            slug=slug,
            task_goal=task_goal,
            info=info,
            events=tuple(events or ()),
            object_a=object_parser(info, "{A}", "target object"),
            object_b=object_parser(info, "{B}", "target object"),
            object_c=object_parser(info, "{C}", "target object"),
            arm_a=arm_parser(info, "{a}", None),
            arm_b=arm_parser(info, "{b}", None),
            arm_c=arm_parser(info, "{c}", None),
        )


TaskBuilder = Callable[[TaskContext], list[StepSpec]]
