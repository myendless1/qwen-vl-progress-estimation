from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path


CURATION_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CURATION_DIR))

try:
    import numpy  # noqa: F401
except ModuleNotFoundError:
    numpy_stub = types.ModuleType("numpy")
    numpy_stub.ndarray = object
    sys.modules["numpy"] = numpy_stub

from robotwin_vlm.alignment import (  # noqa: E402
    assign_spans,
    merge_stack_arm_switches,
    publish_actual_arm_labels,
)
from robotwin_vlm.models import GripperEvent, StepSpec  # noqa: E402


class AlignmentTests(unittest.TestCase):
    def test_assign_spans_uses_events_and_covers_episode(self) -> None:
        steps = [
            StepSpec("Grasp the block.", "close", "left"),
            StepSpec("Place the block on the pad.", "open", "left"),
        ]
        events = [
            GripperEvent(10, "left", "close"),
            GripperEvent(30, "left", "open"),
        ]
        spans = assign_spans(steps, events, 50, states=None)
        self.assertEqual(
            [(span["start_frame"], span["end_frame"]) for span in spans],
            [(0, 10), (11, 49)],
        )
        self.assertEqual(spans[0]["boundary_source"], "gripper_left_close")
        self.assertEqual(spans[1]["boundary_source"], "gripper_left_open")

    def test_stack_arm_switch_is_merged_into_next_prompt(self) -> None:
        spans = [
            {
                "subtask_index": 0,
                "subtask_goal": "Place the red block.",
                "start_frame": 0,
                "end_frame": 20,
                "boundary_source": "gripper_left_open",
            },
            {
                "subtask_index": 1,
                "subtask_goal": "Grasp the green block with the right arm.",
                "start_frame": 21,
                "end_frame": 40,
                "boundary_source": "gripper_right_close",
            },
        ]
        merged = merge_stack_arm_switches(spans, "stack_blocks_two")
        self.assertEqual(
            merged[1]["subtask_goal"],
            "Return the left arm to a neutral pose while grasping the green block with the right arm.",
        )

    def test_published_labels_flip_all_annotation_locations(self) -> None:
        anno = {
            "task_goal": "Use the left arm.",
            "subtasks": [
                {
                    "subtask_goal": "Grasp with the right arm.",
                    "boundary_source": "gripper_right_close",
                }
            ],
            "metadata": {
                "detected_gripper_events": [{"frame": 1, "arm": "left", "kind": "close"}],
                "scene_info": {"{a}": "right"},
            },
        }
        published = publish_actual_arm_labels(anno)
        self.assertEqual(published["task_goal"], "Use the right arm.")
        self.assertEqual(published["subtasks"][0]["subtask_goal"], "Grasp with the left arm.")
        self.assertEqual(published["subtasks"][0]["boundary_source"], "gripper_left_close")
        self.assertEqual(published["metadata"]["detected_gripper_events"][0]["arm"], "right")
        self.assertEqual(published["metadata"]["scene_info"]["{a}"], "left")


if __name__ == "__main__":
    unittest.main()
