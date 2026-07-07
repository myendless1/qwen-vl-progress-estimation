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
    numpy = numpy_stub

from robotwin_vlm import build_steps  # noqa: E402
from robotwin_vlm.alignment import (  # noqa: E402
    assign_spans,
    describe_secondary_arm_motion,
    dual_move_break_candidate,
    ensure_arm_mentions,
    merge_stack_arm_switches,
    merge_tiny_post_gripper_motion,
    publish_actual_arm_labels,
    relabel_dual_container_first_place,
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
            [(0, 29), (30, 49)],
        )
        self.assertEqual(spans[0]["boundary_source"], "before_gripper_left_open")
        self.assertEqual(spans[0]["subtask_type"], "close")
        self.assertEqual(spans[0]["truncation_rule"], "gripper_open")
        self.assertEqual(spans[1]["boundary_source"], "episode_end")
        self.assertEqual(spans[1]["subtask_type"], "open")
        self.assertEqual(spans[1]["truncation_rule"], "episode_end")

    def test_move_steps_peek_without_consuming_gripper_events(self) -> None:
        steps = [
            StepSpec("Move to the grasp pose of the block.", "move", "left"),
            StepSpec("Close the gripper.", "close", "left"),
            StepSpec("Move to the place pose of the block on the pad.", "move", "left"),
            StepSpec("Open the gripper.", "open", "left"),
        ]
        events = [
            GripperEvent(10, "left", "close"),
            GripperEvent(30, "left", "open"),
        ]
        spans = assign_spans(steps, events, 50, states=None)
        self.assertEqual(
            [(span["start_frame"], span["end_frame"]) for span in spans],
            [(0, 9), (10, 29), (30, 49)],
        )
        self.assertEqual(spans[0]["boundary_source"], "before_gripper_left_close")
        self.assertEqual(spans[1]["boundary_source"], "before_gripper_left_open")
        self.assertEqual(spans[2]["boundary_source"], "episode_end")

    def test_gripper_span_absorbs_adjacent_stationary_frames(self) -> None:
        if not hasattr(numpy, "zeros"):
            self.skipTest("numpy is not available")
        states = numpy.zeros((20, 16), dtype=float)
        states[:9, 0] = numpy.linspace(0.0, 0.08, 9)
        states[16:, 0] = numpy.linspace(0.1, 0.2, 4)
        steps = [
            StepSpec("Move to the grasp pose of the block.", "move", "left"),
            StepSpec("Close the gripper.", "close", "left"),
            StepSpec("Lift the block.", "final", "left"),
        ]
        events = [
            GripperEvent(11, "left", "close", start_frame=10, end_frame=12),
        ]
        spans = assign_spans(steps, events, 20, states=states)
        self.assertEqual(
            [(span["start_frame"], span["end_frame"]) for span in spans],
            [(0, 9), (10, 15), (16, 19)],
        )

    def test_open_close_ignores_tiny_single_frame_eef_drift(self) -> None:
        if not hasattr(numpy, "zeros"):
            self.skipTest("numpy is not available")
        states = numpy.zeros((30, 16), dtype=float)
        states[14, 0] = 0.00002
        steps = [
            StepSpec("Close the gripper of the left arm.", "close", "left"),
            StepSpec("Lift the object with the left arm.", "move", "left"),
        ]
        events = [GripperEvent(8, "left", "close", start_frame=5, end_frame=10)]
        spans = assign_spans(steps, events, 30, states=states)
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0]["boundary_source"], "episode_end")

    def test_move_prefers_nearby_gripper_boundary_over_minor_other_arm_motion(self) -> None:
        if not hasattr(numpy, "zeros"):
            self.skipTest("numpy is not available")
        states = numpy.zeros((40, 16), dtype=float)
        states[18:21, 8] = [0.003, 0.006, 0.009]
        steps = [
            StepSpec("Move the left arm to the grasp pose of the block.", "move", "left"),
            StepSpec("Close the gripper of the left arm.", "close", "left"),
            StepSpec("Lift the block with the left arm.", "move", "left"),
        ]
        events = [GripperEvent(22, "left", "close", start_frame=22, end_frame=24)]
        spans = assign_spans(steps, events, 40, states=states)
        self.assertEqual(spans[0]["end_frame"], 21)
        self.assertEqual(spans[0]["boundary_source"], "before_gripper_left_close")

    def test_boundary_arm_does_not_overwrite_current_action_arm(self) -> None:
        spans = ensure_arm_mentions(
            [
                {
                    "subtask_index": 0,
                    "subtask_goal": "Close the gripper of the right arm.",
                    "start_frame": 172,
                    "end_frame": 193,
                    "boundary_source": "before_gripper_left_open",
                }
            ]
        )
        self.assertEqual(spans[0]["subtask_goal"], "Close the gripper of the right arm.")

    def test_press_waits_until_next_event_or_episode_end(self) -> None:
        if not hasattr(numpy, "zeros"):
            self.skipTest("numpy is not available")
        states = numpy.zeros((8, 16), dtype=float)
        states[:, 2] = [0.4, 0.3, 0.2, 0.05, 0.1, 0.2, 0.3, 0.4]
        steps = [
            StepSpec("Press the button with the left arm.", "press", "left"),
            StepSpec("Lift the left arm after pressing.", "move", "left"),
        ]
        spans = assign_spans(steps, [], 8, states=states)
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0]["end_frame"], 7)
        self.assertEqual(spans[0]["boundary_source"], "episode_end")

    def test_press_lift_ends_when_pressing_arm_z_increases(self) -> None:
        if not hasattr(numpy, "zeros"):
            self.skipTest("numpy is not available")
        states = numpy.zeros((8, 16), dtype=float)
        states[:, 2] = [0.4, 0.3, 0.2, 0.05, 0.1, 0.2, 0.3, 0.4]
        steps = [
            StepSpec("Press the button with the left arm.", "press", "left", terminates_on=("press_lift",)),
            StepSpec("Lift up the left arm after pressing.", "move", "left"),
        ]
        spans = assign_spans(steps, [], 8, states=states)
        self.assertEqual(len(spans), 2)
        self.assertEqual(spans[0]["end_frame"], 3)
        self.assertEqual(spans[0]["boundary_source"], "before_eef_left_z_increase")
        self.assertEqual(spans[0]["truncation_rule"], "press_lift")
        self.assertEqual(spans[1]["subtask_goal"], "Lift up the left arm after pressing.")
        self.assertEqual(spans[1]["start_frame"], 4)

    def test_dual_move_ends_when_next_arm_starts_independent_motion(self) -> None:
        if not hasattr(numpy, "zeros"):
            self.skipTest("numpy is not available")
        states = numpy.zeros((40, 16), dtype=float)
        states[:, 2] = 0.0
        states[:, 10] = 0.0
        states[10:21, 2] = numpy.linspace(0.0, 0.12, 11)
        states[21:, 2] = 0.12
        states[10:21, 10] = numpy.linspace(0.0, 0.12, 11)
        states[21:, 10] = 0.12
        states[26:, 8] = numpy.linspace(0.0, 0.1, 14)
        steps = [
            StepSpec("Close the grippers of both arms.", "close"),
            StepSpec(
                "Lift both objects to about 10 cm above the table with both arms.",
                "move",
            ),
            StepSpec("Move the right arm to the place pose.", "move", "right"),
            StepSpec("Open the gripper of the right arm.", "open", "right"),
        ]
        events = [
            GripperEvent(9, "left", "close"),
            GripperEvent(9, "right", "close"),
            GripperEvent(35, "right", "open"),
        ]
        spans = assign_spans(steps, events, 40, states=states)
        self.assertEqual(
            [(span["start_frame"], span["end_frame"]) for span in spans],
            [(0, 10), (11, 26), (27, 34), (35, 39)],
        )
        self.assertEqual(spans[1]["subtask_type"], "dual_move")
        self.assertEqual(spans[1]["boundary_source"], "before_eef_right_motion")
        self.assertEqual(spans[1]["truncation_rule"], "other_arm_motion")
        self.assertEqual(
            spans[1]["subtask_goal"],
            "Lift both objects to about 10 cm above the table with both arms.",
        )
        self.assertEqual(spans[2]["subtask_type"], "move")
        self.assertEqual(spans[2]["boundary_source"], "before_gripper_right_open")
        self.assertEqual(spans[2]["subtask_goal"], "Move the right arm to the place pose.")

    def test_dual_move_break_ignores_protection_frames_and_co_motion(self) -> None:
        if not hasattr(numpy, "zeros"):
            self.skipTest("numpy is not available")
        # Coordinated vertical lift 10-20 (both arms), then both still, then
        # only the right (next) arm moves from frame 26. The break must land at
        # 26 (one-based motion frame) -> boundary 25, NOT during the lift.
        states = numpy.zeros((40, 16), dtype=float)
        states[10:21, 2] = numpy.linspace(0.0, 0.12, 11)
        states[21:, 2] = 0.12
        states[10:21, 10] = numpy.linspace(0.0, 0.12, 11)
        states[21:, 10] = 0.12
        states[26:, 8] = numpy.linspace(0.0, 0.1, 14)
        candidate = dual_move_break_candidate(states, "right", 11, 39)
        self.assertIsNotNone(candidate)
        assert candidate is not None  # for type checkers
        self.assertEqual(candidate.frame, 26)
        self.assertEqual(candidate.source, "before_eef_right_motion")
        self.assertEqual(candidate.rule, "other_arm_motion")
        # While both arms keep moving together the detector must not fire.
        co_move = numpy.zeros((40, 16), dtype=float)
        co_move[10:30, 2] = numpy.linspace(0.0, 0.2, 20)
        co_move[10:30, 10] = numpy.linspace(0.0, 0.2, 20)
        self.assertIsNone(dual_move_break_candidate(co_move, "right", 11, 39))

    def test_release_lift_extends_to_lift_completion(self) -> None:
        if not hasattr(numpy, "zeros"):
            self.skipTest("numpy is not available")
        states = numpy.zeros((40, 16), dtype=float)
        states[10:21, 2] = numpy.linspace(0.0, 0.08, 11)
        states[21:, 2] = 0.08
        states[28:, 0] = numpy.linspace(0.0, 0.1, 12)
        steps = [
            StepSpec("Lift the left arm after releasing the object.", "move", "left"),
            StepSpec("Move the left arm to the place pose.", "move", "left"),
            StepSpec("Open the gripper of the left arm.", "open", "left"),
        ]
        events = [GripperEvent(35, "left", "open")]
        spans = assign_spans(steps, events, 40, states=states)
        self.assertEqual(
            [(span["start_frame"], span["end_frame"]) for span in spans],
            [(0, 34), (35, 39)],
        )
        self.assertEqual(spans[0]["boundary_source"], "before_gripper_left_open")

    def test_move_until_next_motion_uses_next_step_arm_motion(self) -> None:
        if not hasattr(numpy, "zeros"):
            self.skipTest("numpy is not available")
        states = numpy.zeros((40, 16), dtype=float)
        states[5:22, 8] = numpy.linspace(0.0, 0.2, 17)
        states[24:, 0] = numpy.linspace(0.0, 0.2, 16)
        steps = [
            StepSpec("Move the right arm to bring the skillet to the placement position.", "move", "right"),
            StepSpec("Move the left arm above the skillet.", "move", "left"),
            StepSpec("Open the gripper of the left arm.", "open", "left"),
        ]
        events = [GripperEvent(35, "left", "open")]
        spans = assign_spans(steps, events, 40, states=states)
        self.assertEqual(
            [(span["start_frame"], span["end_frame"]) for span in spans],
            [(0, 24), (25, 34), (35, 39)],
        )
        self.assertEqual(spans[0]["boundary_source"], "before_eef_left_motion")

    def test_place_motion_is_not_cut_by_concurrent_return_motion(self) -> None:
        if not hasattr(numpy, "zeros"):
            self.skipTest("numpy is not available")
        states = numpy.zeros((50, 16), dtype=float)
        states[5:20, 0] = numpy.linspace(0.0, 0.2, 15)
        states[6:25, 8] = numpy.linspace(0.0, 0.3, 19)
        steps = [
            StepSpec("Move the left arm to the place pose while returning the right arm to a neutral pose.", "move", "left"),
            StepSpec("Open the gripper of the left arm.", "open", "left"),
        ]
        events = [GripperEvent(30, "left", "open", start_frame=30, end_frame=32)]
        spans = assign_spans(steps, events, 50, states=states)
        self.assertEqual(
            [(span["start_frame"], span["end_frame"]) for span in spans],
            [(0, 29), (30, 49)],
        )
        self.assertEqual(spans[0]["boundary_source"], "before_gripper_left_open")

    def test_place_bread_basket_second_place_is_dual_move_gripper_only(self) -> None:
        if not hasattr(numpy, "zeros"):
            self.skipTest("numpy is not available")
        states = numpy.zeros((230, 16), dtype=float)
        states[160:206, 8] = numpy.linspace(0.0, 0.36, 46)
        states[160:206, 0] = numpy.linspace(0.0, 0.38, 46)
        events = [
            GripperEvent(56, "right", "close"),
            GripperEvent(56, "left", "close"),
            GripperEvent(127, "right", "open"),
            GripperEvent(213, "left", "open", start_frame=206, end_frame=220),
        ]
        steps = build_steps(
            "place_bread_basket",
            "Use the dual arm to place the golden bread loaf and the golden brown bread into the basket.",
            {},
            events,
        )
        spans = assign_spans(steps, events, 230, states=states)
        second_place = next(
            span for span in spans if "while returning the" in span["subtask_goal"]
        )
        self.assertEqual(second_place["subtask_type"], "dual_move")
        self.assertIn("neutral pose", second_place["subtask_goal"])
        self.assertEqual(second_place["boundary_source"], "before_gripper_left_open")
        self.assertEqual(second_place["truncation_rule"], "gripper_open")
        self.assertGreater(second_place["end_frame"] - second_place["start_frame"], 1)

    def test_next_grasp_mentions_concurrent_return_motion(self) -> None:
        if not hasattr(numpy, "zeros"):
            self.skipTest("numpy is not available")
        states = numpy.zeros((60, 16), dtype=float)
        states[15:40, 0] = numpy.linspace(0.0, 0.2, 25)
        states[15:34, 8] = numpy.linspace(0.0, 0.18, 19)
        spans = describe_secondary_arm_motion(
            [
                {
                    "subtask_index": 0,
                    "subtask_goal": "Move to the grasp pose of the medium block.",
                    "start_frame": 15,
                    "end_frame": 39,
                    "boundary_source": "before_gripper_left_close",
                }
            ],
            states,
        )
        self.assertEqual(
            spans[0]["subtask_goal"],
            "Move the left arm to the grasp pose of the medium block while returning the right arm to a neutral pose.",
        )

    def test_merge_tiny_post_gripper_motion_at_episode_end(self) -> None:
        if not hasattr(numpy, "zeros"):
            self.skipTest("numpy is not available")
        states = numpy.zeros((227, 16), dtype=float)
        states[222:227, 8] = numpy.linspace(0.0, 0.0035, 5)
        spans = [
            {
                "subtask_index": 7,
                "subtask_goal": "Open the gripper of the left arm.",
                "subtask_type": "open",
                "start_frame": 207,
                "end_frame": 221,
                "boundary_source": "before_eef_left_motion",
                "truncation_rule": "current_arm_motion",
            },
            {
                "subtask_index": 8,
                "subtask_goal": "Retract the left arm to at least 15 cm above the table after releasing the bread.",
                "subtask_type": "move",
                "start_frame": 222,
                "end_frame": 226,
                "boundary_source": "episode_end",
                "truncation_rule": "episode_end",
            },
        ]
        merged = merge_tiny_post_gripper_motion(spans, states)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["start_frame"], 207)
        self.assertEqual(merged[0]["end_frame"], 226)
        self.assertEqual(merged[0]["boundary_source"], "episode_end")
        self.assertEqual(merged[0]["subtask_type"], "open")

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

    def test_relabel_dual_shoes_first_place_always_dual_move(self) -> None:
        if not hasattr(numpy, "zeros"):
            self.skipTest("numpy is not available")
        states = numpy.zeros((40, 16), dtype=float)
        states[10:20, 0] = numpy.linspace(0.0, 0.2, 10)
        spans = [
            {
                "subtask_index": 3,
                "subtask_goal": "Move the left arm to the place pose of the right shoe into the shoe box.",
                "subtask_type": "move",
                "start_frame": 10,
                "end_frame": 19,
                "boundary_source": "before_gripper_left_open",
                "truncation_rule": "gripper_open",
            },
            {
                "subtask_index": 5,
                "subtask_goal": (
                    "Move the right arm to the place pose of the left shoe into the shoe box "
                    "while returning the left arm to a neutral pose."
                ),
                "subtask_type": "dual_move",
                "start_frame": 20,
                "end_frame": 39,
                "boundary_source": "before_gripper_right_open",
                "truncation_rule": "gripper_open",
            },
        ]
        relabeled = relabel_dual_container_first_place(spans, states, "place_dual_shoes")
        self.assertEqual(relabeled[0]["subtask_type"], "dual_move")
        self.assertNotIn("aside", relabeled[0]["subtask_goal"])
        self.assertEqual(relabeled[1]["subtask_goal"], spans[1]["subtask_goal"])

    def test_relabel_dual_shoes_first_place_adds_aside_text(self) -> None:
        if not hasattr(numpy, "zeros"):
            self.skipTest("numpy is not available")
        states = numpy.zeros((40, 16), dtype=float)
        states[10:20, 0] = numpy.linspace(0.0, 0.2, 10)
        states[10:20, 8] = numpy.linspace(0.0, 0.2, 10)
        spans = [
            {
                "subtask_index": 3,
                "subtask_goal": "Move the left arm to the place pose of the right shoe into the shoe box.",
                "subtask_type": "move",
                "start_frame": 10,
                "end_frame": 19,
                "boundary_source": "before_gripper_left_open",
                "truncation_rule": "gripper_open",
            }
        ]
        relabeled = relabel_dual_container_first_place(spans, states, "place_dual_shoes")
        self.assertEqual(relabeled[0]["subtask_type"], "dual_move")
        self.assertIn("while moving the right arm aside", relabeled[0]["subtask_goal"])

    def test_relabel_dual_shoes_first_place_skips_other_tasks(self) -> None:
        spans = [
            {
                "subtask_index": 1,
                "subtask_goal": "Move the left arm to the place pose of the can into the box.",
                "subtask_type": "move",
                "start_frame": 0,
                "end_frame": 9,
            }
        ]
        relabeled = relabel_dual_container_first_place(spans, None, "place_cans_plasticbox")
        self.assertEqual(relabeled[0]["subtask_type"], "move")

    def test_relabel_dual_container_first_place_cans_always_dual_move_no_aside(self) -> None:
        if not hasattr(numpy, "zeros"):
            self.skipTest("numpy is not available")
        states = numpy.zeros((40, 16), dtype=float)
        states[10:20, 0] = numpy.linspace(0.0, 0.2, 10)
        states[10:20, 8] = numpy.linspace(0.0, 0.018, 10)
        spans = [
            {
                "subtask_index": 3,
                "subtask_goal": "Move the left arm to the place pose of the right can into the plastic box.",
                "subtask_type": "move",
                "start_frame": 10,
                "end_frame": 19,
                "boundary_source": "before_gripper_left_open",
                "truncation_rule": "gripper_open",
            }
        ]
        relabeled = relabel_dual_container_first_place(spans, states, "place_cans_plasticbox")
        self.assertEqual(relabeled[0]["subtask_type"], "dual_move")
        self.assertNotIn("aside", relabeled[0]["subtask_goal"])


if __name__ == "__main__":
    unittest.main()
