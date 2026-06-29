from __future__ import annotations

import sys
import unittest
from pathlib import Path


CURATION_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CURATION_DIR))

from evaluate_robotwin_annos import (  # noqa: E402
    is_episode_end_tail_subtask,
    other_arm_motion_threshold,
)


class EpisodeEndTailMinFramesTests(unittest.TestCase):
    def test_last_subtask_truncated_at_episode_end(self) -> None:
        subtask = {
            "start_frame": 224,
            "end_frame": 227,
            "boundary_source": "episode_end",
            "truncation_rule": "episode_end",
        }
        self.assertTrue(
            is_episode_end_tail_subtask(
                subtask,
                position=8,
                num_subtasks=9,
                num_frames=228,
            )
        )

    def test_short_middle_subtask_is_not_episode_end_tail(self) -> None:
        subtask = {
            "start_frame": 10,
            "end_frame": 12,
            "boundary_source": "before_gripper_left_open",
            "truncation_rule": "gripper_open",
        }
        self.assertFalse(
            is_episode_end_tail_subtask(
                subtask,
                position=2,
                num_subtasks=9,
                num_frames=228,
            )
        )

    def test_last_subtask_short_but_not_episode_end_boundary(self) -> None:
        subtask = {
            "start_frame": 220,
            "end_frame": 227,
            "boundary_source": "before_gripper_left_open",
            "truncation_rule": "gripper_open",
        }
        self.assertFalse(
            is_episode_end_tail_subtask(
                subtask,
                position=8,
                num_subtasks=9,
                num_frames=228,
            )
        )


class OtherArmMotionThresholdTests(unittest.TestCase):
    def test_place_bread_tasks_use_higher_threshold(self) -> None:
        self.assertEqual(other_arm_motion_threshold("place_bread_skillet", 0.015), 0.04)
        self.assertEqual(other_arm_motion_threshold("place_bread_basket", 0.015), 0.04)
        self.assertEqual(other_arm_motion_threshold("put_object_cabinet", 0.015), 0.04)
        self.assertEqual(other_arm_motion_threshold("open_laptop", 0.015), 0.015)


if __name__ == "__main__":
    unittest.main()
