import unittest

import numpy as np

from qwenvl.data.robotwin_progress import (
    build_subtask_progress_curve,
    progress_for_subtask,
    progress_from_curve,
    select_frames_by_progress_bucket,
    select_undone_frame_indices,
    time_progress_for_subtask,
)


class RobotWinProgressTest(unittest.TestCase):
    def test_translation_progress_is_independent_of_rotation_scale(self):
        states = np.zeros((11, 16), dtype=np.float32)
        states[:, 3:7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        for frame in range(1, 11):
            states[frame, 0] = frame / 10.0
        subtask = {"start_frame": 0, "end_frame": 10, "subtask_goal": "Move the left arm.", "subtask_type": "move"}
        curve = build_subtask_progress_curve(states, 0, 10, ("left",))

        self.assertAlmostEqual(progress_for_subtask(subtask, 0, curve=curve), 0.0)
        self.assertAlmostEqual(progress_for_subtask(subtask, 10, curve=curve), 1.0)
        self.assertAlmostEqual(progress_for_subtask(subtask, 5, curve=curve), 0.5)

    def test_gripper_progress_uses_total_change_ratio(self):
        states = np.zeros((6, 16), dtype=np.float32)
        states[:, 3:7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        states[:, 7] = [1.0, 1.0, 0.7, 0.4, 0.1, 0.0]
        subtask = {"start_frame": 0, "end_frame": 5, "subtask_goal": "Close the gripper of the left arm.", "subtask_type": "close"}
        curve = build_subtask_progress_curve(states, 0, 5, ("left",))

        self.assertAlmostEqual(progress_for_subtask(subtask, 0, curve=curve), 0.0)
        self.assertAlmostEqual(progress_for_subtask(subtask, 5, curve=curve), 1.0)
        self.assertAlmostEqual(progress_for_subtask(subtask, 2, curve=curve), 0.3)
        self.assertAlmostEqual(progress_for_subtask(subtask, 3, curve=curve), 0.6)

    def test_combined_progress_averages_active_components(self):
        states = np.zeros((6, 16), dtype=np.float32)
        states[:, 3:7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        states[:, 0] = [0.0, 0.5, 1.0, 1.0, 1.0, 1.0]
        states[:, 7] = [1.0, 1.0, 1.0, 0.5, 0.0, 0.0]
        subtask = {"start_frame": 0, "end_frame": 5, "subtask_goal": "Move the left arm.", "subtask_type": "move"}
        curve = build_subtask_progress_curve(states, 0, 5, ("left",))

        trans_only = progress_from_curve(
            build_subtask_progress_curve(
                np.array([[0, 0, 0, 1, 0, 0, 0, 0], [0.5, 0, 0, 1, 0, 0, 0, 0], [1, 0, 0, 1, 0, 0, 0, 0]], dtype=np.float32),
                0,
                2,
                ("left",),
            ),
            1,
        )
        self.assertAlmostEqual(trans_only, 0.5)
        self.assertAlmostEqual(progress_for_subtask(subtask, 3, curve=curve), 0.75)

    def test_zero_motion_falls_back_to_time(self):
        states = np.zeros((6, 16), dtype=np.float32)
        states[:, 3:7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        subtask = {"start_frame": 0, "end_frame": 5, "subtask_goal": "Move the left arm.", "subtask_type": "move"}
        anno = {"metadata": {}}
        self.assertAlmostEqual(progress_for_subtask(subtask, 3, states=states, anno=anno), 0.6)

    def test_progress_bucket_collapses_plateau_frames(self):
        frame_progress = [
            (10, 0.061),
            (11, 0.062),
            (12, 0.062),
            (13, 0.062),
            (14, 0.063),
            (15, 0.071),
        ]
        selected = select_frames_by_progress_bucket(frame_progress, bucket_size=0.01)
        self.assertEqual(selected, [14, 15])

    def test_progress_bucket_prefers_bucket_center_on_tie(self):
        frame_progress = [(0, 0.004), (1, 0.006)]
        selected = select_frames_by_progress_bucket(frame_progress, bucket_size=0.01)
        self.assertEqual(selected, [1])

    def test_select_undone_frame_indices_uses_stride_when_bucketing_disabled(self):
        subtask = {"start_frame": 0, "end_frame": 10, "subtask_goal": "Move the left arm.", "subtask_type": "move"}
        selected = select_undone_frame_indices(
            0,
            10,
            subtask=subtask,
            q2_frame_stride=3,
            q2_progress_bucket_size=0.0,
        )
        self.assertEqual(selected, [0, 3, 6, 9])

    def test_select_undone_frame_indices_buckets_linear_progress(self):
        states = np.zeros((11, 16), dtype=np.float32)
        states[:, 3:7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        for frame in range(1, 11):
            states[frame, 0] = frame / 10.0
        subtask = {"start_frame": 0, "end_frame": 10, "subtask_goal": "Move the left arm.", "subtask_type": "move"}
        curve = build_subtask_progress_curve(states, 0, 10, ("left",))
        all_frames = list(range(0, 10))
        selected = select_undone_frame_indices(
            0,
            9,
            subtask=subtask,
            curve=curve,
            q2_progress_bucket_size=0.1,
        )
        self.assertLess(len(selected), len(all_frames))
        self.assertEqual(selected, sorted(set(selected)))
        self.assertIn(0, selected)
        self.assertIn(9, selected)


if __name__ == "__main__":
    unittest.main()
