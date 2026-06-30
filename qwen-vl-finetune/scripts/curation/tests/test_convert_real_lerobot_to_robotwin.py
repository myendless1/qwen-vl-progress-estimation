import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from convert_real_lerobot_to_robotwin import (
    ARM_LEFT,
    ARM_RIGHT,
    GRIP_LEFT,
    GRIP_RIGHT,
    POSE_LEFT,
    POSE_RIGHT,
    build_actions,
    build_observation_states,
    convert_episode,
    detect_conversion_mode,
    output_episode_path,
    validate_robotwin_parquet,
)


def _write_pose_source(path: Path, num_rows: int = 5) -> None:
    left_pose = [[0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0] for _ in range(num_rows)]
    right_pose = [[0.4, 0.5, 0.6, 0.0, 1.0, 0.0, 0.0] for _ in range(num_rows)]
    table = pa.table(
        {
            POSE_LEFT: left_pose,
            POSE_RIGHT: right_pose,
            GRIP_LEFT: [0.1 * i for i in range(num_rows)],
            GRIP_RIGHT: [0.2 * i for i in range(num_rows)],
            "timestamp": [float(i) for i in range(num_rows)],
            "frame_index": list(range(num_rows)),
            "episode_index": [7] * num_rows,
            "index": list(range(num_rows)),
            "task_index": [0] * num_rows,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def _write_arm_source(path: Path, num_rows: int = 5) -> None:
    left_arm = [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7] for _ in range(num_rows)]
    right_arm = [[0.8, 0.9, 1.0, 0.2, 0.3, 0.4, 0.5] for _ in range(num_rows)]
    table = pa.table(
        {
            ARM_LEFT: left_arm,
            ARM_RIGHT: right_arm,
            GRIP_LEFT: [0.0, 0.25, 0.5, 0.75, 1.0][:num_rows],
            GRIP_RIGHT: [1.0, 0.75, 0.5, 0.25, 0.0][:num_rows],
            "timestamp": [float(i) for i in range(num_rows)],
            "frame_index": list(range(num_rows)),
            "episode_index": [8] * num_rows,
            "index": list(range(100, 100 + num_rows)),
            "task_index": [0] * num_rows,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


class ConvertRealLerobotTest(unittest.TestCase):
    def test_detect_conversion_mode(self):
        self.assertEqual(
            detect_conversion_mode({POSE_LEFT, POSE_RIGHT, GRIP_LEFT, GRIP_RIGHT}),
            "pose",
        )
        self.assertEqual(
            detect_conversion_mode({ARM_LEFT, ARM_RIGHT, GRIP_LEFT, GRIP_RIGHT}),
            "arm",
        )

    def test_build_actions_matches_robotwin_convention(self):
        states = np.arange(32, dtype=np.float64).reshape(2, 16)
        actions = build_actions(states)
        np.testing.assert_array_equal(actions[0], states[1])
        np.testing.assert_array_equal(actions[1], states[1])

    def test_convert_pose_episode(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            source = repo / "data" / "chunk-000" / "episode_000007.parquet"
            output = output_episode_path(repo, source)
            _write_pose_source(source)
            convert_episode(source, output, "pose", overwrite=True, dry_run=False)
            validate_robotwin_parquet(output)
            table = pq.read_table(output)
            self.assertEqual(table.column_names, [
                "observation.state",
                "action",
                "timestamp",
                "frame_index",
                "episode_index",
                "index",
                "task_index",
            ])
            state0 = table.column("observation.state")[0].as_py()
            self.assertEqual(len(state0), 16)
            self.assertAlmostEqual(state0[7], 0.0)
            self.assertAlmostEqual(state0[15], 0.0)

    def test_convert_arm_episode(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            source = repo / "data" / "chunk-000" / "episode_000008.parquet"
            output = output_episode_path(repo, source)
            _write_arm_source(source)
            convert_episode(source, output, "arm", overwrite=True, dry_run=False)
            validate_robotwin_parquet(output)
            states = np.asarray(
                pq.read_table(output, columns=["observation.state"]).column("observation.state").to_pylist(),
                dtype=np.float64,
            )
            self.assertEqual(states.shape, (5, 16))
            left_quat_norm = np.linalg.norm(states[0, 3:7])
            self.assertAlmostEqual(left_quat_norm, 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
