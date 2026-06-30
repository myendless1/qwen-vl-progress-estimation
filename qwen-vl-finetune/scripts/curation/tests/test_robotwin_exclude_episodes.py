import json
import tempfile
import unittest
from pathlib import Path

from qwenvl.data.robotwin_processor import load_robotwin_excluded_episodes


class RobotWinExcludeEpisodesTest(unittest.TestCase):
    def test_load_json(self):
        path = Path(__file__).resolve().parent / "robotwin_anno_eval_exclude_episodes.json"
        excluded = load_robotwin_excluded_episodes(str(path))
        self.assertEqual(len(excluded), 9)
        self.assertIn(("turn_switch-aloha-agilex_randomized_500", 339), excluded)

    def test_load_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "issues.jsonl"
            with path.open("w", encoding="utf-8") as f:
                f.write(json.dumps({"repo": "demo_repo", "episode_index": 7}) + "\n")
            excluded = load_robotwin_excluded_episodes(str(path))
            self.assertEqual(excluded, {("demo_repo", 7)})


if __name__ == "__main__":
    unittest.main()
