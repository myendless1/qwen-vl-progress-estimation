from __future__ import annotations

import hashlib
import json
import sys
import unittest
from pathlib import Path


CURATION_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CURATION_DIR))

from robotwin_vlm import GripperEvent, TASK_BUILDERS, build_steps  # noqa: E402
from robotwin_vlm.prompts import (  # noqa: E402
    clean_object_name,
    parse_a2b_objects_from_prompt,
    prompt_basket_object,
)
from robotwin_vlm.task_rules_fine import EXPECTED_TASK_SLUGS  # noqa: E402


ATOMIC_RULE_HASHES = {
    "adjust_bottle": "574cecc9a495f70cce802ce99b5581f4c6b820cec65a4254846c7766df082b9a",
    "beat_block_hammer": "553e24b28812def60989a0e5011d809fa8f56c9d614a6f209f3b46f7259516ef",
    "blocks_ranking_rgb": "01213e2c60910aee954221df345cf8dbff8b9996169189cbff226feb6ed7056c",
    "blocks_ranking_size": "cdd4d91f1db35c41083460302283ae0fb8afde3cc50546d0d55ba795b8e8841a",
    "click_alarmclock": "61311022522db0c68fe74251a9b1db2be727b67df5ce69103e168a42949ca866",
    "click_bell": "48db4a3b0a7efbd9efbfc65b49ec2cab341ddae9c53a2925213ae8cedadec8ab",
    "dump_bin_bigbin": "8d33f59f13e135d75b8e3b5f441dbab447f1ec16c0113192490246cb43e7daac",
    "grab_roller": "b53822b7210d32071fea28ad22e3c1b483763d86ba983cb69f5921d627f0c0ec",
    "handover_block": "8b360de00b080d0bf26003430d2d75f2934fa3ce53b87e60734cc649b1cf45e8",
    "handover_mic": "0dc11b3324c7bdeb7cffc79fa467035f2ed8c67d208a16e365aaab6d67e850c2",
    "hanging_mug": "1c80457cc76da7559cdd822ffa2fc8ba9b6e2f2b99d1f9d5c60dee97b5374444",
    "lift_pot": "23e9cb7cb1953716d65dc9d2cad481d5eb108a49e0a0373e738c728082d926ce",
    "move_can_pot": "62b1e7b251d9d9658a4100dfc8cfbf4da158d94438d04a78701f6d851a6d07d8",
    "move_pillbottle_pad": "bd443522c31e9a8d8874aad46a5f5d55d5be196da0734f1f99d35071100499a1",
    "move_playingcard_away": "2998c7c76aa614ff72c59bd5ec8791265e69cbef09e8d841e9b49a1896ea03e1",
    "move_stapler_pad": "1035608b2d1c56736eb4f61db70f7a9ab6b55ac1644da92bec9015010bedeca0",
    "open_laptop": "8d29243718da5095c6ebca325344931d166d2e770b88fe3b2ef0041c2362d3b0",
    "open_microwave": "46684f732392ec28a23e2476c5ea417d1041e4649b270be28cf2ccfa03e075b9",
    "pick_diverse_bottles": "67d57e553d7535ab85ed43ed6df421fdf3af12719133d0707e3f28aca81ec910",
    "pick_dual_bottles": "67d57e553d7535ab85ed43ed6df421fdf3af12719133d0707e3f28aca81ec910",
    "place_a2b_left": "e4f76bed0b7582dddff388cb649280e8af740ffb4539ac5440598f1fbbfcb1e0",
    "place_a2b_right": "3aec839bc148d2b97b597e8d4e26143198a61ee21e5ba5603c56a24c99dc8bab",
    "place_bread_basket": "ae8a66941f223bc7767cf2b024086c12d25892ce4e4e3ef1c7ef35abd3999b13",
    "place_bread_skillet": "0dcaafcfbb24ead15748bb4699496014d0c4e3522855454430229d6f2839ad99",
    "place_burger_fries": "75164501356ff318494c1d1c4d2d7cac8e09ad670e183d8b2c3d51f35e599fc3",
    "place_can_basket": "e5a492a0bccfb6ae52a6ddfeca95b90827dd34163f1a0a80a2345912253f0c71",
    "place_cans_plasticbox": "bf54ca60d6645982997768e71e916b8721d89d95f5be055cf0f7aa885a83ec02",
    "place_container_plate": "1530414620422d3108ea047ad5f4807744d4c44b760a722ff299ce82d6b6cc99",
    "place_dual_shoes": "05dc37ea3666d520ae8ac1aa5f335d62234b085bfe70420701bdc9012d173717",
    "place_empty_cup": "241bcfaa1d06e9960c5b4f237bbd05876965c8c64b7024ce33966d1d29ea7d2c",
    "place_fan": "423482459143d2c836eda1fc40571897da5ad45e3874881683745c3c56acbea1",
    "place_mouse_pad": "b55b1d7e2c76e4a14644bf996315a8224a0b02e1bfee771a76da4abf278cc536",
    "place_object_basket": "135026fa923157a2c4ba66393c80a6b63b2ed1bf6584b824d58524ee3e2a4fa5",
    "place_object_scale": "90f31a5e2577c817070db755dea5f3f721873d6f542760243d8fe0a35c2ba33c",
    "place_object_stand": "0fdfb13e0036254c44bd9fa61f6b6b24a32d79991bae7a7f40171ea6e8350383",
    "place_phone_stand": "f9d1101ab6c64e70235daef933ce79f93baeb8f6e3d999868fd042b4532e1cd2",
    "place_shoe": "1035608b2d1c56736eb4f61db70f7a9ab6b55ac1644da92bec9015010bedeca0",
    "press_stapler": "9ad572fc65def4b4665fecf171216ac08efb2dc55ac2a1b87657ff20c1102443",
    "put_bottles_dustbin": "2b3d3ce45ce90ac3d40c5bad4093265be07adad48500ce2e73a43c0a6b7dcda0",
    "put_object_cabinet": "676f90b73e1e3d5a890bfa1013dfd016aae523d54130d6c3640ebab934d186c9",
    "rotate_qrcode": "2d619d2d401f5ea6754acc23eec7a50daf887d9855f96a015898fb782720475d",
    "scan_object": "a4aca6e7e4c99a2dc49c2f336028e93dea4ef506675d20a1b3682e5c2d70c1c7",
    "shake_bottle": "e1a7aec875d105a8c5a0e70df04bd2244072b52a1b22f0465c4eb37e0ad93b31",
    "shake_bottle_horizontally": "0d66c9369c565c39078c5debc54f8ce8c7f5ae4e6b636b8428ca4b9310f48b63",
    "stack_blocks_three": "a8ee63aa57653ddb5063e6b76d2a437e5d40ad88caabeda6445ef3d88d5e79d9",
    "stack_blocks_two": "5ec5dadd73150bbd3a753edd34ba8b4c190122ca8063abdef046b6f9b31122fe",
    "stack_bowls_three": "c9c2bd2504128b779a56cc4e3796cabdbb9f9546fc9ea69b7d5761af5496709a",
    "stack_bowls_two": "4acfe206f16e9a0cb5344ca756df5274328aa57b0fd40fb54fd8a7c576f6bfa8",
    "stamp_seal": "11ce5c44d61f46b81fb741787d8c4f74a70a6f10dd2b86399ecaeaa5feaaf701",
    "turn_switch": "7244442b1d20006a0411da09e9fe81c4c4839bcade863263b6911235734443c8",
}


EVENT_RULE_HASHES = {
    "place_bread_skillet": "dbe9f20bca6618a5e99483086d43150c1119c35ce9df94bb7320cd88153c7de4",
    "place_can_basket": "0dc940d53b9f3e169e5b473132d9ff2bcc168cd77d7c0ed0c5b21af8e3f9a52a",
    "place_object_basket": "0dc940d53b9f3e169e5b473132d9ff2bcc168cd77d7c0ed0c5b21af8e3f9a52a",
    "put_bottles_dustbin": "06ff0559f308f3dd9c710374420d0e70c0868060634fcf6bb01d4625bb75f109",
    "put_object_cabinet": "af3014dfa61f190cdb9b535fb56cb5a5a1eda3581d0f0843c9cc03bc5bccae38",
    "scan_object": "5dcd3b0f7d1644376431871aae5d455350d002e2575df453502facd4f8641b42",
}


ATOMIC_EVENT_KINDS = {"move", "open", "close", "press", "handover", "final"}


def rule_hash(steps: list[object]) -> str:
    payload = [
        {
            "text": step.text,
            "event_kind": step.event_kind,
            "arm": step.arm,
            "terminates_on": list(step.terminates_on),
        }
        for step in steps
    ]
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


class TaskRuleTests(unittest.TestCase):
    def test_all_50_tasks_are_registered_once(self) -> None:
        self.assertEqual(len(TASK_BUILDERS), 50)
        self.assertEqual(set(TASK_BUILDERS), EXPECTED_TASK_SLUGS)
        self.assertEqual(set(ATOMIC_RULE_HASHES), EXPECTED_TASK_SLUGS)

    def test_all_default_rules_match_atomic_contract(self) -> None:
        for slug, expected in ATOMIC_RULE_HASHES.items():
            with self.subTest(slug=slug):
                self.assertEqual(rule_hash(build_steps(slug, "", {})), expected)

    def test_event_driven_rules_match_atomic_contract(self) -> None:
        events = [
            GripperEvent(10, "left", "close"),
            GripperEvent(20, "right", "close"),
            GripperEvent(30, "left", "open"),
            GripperEvent(40, "right", "open"),
        ]
        info = {"{A}": "toycar", "{B}": "basket", "{a}": "left", "{b}": "right"}
        prompt = "Place the toy car into the basket using the left arm."
        for slug, expected in EVENT_RULE_HASHES.items():
            with self.subTest(slug=slug):
                self.assertEqual(rule_hash(build_steps(slug, prompt, info, events)), expected)

    def test_task_rules_only_use_atomic_event_kinds(self) -> None:
        for slug in EXPECTED_TASK_SLUGS:
            with self.subTest(slug=slug):
                for step in build_steps(slug, "", {}):
                    self.assertIn(step.event_kind, ATOMIC_EVENT_KINDS)

    def test_task_rules_do_not_use_ambiguous_arm_references(self) -> None:
        ambiguous = ("first arm", "receiving arm", "other arm")
        for slug in EXPECTED_TASK_SLUGS:
            with self.subTest(slug=slug):
                text = "\n".join(step.text.lower() for step in build_steps(slug, "", {}))
                for phrase in ambiguous:
                    self.assertNotIn(phrase, text)

    def test_press_tasks_start_above_target_with_close(self) -> None:
        expected = {
            "click_alarmclock": "alarm clock button",
            "click_bell": "bell",
            "press_stapler": "stapler",
        }
        for slug, target in expected.items():
            with self.subTest(slug=slug):
                first = build_steps(slug, "", {})[0]
                self.assertEqual(first.event_kind, "close")
                self.assertIn(f"above the {target}", first.text)
                self.assertIn("close the gripper", first.text)

    def test_open_microwave_mentions_63_degree_threshold(self) -> None:
        text = "\n".join(step.text for step in build_steps("open_microwave", "", {}))
        self.assertIn("at least 63 degrees", text)

    def test_handover_rules_use_handover_action(self) -> None:
        for slug in ("handover_block", "handover_mic"):
            with self.subTest(slug=slug):
                steps = build_steps(slug, "", {})
                self.assertTrue(any(step.event_kind == "handover" for step in steps))
                self.assertNotIn("with right arms", "\n".join(step.text for step in steps).lower())

    def test_place_bread_basket_sequential_breads_follow_prompt_names(self) -> None:
        events = [
            GripperEvent(53, "right", "close"),
            GripperEvent(127, "right", "open"),
            GripperEvent(201, "right", "close"),
            GripperEvent(274, "right", "open"),
        ]
        steps = build_steps(
            "place_bread_basket",
            "Shift the golden bread and the light brown bread with ridges into the woven plastic breadbasket",
            {},
            events,
        )
        self.assertEqual(len(steps), 10)
        self.assertEqual(steps[1].event_kind, "close")
        self.assertEqual(steps[6].event_kind, "close")
        text = "\n".join(step.text for step in steps)
        self.assertIn("golden bread", text)
        self.assertIn("light brown bread with ridges", text)
        self.assertIn("woven plastic breadbasket", text)
        self.assertNotIn("right bread", text)
        self.assertNotIn("left bread", text)

    def test_place_bread_basket_single_bread_and_dual_arm_shapes(self) -> None:
        single_steps = build_steps(
            "place_bread_basket",
            "Put the cuboid bread into the white oval breadbasket after grabbing it.",
            {},
            [GripperEvent(56, "left", "close"), GripperEvent(130, "left", "open")],
        )
        self.assertEqual(len(single_steps), 5)
        self.assertIn("cuboid bread", "\n".join(step.text for step in single_steps))

        dual_steps = build_steps(
            "place_bread_basket",
            "Place the fluffy baked bread and the rounded square loaf in the white oval breadbasket using the dual arm",
            {},
            [
                GripperEvent(58, "left", "close"),
                GripperEvent(58, "right", "close"),
                GripperEvent(131, "left", "open"),
                GripperEvent(218, "right", "open"),
            ],
        )
        self.assertEqual(len(dual_steps), 9)
        text = "\n".join(step.text for step in dual_steps)
        self.assertIn("fluffy baked bread", text)
        self.assertIn("rounded square loaf", text)
        self.assertIn("both arms", text)
        self.assertNotIn("both objects", text)
        lift_step = dual_steps[2]
        self.assertIn("with both arms", lift_step.text)
        self.assertEqual(lift_step.event_kind, "move")
        second_place = dual_steps[6]
        self.assertIn("while returning the", second_place.text)
        self.assertIn("neutral pose", second_place.text)
        self.assertEqual(second_place.terminates_on, ("gripper_open",))

    def test_place_bread_skillet_splits_skillet_and_bread_motion(self) -> None:
        steps = build_steps(
            "place_bread_skillet",
            "Take the soft brown bread and place it inside the metal skillet with comfortable grip",
            {},
            [
                GripperEvent(62, "left", "close"),
                GripperEvent(62, "right", "close"),
                GripperEvent(152, "left", "open"),
            ],
        )
        self.assertEqual(len(steps), 7)
        self.assertEqual(steps[3].event_kind, "move")
        self.assertEqual(steps[3].arm, "right")
        self.assertIn("right arm to bring the skillet", steps[3].text)
        self.assertNotIn("while", steps[3].text)
        self.assertEqual(steps[4].event_kind, "move")
        self.assertEqual(steps[4].arm, "left")
        self.assertIn("left arm above the skillet", steps[4].text)
        self.assertNotIn("while", steps[4].text)

    def test_put_object_cabinet_splits_door_pull_and_object_place(self) -> None:
        steps = build_steps(
            "put_object_cabinet",
            "Place the phone inside the cabinet.",
            {},
            [
                GripperEvent(54, "left", "close"),
                GripperEvent(127, "right", "close"),
                GripperEvent(253, "left", "open"),
            ],
        )
        self.assertEqual(len(steps), 7)
        self.assertEqual(steps[4].text, "Pull open the cabinet with the right arm.")
        self.assertEqual(steps[4].event_kind, "move")
        self.assertEqual(steps[4].arm, "right")
        self.assertEqual(steps[5].text, "Move the left arm to the place pose of the phone into the cabinet, resting on the bottom.")
        self.assertEqual(steps[5].event_kind, "move")
        self.assertEqual(steps[5].arm, "left")
        self.assertNotIn("while moving", "\n".join(step.text for step in steps))

    def test_unknown_task_keeps_single_final_prompt(self) -> None:
        steps = build_steps("future_task", "Do the future task.", {})
        self.assertEqual([(step.text, step.event_kind, step.arm) for step in steps], [
            ("Do the future task.", "final", None)
        ])


class PromptParserTests(unittest.TestCase):
    def test_scene_info_aliases(self) -> None:
        self.assertEqual(clean_object_name("12_toycar/base1", "object"), "toy car")
        self.assertEqual(clean_object_name("playingcards", "object"), "playing card box")
        self.assertEqual(clean_object_name(None, "fallback"), "fallback")

    def test_basket_alias_and_a2b_relation(self) -> None:
        self.assertEqual(
            prompt_basket_object("Put the pink car into the basket.", "object"),
            "toy car",
        )
        self.assertEqual(
            parse_a2b_objects_from_prompt(
                "Place the mouse to the left of the phone.",
                "left",
            ),
            ("mouse", "phone"),
        )


if __name__ == "__main__":
    unittest.main()
