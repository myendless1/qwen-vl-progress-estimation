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
from robotwin_vlm.task_rules import EXPECTED_TASK_SLUGS  # noqa: E402


ATOMIC_RULE_HASHES = {
    "adjust_bottle": "b1a8f28829c5baf2b9e8789cb2c351933af617ed3190cbf6d6690791fc8ec1c1",
    "beat_block_hammer": "631240b6c063d9c39dd46c85896ba1a2c0fb36ef816d2db3cbd4df404d8cfe1e",
    "blocks_ranking_rgb": "003824448b0415a5b496391b86cb1c84c04446b5b68db6bcdaad9ace3703c953",
    "blocks_ranking_size": "25fe40ac3b27b83a0eade060b8d20b77b7ad7e75708826904c2ccced6ef72cdc",
    "click_alarmclock": "2dca41ab3e28a96b4efbabe307b469fdf98f5677d93959c5615a1d6dac082f31",
    "click_bell": "7ae7a966b29dc9549525bffd2c929da3aae059e15f9af5ac2ce9bfe32e4cc193",
    "dump_bin_bigbin": "e9e0956123d5c0d0868063fd1a22c8d3cfed7c46425c989fc1eb67e89c0d2698",
    "grab_roller": "163071489fe6f28b95c0aac92502edc2fcd62bb30b8eec9e79643e91f6d9e41a",
    "handover_block": "21d8ad84f89401c359760284f5131e76dda4ad2ca4008e2df67afad09e7bd0de",
    "handover_mic": "f22e1b0c1def6835291207dc9ef294cf24d9bf5e0ae5c61a58bc87e42942cd04",
    "hanging_mug": "77860c83807dd50222985e259d29d26e2586c6d93eaa13435316949948a0a978",
    "lift_pot": "f8e09b141414e2b9040d1672fe322840997ec6ee92e6f3e7a0307ecc76901349",
    "move_can_pot": "62b1e7b251d9d9658a4100dfc8cfbf4da158d94438d04a78701f6d851a6d07d8",
    "move_pillbottle_pad": "bd443522c31e9a8d8874aad46a5f5d55d5be196da0734f1f99d35071100499a1",
    "move_playingcard_away": "2998c7c76aa614ff72c59bd5ec8791265e69cbef09e8d841e9b49a1896ea03e1",
    "move_stapler_pad": "1035608b2d1c56736eb4f61db70f7a9ab6b55ac1644da92bec9015010bedeca0",
    "open_laptop": "8d29243718da5095c6ebca325344931d166d2e770b88fe3b2ef0041c2362d3b0",
    "open_microwave": "410cdf6d7c222c1119f30f02952a0054940c3341e08a447d49f0c4534e510230",
    "pick_diverse_bottles": "f8426f4398fbf7ec05427fd68ca78669881842375993121c762e643cc16e84ac",
    "pick_dual_bottles": "f8426f4398fbf7ec05427fd68ca78669881842375993121c762e643cc16e84ac",
    "place_a2b_left": "3aec839bc148d2b97b597e8d4e26143198a61ee21e5ba5603c56a24c99dc8bab",
    "place_a2b_right": "e4f76bed0b7582dddff388cb649280e8af740ffb4539ac5440598f1fbbfcb1e0",
    "place_bread_basket": "6045baf909fc0e2097ad7e9b7827f74b2271e616630b351cdd3e3e5bb5601ad5",
    "place_bread_skillet": "588d0fc88964fa3ef8db5862e0400b434782f04e9ffc89647951947dba3a8a31",
    "place_burger_fries": "17a77168aa649443463c8ca28081800ac8bf3ac7f28eecbbfd00ab56a78daadd",
    "place_can_basket": "6859edbf0c0214d9671b8c14dcc168fe3705cad57ca919b0505c19b279d8cb49",
    "place_cans_plasticbox": "91b90aba1708c1bc19198436895d662cc0b6fadfb7705e7c2053815e58eac562",
    "place_container_plate": "bb740149952ce49704780a987a01b85c6a458db4ffd2dc1c465300648ff8d85d",
    "place_dual_shoes": "b71c6c1bf9a03bbe83b3f8ed69818bcef438c84f7520fe1b4e214aa6cf70cd02",
    "place_empty_cup": "f654ad9d7a7a19da9e66cae33250fee9729aa2ac7fb369e7aedf5e447806b937",
    "place_fan": "423482459143d2c836eda1fc40571897da5ad45e3874881683745c3c56acbea1",
    "place_mouse_pad": "b55b1d7e2c76e4a14644bf996315a8224a0b02e1bfee771a76da4abf278cc536",
    "place_object_basket": "4a620949477b85003106fccdc1fdc2b618fb5465abf088809650ef3a75b7c6b8",
    "place_object_scale": "90f31a5e2577c817070db755dea5f3f721873d6f542760243d8fe0a35c2ba33c",
    "place_object_stand": "0fdfb13e0036254c44bd9fa61f6b6b24a32d79991bae7a7f40171ea6e8350383",
    "place_phone_stand": "f9d1101ab6c64e70235daef933ce79f93baeb8f6e3d999868fd042b4532e1cd2",
    "place_shoe": "1035608b2d1c56736eb4f61db70f7a9ab6b55ac1644da92bec9015010bedeca0",
    "press_stapler": "b9655d6c9fd0fb4e44430bbcdeb5eae5c47a546d1e0bbe2eaf2d02a62240d39f",
    "put_bottles_dustbin": "790cba299a36138772b6845b6a24d2837972ac300c93750725c08741b50873c3",
    "put_object_cabinet": "f9ccf7a9f4311403d72d0d347527eb0611e218836a21a0ebce803dfe02ee917e",
    "rotate_qrcode": "2ca4ec8959002748030625032cb1db79feba56850f51b2a6c7428a36804ac816",
    "scan_object": "5a3e3a6c56831290c66a5802a3ac203e6a8c51e09821da1bd81ec72d29a0eb28",
    "shake_bottle": "e1a7aec875d105a8c5a0e70df04bd2244072b52a1b22f0465c4eb37e0ad93b31",
    "shake_bottle_horizontally": "0d66c9369c565c39078c5debc54f8ce8c7f5ae4e6b636b8428ca4b9310f48b63",
    "stack_blocks_three": "0bf61dc5e79cbfd3aca0a39419b9618cf3a921b769faea70409cb9d7edd29ce2",
    "stack_blocks_two": "6faeb3d2ad4b0f4f429bbc3f90fe7f9a17a008a3232d1515b2d73d520cfc409d",
    "stack_bowls_three": "4f73bb27015e40d4c518b6edef72bb27cd041997222abe82d498f4e729307378",
    "stack_bowls_two": "3d60f01de33eae48631c9c00c9a7bdb7e0f0b7ff64eea1f4c9de54cb6b536353",
    "stamp_seal": "11ce5c44d61f46b81fb741787d8c4f74a70a6f10dd2b86399ecaeaa5feaaf701",
    "turn_switch": "7244442b1d20006a0411da09e9fe81c4c4839bcade863263b6911235734443c8",
}

EVENT_RULE_HASHES = {
    "place_bread_skillet": "39e806e7e5fb96f534607069a408e058a23ee362680e766b2f9261a62cfe0030",
    "place_can_basket": "5dbe1737776b2408c92f2c730ec7e9eb2632d12b5e4f78828afbd2b23323b759",
    "place_object_basket": "5dbe1737776b2408c92f2c730ec7e9eb2632d12b5e4f78828afbd2b23323b759",
    "put_bottles_dustbin": "71f0e2cafe15937525bbf77f1b791826c245b061b33f1cec08f34c5b581bff1c",
    "put_object_cabinet": "72dcd5c57880e6f6acfa34ea9609921c33548d67cbbf6244bbcae45df4902c86",
    "scan_object": "74dbb72c820d0b1548e915b57290b82284cda1229d2659570d4222a1877a3f07",
}

ATOMIC_EVENT_KINDS = {"move", "open", "close", "press", "final"}


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
        self.assertEqual(len(steps), 8)
        self.assertEqual(steps[1].event_kind, "close")
        self.assertEqual(steps[5].event_kind, "close")
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
        self.assertEqual(len(single_steps), 4)
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
        self.assertEqual(steps[5].text, "Move the left arm to the place pose of the phone inside the cabinet.")
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
