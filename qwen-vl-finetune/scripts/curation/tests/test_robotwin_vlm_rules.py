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


LEGACY_RULE_HASHES = {
    "adjust_bottle": "58e66eef4e55b9f472f1165f7211c5dbbd231fde514f61978dc374ac02dd49e9",
    "beat_block_hammer": "9f26106f5c75cd19ecd3dba45534d57f783e8965db82a373473dd94ff7c51c3d",
    "blocks_ranking_rgb": "8b41a1d9de7bff95f8eb5218339aefb47caf81d402ea2c2bc45e1dc7fe1b26c2",
    "blocks_ranking_size": "a38d52ee087b7a7628df183091b800a3f51a0f14442865eea72de0fa4ed288a4",
    "click_alarmclock": "343f7d587392db43ce5f70e304c8fd9f1eeb447da03f7b6fc85121d56269300f",
    "click_bell": "eb31ceabecc9b06a7dbb134035cbd08a7b501158ae9b7b2fa0674794ef786eba",
    "dump_bin_bigbin": "15c2d01703e839cc2d0e3ddfe9dffe9f1980d0731c43cd655a507aa716a6a0d7",
    "grab_roller": "6170edf3248fb4d20fb26e2dbfd960161ef845f413fce006c340255161502fc1",
    "handover_block": "ef8e6722b4e64be971926506ff50418ac7d70494402c8b0f7cc61089d13cff0b",
    "handover_mic": "cd00705d0cbc00e40e502c9e90e4b72e83428f2975ff39cd820886588f161840",
    "hanging_mug": "eb08df613ea4b33efb96d07a0f89780db06a5beef2535dcdb659d164a212c8ad",
    "lift_pot": "87b36bd24fe49ec483e41467f2e0e62d37b957a0555c874279412ff536739769",
    "move_can_pot": "1ccc321c17cd7fc87d5113d0cb72568441d802b561e082da4bf6bac12cbdbbf8",
    "move_pillbottle_pad": "5e29d1a4d646dca8e416093e61c1aca13c672e20b6300f1fe01153452d2dc425",
    "move_playingcard_away": "83a46afa87bf47cdd52a4cb5e3b32e66847d0866bc976fa7d4b3a198de374e9b",
    "move_stapler_pad": "75986804c67c147b034571d6da929a9fa504db42da466d569acf5952f312535c",
    "open_laptop": "1fe918c42d75e9c41b25655c8b81fef73086c9f99444b8014dba8b6843af0238",
    "open_microwave": "8b33498c8ed92f9472dc033aebf3745c3d1af978d58096d27a962f1ac805e671",
    "pick_diverse_bottles": "f98520f4807fa41f79167f35c9660c4c23523e8f8ea99e8ca7c3559ec572d17b",
    "pick_dual_bottles": "f98520f4807fa41f79167f35c9660c4c23523e8f8ea99e8ca7c3559ec572d17b",
    "place_a2b_left": "63052712352f7ab13730a13a42cb25ecdb248ba523de6d0ddcc1c36a1513eb74",
    "place_a2b_right": "9124c26ff1ef324975f001e05bae5f44d4a0f53ad057526709dae8ef11e4ad31",
    "place_bread_basket": "061ab2b76f30c2b2fff6f005180b4a2c8d746ea0efe38c81b2392364faf81c78",
    "place_bread_skillet": "2157feeb34f16829d1057f81fc9d6ccd18d878ed3566a70bee9345668f89e6c3",
    "place_burger_fries": "0bba033c8ba1fa1318a6e7a11380453701990db2019e43c8c546cf7093e7462b",
    "place_can_basket": "3d9c727b4547de79987a8a21baebb4a2bd66f709ae614e24cac547100ca1c414",
    "place_cans_plasticbox": "9a3bf7263b3d0792a89b89c4361187bd8f9e63b027ef635c2ff780d33a27438b",
    "place_container_plate": "f6cf832c510ecb4b3205edc4920ffe8910b78949f3474a3e788933c2b6c6b42a",
    "place_dual_shoes": "4ba4331bfdb28df33a73e9fccedde8c8bc2a9a050a8fbc48572123f427ec3a57",
    "place_empty_cup": "d470d377c91dd62b5b44b5a9223dd71e5580c4657554405c1dbee4c527f5cd6a",
    "place_fan": "95aa8d90db46332a0bef5d0ee90b221ef710125a6c7aac198aeeda942065e434",
    "place_mouse_pad": "780c2ce142a69bf5853dbf86687a609327e3e36dbe7f9dcdac5773b527eff903",
    "place_object_basket": "c17e61ec14f757caff1ff29f69e2e66d5ed42dadbe731b5ef0e786eea5c460e0",
    "place_object_scale": "f3d9cc5c4137d52e6b66c0b0c93e58618db8ad51a6fd84cb6e14728f2e3b9f8f",
    "place_object_stand": "9f16cebe4d0f62da1678a5c12ee4e93137463f3ff1e63adeacf619a835a0d2ed",
    "place_phone_stand": "8d10ba60f915d58d42d11da1e626b60e50c9943ddd829a50d55be2e2ff8478f0",
    "place_shoe": "75986804c67c147b034571d6da929a9fa504db42da466d569acf5952f312535c",
    "press_stapler": "195d40c6ee9d604d1baeb64a221f59a9d95013c0805983a3a00e1441d0c2322b",
    "put_bottles_dustbin": "6711afe25060bfc8832e8c30e9029000ab003997d3b171292f3b8c4bacc6d637",
    "put_object_cabinet": "a90c469a2a8385663b1b04daacf1a0d7079be4fb0fc3b5c0caf74e0bc6cfd824",
    "rotate_qrcode": "382a00eba5afe630a061234616d055a12cc4632c8c07136334948babe172357b",
    "scan_object": "2ce9df0a914be9f68fca797c7e19d551a5d635288a956b6ba463383b4412c2a2",
    "shake_bottle": "e3137bcddd726b5a1cc075b08dca4c44394d8ade0e5216e30ace3e13193d91c8",
    "shake_bottle_horizontally": "509260597b832d1f88bf8d8ae89d19d8381ce43d73c81f14c48fd9d7d8627b57",
    "stack_blocks_three": "713bcd054594deb665b5221425a73036e82515b96c1b282e3a74e8f478381012",
    "stack_blocks_two": "d8bd9f2c2923099269cce1589618ab6f36007569cdf30dec955ef3bf3205a29d",
    "stack_bowls_three": "22d88a0e6b5f8a5a9b77bbe267e539287d652151464a54ca10d107bbc3ab6c4a",
    "stack_bowls_two": "946f6f14082df82021517d4f8b17fbd29e4044654f6eb301b0e3722cccbde8db",
    "stamp_seal": "ec980564da38b8fb59dfa077e4c1b44edb4e34c3cf0ea9d3916a76c0b98b14aa",
    "turn_switch": "bf93c47276e0fbd46b58b4d28c5cf309fdbc5286f9421a25caf3fdb0d1336f71",
}

EVENT_RULE_HASHES = {
    "place_can_basket": "8fbc2f513361838d7f7d65e7c3202fc31ad03f02beb71d149a4a0be906c88370",
    "place_object_basket": "8fbc2f513361838d7f7d65e7c3202fc31ad03f02beb71d149a4a0be906c88370",
    "put_bottles_dustbin": "3b9b0931b153dfe15aef65111c42305904dd3590803bacdf228d5a51df63940e",
    "put_object_cabinet": "f856dac81a14cf50f6a7dbf8110cd7a7601c6f90aeff0167782911d53fb50463",
    "scan_object": "0c2e82c3b80c8761e4515abf45dc07b2831fdf70ec62e51760278319c0f75fd7",
    "place_bread_skillet": "efde1ef03c6305986c0dfbb95d461c282469e56363d8ed10445f36c3ea1c35b5",
}


def rule_hash(steps: list[object]) -> str:
    payload = [
        {"text": step.text, "event_kind": step.event_kind, "arm": step.arm}
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
        self.assertEqual(set(LEGACY_RULE_HASHES), EXPECTED_TASK_SLUGS)

    def test_all_default_rules_match_legacy_contract(self) -> None:
        for slug, expected in LEGACY_RULE_HASHES.items():
            with self.subTest(slug=slug):
                self.assertEqual(rule_hash(build_steps(slug, "", {})), expected)

    def test_event_driven_rules_match_legacy_contract(self) -> None:
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
