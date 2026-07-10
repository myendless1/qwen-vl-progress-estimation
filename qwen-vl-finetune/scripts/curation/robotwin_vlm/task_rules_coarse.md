# RoboTwin VLM Coarse Rules

Coarse annotations use seven subtask types:

- `single_move`
- `single_grasp`
- `single_place`
- `dual_move`
- `dual_grasp`
- `dual_place`
- `open_move`
- `handover`

Each subtask contains continuous motion plus at most one terminal gripper event.
`open_move` is the exception used for release-and-retreat motions. `handover`
covers the transfer phase where both grippers hold the object, one gripper
opens, and the receiving arm moves the object until the next gripper motion
ends the transfer.

## Boundary Rules

Coarse alignment has only two split rules:

1. A gripper event group ends.
2. A `single_move` changes to another `single_move`, i.e. left-only motion
   becomes right-only motion or right-only motion becomes left-only motion.

No other active-arm transition creates a boundary. In particular:

- single-to-dual does not split.
- dual-to-single does not split.
- dual motion is treated as transition/context and is absorbed into the
  surrounding coarse subtask.
- A standalone move while the gripper is already open is not emitted as its own
  subtask. Merge it into the previous open/place/handover subtask, including a
  final pure move at episode end, e.g. release plus retract/return/lift-up. A
  standalone move after a close gripper event may be a lift or carry step and
  should remain separate.

Episode end naturally closes the last subtask.

## Type Rules

The terminal gripper event decides grasp/place:

- motion + close -> `single_grasp` or `dual_grasp`
- motion + open -> `single_place` or `dual_place`
- motion without terminal gripper -> `single_move` or `dual_move`

Pending gripper events before a motion segment are included in the subtask time
range but do not decide the subtask type.

`open_move` covers an open gripper event followed by same-arm retreat. It ends
when another arm starts moving.

`handover` covers release by one arm while the other arm carries the object.
It starts when both arms are holding the object, one gripper opens, and the
receiving arm starts/continues motion. It ends when the next gripper motion for
the receiving arm finishes. In practice, the release-open span and the receiver
move/place span are merged into one `handover` subtask, and the instruction
combines the release and receiver motion with `while`.

## Text Rules

Task-specific coarse rule text should prefer semantic instructions:

- `Grasp the red block with the right arm.`
- `Place the red block at the right position with the right arm.`
- `Lift the bottle with the right arm.`
- `Open and retreat the right arm.`

Do not expose terminal gripper mechanics in ordinary grasp/place instructions:

- `*_grasp` should say `grasp ...`, not `move ... then close ...`.
- `*_place` should say `place ...`, not `move ... then open ...`.

For dual-arm object setup, name each arm/object when the task semantics require
it, for example `Grasp the bread with the left arm while grasping the skillet
with the right arm.` For container placement tasks, mention the actual
placement destination, for example `Place the bread into the breadbasket,
resting on the bottom.`
