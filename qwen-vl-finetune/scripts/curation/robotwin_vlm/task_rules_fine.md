# RoboTwin VLM Fine Rules

Fine annotations decompose each task into atomic action-style subtasks.

The common atomic action kinds are:

- `move`
- `close`
- `open`
- `press`
- `handover`
- `final`

Task builders in `task_rules_fine.py` compose these actions with reusable
primitives such as `grasp_steps`, `place_steps`, `pair_steps`, and dual-arm
variants. A typical pick-and-place task is therefore represented as:

1. move to grasp pose
2. close gripper
3. move to place pose
4. open gripper

`handover` is used when both grippers hold the same object, one arm opens, and
the receiving arm moves the object until the next gripper motion ends the
transfer.

Fine alignment uses the ordered rule steps plus detected gripper events and
state motion candidates. Boundaries may terminate on gripper open/close,
current-arm motion, other-arm motion, both-arm settling, press lift, or episode
end depending on the rule step.

Post-processing keeps fine annotations atomic while correcting common alignment
details, including stack arm-switch wording, open-state move merging,
dual-container first-place relabeling, and final arm-label publication.
