Checkpoint name: checkpoint9
Created from current working tree.

Purpose:
- Restore the project to the exact source/configuration state that existed
  when the user said to save the current status as "checkpoint9".

Key changes in this checkpoint (relative to checkpoint8):
- Added rule-based lane-safety and temporary-destination planner modules in
  `behavior_planner/`.
- Updated intersection behavior so the blue dot changes lanes only from
  behavior-planner decisions, follows the global route inside intersections,
  and switches to `Lane Follow` while traversing an active turn branch.
- Updated the runner and global-route summary so maneuver/mode decisions are
  driven from the blue dot's route position and current turn block.
- Added regression tests for intersection behavior, blue-dot mode logic, and
  route-summary progression.

Restore method:
- Sync `checkpoints/checkpoint9/worktree/` back into the repo root.
- Use delete mode so files added later are removed during restore.

Notes:
- `.git/`, `checkpoints/`, `__pycache__/`, and `.claude/` are intentionally
  excluded from the snapshot.
