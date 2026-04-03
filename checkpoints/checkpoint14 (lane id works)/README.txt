Checkpoint name: checkpoint14 (lane id works)
Created from current working tree.

Purpose:
- Restore the project to the exact source/configuration state that existed
  when the user said to save the current status as
  "checkpoint 14 (lane id works)".

Key changes in this checkpoint (relative to checkpoint13):
- Normalized lane-ID handling across the planner/runtime path so route,
  behavior-planner, temp-destination, and HUD lane information use one
  consistent project convention.
- Tightened lane-change decision gating so the planner only commands left or
  right changes when that adjacent lane actually exists.
- Kept temporary-destination lane changes under behavior-planner control so
  the blue-dot lane only changes when the planner explicitly decides it.

Restore method:
- Sync `checkpoints/checkpoint14 (lane id works)/worktree/` back into the
  repo root.
- Use delete mode so files added later are removed during restore.

Notes:
- `.git/`, `checkpoints/`, and `.claude/` are intentionally excluded from the
  snapshot.
