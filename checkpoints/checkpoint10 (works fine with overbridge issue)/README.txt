Checkpoint name: checkpoint10
Created from current working tree.

Purpose:
- Restore the project to the exact source/configuration state that existed
  when the user said to save the current status as "checkpoint10".

Key changes in this checkpoint (relative to checkpoint9):
- Kept NORMAL mode behavior unchanged.
- Changed INTERSECTION mode so the blue dot is locked to the stored global
  route for the full intersection traversal.
- Added a latched intersection-state flag so INTERSECTION mode stays active
  until the ego has entered and then exited the junction.
- Updated the MPC reference generation to stay route-locked during the full
  intersection traversal as well.
- Added regression tests covering the intersection latch behavior.

Restore method:
- Sync `checkpoints/checkpoint10/worktree/` back into the repo root.
- Use delete mode so files added later are removed during restore.

Notes:
- `.git/`, `checkpoints/`, `__pycache__/`, and `.claude/` are intentionally
  excluded from the snapshot.
