Checkpoint name: checkpoint11(works but has issue with maneuver update)
Created from current working tree.

Purpose:
- Restore the project to the exact source/configuration state that existed
  when the user said to save the current status as
  "checkpoint11(works but has issue with maneuver update)".

Key changes in this checkpoint (relative to checkpoint10):
- Updated lane-context and route-summary handling so lane IDs and upcoming
  maneuver reporting use the current runtime route position more consistently.
- Reworked live lane-safety scoring to run synchronously from the current
  obstacle set and filter obstacles by the ego driving corridor.
- Improved MPC curved-road tracking by keeping stage-to-reference matching
  local along the lane-center reference and by making destination XY tracking
  symmetric in the cost configuration.

Restore method:
- Sync `checkpoints/checkpoint11(works but has issue with maneuver update)/worktree/`
  back into the repo root.
- Use delete mode so files added later are removed during restore.

Notes:
- Checkpoint label retained exactly from the user request.
- `.git/`, `checkpoints/`, `__pycache__/`, and `.claude/` are intentionally
  excluded from the snapshot.
