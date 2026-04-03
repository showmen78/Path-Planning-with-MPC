Checkpoint name: checkpoint13 (reroute works)
Created from current working tree.

Purpose:
- Restore the project to the exact source/configuration state that existed
  when the user said to save the current status as "checkpoint13 (reroute works)".

Key changes in this checkpoint (relative to checkpoint12):
- Tightened the cooperative reroute flow so lane-closure messages resolve the
  blocked lane from the snapped workzone position and use the CARLA global
  planner's blocked-lane reroute path as the primary route update.
- Updated the `high_level_route_planning` scenario to use the environment cube
  named `workzone` as the lane-closure source and removed the virtual workzone
  obstacle injection so reroute is no longer coupled to an MPC potential field.
- Added/updated regression coverage for workzone lookup, raw road/lane ID
  resolution, and reroute behavior.

Restore method:
- Sync `checkpoints/checkpoint13 (reroute works)/worktree/` back into the repo root.
- Use delete mode so files added later are removed during restore.

Notes:
- `.git/` and `checkpoints/` are intentionally excluded from the snapshot.
