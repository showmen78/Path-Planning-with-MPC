Checkpoint name: checkpoint15 (Everything works except for signal detection)
Created from current working tree.

Purpose:
- Restore the project to the exact source/configuration state that existed
  when the user said to save the current status as
  "checkpoint 15 (Everything works except for signal detection)".

Key changes in this checkpoint (relative to checkpoint14):
- Restored the rule-based behavior planner loop with planner-owned blue-dot
  lane control, cooperative reroute handling, and traffic-light stop latching.
- Fixed the route-optimal-lane lookup regression and reintroduced tracked
  signal context for the next relevant traffic light and stop target.
- Corrected intersection handoff so the blue dot follows the global route once
  the planner-owned lane already matches the route-optimal lane.

Restore method:
- Sync `checkpoints/checkpoint15 (Everything works except for signal detection)/worktree/`
  back into the repo root.
- Use delete mode so files added later are removed during restore.

Notes:
- `.git/`, `checkpoints/`, and `.claude/` are intentionally excluded from the
  snapshot.
