Checkpoint name: checkpoint12
Created from current working tree.

Purpose:
- Restore the project to the exact source/configuration state that existed
  when the user said to save the current status as "checkpoint12".

Key changes in this checkpoint (relative to checkpoint11):
- Extended the rule-based planning stack with route-first lane selection,
  updated lane-safety gating, and scenario-specific runtime hooks in the
  shared CARLA runner.
- Added the `roadway_hazard` and `high_level_route_planning` CARLA scenarios
  plus their scenario-local Python modules and YAML configurations.
- Added presentation/demo assets, including the rule-based behavior planner
  LaTeX slide deck and pygame visualization scripts for geometric shape
  comparisons.

Restore method:
- Sync `checkpoints/checkpoint12/worktree/` back into the repo root.
- Use delete mode so files added later are removed during restore.

Notes:
- `.git/`, `checkpoints/`, `__pycache__/`, `.claude/`, and `.codex/` are
  intentionally excluded from the snapshot.
