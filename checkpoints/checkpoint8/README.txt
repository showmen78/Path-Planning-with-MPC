Checkpoint name: checkpoint8
Created from current working tree.

Purpose:
- Restore the project to the exact source/configuration state that existed
  when the user said to save the current status as "checkpoint8".

Key changes in this checkpoint (relative to checkpoint7):
- Fixed route mismatch in behavior planner prompt: prompt_builder now uses
  get_current_route_info() instead of plan_route() so the route in the prompt
  matches the global planner's stored route.
- Fixed temporary destination placement at intersections: switched from
  lane-center-waypoint-based branch derivation to route-polyline-based sampling.
  - base_destination_state uses compute_temporary_destination_state_from_route()
  - Behavior planner override temp destination also uses route-based version
  - MPC reference trajectory uses build_route_reference_samples()
- Added build_route_reference_samples() to MPC/local_goal.py and exported it
  from MPC/__init__.py.
- All three changes fall back to the old approach when route_waypoints < 2.

Restore method:
- Sync `checkpoints/checkpoint8/worktree/` back into the repo root.
- Use delete mode so files added later are removed during restore.

Notes:
- `.git/`, `checkpoints/`, `__pycache__/`, and `.claude/` are intentionally
  excluded from the snapshot.
