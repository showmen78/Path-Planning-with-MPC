Checkpoint name: checkpoint1
Created from current working tree.

Purpose:
- Restore the project to the exact source/configuration state that existed
  when the user said to remember the current project setting as "checkpoint1".

Restore method:
- Sync `checkpoints/checkpoint1/worktree/` back into the repo root.
- Use delete mode so files added later are removed during restore.

Notes:
- `.git/` and `checkpoints/` are intentionally excluded from the snapshot.
