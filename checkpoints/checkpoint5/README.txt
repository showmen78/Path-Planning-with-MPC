Checkpoint name: checkpoint5
Created from current working tree.

Purpose:
- Restore the project to the exact source/configuration state that existed
  when the user said to save the current status as "checkpoint5".

Restore method:
- Sync `checkpoints/checkpoint5/worktree/` back into the repo root.
- Use delete mode so files added later are removed during restore.

Notes:
- `.git/` and `checkpoints/` are intentionally excluded from the snapshot.
