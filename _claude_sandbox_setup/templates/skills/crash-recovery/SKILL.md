---
name: crash-recovery
description: Nighttime session start — assess git state, recover interrupted tasks, restore tracker consistency
---

# Crash Recovery

Run this at the start of every nighttime session, after the health check. Assess git state
and tracker.json to recover from interrupted sessions.

This skill is for **nighttime use only**. Daytime sessions do not need crash recovery.

---

## Step 1 — Assess git state

Run these commands and capture output:

```
git status --porcelain
git branch --show-current
git stash list
```

Determine the state:

| Branch | Working tree | State |
|---|---|---|
| `main` | clean | **NORMAL** — no recovery needed |
| `main` | dirty | **UNEXPECTED DIRTY** |
| `night/task-*` | clean | **INTERRUPTED TASK (clean)** |
| `night/task-*` | dirty | **CRASHED MID-WRITE** |
| `night/sweep-*` | clean | **INTERRUPTED SWEEP (clean)** |
| `night/sweep-*` | dirty | **CRASHED MID-SWEEP** |
| other | any | **UNKNOWN BRANCH** |

---

## Step 2 — Handle NORMAL

No recovery needed. Skip to Step 9.

---

## Step 3 — Handle UNEXPECTED DIRTY (main + dirty working tree)

Something was left uncommitted on main. This shouldn't happen in normal operation.

1. Run `git diff` to see what's uncommitted.
2. Stash the changes:
   ```
   git stash push -m "crash-recovery: dirty main at <ISO timestamp>"
   ```
3. Log: `[<timestamp>] CRASH-RECOVERY: stashed dirty changes on main`
4. Continue to Step 9.

---

## Step 4 — Handle INTERRUPTED TASK (night/task-* + clean)

The session ended on a task branch with no uncommitted changes.

1. Extract the task-id from the branch name (e.g., `night/task-003-add-auth` → `task-003`).
2. Check for commits: `git log main..HEAD --oneline`

**If commits exist:**
- Check if `WrittenByNighttime/<task-dir>/result.md` exists.
- **result.md exists:** Implementation is done — bookkeeping was interrupted. Complete the
  bookkeeping now:
  - Run `git rev-parse HEAD` to get the SHA
  - Update tracker.json: `status: "done"`, `branch`, `commit_sha`, `nighttime_completed`
  - Move files from `WrittenByDaytime/<task-dir>/` to `WrittenByNighttime/<task-dir>/`
  - Log to nighttime.log
  - `git checkout main`
- **result.md missing:** Implementation was in progress when the session ended.
  Read `WrittenByNighttime/<task-dir>/plan.md` to understand where you were.
  Resume implementation from the current state — do NOT start over.

**If no commits:**
- The branch was created but work never started.
- Check if `WrittenByNighttime/<task-dir>/plan.md` exists.
- If plan exists: resume from the plan.
- If no plan: start the task fresh (stay on the branch).

---

## Step 5 — Handle CRASHED MID-WRITE (night/task-* + dirty)

The session crashed while writing files on a task branch.

1. Extract the task-id from the branch name.
2. Run `git diff` and `git diff --cached` to see uncommitted/staged changes.
3. **Do NOT discard these changes.** They are in-progress work.
4. Commit them:
   ```
   git add -A
   git commit -m "crash-recovery: preserve interrupted work on <task-id>"
   ```
5. Now handle as **INTERRUPTED TASK with commits** (Step 4, "if commits exist").

---

## Step 6 — Handle INTERRUPTED SWEEP (night/sweep-* + clean)

Sweeps are independent — no need to resume a partial sweep.

1. Check for commits: `git log main..HEAD --oneline`
2. **If commits exist:** Sweep work was done. `git checkout main`.
3. **If no commits:** Empty sweep branch — delete it and checkout main:
   ```
   git checkout main
   git branch -d <branch>
   ```

---

## Step 7 — Handle CRASHED MID-SWEEP (night/sweep-* + dirty)

1. Commit the dirty changes to preserve the work:
   ```
   git add -A
   git commit -m "crash-recovery: preserve sweep work on <branch>"
   ```
2. `git checkout main`
3. Log: `[<timestamp>] CRASH-RECOVERY: preserved interrupted sweep on <branch>`

---

## Step 8 — Handle UNKNOWN BRANCH

An unexpected branch that doesn't match `night/*` naming.

1. Log: `[<timestamp>] CRASH-RECOVERY: found unexpected branch <name> — switching to main`
2. `git checkout main`
3. Do **not** delete the branch — it may be intentional work.

---

## Step 9 — Check tracker for in_progress tasks

Read `DaytimeNighttimeHandOff/tracker.json`. Find any tasks with `"status": "in_progress"`.

For each:
1. Check if the corresponding branch exists: `git branch --list "night/<task-id>*"`
2. **Branch exists:** The git-state handlers above already dealt with it — no further action.
3. **No branch:** The task was marked in_progress but the branch was never created (or was
   already cleaned up). Reset `"status": "todo"` so it gets picked up normally.

Log any corrections.

---

## Step 10 — Check for orphaned stashes

If `git stash list` shows stashes (especially from previous crash-recovery runs):

Log: `[<timestamp>] CRASH-RECOVERY: N orphaned stash(es) found — morning should review with git stash list`

Do not pop or drop stashes — let the user review them.

---

## Step 11 — Summary log

If any recovery action was taken (anything other than NORMAL state), append one line to
`DaytimeNighttimeHandOff/nighttime.log`:

```
[<timestamp>] CRASH-RECOVERY: <state found> — <action taken>
```

If state was NORMAL, do not log anything.

---

## Done

Return to the nighttime supplement task loop. Git state and tracker.json are now consistent.
