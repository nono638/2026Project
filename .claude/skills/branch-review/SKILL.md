---
name: branch-review
description: Structured review of night branches — diffs, test results, flags, and merge workflow
---

# Branch Review

Review night branches against main, summarize changes, flag issues, and offer merge options.
This skill is for **daytime use only** — it is interactive and requires the user.

---

## Step 1 — Identify branches to review

Run `git branch --list "night/*"` to find all night branches.

Separate into:
- **Task branches** (`night/task-*`)
- **Sweep branches** (`night/sweep-*`)

For each task branch, check `tracker.json` — if the task has `"daytime_reviewed"` set and
the branch has already been merged into main, skip it. Only review branches with unreviewed
or unmerged work.

If no branches need review, report "No night branches to review." and return.

---

## Step 2 — Review each task branch

For each unreviewed task branch:

1. **Get the commit list:**
   ```
   git log main..<branch> --oneline
   ```

2. **Get the change summary:**
   ```
   git diff main...<branch> --stat
   ```

3. **Read the night instance's report:**
   Read `DaytimeNighttimeHandOff/WrittenByNighttime/<task-dir>/result.md`

4. **Check tracker.json** for the task entry — note `flags[]`, `tests_passed`,
   `attempted_approaches`, and `nighttime_comments`.

5. **Present to the user:**
   ```
   ### task-NNN: <description>
   Branch: night/task-NNN-name | N commits | M files changed

   **Changes:** [2-3 sentence summary of what was done, grouped by concern]

   **Tests:** passed/failed [detail if failed]

   **Decisions made:** [from result.md — anything non-obvious]

   **Flags:** [from result.md and tracker.json, or "None"]
   ```

6. **Flag anything suspicious:**
   - Files changed that weren't mentioned in the original spec
   - Large diffs (>500 lines changed) — possible scope creep
   - Failed tests
   - New TODO/FIXME/HACK comments added
   - New dependencies imported that weren't in the spec

---

## Step 3 — Review each sweep branch

For each sweep branch:

1. Run `git log main..<branch> --oneline` for commit list.
2. Run `git diff main...<branch> --stat` for change summary.
3. Summarize briefly — sweeps are routine, don't need deep review:
   ```
   ### sweep: <name>
   N commits | M files changed
   [one-line summary of what the sweep did]
   ```

If the sweep branch has no commits (empty sweep), note it and move on.

---

## Step 4 — Merge workflow

After presenting all branches, offer the user options for each:

> **A** *(recommended)* — Merge all clean branches to main now.
> **B** — Let me pick which ones to merge.
> **C** — Show me the full diff for [branch] first.
> **D** — Leave all for later.

When merging:
```
git checkout main
git merge <branch> --no-ff -m "merge: <task-id> <description>"
```

After a successful merge, offer to delete the branch:
```
git branch -d <branch>
```

Do **not** force-delete (`-D`). If `-d` fails, the branch has unmerged commits — warn the
user and leave it.

---

## Step 5 — Update tracker

For each task whose branch was merged, ensure `tracker.json` has `"daytime_reviewed"` set
to the actual wall-clock time. Get the real time by running:
`python -c "from datetime import datetime; print(datetime.now().strftime('%Y-%m-%dT%H:%M:%S'))"`
Do NOT estimate or fabricate timestamps. Skip if already set during session opening.

---

## Done

Return to the calling supplement. Summarize what was merged and what's still pending.
