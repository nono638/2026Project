---
name: project-snapshot
description: On-demand project status report — task queue, branches, recent activity, environment, health
---

# Project Snapshot

Generate a status overview of the project. Either agent can call this on demand — typically
when the user asks "what's the project status?" or wants to orient after time away.

---

## Step 1 — Task queue summary

Read `DaytimeNighttimeHandOff/tracker.json` and count tasks by status:

| Status | Count | Detail |
|---|---|---|
| todo | N | |
| in_progress | N | (should be 0 outside an active nightrun — flag if not) |
| done | N | note how many have/haven't been `daytime_reviewed` |
| skipped | N | |
| blocked | N | list `blocked_reason` for each |
| cancelled | N | |

If there are **blocked** tasks, list each with its `blocked_reason` — these need user input.

If there are **done but unreviewed** tasks, flag them — the user should review before
queuing more work.

---

## Step 2 — Open branches

Run:
```
git branch --list "night/*"
git branch --list "day/*"
```

For each branch:
1. Count commits ahead of main: `git rev-list main..<branch> --count`
2. Check if the corresponding tracker entry is done/in_progress/etc.
3. Note whether the branch has been merged into main:
   `git branch --merged main | grep <branch>`

Flag:
- **Orphaned branches** — no matching tracker entry, or tracker shows done but branch
  was never merged
- **Stale branches** — branches with no corresponding active task

---

## Step 3 — Recent activity

Read the last 20 lines of `DaytimeNighttimeHandOff/nighttime.log`.

Summarize:
- When was the last nighttime session?
- How many tasks were completed in the last run?
- Any CRASH-RECOVERY or SELF-HEAL entries?
- Any flags logged?

---

## Step 4 — Environment status

| Check | How |
|---|---|
| Venv exists? | Look for `.venv/`, `venv/`, `env/` |
| Python version | `python --version` (inside venv if it exists) |
| requirements.txt | Exists? How many packages pinned? |
| ENVIRONMENT.md | Exists? |

Report any gaps (e.g., "venv exists but ENVIRONMENT.md is missing").

---

## Step 5 — Disk usage

Use Python to check directory sizes (works on all platforms):

```python
import os
def dir_size(path):
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    return total
```

Report sizes for:
- Project root total
- `.venv/` (if exists)
- `DaytimeNighttimeHandOff/`
- `_claude_sandbox_setup/`
- `.git/`

Flag anything unexpected (venv > 500MB, handoff > 100MB, git > 200MB).

---

## Step 6 — Handoff structure health

Quick check that the handoff structure is intact:
- `WrittenByDaytime/` — list any task directories present (specs awaiting implementation)
- `WrittenByNighttime/` — list any task directories present (completed work)
- `DaytimeOnly/` — confirm core files exist (project_overview.md, inbox.md, incubating.md)
- `tracker.json` — confirm it parses as valid JSON

---

## Step 7 — Present the report

Format as a structured summary:

```
# Project Snapshot — <date>

## Task Queue
| Status | Count |
|--------|-------|
| todo   | N     |
| done   | N (M unreviewed) |
| blocked| N     |
| ...    | ...   |

[Blocked task details if any]

## Open Branches
- night/task-001-foo (3 commits, done, unmerged)
- night/sweep-dry (1 commit, unmerged)
- ...
[or "No open branches."]

## Recent Activity
[Last session: <date>, N tasks completed]
[Notable log entries]

## Environment
Python X.Y | .venv/ | N packages | ENVIRONMENT.md: yes/no

## Disk Usage
Total: X MB | venv: X MB | handoff: X MB | git: X MB

## Health
[Any issues found, or "All checks passed."]
```

---

## Done

In daytime: present the report to the user.
In nighttime: write it to nighttime.log as a summary entry if requested.
Return to the calling context.
