
---
<!-- NIGHTTIME SANDBOX RULES — DO NOT EDIT BELOW THIS LINE -->
---

# ClaudeDayNight — Nighttime Mode

## SESSION START — STRUCTURE HEALTH CHECK

On every session start, run the health check skill before doing anything else:
read `.claude/skills/health-check/SKILL.md` and follow all instructions.

If the skill file itself is missing, copy it from
`_claude_sandbox_setup/templates/skills/health-check/SKILL.md` first. If that's
also missing, STOP and tell the user the setup folder is damaged.

Then proceed to the FIRST RUN SETUP section below.

## FIRST RUN SETUP — DO THIS BEFORE ANYTHING ELSE

**PROJECT_ROOT: NOT YET CONFIGURED**

Claude, read this carefully. Before doing ANY other work, check the PROJECT_ROOT line above.

**If it says "NOT YET CONFIGURED"**, do the following steps in order:

1. Run `pwd` in Bash to get the absolute path of the current working directory.

2. Replace the PROJECT_ROOT line above (the one that says "NOT YET CONFIGURED") with the
   actual absolute path. For example:
   `**PROJECT_ROOT: C:\Users\noahc\Projects\my-project**`

3. Open `.claude/hooks/directory_guard.py` and find the line:
   `HARDCODED_PROJECT_DIR = None`
   Replace it with the actual path as a raw string. For example:
   `HARDCODED_PROJECT_DIR = r"C:\Users\noahc\Projects\my-project"`

4. Tell the user:
   ```
   Sandbox setup complete.
   Project directory locked to: <the absolute path>
   The directory guard hook and CLAUDE.md have both been updated.
   All file access and commands are now restricted to this directory.
   ```

5. THEN proceed with the nighttime task loop below.

**If PROJECT_ROOT already has a real path**, skip this section — setup was already done.
Confirm the path matches your current working directory. If it doesn't, STOP and tell the
user there's a mismatch — the project may have been moved.

---

## NIGHTTIME TASK LOOP

You are running in **nighttime/unattended mode**. The user is not watching. Your job is to
work through all pending tasks in `DaytimeNighttimeHandOff/tracker.json` until none remain.

**On session start, read these for context:**
- `DaytimeNighttimeHandOff/DaytimeOnly/project_overview.md` — what the project is and where
  it's headed. This helps you understand the bigger picture behind the specs you're implementing.
- `DaytimeNighttimeHandOff/DaytimeOnly/reference/architecture-decisions.md` (if it exists) —
  decisions already made and why. Follow these; do not relitigate them.

**DO NOT read** `inbox.md`, `incubating.md`, or `archive/` in DaytimeOnly/. Those contain
unprocessed ideas and project management state that would distract from implementation.
Your primary inputs are `tracker.json` and the spec files in `WrittenByDaytime/`.

### On Session Start

**Step 0 — Crash recovery check**

Run the crash-recovery skill: read `.claude/skills/crash-recovery/SKILL.md` and follow
all instructions. This handles interrupted sessions, dirty working trees, orphaned branches,
and in_progress tasks from previous runs.

If the skill file is missing, copy it from
`_claude_sandbox_setup/templates/skills/crash-recovery/SKILL.md` first.

Then proceed to Step 1 below.

1. Read `DaytimeNighttimeHandOff/tracker.json`
2. Skip any tasks with `"status": "blocked"` — these need human input. You cannot proceed on them.
   Check if the daytime session has updated `blocked_reason` or `daytime_comments` with new
   information. If the blocker has been resolved, the human will have changed the status back to
   `"todo"` — otherwise, leave it.
4. Check `depends_on` for each `"todo"` task — if any listed dependency is not `"done"`, skip
   the task this session (it's not ready to start). Do not mark it `"blocked"` — just leave it `"todo"`.
5. Pick up remaining `"status": "todo"` tasks in order (lowest task_id first).
6. If all tasks are `done`, `skipped`, or `blocked`, proceed to the
   **END-OF-NIGHT SWEEPS** section below before ending the session.

### Per-Task Workflow

For each task, follow these steps exactly:

**Step 1 — Read the spec**
Read `DaytimeNighttimeHandOff/WrittenByDaytime/<task-dir>/spec.md` and all test files in
`DaytimeNighttimeHandOff/WrittenByDaytime/<task-dir>/tests/`.

**Step 2 — Write an implementation plan**
Before touching any code, write your plan to
`DaytimeNighttimeHandOff/WrittenByNighttime/<task-dir>/plan.md`.
Include: which files you'll modify, your approach, and any ambiguities you noticed in the spec.
Create the `WrittenByNighttime/<task-dir>/` directory if it doesn't exist.

**Step 3 — Update tracker to in_progress (on main, before branching)**
While still on main, update `tracker.json`: set `"status": "in_progress"` and
`"nighttime_started": "<ISO timestamp>"`. Commit it immediately:
```
git add DaytimeNighttimeHandOff/tracker.json
git commit -m "tracker: <task-id> in_progress"
```
This keeps tracker.json linear on main and avoids merge conflicts across branches.

**Step 4 — Create a git branch**
```
git checkout -b night/<task-id>-<short-name>
```
If the branch already exists (e.g., from a previous crashed session), check it out instead:
```
git checkout night/<task-id>-<short-name>
```
Then assess its state the same way as Step 0 — check for existing commits and dirty working tree
before deciding whether to resume or start fresh.

**Step 5 — Implement**
Implement exactly what the spec says. If you notice gaps or ambiguities, log them in plan.md
and make the simplest reasonable choice — do NOT deviate from the spec's intent.

**Step 6 — Run tests**
Run the pre-written test files from `WrittenByDaytime/<task-dir>/tests/`. Note: pass/fail.
Also run any existing project tests to check for regressions.

**Step 7 — Write result.md**
Write `DaytimeNighttimeHandOff/WrittenByNighttime/<task-dir>/result.md` using this format:

```markdown
# Result: <task-id> — <short description>
**Status:** done | skipped | blocked
**Completed:** <ISO timestamp>

## Commits
- `<sha>` — <commit message>

## Test Results
- Command run: `<exact command>`
- Outcome: X passed, Y failed
- Failures: [list any failing tests with one-line reason, or "none"]

## Decisions Made
[Any choice you made that wasn't explicit in the spec — explain why you chose it]

## Flags for Morning Review
[Things that need human eyes — surprises, concerns, quality issues. Be specific.]
[If none: "None."]

## Attempted Approaches (if skipped/blocked)
[What you tried, why each failed — so morning doesn't retry dead ends]
```

**Step 8 — Commit on branch, then update tracker on main**

Stage only the files you created or modified for this task. **Never use `git add -A` or
`git add .`** — these will pick up runtime files that must not be committed.

```
git add src/<files you changed>
git add tests/<test files>
git add DaytimeNighttimeHandOff/WrittenByNighttime/<task-dir>/
git add requirements.txt  # only if you installed new packages
git commit -m "night: <task-id> <short description>"
git rev-parse HEAD   # capture the SHA
git checkout main
```

**Never stage or commit** files under `.claude/` — these are runtime configuration managed
by the launcher scripts. Also never commit `pip_output.txt`, `*.bak`, `audit.jsonl`, or
`nighttime.log`.

Now update `tracker.json` **on main** and commit it:
```
# edit tracker.json with the fields below
git add DaytimeNighttimeHandOff/tracker.json
git commit -m "tracker: <task-id> done"
```

Tracker fields to update:
- `"status": "done"` (or `"skipped"` or `"blocked"` — see below)
- `"nighttime_completed": "<ISO timestamp>"`
- `"nighttime_comments": "<one-sentence summary>"`
- `"branch": "night/<task-id>-<short-name>"`
- `"commit_sha": "<full SHA from git rev-parse HEAD>"`
- `"tests_passed": true/false`
- `"attempted_approaches": ["<approach 1 if anything failed>"]` (leave `[]` if nothing failed)
- `"flags": ["<any flags for morning review>"]`
- `"blocked_reason": "<what the human needs to provide>"` (only if status is `"blocked"`)

**When to use `blocked` instead of `skipped`:** If you cannot proceed because you need
a specific piece of human input (a missing credential, an ambiguous requirement you can't
resolve, a decision about approach), set `status: "blocked"` and write a clear
`blocked_reason`. The human can read this in the morning and unblock you. Use `skipped`
only when there's no clear path forward even with human help — true dead ends.

**Step 9 — Copy task files from WrittenByDaytime to WrittenByNighttime**
The `WrittenByNighttime/<task-dir>/` directory already exists (you created it in Step 2 for
plan.md and result.md). **Copy** (do not move) the spec and tests into it:
```
cp DaytimeNighttimeHandOff/WrittenByDaytime/<task-dir>/spec.md DaytimeNighttimeHandOff/WrittenByNighttime/<task-dir>/spec.md
cp -r DaytimeNighttimeHandOff/WrittenByDaytime/<task-dir>/tests DaytimeNighttimeHandOff/WrittenByNighttime/<task-dir>/tests
```
Do NOT delete the WrittenByDaytime directory — leave the originals in place. Deleting them
on the branch creates merge conflicts when multiple branches are merged. The daytime agent
will clean up WrittenByDaytime/ during morning review after merging.

**Step 10 — Log to nighttime.log**
Append a one-line summary to `DaytimeNighttimeHandOff/nighttime.log`:
```
[<timestamp>] <task-id>: <done/skipped> — <one sentence summary> [tests: pass/fail]
```

**Step 11 — Pick up next task**
Return to Step 1 of the task loop with the next pending task.

---

## END-OF-NIGHT SWEEPS

After all queued tasks are `done`, `skipped`, or `blocked`, run the end-of-night sweeps
skill: read `.claude/skills/end-of-night-sweeps/SKILL.md` and follow all instructions.

The sweeps are loaded as a skill (not inlined here) to keep this supplement lean during
the task implementation phase. The skill contains 6 sweeps: test fixes, bug sweep, DRY
refactoring, type hints/docstrings, dead code cleanup, and security scan.

If the context monitor hook has already fired a WARNING or CRITICAL alert, skip sweeps
entirely — preserve remaining context for committing and updating tracker.json.

---

## HARD CONSTRAINTS (enforced by .claude/settings.json and hooks — you cannot override these)

The `.claude/settings.json` and `.claude/hooks/directory_guard.py` in this project enforce
OS-level restrictions that you cannot bypass. They will block any attempt to read, write,
search, or run commands referencing paths outside this project directory. Do not attempt to
work around them.

The `no_ask_human.py` hook will block any attempt to ask the user a question. If it fires,
you must decide on your own and keep working. Do not try to ask again.

Network access is blocked. `curl`, `wget`, `WebFetch`, `WebSearch`, and PowerShell web
cmdlets are all denied. Do not attempt to download files or fetch URLs.

Secrets files (`.env`, `.key`, `.pem`, `credentials`, etc.) are blocked from reading.
If you need configuration values, check spec.md — they should have been specified there.

## SOFT CONSTRAINTS (you must follow these — they cover gaps the hooks can't catch perfectly)

1. **Stay in this directory.** Every file you read, write, edit, search, or reference must be
   within this project root. Use relative paths for everything. Do not use absolute paths
   unless they point inside this directory.

2. **Do not navigate out.** Do not `cd ..` above the project root. Do not reference parent
   directories. Do not use `~`, `$HOME`, `%USERPROFILE%`, or any environment variable that
   resolves outside this directory.

3. **Do not construct paths dynamically to escape the project.** Do not use string
   concatenation, environment variables, or any other technique to build a path that
   resolves outside this directory.

4. **Do not modify the guard.** After first-run setup is complete, do not edit, delete,
   move, or rename `.claude/settings.json`, `.claude/hooks/`, or any file within `.claude/`.
   These are security boundaries.

5. **Respect existing files.** Do not delete, overwrite, or reorganize existing files unless
   the spec explicitly requires it.

6. **If blocked, stop and explain.** If the directory guard blocks a tool call, do not retry
   or try to work around it. Log what happened in nighttime.log and move on.

---

## UNATTENDED OPERATION

This project is running without a human watching. Follow these rules:

- **Never ask questions — decide and log.** If you are unsure about something, make the
  simplest reasonable choice and record your decision in result.md. The `no_ask_human.py`
  hook will block questions anyway.

- **Git checkpoint before large changes.** Before making large-scale changes, commit the
  current state with a message like `checkpoint: before <description>`.

- **3 attempts, then move on.** If something fails, try up to 3 meaningfully different
  approaches. If all 3 fail, set the task status to `"skipped"` in tracker.json, document
  what you tried in result.md, and move to the next task.

- **No network access.** Do not attempt to download files, fetch URLs, search the web,
  or make any outbound network requests.

- **Manage context proactively.** Update tracker.json and nighttime.log before any long
  operation so task state is written to disk. Between tasks, run `/compact` to free context
  space — this preserves a summary of what you've done so far (unlike `/clear` which erases
  everything). Multiple compactions are fine; you'll retain a thread of the session history.

---

## ENVIRONMENT AND PACKAGES

- **Always use a virtual environment.** If a `.venv` or `venv` folder exists, activate it.
  If one doesn't exist and you need Python packages, create one with `python -m venv .venv`
  and activate it before installing anything.
- **Never install packages globally.** No `pip install` without an active venv. No
  `npm install -g`. All dependencies go into the project directory.
- **Respect existing dependency files.** If `requirements.txt`, `pyproject.toml`,
  `package.json`, etc. exist, install from them before adding new packages.
- **Keep requirements.txt up to date with pinned versions.** After installing any new package,
  run `pip freeze > requirements.txt` (or update the existing file) so that every dependency
  has an exact pinned version (e.g., `requests==2.31.0`, not `requests`). If `requirements.txt`
  already exists, merge new packages into it — don't overwrite existing pins unless upgrading.
- **Maintain an environment README.** If a file named `ENVIRONMENT.md` (or an environment
  section in an existing `README.md`) exists, update it when the environment changes. If
  neither exists, create `ENVIRONMENT.md` in the project root with:
  - Virtual environment name and location (e.g., `.venv/`)
  - Python version (output of `python --version`)
  - How to activate (e.g., `source .venv/bin/activate` or `.venv\Scripts\activate`)
  - How to install dependencies (e.g., `pip install -r requirements.txt`)
  - Any other runtime requirements (Node version, system packages, etc.)
- **For full environment setup or repair**, run the environment-bootstrap skill: read
  `.claude/skills/environment-bootstrap/SKILL.md` and follow all instructions.

---

## ERROR HANDLING

- **Try multiple approaches before giving up.** If something fails, try a different approach
  — different syntax, different library, different algorithm. Make at least 3 meaningfully
  different attempts.
- **If stuck, mark skipped and move on.** If you've tried 3 approaches and can't proceed,
  set `"status": "skipped"` in tracker.json, write what failed in result.md, and continue to
  the next task.
- **Don't loop forever.** Never retry the same failing command more than twice.
- **If you need test data, keep it small.** Under 100KB. Do not create multi-megabyte files.

---

## GIT BEHAVIOR (NIGHTTIME OVERRIDE)

Nighttime mode uses **branch-per-task** workflow. This overrides the "don't create branches"
default.

- **Create a branch for every task**: `git checkout -b night/<task-id>-<short-name>`
- **Commit after implementing each task**: clear commit messages, reference task-id.
- **Return to main after each task**: `git checkout main`
- **Don't push.** Commit locally only. The user reviews branches in the morning.
- **Don't amend, rebase, or rewrite history.** Create new commits only.
- **Checkpoint before risky changes.** Before large rewrites, commit current state.

---

## CODE QUALITY STANDARDS

All code you write must meet these standards:

- **Type hints on every function** — parameters and return types. Use `from __future__ import
  annotations` where needed for forward references.
- **Docstrings on every function** — at minimum a one-line summary. For non-trivial functions,
  include Args, Returns, and Raises sections.
- **Comment the *why*, not the *what*.** Every non-obvious decision should have an inline
  comment explaining the rationale. Someone reading the code for the first time should
  understand your reasoning without needing the spec.
- **Carry rationale from the spec into comments.** The spec's "Rationale" or "Why" sections
  exist specifically so you can embed that reasoning at the point of use in the code. If the
  spec says "use httpx because it supports async natively," put that in a comment where httpx
  is imported or instantiated.
- **Include URLs from the spec in comments when present.** If the spec cites a URL
  (documentation, benchmark, security advisory), include it in a comment near the relevant
  code. Not every spec will have URLs — only include them when the spec provides them:
  ```python
  # Using exponential backoff with jitter per AWS architecture recommendations
  # See: https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/
  delay = base_delay * (2 ** attempt) + random.uniform(0, jitter)
  ```
- **These standards apply to all code you write**, including test helpers and utility functions.
  Do not skip them for "simple" functions.

---

## SCOPE AND AUTONOMY

- **Implement what the spec says, nothing more.** Do not add features, refactor unrelated
  code, or "improve" things beyond the spec's scope.
- **When spec is ambiguous, pick the simplest option.** Log your interpretation in result.md
  so the user can course-correct in the next daytime session.

---

## RESOURCE SAFETY

- **Don't leave processes running.** Kill any servers or background jobs when done testing.
- **Don't write infinite loops.** Use bounded loops with clear exit conditions.
- **Don't create large files.** Test data under 100KB.

---

## How to work in this project

- Use **relative paths** for all file operations
- Run commands from the project root
- If you need a tool or command that is blocked, log it in nighttime.log and move on —
  do not work around it
- You have full access to powershell, cmd, bash, python, git, and common dev tools within
  this directory
