
---
<!-- SANDBOX RULES — DO NOT EDIT BELOW THIS LINE -->
---

# Sandboxed Project — Directory-Restricted Mode

## SESSION START — CHECK SANDBOX HEALTH

On every session start, check if `_claude_sandbox_setup/SETUP.md` exists AND `.claude/hooks/directory_guard.py`
contains `HARDCODED_PROJECT_DIR = None`. If BOTH are true, the sandbox was never set up — read
`_claude_sandbox_setup/SETUP.md` and follow all steps before doing anything else.

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
   This lets the user confirm you have the right directory.

5. THEN proceed with whatever task the user asked for.

**If PROJECT_ROOT already has a real path**, skip this section — setup was already done.
Confirm the path matches your current working directory. If it doesn't, STOP and tell the
user there's a mismatch — the project may have been moved.

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
If you need configuration values, ask the user to provide them in the task prompt or
in a non-sensitive config file.

## SOFT CONSTRAINTS (you must follow these — they cover gaps the hooks can't catch perfectly)

1. **Stay in this directory.** Every file you read, write, edit, search, or reference must be
   within this project root. Use relative paths for everything. Do not use absolute paths
   unless they point inside this directory.

2. **Do not navigate out.** Do not `cd ..` above the project root. Do not reference parent
   directories. Do not use `~`, `$HOME`, `%USERPROFILE%`, or any environment variable that
   resolves outside this directory.

3. **Do not construct paths dynamically to escape the project.** Do not use string
   concatenation, environment variables, or any other technique to build a path that
   resolves outside this directory. This applies to all shells — bash, powershell, cmd.

4. **Do not modify the guard.** After the first-run setup is complete, do not edit, delete,
   move, or rename `.claude/settings.json`, `.claude/hooks/`, or any file within `.claude/`.
   These are security boundaries. The ONLY exception is the one-time first-run setup above.

5. **Respect existing files.** This directory may already contain project files. Do not
   delete, overwrite, or reorganize existing files unless the user explicitly asks you to.
   Read and understand existing structure before making changes.

6. **If blocked, stop and explain.** If the directory guard blocks a tool call, do not retry
   or try to work around it. Stop and tell the user what you were trying to do and ask how
   they'd like to proceed.

---

## UNATTENDED OPERATION

This project may be running without a human watching. Follow these rules:

- **Never ask questions — decide and log.** If you are unsure about something, do not ask.
  Pick the simplest reasonable option, proceed, and log your decision and reasoning in
  `CLAUDE_LOG.md`. The `no_ask_human.py` hook will block questions anyway, but follow this
  rule even if the hook is not present.

- **Git checkpoint before large changes.** Before making large-scale changes (refactoring
  multiple files, deleting code, changing architecture), commit the current state first
  with a message like `checkpoint: before <description>`. This gives the user a safe
  rollback point.

- **3 attempts, then move on.** If something fails, try up to 3 meaningfully different
  approaches. If all 3 fail, log the problem to `CLAUDE_LOG.md` with what you tried and
  why each attempt failed, then move to the next part of the task. Do not retry the same
  approach more than once. Do not loop, sleep, or poll.

- **No network access.** Do not attempt to download files, fetch URLs, search the web,
  or make any outbound network requests. All dependencies and resources must already be
  available locally or installable via package managers (pip, npm, etc.).

- **Preserve state across context compaction.** Update `CLAUDE_LOG.md` before any
  long operation so task state is written to disk regardless of what gets compacted.
  The most critical things to preserve: current task status, files modified so far,
  autonomous decisions made, and any errors encountered.

---

## ENVIRONMENT AND PACKAGES

- **Always use a virtual environment.** If a `.venv` or `venv` folder exists, activate it.
  If one doesn't exist and you need Python packages, create one with `python -m venv .venv`
  and activate it before installing anything.
- **Never install packages globally.** No `pip install` without an active venv. No
  `npm install -g`. All dependencies go into the project — `pip install` into the venv,
  `npm install` into the local `node_modules`, etc.
- **Respect existing dependency files.** If `requirements.txt`, `pyproject.toml`,
  `package.json`, `Cargo.toml`, or similar files exist, use them. Install from them first
  before adding new packages. If you add a new dependency, add it to the appropriate file.

---

## ERROR HANDLING

- **Try multiple approaches before giving up.** If something fails, don't retry the exact
  same thing. Try a different approach — different syntax, different library, different
  algorithm, different tool. Make at least 3 meaningfully different attempts.
- **If you're stuck, move on.** If you've tried several different approaches and can't get
  past a problem, skip it and move to the next part of the task. Leave a clear comment in
  the code (e.g., `# TODO: <what failed and why>`) so the user knows what still needs work.
- **Don't loop forever.** Never retry the same failing command more than twice. Never write
  a retry loop or sleep-and-poll pattern to wait out a transient error. If it failed twice
  the same way, it's not transient — try something different or move on.
- **Log what went wrong.** If you skip something due to errors, write a brief note in a file
  called `CLAUDE_LOG.md` at the project root. Include what you tried, what failed, and what
  you skipped. The user will read this when they come back.
- **If you need test data, keep it small.** Generate small representative samples under
  100KB. Do not create multi-megabyte files or download datasets.

---

## SCOPE AND AUTONOMY

- **Do what was asked, then stop.** Complete the user's task. Do not look for additional
  work, refactor unrelated code, add features that weren't requested, or "improve" things
  beyond the scope of the task.
- **Don't reorganize the project.** Don't rename files, restructure directories, or change
  coding conventions unless the user specifically asked for it.
- **When in doubt, pick the simplest option.** If the user's instructions are ambiguous and
  you can't ask for clarification (because they may not be watching), choose the most
  straightforward interpretation. Do the simple thing. Don't build the complex thing.
  Log your interpretation in `CLAUDE_LOG.md` so the user can course-correct.
- **Don't add unnecessary extras.** No extra docstrings for code you didn't write. No type
  annotations you weren't asked for. No README files. No config files. No test files.
  Unless the user asked for them.

---

## GIT BEHAVIOR

- **Commit your work when you finish a logical unit.** Don't leave everything uncommitted.
  Commit after completing each distinct piece of the task with a clear commit message
  describing what you did.
- **Checkpoint before risky changes.** Before refactoring, deleting, or rewriting significant
  code, commit the current state with a message like `checkpoint: before <description>`.
- **Don't push.** Commit locally only. The user will review and push when ready.
- **Don't create branches** unless the user asked for one. Work on whatever branch is
  currently checked out.
- **Don't amend, rebase, or rewrite history.** Create new commits only.

---

## RESOURCE SAFETY

- **Don't leave processes running.** If you start a dev server, test runner, or any
  long-running process to test something, kill it when you're done. Don't leave servers,
  watchers, or background jobs running after your task is complete.
- **Don't write infinite loops.** If you need to test something repeatedly, use a bounded
  loop with a clear exit condition.
- **Don't create large files.** If you need test data, generate small representative
  samples (under 100KB). Do not create multi-megabyte files or download datasets.

---

## PROGRESS REPORTING

- **Verify before declaring done.** Before writing your completion summary, run the
  project's tests, linter, or type-checker to confirm the work actually succeeds. If
  tests don't exist, do a quick sanity check — import the module, run the script, or
  check for obvious errors. If verification fails, fix it first. If you can't fix it,
  document the failure clearly rather than silently marking the task complete.

- **Write a summary when you finish.** When your task is complete, create or update a file
  called `CLAUDE_LOG.md` at the project root with a brief summary of what you did. Include:
  - What was asked
  - What you did
  - What worked
  - What didn't work or was skipped (if anything)
  - Any decisions you made autonomously (and why)
  - What the user should review or test
- This is how the user catches up when they come back. Keep it concise — bullet points, not
  paragraphs.

---

## How to work in this project

- Use **relative paths** for all file operations (e.g., `src/main.py` not `C:\Users\...\src\main.py`)
- Run commands from the project root
- If you need a tool or command that is blocked, log it in `CLAUDE_LOG.md` and move on —
  do not work around it
- You have full access to powershell, cmd, bash, python, git, and common dev tools within
  this directory
- Work freely and autonomously on whatever the user needs — the restrictions are guardrails,
  not obstacles
