# Day/Night System Issues

Tracking issues discovered while using ClaudeDayNight in this project.
When fixes are applied here first, they need to be transposed back to the
main `_claude_sandbox_setup` repository so other projects get them too.

---

## Fixed (2026-03-17)

These 10 issues were found during the first night run and fixed in both
this project's `_claude_sandbox_setup/` and documented here.

### 1. `git add -A` commits .claude/ runtime files
**Root cause:** Nighttime supplement Step 8 said `git add -A`, which staged
`.claude/active_mode.md`, `.claude/settings.json`, `.claude/hooks/*.py`,
`pip_output.txt`, and `audit.jsonl` — all runtime artifacts the launcher
scripts had already modified before Claude launched.
**Fix:** Changed to explicit `git add <specific files>` with a prohibition
on staging anything under `.claude/`.

### 2. tracker.json race condition across branches
**Root cause:** Each branch got a stale snapshot of tracker.json at branch
creation time, then committed its own updates. Merging N branches = N
tracker conflicts.
**Fix:** tracker.json is now updated only on main — committed before
branching (in_progress) and after checkout main (done).

### 3. audit.jsonl grows unboundedly and gets committed
**Root cause:** Audit hook fires on every tool call, no rotation, not
gitignored.
**Fix:** Added to `.gitignore`. Consider adding rotation in future.

### 4. Mode switching dirties git working tree
**Root cause:** dayrun/nightrun overwrite `.claude/` files that were
tracked by git.
**Fix:** `.claude/` runtime files added to `.gitignore` and untracked
with `git rm --cached`.

### 5. .gitignore too minimal
**Root cause:** Only had `__pycache__/`, `*.pyc`, `.venv/`.
**Fix:** Added `.claude/active_mode.md`, `.claude/settings.json`,
`.claude/settings.local.json`, `.claude/hooks/`, `audit.jsonl`,
`nighttime.log`, `*.bak`, `pip_output.txt`.

### 6. No instruction to exclude .claude/ from commits
**Root cause:** Soft constraints said "don't modify .claude/" but the
launcher already modified them. `git add -A` committed the dirty state.
**Fix:** Explicit prohibition added to Step 8 of nighttime supplement.

### 7. Spec movement creates merge conflicts
**Root cause:** Step 9 moved specs from WrittenByDaytime/ to
WrittenByNighttime/ via `mv`. Each branch did this independently,
creating delete/add conflicts on merge.
**Fix:** Changed to `cp` (copy). Originals stay in WrittenByDaytime/
for the daytime agent to clean up after merging.

### 8. Hardcoded machine-specific path in directory_guard.py
**Root cause:** First-run setup writes the absolute path into the guard.
If committed, breaks on clone/move.
**Fix:** Template keeps `HARDCODED_PROJECT_DIR = None`. Now that
`.claude/hooks/` is gitignored, the runtime copy with the real path
is never committed.

### 9. dayrun.bat cleanup is best-effort
**Root cause:** If killed before exit, nighttime settings aren't restored.
**Fix:** Documented as safe — nightrun.bat always reinstalls settings
before launching. No manual intervention needed.

### 10. Directory guard blocks daytime agent
**Root cause:** Guard had no mode awareness — blocked everything
unconditionally, including the daytime agent writing memory files or
running commands with `/dev/null` redirects.
**Fix:** Guard now reads `.claude/active_mode.md` and exits 0 in
daytime mode. Also strips `/dev/null` redirects before checking for
system paths.

---

## Open / Watch List

### nightrun summary showed "Done: 11" instead of "Done tonight: 6"
**Status:** Fixed in `nightrun_helper.py` and `nightrun.bat`/`.sh`.
The summary now captures a session start timestamp and splits done tasks
into "completed this session" vs "previously completed".
**Needs transpose:** Yes — the helper and both launcher scripts were
updated in this project. Copy back to main repo.

### .pyc files were committed on night branches
**Status:** Fixed by `.gitignore` having `__pycache__/` and `*.pyc`.
However, existing branches already had them tracked. The merge removed
them from tracking via `git rm --cached`.
**Watch for:** If new projects start from a state where pyc files are
already tracked, they'll need the same `git rm --cached` cleanup.

### Audit hook catch-22 during merges
**Status:** Mitigated by gitignoring audit.jsonl. Previously, the hook
appended to the file on every git command, making it permanently dirty
and blocking merges. With gitignore, git doesn't care about changes to it.
**Watch for:** If someone force-tracks audit.jsonl again, the catch-22
returns.

### No dependency step in spec-writing process
**Found:** 2026-03-17
**Severity:** causes confusion
**Description:** The spec-writer skill and daytime quality checklist have no step for
identifying and installing new Python dependencies before queuing a task. When the night
agent runs, it either installs packages itself (wasting time, risking errors without
human oversight) or encounters import errors from unspecified dependencies.
**Fix:** Add a "New Dependencies" section to the spec template, add a dependency
installation step to the spec-writer skill, and add a checklist item to the daytime
quality checklist.
**Transposed to main repo:** no

### Nightrun summary lists all previously completed tasks
**Found:** 2026-03-17
**Severity:** causes confusion
**Description:** The nightrun summary's "Previously completed" section lists every done
task from all previous sessions, not just tonight's work. When 12 tasks are done and only
1 was tonight's, the user sees 12 lines of output and has to scan to find the 1 that
matters. The "Done tonight" / "Previously" split exists but the detail listing for
"Previously" is too verbose.
**Fix:** Show only the count for previously completed tasks, not the full listing.
**Transposed to main repo:** no

### Night instance fabricates timestamps instead of using wall-clock time
**Found:** 2026-03-17
**Severity:** causes confusion
**Description:** The night instance generates plausible-looking ISO timestamps that are
~11 hours off from actual wall-clock time. For example, tasks running at 13:03-13:29 EDT
got `nighttime_completed` values of 23:47-00:05. This caused 7/14 tasks to have
`daytime_reviewed` timestamps that predated `nighttime_completed` — chronologically
impossible. Root cause: instruction files used vague `<ISO timestamp>` placeholders instead
of requiring a specific datetime command, so the LLM estimated rather than measuring.
**Fix:** Updated 8 instruction files (4 active + 4 templates) to replace every
`<ISO timestamp>` with an explicit
`python -c "from datetime import datetime; print(datetime.now().strftime('%Y-%m-%dT%H:%M:%S'))"`
command and added "do NOT estimate or fabricate timestamps" warnings. Also corrected the 7
bad timestamps in tracker.json to postdate their `nighttime_completed` values.
**Files changed:** `_claude_sandbox_setup/templates/nighttime_supplement.md`,
`_claude_sandbox_setup/templates/daytime_supplement.md`,
`.claude/active_mode.md`, `.claude/skills/spec-writer/SKILL.md` (active + template),
`.claude/skills/branch-review/SKILL.md` (active + template)
**Transposed to main repo:** no

### Pre-written tests in WrittenByDaytime/ cause pytest collection errors on main
**Found:** 2026-03-17
**Severity:** causes confusion
**Description:** Tests written by daytime in WrittenByDaytime/ import modules that only
exist on specific task branches. When the night agent runs pytest globally for regression
checking (Step 6), pytest tries to collect these test files and fails on import errors.
This creates recurring "missing deps" flags that are really about unmerged branch code,
not missing packages. Example: tests for CrossEncoderFilter import xgboost, which is on
the task-011 branch but not on main.
**Fix idea:** Add `--ignore=DaytimeNighttimeHandOff/` to the regression pytest command
in the nighttime supplement Step 6 instructions. Pre-written tests should only be run
when targeted explicitly (`pytest WrittenByDaytime/<task>/tests/`), not swept up by
global regression checks.
**Transposed to main repo:** no

---

## How to Report New Issues

Add entries below the Open / Watch List section with:
```
### Short title
**Found:** YYYY-MM-DD
**Severity:** blocks workflow / causes confusion / minor
**Description:** What happens and why
**Fix idea:** If you have one
**Transposed to main repo:** yes/no
```
