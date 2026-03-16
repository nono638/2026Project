# Nighttime Mode

You are now in **nighttime / unattended mode**. The user is not watching. Work through all
pending tasks autonomously. Do not ask questions — decide and log.

## Switch to nighttime settings

Before doing anything else, swap the settings and mode files:

1. Copy `_claude_sandbox_setup/templates/nighttime_supplement.md` to `.claude/active_mode.md`
   (overwrite the existing file).
2. Copy `_claude_sandbox_setup/templates/nighttime_settings.json` to `.claude/settings.json`
   (overwrite the existing file).

**Important:** The settings.json swap activates the directory guard hook, blocks network
access, and restricts permissions. This takes effect on subsequent tool calls. If it doesn't
seem to take effect, tell the user they may need to restart the session.

## Immediate actions — start now

1. Read `DaytimeNighttimeHandOff/tracker.json`
2. Check for `"status": "in_progress"` tasks — resume these first (check out their branch,
   read plan.md, continue implementation)
3. Then pick up `"status": "todo"` tasks in order (lowest task_id first)
4. If nothing pending, append to `DaytimeNighttimeHandOff/nighttime.log` and stop

## Per-task workflow

For each task:

1. **Read** `WrittenByDaytime/<task-dir>/spec.md` and all test files
2. **Plan** — write `WrittenByNighttime/<task-dir>/plan.md` before touching any code
3. **Update tracker** — set `status: "in_progress"`, `nighttime_started: <timestamp>`
4. **Branch** — `git checkout -b night/<task-id>-<short-name>`
5. **Tests first** — if spec has acceptance tests, implement them as pytest before coding
6. **Implement** — exactly what spec.md says, nothing more
7. **Test** — run the pre-written tests + existing project tests
8. **Result** — write `WrittenByNighttime/<task-dir>/result.md`
9. **Commit** — `git add -A && git commit -m "night: <task-id> <description>"`
10. **Return to main** — `git checkout main`
11. **Update tracker** — `status: "done"`, fill in nighttime fields
12. **Move task** — move spec.md and tests/ from WrittenByDaytime/ to WrittenByNighttime/
13. **Log** — append to `nighttime.log` with timestamp, status, and duration
14. **Next task** — return to step 1

## Rules

- 3 attempts max on any failure, then mark `status: "skipped"` and move on
- No questions, no network, no pushing to remote
- Branch per task, commit per task, return to main between tasks
- Log everything in result.md and nighttime.log
- Use medium effort — do competent work, but don't over-think blockers

Refer to your `.claude/active_mode.md` nighttime supplement for full behavioral rules.
