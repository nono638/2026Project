# Nighttime Mode

You are now in **nighttime / unattended mode**. The user is not watching. Work through all
pending tasks autonomously. Do not ask questions — decide and log.

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
5. **Implement** — exactly what spec.md says, nothing more
6. **Test** — run the pre-written tests + existing project tests
7. **Result** — write `WrittenByNighttime/<task-dir>/result.md`
8. **Commit** — `git add -A && git commit -m "night: <task-id> <description>"`
9. **Return to main** — `git checkout main`
10. **Update tracker** — `status: "done"`, fill in nighttime fields
11. **Move task** — move spec.md and tests/ from WrittenByDaytime/<task-dir>/ to WrittenByNighttime/<task-dir>/
12. **Log** — append one line to `nighttime.log`
13. **Next task** — return to step 1

## Rules

- 3 attempts max on any failure, then mark `status: "skipped"` and move on
- No questions, no network, no pushing to remote
- Branch per task, commit per task, return to main between tasks
- Log everything in result.md and nighttime.log

Refer to your CLAUDE.md nighttime supplement for full behavioral rules.
