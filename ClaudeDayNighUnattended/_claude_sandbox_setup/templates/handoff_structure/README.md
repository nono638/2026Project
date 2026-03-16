# DaytimeNighttimeHandOff

This directory is the handoff point between daytime planning sessions and nighttime
implementation sessions.

## Structure

```
DaytimeNighttimeHandOff/
├── DaytimeOnly/                    # Daytime project management — night instance does NOT read these
│   ├── project_overview.md         # What the project is, goals, scope, change history
│   ├── inbox.md                    # Zero-friction capture — cleared every session-open triage
│   ├── incubating.md               # Ideas not yet ready to spec — each has a "next trigger"
│   ├── reference/                  # Non-actionable knowledge: decisions, patterns, research
│   └── archive/                    # Completed or abandoned incubating items (never delete)
├── WrittenByDaytime/               # Daytime writes task specs here
│   └── task-001-retry-logic/
│       ├── spec.md                 # Implementation spec
│       └── tests/                  # Pre-written test files
├── WrittenByNighttime/             # Nighttime writes results here (and moves tasks here when done)
│   └── task-001-retry-logic/
│       ├── spec.md                 # Original spec (moved from WrittenByDaytime)
│       ├── tests/                  # Original tests (moved from WrittenByDaytime)
│       ├── plan.md                 # Night's implementation plan
│       └── result.md               # What happened: commits, test results, flags for morning
├── tracker.json                    # Task execution queue — both instances read and write this
├── audit.jsonl                     # Tool call log written by audit_log.py hook (auto-generated)
└── nighttime.log                   # Running log of all nighttime activity
```

## The pipeline

```
mid-session capture → inbox.md
                           ↓  [session-open triage]
             ┌─────────────┼──────────────┬──────────┐
             ↓             ↓              ↓          ↓
       incubating.md   reference/    WrittenByDaytime/ drop
             ↓                       + tracker.json (todo)
      [trigger fires]                      ↓
             ↓                       [nightrun.sh]
       WrittenByDaytime/                   ↓
       + tracker.json (todo)    WrittenByNighttime/result.md
                                           ↓  [morning review]
                                flags/blocked → inbox.md
                                branches reviewed + merged
                                archive/ (skipped/abandoned)
```

## tracker.json format

```json
[
  {
    "task_id": "task-001",
    "description": "Add retry logic to API client",
    "daytime_created": "2026-03-14T14:30:00",
    "daytime_comments": "Max 3 retries, exponential backoff, skip 4xx errors",
    "depends_on": null,
    "status": "todo",
    "nighttime_started": null,
    "nighttime_completed": null,
    "nighttime_comments": null,
    "branch": null,
    "commit_sha": null,
    "tests_passed": null,
    "attempted_approaches": [],
    "blocked_reason": null,
    "flags": [],
    "daytime_reviewed": null
  }
]
```

### Field notes

- **`depends_on`** — null, or an array of task_ids that must be `done` before this task starts.
  Example: `["task-001", "task-002"]`. Leave null if independent.
- **`commit_sha`** — the SHA of the final commit for this task (more durable than branch name).
- **`attempted_approaches`** — list of approaches tried if something failed. Prevents the next
  nighttime run from re-trying dead ends. Example: `["tried library X — import error", "tried subprocess approach — permission denied"]`.
- **`blocked_reason`** — set when status is `blocked`. Human-readable explanation of what
  nighttime needs from the user before it can proceed. See `blocked` status below.
- **`daytime_reviewed`** — ISO timestamp set by daytime after processing a done/skipped/blocked
  task's results. Prevents re-processing the same results across consecutive daytime sessions.
  Cleared when a blocked task is reset to `todo`.

### Status values

| Status | Meaning |
|--------|---------|
| `todo` | Created by daytime, not yet picked up by nighttime |
| `in_progress` | Nighttime is currently working on it |
| `done` | Completed successfully |
| `blocked` | Nighttime hit a blocker requiring human input — see `blocked_reason` |
| `skipped` | Nighttime gave up after 3 attempts with no path forward — see result.md |
| `cancelled` | Cancelled by daytime before nighttime picked it up |

**`blocked` vs `skipped`:** Use `blocked` when a specific piece of human input would unblock
the task (a missing credential, an ambiguous requirement, a decision about approach). Use
`skipped` when the task is genuinely stuck with no clear path forward.

## Daytime writes to

- `DaytimeOnly/inbox.md` (mid-session captures)
- `DaytimeOnly/incubating.md` (ideas with next triggers)
- `DaytimeOnly/project_overview.md` (scope and change history)
- `DaytimeOnly/reference/*.md` (non-actionable knowledge)
- `DaytimeOnly/archive/` (completed/abandoned incubating items)
- `WrittenByDaytime/<task-id>/spec.md`
- `WrittenByDaytime/<task-id>/tests/<test files>`
- `tracker.json` (new entries `status: "todo"`, or `status: "cancelled"`)

## Nighttime writes to

- `WrittenByNighttime/<task-id>/plan.md` (before starting)
- `WrittenByNighttime/<task-id>/result.md` (after completing)
- `tracker.json` (updates status, fills in nighttime fields)
- `nighttime.log` (running log)
- `audit.jsonl` (tool call log — via hook, automatic)
- Moves `WrittenByDaytime/<task-id>/` contents into `WrittenByNighttime/<task-id>/`
