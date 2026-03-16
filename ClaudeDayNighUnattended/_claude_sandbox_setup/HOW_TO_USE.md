# ClaudeDayNight — How To Use

## The Workflow in One Sentence

Brainstorm with Claude during the day, let Claude implement overnight while you sleep.

---

## Quick Start

### First time: install into your project

See `NEW_PROJECT_SETUP_CHEAT_SHEET.md` in the project root for the full step-by-step
walkthrough (includes git init, GitHub connection, and troubleshooting).

Short version: copy `_claude_sandbox_setup/` into your project, init git, run `claude`,
tell it to read `_claude_sandbox_setup/SETUP.md`.

### Every daytime session

```bash
bash _claude_sandbox_setup/scripts/dayrun.sh
```

Windows (if `bash` isn't available):
```
_claude_sandbox_setup\scripts\dayrun.bat
```

Claude enters collaborative mode with daytime settings pre-loaded. Tell it what you want
to build. It researches, pushes back, writes task specs, and can do small code fixes
directly. On exit, settings automatically restore to nighttime-safe mode.

### Every nighttime session

```bash
bash _claude_sandbox_setup/scripts/nightrun.sh
```

Walk away. Claude implements all pending tasks, creates branches, runs tests, logs results.
After all tasks, runs end-of-night sweeps (tests, bugs, DRY, type hints, dead code, security).
Auto-resumes if it hits a usage cap.

### Morning review

1. Check the terminal — nightrun prints a summary (done/skipped/blocked counts, branches, flags)
2. Run `dayrun.sh` — daytime Claude reads everything and briefs you
3. Review `night/` task branches and `night/sweep-*` cleanup branches
4. Merge what looks good, handle any `blocked` tasks
5. Queue up tonight's work

---

## What's in this folder

```
_claude_sandbox_setup/
├── SETUP.md                         ← Claude's self-installation instructions
├── HOW_TO_USE.md                    ← This file
├── docs/
│   ├── DangerousClaudeFlagReadme.md ← --dangerously-skip-permissions rationale
│   ├── DayNightWorkflowDesign.md    ← Architecture and design decisions
│   └── ContextAndClaudeMDBestPractices.md
├── templates/
│   ├── nighttime_supplement.md      ← Behavioral rules for nighttime mode
│   ├── daytime_supplement.md        ← Behavioral rules for daytime mode
│   ├── nighttime_settings.json      ← Tight permissions for unattended nighttime
│   ├── daytime_settings.json        ← Relaxed permissions for interactive daytime
│   ├── tracker_template.json        ← Empty tracker for new projects
│   ├── skills/
│   │   └── end-of-night-sweeps/SKILL.md  ← Codebase sweeps loaded after tasks done
│   ├── handoff_structure/
│   │   └── README.md                ← Template README for DaytimeNighttimeHandOff/
│   ├── commands/
│   │   ├── day.md                   ← /day slash command
│   │   └── night.md                 ← /night slash command
│   ├── claudeMDsupplement.md        ← Legacy unified supplement (kept for compat)
│   └── settings_to_merge.json       ← Legacy settings (kept for compat)
├── hooks/
│   ├── directory_guard.py           ← Blocks file access outside the project
│   ├── no_ask_human.py              ← Blocks questions in nighttime (decides autonomously)
│   ├── audit_log.py                 ← Logs every tool call to audit.jsonl
│   ├── stop_quality_gate.py         ← Prevents stopping with todo tasks remaining
│   ├── notification_log.py          ← Logs notification events to nighttime.log
│   ├── context_monitor.py           ← Graduated context warnings, prompts /compact
│   └── syntax_check.py              ← Validates Python/JSON/YAML after every file edit
├── scripts/
│   ├── verify_environment.py        ← Checks Python, venv, pytest, git
│   ├── dayrun.sh / dayrun.bat       ← Daytime launcher (swaps settings, restores on exit)
│   ├── nightrun.sh / nightrun.bat   ← Nighttime launcher with auto-resume
│   └── repairrun.sh / repairrun.bat ← Resets project to nighttime-safe state
└── tests/
    └── test_sandbox_setup.py        ← Verifies everything is wired up correctly
```

---

## The Two Modes

### Daytime mode

You're present. Claude is collaborative.

| Capability | Status |
|---|---|
| Ask you questions | Yes |
| WebSearch / WebFetch | Yes |
| Read codebase | Yes |
| Write specs + tests | Yes (to WrittenByDaytime/) |
| Small code fixes | Yes (bug fixes, typos, quick refactors) |
| Implement large features | Queue for nighttime instead |
| git push | No |
| Network tools (curl, wget) | No |
| Secrets files | No |

**Daytime Claude's job:** strategist and PM. It plans features, writes specs with
rationale and research URLs, routes information to the right files (goals → project_overview,
decisions → architecture-decisions, ideas → incubating, tasks → specs + tracker), and can
handle small code fixes directly.

**How to start**: `bash _claude_sandbox_setup/scripts/dayrun.sh`

### Nighttime mode

You're asleep. Claude is autonomous.

| Capability | Status |
|---|---|
| Ask you questions | **Blocked** (decides autonomously) |
| WebSearch / WebFetch | **Blocked** |
| Read specs and implement | Yes |
| Read project_overview + architecture-decisions | Yes |
| Create git branches | Yes (one per task + sweep branches) |
| Run tests | Yes |
| Run end-of-night sweeps | Yes (tests, bugs, DRY, types, dead code, security) |
| git push | No |
| Network tools (curl, wget, gh, PowerShell) | No |
| Secrets files | No |

**Nighttime Claude's job:** implementer. Works through all todo tasks, writes well-commented
code with type hints and docstrings, carries rationale and research URLs from specs into
code comments, then runs 6 codebase sweeps.

**How to start**: `bash _claude_sandbox_setup/scripts/nightrun.sh`

---

## The Handoff Directory

```
DaytimeNighttimeHandOff/
├── DaytimeOnly/
│   ├── project_overview.md    (what the project is — night reads this for context)
│   ├── inbox.md               (zero-friction capture, cleared each session)
│   ├── incubating.md          (ideas not ready to spec, each has a "next trigger")
│   ├── reference/
│   │   ├── architecture-decisions.md  (night reads this — follows established patterns)
│   │   ├── research.md               (night ignores — daytime working memory)
│   │   └── known-issues.md           (night ignores — daytime working memory)
│   └── archive/               (promoted or abandoned incubating items)
├── WrittenByDaytime/          ← Specs + tests ready for nighttime
│   └── task-001-retry-logic/
│       ├── spec.md
│       └── tests/
├── WrittenByNighttime/        ← Results + moved tasks
│   └── task-001-retry-logic/
│       ├── spec.md            (moved from WrittenByDaytime)
│       ├── tests/             (moved from WrittenByDaytime)
│       ├── plan.md            (night's implementation plan)
│       └── result.md          (commits, test results, decisions, flags)
├── tracker.json               ← Task state (both modes read/write)
├── audit.jsonl                ← Tool call log (auto-generated by hook)
└── nighttime.log              ← Running log of nighttime activity
```

### tracker.json example

```json
[
  {
    "task_id": "task-001",
    "description": "Add retry logic to API client",
    "daytime_created": "2026-03-14T14:30:00",
    "daytime_comments": "Max 3 retries, exponential backoff, skip 4xx errors",
    "depends_on": null,
    "status": "done",
    "nighttime_started": "2026-03-15T00:12:00",
    "nighttime_completed": "2026-03-15T01:45:00",
    "nighttime_comments": "Implemented. Added default 30s timeout (not in spec — flagged).",
    "branch": "night/task-001-retry-logic",
    "commit_sha": "a3f9c12e",
    "tests_passed": true,
    "attempted_approaches": [],
    "blocked_reason": null,
    "flags": ["Added default timeout - not in spec, review needed"],
    "daytime_reviewed": "2026-03-15T08:30:00"
  }
]
```

**Status values:** `todo` → `in_progress` → `done` / `skipped` / `blocked`

- **`blocked`** — nighttime needs something from you. Read `blocked_reason`, update
  `daytime_comments` with the answer, set status back to `todo`.
- **`skipped`** — nighttime gave up after 3 attempts. Read `attempted_approaches`.
- **`depends_on`** — list of task_ids that must be `done` first (null if independent).
- **`daytime_reviewed`** — timestamp set by daytime after processing results. Prevents
  re-processing the same results across consecutive daytime sessions.

---

## What happens to existing project files

| Your project has... | What Claude does |
|---|---|
| No CLAUDE.md | Creates one with `@.claude/active_mode.md` import line |
| Existing CLAUDE.md | Adds import line at top, keeps your content, removes contradictions |
| No .claude/settings.json | Creates one from nighttime_settings.json |
| Existing .claude/settings.json | Merges rules in (additive only, never removes your rules) |
| Existing .claude/hooks/ | Copies sandbox hooks in alongside existing hooks |
| No DaytimeNighttimeHandOff/ | Creates full structure |
| Existing DaytimeNighttimeHandOff/ | Verifies structure, adds missing pieces |

---

## Auto-resume on usage cap

Claude Max has hourly usage limits. If Claude hits the cap mid-session:

1. tracker.json shows the interrupted task as `"status": "in_progress"`
2. nightrun backs up tracker.json, waits the cooldown (default: 5 minutes), relaunches
3. The new session checks git state, resumes in_progress tasks, then continues

To adjust:
```bash
bash _claude_sandbox_setup/scripts/nightrun.sh . 600 2000    # 10 min cooldown, 2000 turns
```

Or environment variables:
```bash
NIGHTRUN_COOLDOWN=600 NIGHTRUN_MAX_TURNS=2000 NIGHTRUN_MAX_RELAUNCHES=10
```

---

## Safety hooks

| Hook | When | What it does |
|---|---|---|
| directory_guard.py | Before every tool call | Blocks file/command access outside project |
| no_ask_human.py | Before AskUserQuestion | Blocks questions, logs them, tells Claude to decide |
| stop_quality_gate.py | When Claude tries to stop | Blocks if todo tasks remain |
| context_monitor.py | After every tool call | Graduated warnings at 400/600/800 calls, prompts /compact |
| syntax_check.py | After Edit/Write/MultiEdit | Validates Python, JSON, YAML syntax immediately |
| audit_log.py | After every tool call | Logs to audit.jsonl (async, never blocks) |
| notification_log.py | On notifications | Logs to nighttime.log (async, never blocks) |

---

## If you move the project

1. Open `.claude/active_mode.md` → change PROJECT_ROOT to `NOT YET CONFIGURED`
2. Open `.claude/hooks/directory_guard.py` → change HARDCODED_PROJECT_DIR to `None`
3. Relaunch Claude — it redoes first-run setup

---

## Troubleshooting

| Problem | Fix |
|---|---|
| Claude says "BLOCKED" | Directory guard caught an out-of-bounds path. Check the path. |
| `/day` doesn't work | Check `.claude/commands/day.md` exists. Re-run SETUP.md. |
| `/night` doesn't start loop | Check tracker.json exists and has `todo` tasks. |
| nightrun exits immediately | All tasks done/skipped/blocked. Check tracker.json. |
| Nighttime browsing web or asking questions | Daytime settings left behind. Run `repairrun.sh`. |
| Task stuck on `in_progress` | Run `repairrun.sh` — resets stuck tasks to `todo`. |
| nightrun fails before Claude launches | Check git is initialized. Run `repairrun.sh`. |
| WebSearch blocked in daytime | Run `dayrun.sh` (not bare `claude`) to swap settings. |
| Tests failing after setup | Run `python -m pytest _claude_sandbox_setup/tests/ -v`. |
| SYNTAX CHECK warnings | Claude introduced a syntax error. It should fix it immediately. |
| CONTEXT CAUTION/WARNING | Claude should run /compact. If critical, it wraps up and stops. |

---

## Known limitations

1. **Hook checks command strings, not runtime behavior.** Python's stdlib can still access
   the network. Package managers (pip, npm) also have network access.

2. **`cmd /c` can chain to programs** that bypass deny rules. Documented, not fixable
   without OS-level sandboxing (coming for Windows in a future Claude Code release).

3. **nightrun uses --print mode** for non-interactive operation. The nighttime supplement
   and skills guide Claude's behavior.

4. **Single project scope.** The handoff directory is per-project.
