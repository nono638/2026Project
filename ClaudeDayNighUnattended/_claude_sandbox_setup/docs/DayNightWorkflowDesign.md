# ClaudeDayNight — Architecture and Design Rationale

## What This Is

ClaudeDayNight is a workflow system for async human-AI collaboration on software projects.
The core idea: a developer brainstorms and plans during the day, and Claude implements
overnight while they sleep.

Two operational modes, one toolkit:

- **Daytime**: Interactive, research-enabled. Claude asks questions, pushes back on ideas,
  and writes structured task specs with pre-written tests.
- **Nighttime**: Unattended, strictly bounded. Claude implements specs exactly as written,
  runs tests, logs results, and leaves branches for morning review.

---

## Why This Approach

### The Problem

Claude Code's `--dangerously-skip-permissions --max-turns 2000` flag combination enables
long unattended sessions, but without structure:
- Claude doesn't know what to work on
- There's no handoff mechanism between planning and implementation
- Progress state is lost on session restart
- No clear separation between "think about this" and "build this"

### The Solution

A structured handoff directory and two behavioral profiles solve each problem:

| Problem | Solution |
|---------|---------|
| Claude doesn't know what to work on | `tracker.json` with explicit task queue |
| No planning/implementation separation | WrittenByDaytime/ vs WrittenByNighttime/ directories |
| State lost on restart | tracker.json persists `in_progress` status, nightrun scripts relaunch |
| No behavioral guardrails for unattended work | `nighttime_supplement.md` in CLAUDE.md |

---

## Design Decisions

### JSON Tracker Over CSV

tracker.json is JSON rather than CSV because:
- Claude handles JSON more reliably via the Edit tool (no column alignment or escaping issues)
- The schema is self-documenting
- Easy to add new fields without breaking existing entries
- Python's json module makes it trivial to filter/update from wrapper scripts

### Separate Directories for Daytime and Nighttime Output

`WrittenByDaytime/` and `WrittenByNighttime/` are separate because:
- Clear ownership: daytime writes specs, nighttime writes results
- No accidental overwrites — nighttime can safely scan `WrittenByDaytime/` for pending work
- Morning review is simple: everything in `WrittenByNighttime/` is done work
- The move step (daytime → nighttime on completion) provides a clear "task is done" signal

### Branch-Per-Task

Each task gets its own git branch (`night/task-NNN-name`) because:
- Morning review is granular — each branch can be inspected, merged, or discarded independently
- Bad implementations don't contaminate main
- The branch name encodes the task ID for traceability

### No Branch Push

Nighttime commits locally only. Pushing is the human's job after morning review. This:
- Prevents pushing broken code to shared repositories
- Keeps human judgment in the loop for the final approval step
- Avoids needing git credentials/SSH keys in the unattended environment

### Pre-Written Tests

Daytime writes tests, nighttime runs them, because:
- Tests written by the specifier capture intent better than tests written by the implementer
- Nighttime Claude has something objective to verify against
- Test failures are clearly signaled in result.md for morning review
- Prevents nighttime Claude from writing tests that pass by construction

### Three-Attempt Rule

If nighttime gets stuck on something, it tries 3 meaningfully different approaches then moves
on. This:
- Prevents infinite loops on unresolvable problems
- Ensures the rest of the tasks still get done
- Creates clear documentation of what was tried (in result.md)
- Is already the Claude Code convention — we're just making it explicit

### Slash Commands for Mode Switching

`/day` and `/night` are custom slash commands (files in `.claude/commands/`) rather
than separate CLAUDE.md files because:
- One CLAUDE.md that includes the base rules for both modes
- Slash commands inject behavioral context on demand
- Users can switch modes within a session if needed
- The slash commands themselves serve as documentation for what each mode does

### Auto-Resume via Wrapper Scripts

`nightrun.bat` / `nightrun.sh` handle session restarts because Claude Max has usage limits:
- tracker.json persists state, so a restarted session picks up where it left off
- In-progress tasks are resumed before new ones are started
- The wrapper script abstracts the restart loop — the user just runs nightrun and walks away
- Configurable cooldown (default: 5 minutes) to avoid hammering the API

---

## File Hierarchy

```
_claude_sandbox_setup/
├── SETUP.md                   ← Claude's self-installation instructions
├── HOW_TO_USE.md              ← Human quickstart guide
├── docs/
│   ├── DangerousClaudeFlagReadme.md  ← --dangerously-skip-permissions rationale
│   ├── DayNightWorkflowDesign.md     ← This file
│   └── ContextAndClaudeMDBestPractices.md
├── templates/
│   ├── nighttime_supplement.md  ← CLAUDE.md content for nighttime mode
│   ├── daytime_supplement.md    ← CLAUDE.md content for daytime mode
│   ├── nighttime_settings.json  ← Tight permissions for unattended nighttime
│   ├── daytime_settings.json    ← Relaxed permissions for interactive daytime
│   ├── tracker_template.json    ← Empty tracker to copy to new projects
│   ├── claudeMDsupplement.md    ← Legacy: original unified supplement (kept for compat)
│   ├── settings_to_merge.json   ← Legacy: original settings (kept for compat)
│   ├── handoff_structure/
│   │   └── README.md            ← Template README for DaytimeNighttimeHandOff/
│   └── commands/
│       ├── day.md               ← /day slash command
│       └── night.md             ← /night slash command
├── hooks/
│   ├── directory_guard.py       ← Blocks file access outside project root
│   └── no_ask_human.py          ← Blocks Claude from asking questions
├── scripts/
│   ├── verify_environment.py    ← Checks Python, venv, pytest, handoff structure
│   ├── nightrun.bat             ← Windows nighttime session launcher with auto-resume
│   └── nightrun.sh              ← Unix nighttime session launcher with auto-resume
└── tests/
    ├── conftest.py
    └── test_sandbox_setup.py    ← Verifies all components are in place
```

---

## Daytime/Nighttime Permissions Comparison

| Capability | Daytime | Nighttime |
|---|---|---|
| Read files | Yes | Yes |
| Write files | Yes | Yes |
| WebSearch | Yes | **No** |
| WebFetch | Yes | **No** |
| AskUserQuestion | Yes | **No** (blocked by hook) |
| git push | No | No |
| rm -rf | No | No |
| Secrets files | No | No |
| directory_guard hook | Yes | Yes |
| no_ask_human hook | **No** | Yes |

Daytime removes the no_ask_human hook and adds WebSearch/WebFetch/AskUserQuestion to the
allow list. Everything else is the same.

---

## Morning Review Checklist

When you wake up after a nighttime session:

1. **Check overall status**: `cat DaytimeNighttimeHandOff/tracker.json | python -m json.tool`
2. **Check the log**: `cat DaytimeNighttimeHandOff/nighttime.log`
3. **Per-task review**: For each completed task, read `WrittenByNighttime/<task>/result.md`
4. **Inspect branches**: `git branch | grep night/`
5. **Merge or discard**: For each branch, decide: `git merge night/task-NNN` or `git branch -d night/task-NNN`
6. **Note flags**: Anything in `"flags"` in tracker.json needs human review
7. **Plan daytime**: Use flags and skipped tasks as input for next daytime session

---

## Future Considerations

- **Task dependencies**: Add a `depends_on: ["task-NNN"]` field to tracker.json entries.
  Nighttime skips tasks whose dependencies aren't yet done.
- **Multi-project**: Currently one handoff directory per project. Could be extended with
  a global task queue, but per-project is simpler and avoids cross-contamination.
- **Test framework detection**: verify_environment.py could auto-detect whether to run
  pytest, jest, cargo test, etc. based on project files.
- **Daytime session notes**: Add a `daytime_session_log.md` for the daytime instance to
  leave context that doesn't fit in a spec (research findings, rejected approaches, etc.).

---

## Evolution from ClaudeUnattended

This toolkit evolved from `ClaudeUnattended`, which focused purely on sandboxed unattended
execution. The key evolution:

| ClaudeUnattended | ClaudeDayNight |
|---|---|
| Single operational mode (unattended only) | Two modes: daytime (collaborative) + nighttime (unattended) |
| No structured task handoff | Explicit WrittenByDaytime/WrittenByNighttime handoff |
| No planning phase | Daytime mode: research, pushback, spec writing |
| Manual restart on usage cap | nightrun scripts with auto-resume |
| Single CLAUDE.md supplement | Split into daytime_supplement + nighttime_supplement |
| Single settings.json | Split into daytime_settings + nighttime_settings |

The sandbox safety infrastructure (directory_guard.py, no_ask_human.py, settings deny lists)
is unchanged and remains the foundation of the nighttime profile.
