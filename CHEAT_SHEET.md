# ClaudeDayNight Cheat Sheet

## How to open a terminal

Right-click your project folder in File Explorer → **"Open in Terminal"**.
This opens Windows Terminal. It usually defaults to PowerShell — that's fine,
both `.bat` and `bash` commands work from there (as long as Git is installed,
which gives you `bash`).

## Quick reference

| Command | What it does |
|---------|-------------|
| `bash _claude_sandbox_setup/scripts/dayrun.sh` | Launch daytime session |
| `bash _claude_sandbox_setup/scripts/nightrun.sh` | Launch nighttime session (auto-resumes) |
| `bash _claude_sandbox_setup/scripts/repairrun.sh` | Reset project to known-good state |
| `/day` | Activate daytime mode in a manual session |
| `/night` | Activate nighttime mode in a manual session |
| `/compact` | Compress context (keeps summary of what happened) |
| `/clear` | Wipe conversation entirely and start fresh |

If `bash` isn't recognized, use the `.bat` versions instead:
`_claude_sandbox_setup\scripts\dayrun.bat`, `nightrun.bat`, `repairrun.bat`.

---

## Daytime session (you're present)

```bash
bash _claude_sandbox_setup/scripts/dayrun.sh
```
Launches Claude in collaborative mode. WebSearch, WebFetch, and questions allowed.
Settings swap automatically — no need to type `/day`.

Daytime Claude is your **strategist + PM**. It:
- Plans features, writes specs, queues tasks for nighttime
- Can do small code fixes directly (bug fixes, typos, quick refactors)
- Researches decisions and captures rationale + URLs in specs
- Routes information: goals → project_overview, decisions → architecture-decisions,
  ideas → incubating, actionable work → specs + tracker

*On exit, settings automatically restore to nighttime-safe mode.*

---

## Nighttime session (walk away)

```bash
bash _claude_sandbox_setup/scripts/nightrun.sh
```
Launches Claude unattended. Runs all pending tasks, auto-resumes on usage cap.
Network blocked, no questions, no push. Everything on `night/` branches.

After all tasks: runs **end-of-night sweeps** (test fixes, bug sweep, DRY refactoring,
type hints, dead code cleanup, security scan) — each on its own `night/sweep-*` branch.

**Safety hooks active:**
- Directory guard — blocks all access outside project
- No-ask-human — decides autonomously, logs decisions
- Stop quality gate — can't quit while todo tasks remain
- Context monitor — graduated warnings as context fills, prompts `/compact`
- Syntax check — validates Python/JSON/YAML after every file edit
- Audit log — records every tool call to audit.jsonl

---

## Nighttime manual (no auto-resume)

```bash
claude --dangerously-skip-permissions --max-turns 2000
```
Then type `/night`. Same behavior but won't restart on usage cap.

---

## Repair (when something seems wrong)

```bash
bash _claude_sandbox_setup/scripts/repairrun.sh
```

Run this when:
- Nighttime is asking questions or browsing the web (daytime settings left behind)
- A task is stuck on `in_progress` (nighttime crashed mid-task)
- Hook scripts seem missing or not firing
- You're not sure what state things are in — just run it, it's safe

What it does: restores nighttime settings + rules, re-copies hooks, resets stuck
`in_progress` tasks back to `todo`.

---

## Morning review

1. Check the terminal — nightrun prints a summary when it finishes
2. Run `dayrun.sh` — daytime Claude reads everything and briefs you
3. Review `night/` branches, merge what's good
4. Review `night/sweep-*` branches for cleanup changes
5. Handle any `blocked` tasks (answer the question, set back to `todo`)

Per-task details: `DaytimeNighttimeHandOff/WrittenByNighttime/<task>/result.md`
Full state: `DaytimeNighttimeHandOff/tracker.json`
Audit trail: `DaytimeNighttimeHandOff/audit.jsonl`

---

## First time setup (once per project)

See **`NEW_PROJECT_SETUP_CHEAT_SHEET.md`** for the full step-by-step walkthrough
(includes git init, GitHub connection, and troubleshooting).

---

## How it all fits together

### The two Claudes

```
┌─────────────────────────────────┐   ┌─────────────────────────────────┐
│           DAYTIME               │   │           NIGHTTIME             │
│                                 │   │                                 │
│  You're present, talking to it  │   │  You're gone. It works alone.   │
│                                 │   │                                 │
│  Role: strategist + PM          │   │  Role: implementer              │
│  Can: search web, ask you stuff │   │  Can: code, git, run tests      │
│  Can: do small code fixes       │   │  Can't: ask questions, use web  │
│  Queues big jobs for night      │   │  Runs sweeps after all tasks    │
│                                 │   │                                 │
│  Launch: dayrun.sh              │   │  Launch: nightrun.sh            │
└─────────────────────────────────┘   └─────────────────────────────────┘
```

### The idea pipeline (daytime manages this)

```
Something comes up mid-chat
         │
         ▼
    inbox.md  ◄── zero-friction, no judgment, just capture it
         │
         │  [at next session open, triage each item]
         │
    ┌────┴─────────────────────────┬──────────────┬──────────┐
    ▼                              ▼              ▼          ▼
incubating.md               reference/      WrittenByDaytime/  (drop)
"not ready yet,           "good to know,    "ready to build"
 needs a trigger"          not actionable"
    │
    │  [trigger fires — "this is ready now"]
    ▼
WrittenByDaytime/task-NNN/
  spec.md    ← what, when, why + rationale + research URLs
  tests/     ← pre-written tests nighttime runs
    +
tracker.json ← status: "todo"  ← nighttime's work queue
```

### The nighttime task loop

```
nightrun.sh starts
     │
     ▼
Check git state           ← crash recovery orientation
     │
     ▼
Read tracker.json + project_overview.md + architecture-decisions.md
     │
     ├── in_progress tasks? → resume (crash recovery)
     ├── blocked tasks?     → skip (need your input)
     ├── todo tasks?        → work through them in order
     └── nothing pending?   → run end-of-night sweeps, then stop
          │
          ▼  [for each todo task]
     Read spec.md + tests
          │
          ▼
     Write plan.md
          │
          ▼
     tracker: in_progress
          │
          ▼
     git checkout -b night/task-NNN-name
          │
          ▼
     Implement + test
          │
          ▼
     Write result.md (commits, test results, decisions, flags)
          │
          ▼
     git commit + checkout main
          │
          ▼
     tracker: done/skipped/blocked
          │
          ▼
     Move spec/tests → WrittenByNighttime/
          │
          ▼
     /compact if context is getting large
          │
          ▼
     Next task ──────────────────────────►

          [all tasks done]
               │
               ▼
          End-of-night sweeps (6 sweeps, each on night/sweep-* branch)
               │
               ▼
          Done — nightrun prints summary
```

### The handoff files — who reads what

```
File/Folder                              Day reads?   Night reads?   Who writes?
──────────────────────────────────────────────────────────────────────────────────
DaytimeOnly/project_overview.md             ✓             ✓           Day
DaytimeOnly/reference/architecture-*        ✓             ✓           Day
DaytimeOnly/inbox.md                        ✓             ✗           Day
DaytimeOnly/incubating.md                   ✓             ✗           Day
DaytimeOnly/reference/ (other files)        ✓             ✗           Day
WrittenByDaytime/<task>/spec.md             ✓             ✓           Day
WrittenByDaytime/<task>/tests/              ✓             ✓           Day
tracker.json                                ✓             ✓           Both
WrittenByNighttime/<task>/plan.md           ✓             ✓           Night
WrittenByNighttime/<task>/result.md         ✓             ✓           Night
nighttime.log                               ✓             ✓           Night
audit.jsonl                                 ✓             ✗           Hook (auto)
```

### tracker.json task lifecycle

```
  [daytime writes spec]     [nighttime works]              [daytime reviews]
       todo  ──────────►  in_progress  ──────────►  done      ──► daytime_reviewed ──► merge branch
                                       ──────────►  skipped   ──► daytime_reviewed ──► retry or drop
                                       ──────────►  blocked   ──► answer blocked_reason → todo
  [daytime cancels]
       todo  ──────────►  cancelled
```

### Information routing (daytime decides where things go)

```
User says something
    │
    ├─ Project goals, vision, scope?       → project_overview.md      (night reads this)
    ├─ Architecture or design decision?    → reference/architecture-*  (night reads this)
    ├─ Research finding, tool evaluation?  → reference/research.md     (night ignores)
    ├─ Known bug or limitation?            → reference/known-issues.md (night ignores)
    ├─ Ready to build now?                 → spec + tracker.json       (night implements)
    ├─ Idea, not ready yet?                → incubating.md             (night ignores)
    ├─ Quick aside, random thought?        → inbox.md                  (triaged next session)
    └─ Not sure?                           → day agent asks you (A/B/C)
```
