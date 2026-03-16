# ClaudeDayNight — Self-Setup Instructions

Claude, follow these steps exactly in order. This folder (`_claude_sandbox_setup/`)
contains everything needed to set up this project for day/night workflow operation.

Do all steps from the project root (the parent of this `_claude_sandbox_setup/` folder).

---

## Step 0: Verify environment

Run the environment verification script:

```bash
python _claude_sandbox_setup/scripts/verify_environment.py
```

This checks (and only acts if something is missing):
- Python is on PATH and is 3.10+
- A virtual environment exists (creates `.venv/` only if none found)
- pip is available (bootstraps only if missing)
- pytest is installed (installs only if missing)
- If the venv was just created and requirements.txt/pyproject.toml exist,
  dependencies are installed into the new venv
- Reports on handoff directory status (does NOT create it — that's Step 3)

Running this script multiple times is safe — it skips anything already in place.

**If it fails**, stop and tell the user what went wrong. Do not proceed until the
environment is ready.

**If it succeeds**, note the activation commands it prints. Activate the venv:
- Windows (bash/git bash): `source .venv/Scripts/activate`

---

## Step 1: Set up CLAUDE.md and .claude/active_mode.md

Mode-specific rules (nighttime vs. daytime) live in `.claude/active_mode.md`. `CLAUDE.md`
imports it via `@.claude/active_mode.md` and holds any project-specific rules below that
line. The run scripts (`dayrun.sh`, `nightrun.sh`) swap `active_mode.md` at launch time —
`CLAUDE.md` itself never changes, keeping it clean in version control.

**Step 1a — Set up `.claude/active_mode.md`:**

Create `.claude/` if it doesn't exist.
Copy `_claude_sandbox_setup/templates/nighttime_supplement.md` to `.claude/active_mode.md`.
This is the safe default — nighttime rules are active whenever no script has swapped the file.

**Step 1b — Set up `CLAUDE.md`:**

**If `CLAUDE.md` does not exist**: create it with just the import line:
```
@.claude/active_mode.md
```
No contradiction scan needed.

**If `CLAUDE.md` already starts with `@.claude/active_mode.md`**: already set up with the
current architecture. Skip this step.

**If `CLAUDE.md` exists and contains `NIGHTTIME SANDBOX RULES`** (old architecture —
rules were prepended directly into the file):
1. Read the full file. Identify where the sandbox block ends.
2. Extract everything after the sandbox block as project-specific content.
3. Rewrite `CLAUDE.md` as:
   ```
   @.claude/active_mode.md

   [extracted project-specific content]
   ```
4. Log the migration in `CLAUDE_LOG.md` under `## Sandbox Setup — CLAUDE.md Changes`.

**If `CLAUDE.md` exists with project-specific content but no sandbox rules**:
1. Read the full file. Scan for contradictions (anything that conflicts with the sandbox
   rules listed below). Remove or comment out contradictions with:
   `<!-- OVERRIDDEN BY SANDBOX RULES: [brief reason] -->`
2. Rewrite `CLAUDE.md` as:
   ```
   @.claude/active_mode.md

   [existing content, with contradictions removed]
   ```
3. Log every change in `CLAUDE_LOG.md`.

Contradictions to remove from project content:
- Tells Claude to ask, confirm, pause, or wait for human approval before acting
- Permits network access, web browsing, or fetching external URLs
- Permits reading or writing files outside the project directory
- Permits installing packages globally (e.g. `pip install` without a venv, `npm install -g`)
- Permits `git push`, `git reset --hard`, branch creation, or history rewriting
- Uses permissive language like "feel free to..." that conflicts with a hard constraint

---

## Step 2: Set up .claude/settings.json

Read `_claude_sandbox_setup/templates/nighttime_settings.json`.

- **If `.claude/settings.json` already exists at the project root**: merge the nighttime
  settings INTO the existing file. The merge must be **additive only**:
  - **allow list**: add any entries from the template that aren't already present.
  - **deny list**: add any entries from the template that aren't already present.
    **Never remove existing deny rules.**
  - **hooks**: add the hook entries (directory_guard.py and no_ask_human.py) if they
    aren't already present. Keep all existing hooks.
- **If no `.claude/settings.json` exists**: create `.claude/settings.json` with the
  exact contents of `templates/nighttime_settings.json`.

---

## Step 3: Copy hook scripts

Create `.claude/hooks/` if it doesn't exist.

Copy these files from `_claude_sandbox_setup/hooks/` into `.claude/hooks/`:
- `directory_guard.py`
- `no_ask_human.py`
- `audit_log.py`
- `stop_quality_gate.py`
- `notification_log.py`

If the files already exist in `.claude/hooks/`, overwrite them (they may have been updated).

---

## Step 4: Copy slash commands

Create `.claude/commands/` if it doesn't exist.

Copy these files from `_claude_sandbox_setup/templates/commands/` into `.claude/commands/`:
- `day.md`
- `night.md`

If the files already exist in `.claude/commands/`, overwrite them (they may have been updated).

These commands become available as `/day` and `/night` in Claude Code sessions.

---

## Step 5: Create handoff directory structure

Check if `DaytimeNighttimeHandOff/` exists at the project root.

**If it does NOT exist**, create it:
```
DaytimeNighttimeHandOff/
├── DaytimeOnly/
│   ├── project_overview.md   (copy from templates/handoff_structure/DaytimeOnly/project_overview_template.md)
│   ├── inbox.md              (copy from templates/handoff_structure/DaytimeOnly/inbox_template.md)
│   ├── incubating.md         (copy from templates/handoff_structure/DaytimeOnly/incubating_template.md)
│   ├── reference/            (create empty directory)
│   └── archive/              (create empty directory)
├── WrittenByDaytime/
├── WrittenByNighttime/
├── tracker.json              (copy from templates/tracker_template.json — empty array)
└── nighttime.log             (create as empty file)
```

Also copy `_claude_sandbox_setup/templates/handoff_structure/README.md` to
`DaytimeNighttimeHandOff/README.md`.

**If it already exists**, verify that each of the following is present. Create any that are missing:
- `DaytimeOnly/` directory (copy template files if creating fresh)
- `DaytimeOnly/project_overview.md`
- `DaytimeOnly/inbox.md`              (copy from inbox_template.md if missing)
- `DaytimeOnly/incubating.md`         (copy from incubating_template.md if missing)
- `DaytimeOnly/reference/`            (create empty directory if missing)
- `DaytimeOnly/archive/`              (create empty directory if missing)
- `WrittenByDaytime/`
- `WrittenByNighttime/`
- `tracker.json`

---

## Step 6: First-run setup

Now read `.claude/active_mode.md`. Follow the **FIRST RUN SETUP** section:
1. Run `pwd` to get the absolute path.
2. Write the path into `.claude/active_mode.md`'s PROJECT_ROOT line.
3. Write the path into `.claude/hooks/directory_guard.py`'s HARDCODED_PROJECT_DIR.

---

## Step 7: Run verification tests

Run the test suite to confirm everything is wired up correctly:

```bash
python -m pytest _claude_sandbox_setup/tests/ -v
```

**If all tests pass**, proceed to Step 8.

**If any tests fail**, read the failure messages — they explain exactly what's wrong.
Fix the issues and re-run the tests. If you can't fix something after 3 attempts,
log it in `CLAUDE_LOG.md` and proceed.

---

## Step 8: Confirm to user

Tell the user:

```
ClaudeDayNight setup complete.

Environment:
- Python: [version]
- venv: [name — created / existing]
- Activate with: [activation command]
- pytest: installed

Sandbox:
- CLAUDE.md: [created / updated] (imports .claude/active_mode.md)
- .claude/active_mode.md: nighttime profile (default)
- .claude/settings.json: [created / merged]
- Hooks installed: directory_guard.py, no_ask_human.py
- Slash commands installed: /day, /night
- Project directory locked to: [absolute path]

Handoff directory:
- DaytimeNighttimeHandOff/ [created / already exists]
  - WrittenByDaytime/ ✓
  - WrittenByNighttime/ ✓
  - tracker.json ✓
  - nighttime.log ✓

Nighttime protections active:
- File access restricted to this directory
- Network tools (curl, wget, WebFetch, WebSearch) blocked
- Secrets files blocked from reading
- Questions blocked (auto-decided and logged)
- git push blocked (commit only)

Daytime mode (run interactively — type /day to activate):
- WebSearch and WebFetch allowed
- AskUserQuestion allowed
- Same directory and secrets restrictions apply

How to use:
  Daytime:  cd <project> && claude        → then type /day
  Nighttime: cd <project> && nightrun.bat  (or ./nightrun.sh on Unix)
             OR: claude --dangerously-skip-permissions --max-turns 2000
                 then type /night

You can now give me tasks or walk away.
```

---

## Step 9: Proceed with user's task

If the user gave you a task along with the setup request, do it now.
The sandbox rules from CLAUDE.md are now active — follow them.

---

## Do NOT delete this folder

Keep `_claude_sandbox_setup/` in the project. It contains:
- Reference documentation (`docs/`, `HOW_TO_USE.md`)
- Original template files in `templates/` (for re-installation or updates)
- Verification tests (re-run anytime to check setup health)
- nightrun scripts for automated nighttime sessions

You may add `_claude_sandbox_setup/` to `.gitignore` if the user wants to keep it
out of version control.
