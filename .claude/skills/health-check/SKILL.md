---
name: health-check
description: Session-start verification — check all sandbox components are present and configured, self-heal from templates if possible
---

# Health Check Skill

Run this on every session start. Fix what you can silently from the templates;
only tell the user if something is unfixable.

---

## Step 1 — Is the setup folder present?

Check if `_claude_sandbox_setup/SETUP.md` exists.

- **If missing:** STOP. Tell the user: "The `_claude_sandbox_setup/` folder is missing
  or damaged. I can't self-heal without it. Re-copy it from the template repo or run
  the setup cheat sheet again."
- **If present:** Continue.

---

## Step 2 — Was first-run setup completed?

Read `.claude/hooks/directory_guard.py` and check the `HARDCODED_PROJECT_DIR` line.

- **If `HARDCODED_PROJECT_DIR = None`:** First-run setup was never completed. Read
  `_claude_sandbox_setup/SETUP.md` and follow all steps. Then return here and continue.
- **If it has a real path:** Confirm it matches your current working directory (`pwd`).
  If it doesn't match, STOP and tell the user there's a path mismatch — the project
  may have been moved.

---

## Step 3 — Verify required files and folders

Check each item below. If missing, fix it using the source listed. Do this silently —
don't narrate each check to the user.

### .claude/ structure

| File | Source if missing |
|---|---|
| `.claude/hooks/directory_guard.py` | `_claude_sandbox_setup/hooks/directory_guard.py` — then set `HARDCODED_PROJECT_DIR` to pwd |
| `.claude/hooks/audit_log.py` | `_claude_sandbox_setup/hooks/audit_log.py` |
| `.claude/hooks/context_monitor.py` | `_claude_sandbox_setup/hooks/context_monitor.py` |
| `.claude/hooks/no_ask_human.py` | `_claude_sandbox_setup/hooks/no_ask_human.py` |
| `.claude/hooks/notification_log.py` | `_claude_sandbox_setup/hooks/notification_log.py` |
| `.claude/hooks/stop_quality_gate.py` | `_claude_sandbox_setup/hooks/stop_quality_gate.py` |
| `.claude/hooks/syntax_check.py` | `_claude_sandbox_setup/hooks/syntax_check.py` |
| `.claude/commands/day.md` | `_claude_sandbox_setup/templates/commands/day.md` |
| `.claude/commands/night.md` | `_claude_sandbox_setup/templates/commands/night.md` |
| `.claude/skills/architecture-advisor/SKILL.md` | `_claude_sandbox_setup/templates/skills/architecture-advisor/SKILL.md` |
| `.claude/skills/branch-review/SKILL.md` | `_claude_sandbox_setup/templates/skills/branch-review/SKILL.md` |
| `.claude/skills/coverage-check/SKILL.md` | `_claude_sandbox_setup/templates/skills/coverage-check/SKILL.md` |
| `.claude/skills/crash-recovery/SKILL.md` | `_claude_sandbox_setup/templates/skills/crash-recovery/SKILL.md` |
| `.claude/skills/dependency-audit/SKILL.md` | `_claude_sandbox_setup/templates/skills/dependency-audit/SKILL.md` |
| `.claude/skills/distill/SKILL.md` | `_claude_sandbox_setup/templates/skills/distill/SKILL.md` |
| `.claude/skills/end-of-night-sweeps/SKILL.md` | `_claude_sandbox_setup/templates/skills/end-of-night-sweeps/SKILL.md` |
| `.claude/skills/environment-bootstrap/SKILL.md` | `_claude_sandbox_setup/templates/skills/environment-bootstrap/SKILL.md` |
| `.claude/skills/health-check/SKILL.md` | `_claude_sandbox_setup/templates/skills/health-check/SKILL.md` |
| `.claude/skills/install-package/SKILL.md` | `_claude_sandbox_setup/templates/skills/install-package/SKILL.md` |
| `.claude/skills/intake/SKILL.md` | `_claude_sandbox_setup/templates/skills/intake/SKILL.md` |
| `.claude/skills/pre-commit-review/SKILL.md` | `_claude_sandbox_setup/templates/skills/pre-commit-review/SKILL.md` |
| `.claude/skills/project-snapshot/SKILL.md` | `_claude_sandbox_setup/templates/skills/project-snapshot/SKILL.md` |
| `.claude/skills/spec-writer/SKILL.md` | `_claude_sandbox_setup/templates/skills/spec-writer/SKILL.md` |
| `.claude/skills/tdd-enforcer/SKILL.md` | `_claude_sandbox_setup/templates/skills/tdd-enforcer/SKILL.md` |

**Do NOT overwrite** `.claude/settings.json` or `.claude/active_mode.md` — these are
mode-specific and managed by the run scripts. Only check that they exist. If missing,
log it as a problem but don't guess which mode to install.

### Handoff structure

| Path | Fix if missing |
|---|---|
| `DaytimeNighttimeHandOff/` | `mkdir -p` |
| `DaytimeNighttimeHandOff/DaytimeOnly/` | `mkdir -p` |
| `DaytimeNighttimeHandOff/DaytimeOnly/reference/` | `mkdir -p` |
| `DaytimeNighttimeHandOff/DaytimeOnly/archive/` | `mkdir -p` |
| `DaytimeNighttimeHandOff/WrittenByDaytime/` | `mkdir -p` |
| `DaytimeNighttimeHandOff/WrittenByNighttime/` | `mkdir -p` |
| `DaytimeNighttimeHandOff/tracker.json` | Create with contents: `[]` |
| `DaytimeNighttimeHandOff/nighttime.log` | Create empty |
| `DaytimeNighttimeHandOff/DaytimeOnly/project_overview.md` | Copy from `_claude_sandbox_setup/templates/handoff_structure/DaytimeOnly/project_overview_template.md` |
| `DaytimeNighttimeHandOff/DaytimeOnly/inbox.md` | Copy from `_claude_sandbox_setup/templates/handoff_structure/DaytimeOnly/inbox_template.md` |
| `DaytimeNighttimeHandOff/DaytimeOnly/incubating.md` | Copy from `_claude_sandbox_setup/templates/handoff_structure/DaytimeOnly/incubating_template.md` |

### Root files

| File | Fix if missing |
|---|---|
| `CLAUDE.md` | Create with single line: `@.claude/active_mode.md` |

### Environment files

| Check | Fix if missing |
|---|---|
| `.venv/` or `venv/` exists | Do NOT create — just note it. The first task that needs packages will create it. |
| `requirements.txt` exists (if a venv exists) | Create an empty `requirements.txt`. On next `pip install`, the agent will populate it. |
| `ENVIRONMENT.md` exists (if a venv exists) | Create with Python version (`python --version`), venv location, activation command, and `pip install -r requirements.txt` instruction. |

If a venv exists but `requirements.txt` is missing or empty, run
`pip freeze > requirements.txt` (activating the venv first) to capture current state.

---

## Step 4 — Validate tracker.json

Try to parse `DaytimeNighttimeHandOff/tracker.json` as JSON.

- **Parses OK:** Continue.
- **Parse fails:** Check if `DaytimeNighttimeHandOff/tracker.json.bak` exists (nightrun
  creates backups before each launch). If the backup parses, copy it over the corrupt
  file. If no backup or backup is also corrupt, create a fresh `[]` file.
  Log the recovery or loss either way.

---

## Step 5 — Run the setup test suite

Run the verification tests:
```bash
python -m pytest _claude_sandbox_setup/tests/ -q 2>&1
```

**Interpreting results:**
- Tests in `TestClaudeMd`, `TestSettingsJson`, `TestHooks`, `TestVenv` check deployed files.
  If these fail, it means files are missing from `.claude/` or the project root. Go back to
  Step 3 and fix whatever is missing.
- Tests in `TestDayNightTemplates`, `TestSkills`, `TestScripts` check the template directory.
  If these fail, the `_claude_sandbox_setup/` folder is damaged or outdated.
- Tests in `TestNighttimeNewFeatures`, `TestDaytimeNewFeatures`, `TestDaytimeSettingsPermissions`
  check feature configuration. If these fail, templates may need to be re-copied from the
  source repo.

**Self-healing from test failures:**
1. If a test fails because a file is missing → copy it from the template (Step 3 tables)
2. If a test fails because of wrong content (e.g., wrong defaultMode) → copy the template
   version to overwrite the corrupt file
3. If a test fails and you can't fix it → log the failure and continue. Don't block the
   entire session over a test failure.

After fixing, re-run the tests to confirm. If all pass (or only expected skips remain),
proceed. If failures persist after one fix attempt, log them and move on.

---

## Step 6 — Log any repairs

If you fixed anything in Steps 1–5, append one summary line to
`DaytimeNighttimeHandOff/nighttime.log`:

```
[<ISO timestamp>] SELF-HEAL: restored <comma-separated list of what was missing>
```

If everything was healthy, don't log anything.

---

## Done

Return to the calling mode supplement and continue with the next step in the session
protocol. Do not mention the health check to the user unless you found and fixed
something — in that case, briefly note what was repaired.
