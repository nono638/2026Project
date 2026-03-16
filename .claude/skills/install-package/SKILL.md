---
name: install-package
description: Safely install Python packages into the project venv only — never globally. Updates requirements.txt and logs changes to ENVIRONMENT.md.
---

# Install Package

Safely install one or more Python packages into the project's virtual environment. This
skill enforces that nothing is ever installed globally on the system. Both agents (day and
night) must use this skill whenever they need to install a package.

**Never run `pip install` directly.** Always go through this skill.

---

## Step 0 — Pre-flight: verify venv is active

Before anything else, confirm you're in a virtual environment:

```bash
python -c "import sys; print(sys.executable)"
```

The output **must** contain `.venv` (or `venv` or `env`). If it points to a system Python
(e.g., `C:\Python312\python.exe` or `/usr/bin/python3`), **STOP. Do not install anything.**

If no venv exists or it's not active:
1. Run the environment-bootstrap skill first: read `.claude/skills/environment-bootstrap/SKILL.md`
2. Return here after the venv is active

If the venv exists but isn't activated:
- **Windows (bash/git bash):** `source .venv/Scripts/activate`
- **Linux/macOS:** `source .venv/bin/activate`

Then re-run the sys.executable check. Only proceed when it points to the venv.

---

## Step 1 — Read ENVIRONMENT.md

Read `ENVIRONMENT.md` in the project root. If it doesn't exist, note that — you'll create
it in Step 6.

Check the **Change Log** section (if it exists) for:
- Previous install attempts that failed (avoid repeating mistakes)
- Version constraints noted from past issues
- Any packages that were intentionally removed or avoided (and why)

---

## Step 2 — Check if already installed

Before installing, check if the package is already available:

```bash
pip show <package-name>
```

If already installed:
- Check if the installed version satisfies the requirement
- If yes: skip install, report "already installed at version X.Y.Z"
- If a specific version was requested and differs: proceed with upgrade/downgrade

Also check `requirements.txt` — if the package is listed there but `pip show` says it's
not installed, that's a broken state. Flag it and install from requirements.txt first.

---

## Step 3 — Install the package

Install with a pinned version when possible:

```bash
pip install <package-name>==<version>
```

If no specific version was requested, install latest and capture the version:

```bash
pip install <package-name>
pip show <package-name> | grep Version
```

**For multiple packages:** Install one at a time so you can track which succeeded and which
failed. Do not `pip install pkg1 pkg2 pkg3` in one command.

**If install fails:**
1. Read the error message carefully
2. Common issues:
   - Version conflict → check what's conflicting, note it in ENVIRONMENT.md
   - Build tools missing → note in ENVIRONMENT.md as a system prerequisite
   - Package not found → check spelling, check if it's named differently on PyPI
3. Log the failure in ENVIRONMENT.md (Step 6) even if you can't resolve it
4. Do NOT attempt to install globally as a workaround. Ever.

---

## Step 4 — Verify the install

After installation, verify it actually works:

```bash
python -c "import <module_name>; print(<module_name>.__version__)"
```

Note: the import name sometimes differs from the package name (e.g., `pip install Pillow`
but `import PIL`). If the import name is unclear, check with:

```bash
pip show <package-name> | grep -i "name\|location"
```

If the import fails despite pip saying it's installed, that's a problem — log it in
ENVIRONMENT.md and flag it for morning review.

---

## Step 5 — Update requirements.txt

Read the current `requirements.txt`. Add the new package with its pinned version:

```
<package-name>==X.Y.Z
```

**Rules:**
- Always pin to exact version (`==`), never use `>=` or unpinned
- Keep the file sorted alphabetically for readability
- Do not remove or modify existing entries unless resolving a conflict
- If a version conflict forced a change to an existing pin, note the old and new versions
  in your ENVIRONMENT.md log entry

**If requirements.txt doesn't exist:** Create it with all currently installed packages:

```bash
pip freeze > requirements.txt
```

Then review it — remove any packages that are clearly unrelated to the project (pip, setuptools,
wheel are fine to keep since they're standard).

---

## Step 6 — Log to ENVIRONMENT.md

Open `ENVIRONMENT.md`. If it doesn't exist, create it using the template from the
environment-bootstrap skill, then add the sections below.

### Add or update the Change Log section

If `ENVIRONMENT.md` doesn't have a `## Change Log` section, add one at the bottom.

Append an entry:

```markdown
### YYYY-MM-DD — Installed <package-name> <version>
- **Why:** <one sentence — why this package is needed, what feature/task requires it>
- **Requested by:** <task-id from tracker.json, or "daytime session">
- **Import name:** `<module_name>` (if different from package name)
- **Notes:** <any issues encountered, version constraints, or compatibility notes>
```

If the install **failed**, still log it:

```markdown
### YYYY-MM-DD — FAILED: <package-name>
- **Why:** <why it was needed>
- **Error:** <one-line summary of the error>
- **Attempted:** <what you tried>
- **Resolution:** <pending / workaround found / not needed>
```

This log is critical — it prevents both agents from hitting the same issue repeatedly and
gives the user a clear history of what happened to the environment.

### Update the Dependencies section (if it exists)

If ENVIRONMENT.md has a section listing key dependencies, add the new package there too
with a brief description of its purpose.

---

## Step 7 — Report

Summarize what happened:

> "Installed <package> <version> into .venv. requirements.txt updated. Logged to ENVIRONMENT.md."

Or if it failed:

> "Failed to install <package>: <reason>. Logged to ENVIRONMENT.md. [Needs human input /
> will try alternative / task blocked]."

---

## Nighttime-specific behavior

If running in nighttime mode:
- Also log to `DaytimeNighttimeHandOff/nighttime.log`:
  ```
  [<ISO timestamp>] INSTALL <package-name>==<version> — <why>
  ```
- If install fails and the package is required by the spec, mark the task as `blocked`
  with a clear `blocked_reason` explaining what's needed
- Do NOT attempt to install system-level packages, compile from source, or use
  `--user` flag as workarounds

---

## Done

Return to the calling context.
