---
name: dependency-audit
description: Audit Python dependencies for outdated packages, known vulnerabilities, and unused imports — logs findings to ENVIRONMENT.md
---

# Dependency Audit

Audit the project's Python dependencies for outdated packages, known vulnerabilities,
unused imports, and dependency conflicts. Both agents (day and night) can use this skill.
During nighttime, it can be invoked as part of end-of-night sweeps. During daytime, the
user might invoke it explicitly.

---

## Step 1 — Pre-flight: verify venv is active

Before anything else, confirm you're in a virtual environment:

```bash
python -c "import sys; print(sys.executable)"
```

The output **must** contain `.venv` (or `venv` or `env`). If it points to a system Python
(e.g., `C:\Python312\python.exe` or `/usr/bin/python3`), **STOP. Do not proceed.**

If no venv exists or it's not active:
1. Run the environment-bootstrap skill first: read `.claude/skills/environment-bootstrap/SKILL.md`
2. Return here after the venv is active

If the venv exists but isn't activated:
- **Windows (bash/git bash):** `source .venv/Scripts/activate`
- **Linux/macOS:** `source .venv/bin/activate`

Then re-run the sys.executable check. Only proceed when it points to the venv.

---

## Step 2 — Read context files

Read these files for context before running checks:

- **`requirements.txt`** — the declared dependencies and their pinned versions
- **`ENVIRONMENT.md`** — the environment state, previous audit results, change log

If either file is missing, note it. `requirements.txt` missing means there's nothing to
audit — log that to ENVIRONMENT.md and stop. `ENVIRONMENT.md` missing is fine; you'll
create the audit section when you write results.

---

## Step 3 — Check for outdated packages

Run:

```bash
pip list --outdated --format=columns
```

For each outdated package, categorize by severity:

- **High priority:** Major version behind (e.g., installed 2.x, latest 3.x)
- **Medium priority:** Minor version behind (e.g., installed 2.1.x, latest 2.3.x)
- **Low priority:** Patch version behind (e.g., installed 2.1.0, latest 2.1.3)

Also flag any packages pinned to versions more than 1 year behind the latest release.
You can estimate this from major/minor version gaps — you don't need to look up exact
release dates.

Store results for the summary in Step 7.

---

## Step 4 — Check for known vulnerabilities

First, check if `pip-audit` is installed:

```bash
pip show pip-audit
```

**If pip-audit is available:**

```bash
pip-audit
```

Capture the output. Flag any packages with known CVEs — note the CVE identifier, the
affected package, and the fixed version if one exists.

**If pip-audit is NOT available:**

Attempt to install it using the install-package skill:
1. Read `.claude/skills/install-package/SKILL.md`
2. Install `pip-audit`
3. Return here and run `pip-audit`

**If pip-audit can't be installed (install fails):**

Fall back to dependency conflict checking only:

```bash
pip check
```

Note in the results that vulnerability scanning was skipped because pip-audit is
unavailable. This is a degraded audit — recommend installing pip-audit for future runs.

---

## Step 5 — Check for unused dependencies

Compare `requirements.txt` entries against actual imports in the codebase.

For each package listed in `requirements.txt`:

1. Determine the import name (often the same as the package name, but not always —
   e.g., `Pillow` imports as `PIL`, `python-dateutil` imports as `dateutil`)
2. Search the codebase for import statements:
   ```bash
   grep -r "import <module_name>" --include="*.py" .
   grep -r "from <module_name>" --include="*.py" .
   ```
3. Exclude `.venv/`, `venv/`, `__pycache__/`, and `.git/` from the search

**Flag as "potentially unused"** any package in requirements.txt with zero import matches.

Add a note that some flagged packages may be indirect dependencies (required by other
packages but never directly imported) or used via entry points, plugins, or dynamic
imports. Do not recommend removing them without human review.

---

## Step 6 — Check for dependency conflicts

Run:

```bash
pip check
```

This reports packages with incompatible version requirements. Flag any conflicts found,
noting which packages are involved and what versions are expected vs. installed.

---

## Step 7 — Write findings to ENVIRONMENT.md

Open `ENVIRONMENT.md`. If it doesn't exist, create it with a minimal header first.

Add or **replace** a `## Dependency Audit` section (if a previous audit section exists,
replace it entirely so there's only one — keep the latest results):

```markdown
## Dependency Audit

### Audit — YYYY-MM-DD
- **Outdated:** N packages (X high/major, Y medium/minor, Z low/patch)
  - [list each: package installed_version -> latest_version (priority)]
- **Vulnerabilities:** N found
  - [list each: package CVE-ID — fixed in version X.Y.Z]
  - (or "pip-audit not available — vulnerability scan skipped")
- **Unused:** N potentially unused
  - [list each: package — no imports found (may be indirect dep)]
- **Conflicts:** N found
  - [list each: package requires X but Y is installed]
- **Action items:**
  - [specific recommendations, e.g., "Upgrade requests from 2.28.0 to 2.31.0 (security fix)"]
  - [e.g., "Review potentially unused packages before removing"]
  - [e.g., "Resolve conflict between X and Y by pinning Z to version A.B.C"]
```

If everything is clean (no outdated, no vulnerabilities, no unused, no conflicts), write:

```markdown
## Dependency Audit

### Audit — YYYY-MM-DD
- **Status:** All clear. No issues found.
```

---

## Step 8 — Nighttime logging

If running in nighttime mode (check `.claude/active_mode.md` for mode), also append to
`DaytimeNighttimeHandOff/nighttime.log`:

```
[<ISO timestamp>] AUDIT dependencies — <N> outdated, <N> vulnerabilities, <N> conflicts
```

If the audit found high-priority issues (vulnerabilities or major version gaps), also
note them:

```
[<ISO timestamp>] AUDIT WARNING — <brief description of critical finding>
```

---

## Step 9 — Report summary

Report a concise summary to the calling context:

> "Dependency audit complete. N outdated (X high, Y medium, Z low), N vulnerabilities,
> N potentially unused, N conflicts. Details written to ENVIRONMENT.md."

Or if clean:

> "Dependency audit complete. No issues found. Logged to ENVIRONMENT.md."

---

## Done

Return to the calling context.
