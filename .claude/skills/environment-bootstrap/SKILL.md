---
name: environment-bootstrap
description: Create or verify Python virtual environment, freeze requirements, write ENVIRONMENT.md
---

# Environment Bootstrap

Set up or verify the project's Python environment. Either agent can call this when a venv
is needed or environment documentation is missing.

---

## Step 1 — Assess current state

Check what exists:

| Check | Command |
|---|---|
| Venv exists? | Look for `.venv/`, `venv/`, or `env/` (also check for `pyvenv.cfg` inside each) |
| requirements.txt? | Check root directory |
| pyproject.toml? | Check root directory |
| ENVIRONMENT.md? | Check root directory |
| Python version? | `python --version` (also try `python3 --version`) |

Determine which scenario applies:

| Scenario | Venv | requirements.txt | ENVIRONMENT.md |
|---|---|---|---|
| A — Fresh setup | missing | missing | missing |
| B — Docs gap | exists | exists | missing |
| C — Freeze needed | exists | missing or empty | any |
| D — Install needed | missing | exists | any |
| E — Verify only | exists | exists | exists |

---

## Step 2 — Create venv if needed (scenarios A, D)

```
python -m venv .venv
```

If that fails, try `python3 -m venv .venv`.

Verify: check that `.venv/pyvenv.cfg` exists after creation.

If creation fails, log the error and stop — do not attempt workarounds.

---

## Step 3 — Activate and install

Activate the venv:
- **Windows (bash/git bash):** `source .venv/Scripts/activate`
- **Windows (cmd):** `.venv\Scripts\activate.bat`
- **Linux/macOS:** `source .venv/bin/activate`

Then install:
- If `requirements.txt` exists and is non-empty: `pip install -r requirements.txt`
- If `pyproject.toml` exists (and no requirements.txt): `pip install -e .`
- Check if `pytest` is available: `python -m pytest --version`
- If pytest is missing: `pip install pytest`

---

## Step 4 — Freeze requirements (scenarios A, C)

If `requirements.txt` is missing or empty and the venv has installed packages:

```
pip freeze > requirements.txt
```

If `requirements.txt` exists but may be incomplete, run `pip freeze` and compare.
Add any missing packages with their pinned versions. Do not remove existing pins
unless you know the package was uninstalled.

Every entry must have a pinned version: `package==X.Y.Z`

---

## Step 5 — Write ENVIRONMENT.md (if missing)

Only create this file if `ENVIRONMENT.md` does not already exist. If a `README.md` exists
and already has an environment section, skip this step.

Write `ENVIRONMENT.md` in the project root:

```markdown
# Environment Setup

## Python
- **Version:** <output of `python --version`>
- **Virtual environment:** `.venv/`

## Activation
| Shell | Command |
|---|---|
| bash (Git Bash / WSL) | `source .venv/Scripts/activate` (Windows) or `source .venv/bin/activate` (Linux/macOS) |
| cmd | `.venv\Scripts\activate.bat` |
| PowerShell | `.venv\Scripts\Activate.ps1` |

## Dependencies
- **Install all:** `pip install -r requirements.txt`
- **Add new package:** `pip install <package>` then `pip freeze > requirements.txt`
- **Pinning:** All versions are pinned in requirements.txt. Do not use unpinned installs.

## Notes
- Never install packages globally — always activate the venv first.
- <any other project-specific notes>
```

---

## Step 6 — Verify

Run these checks (all must pass):

1. `python -c "import sys; print(sys.executable)"` — must point to `.venv/`
2. `python -m pytest --version` — pytest must be available
3. If requirements.txt has entries, spot-check: `python -c "import <first-package>"`

If any check fails, log the issue clearly. Do not silently proceed with a broken environment.

---

## Step 7 — Log and return

If anything was created or fixed, log to `DaytimeNighttimeHandOff/nighttime.log`:
```
[<ISO timestamp>] ENV-BOOTSTRAP: <what was done>
```

Report to calling context:
> "Environment ready: Python X.Y, .venv/, N packages installed."

---

## Done

Return to the calling supplement.
