#!/usr/bin/env python3
"""
Environment verification and setup for sandboxed Claude Code projects.

This script is run by Claude during sandbox setup (Step 0 in SETUP.md).
It ensures the project has a working Python environment before the hooks
and sandbox rules are installed.

Every step checks first, then acts only if needed. Running this script
multiple times is safe — it won't redo work that's already done.

What it does:
  1. Checks Python is on PATH and is 3.10+
  2. Finds existing venv (.venv/, venv/, env/, or custom name via pyvenv.cfg)
     Only creates .venv/ if NO existing venv is found
  3. Ensures pip is available in the venv
  4. Ensures pytest is installed in the venv
  5. Installs dependencies from requirements.txt/pyproject.toml
     (only on first run — skips if packages are already installed)
  6. Reports README and environment status for Claude to relay to user

Exit codes:
  0 = everything OK (environment ready)
  1 = something failed (error printed to stderr)
"""
import os
import subprocess
import sys


MIN_PYTHON = (3, 10)


def main():
    """Run all environment checks in order and print a summary.

    Execution order matters: later steps (pip, pytest, deps) depend on
    the venv being located/created first. Exits with code 1 on any
    unrecoverable error; exits 0 when the environment is ready.
    """
    project_dir = find_project_dir()
    print(f"Project directory: {project_dir}")

    check_python_version()
    check_git(project_dir)
    venv_dir, venv_status = ensure_venv(project_dir)
    venv_python = get_venv_python(venv_dir)
    ensure_pip(venv_python)
    ensure_pytest(venv_python)

    # Only install deps when we just created the venv — existing venvs
    # presumably already have their deps installed
    if venv_status == "created":
        install_existing_deps(venv_python, project_dir)

    readme_status = check_readme(project_dir, venv_dir)
    handoff_status = check_handoff_directory(project_dir)

    print_summary(project_dir, venv_dir, venv_status, venv_python, readme_status, handoff_status)


def find_project_dir():
    """Find the project root (parent of _claude_sandbox_setup/)."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    setup_dir = os.path.dirname(script_dir)   # _claude_sandbox_setup/
    project_dir = os.path.dirname(setup_dir)  # project root

    if os.path.basename(setup_dir) == "_claude_sandbox_setup":
        return project_dir

    return os.getcwd()


def check_python_version():
    """Verify the running Python interpreter meets the minimum version requirement.

    Calls fail() and exits with code 1 if the version is too old.
    """
    major, minor = sys.version_info[:2]
    print(f"Python version: {sys.version.split()[0]}")

    if (major, minor) < MIN_PYTHON:
        fail(
            f"Python {major}.{minor} is too old. "
            f"This sandbox requires Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+.\n"
            f"Install a newer Python from https://www.python.org/downloads/"
        )

    print(f"  OK (>= {MIN_PYTHON[0]}.{MIN_PYTHON[1]})")


def check_git(project_dir):
    """Verify git is available and the project is a git repository.

    The nighttime workflow requires git for branch-per-task, commits,
    and crash recovery. Warns (but does not fail) if git is missing
    since setup can proceed without it — but nightrun will fail.
    """
    import shutil
    print("git: checking...")

    if not shutil.which("git"):
        print("  WARNING: 'git' not found on PATH.")
        print("  The nighttime workflow requires git. Install it before running nightrun.")
        return

    result = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"],
        capture_output=True, text=True, cwd=project_dir
    )
    if result.returncode != 0:
        print("  WARNING: Not a git repository.")
        print("  Run 'git init && git add -A && git commit -m \"initial commit\"' before nightrun.")
    else:
        print("  OK (git repository)")


def ensure_venv(project_dir):
    """Find an existing venv or create one. Returns (venv_path, status).

    Searches common names first, then scans top-level dirs for pyvenv.cfg.
    Only creates .venv/ if nothing is found.
    """
    # Check common venv directory names first (fast path)
    for name in [".venv", "venv", "env"]:
        candidate = os.path.join(project_dir, name)
        if is_venv(candidate):
            print(f"Virtual environment: found existing at {name}/")
            return candidate, "existing"

    # Scan all top-level directories for pyvenv.cfg (catches custom names)
    try:
        for entry in os.scandir(project_dir):
            if entry.is_dir() and not entry.name.startswith("_"):
                if is_venv(entry.path):
                    print(f"Virtual environment: found existing at {entry.name}/")
                    return entry.path, "existing"
    except PermissionError:
        pass

    # Check for conda environment marker
    for conda_file in ["environment.yml", "environment.yaml"]:
        if os.path.isfile(os.path.join(project_dir, conda_file)):
            print(f"WARNING: Found {conda_file} but no venv.")
            print("  This project may use conda. Creating .venv/ anyway for hooks/pytest.")
            print("  You may want to use the conda environment instead.")
            break

    # No venv found — create one
    venv_dir = os.path.join(project_dir, ".venv")
    print("Virtual environment: none found, creating .venv/")

    result = subprocess.run(
        [sys.executable, "-m", "venv", venv_dir],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        fail(f"Failed to create virtual environment:\n{result.stderr}")

    if not is_venv(venv_dir):
        fail("Created .venv/ but it doesn't look like a valid virtual environment")

    print(f"  Created .venv/ at {venv_dir}")
    return venv_dir, "created"


def is_venv(path):
    """Check if a directory looks like a Python virtual environment."""
    if not os.path.isdir(path):
        return False
    if os.path.isfile(os.path.join(path, "pyvenv.cfg")):
        return True
    if os.path.isfile(os.path.join(path, "Scripts", "python.exe")):
        return True
    if os.path.isfile(os.path.join(path, "bin", "python")):
        return True
    return False


def get_venv_python(venv_dir):
    """Return the path to the Python executable inside the venv.

    Checks Windows path (Scripts/python.exe) before Unix path (bin/python).
    Calls fail() and exits with code 1 if neither is found.

    Args:
        venv_dir: Absolute path to the virtual environment root.

    Returns:
        Absolute path to the Python executable.
    """
    win_python = os.path.join(venv_dir, "Scripts", "python.exe")
    if os.path.isfile(win_python):
        return win_python

    unix_python = os.path.join(venv_dir, "bin", "python")
    if os.path.isfile(unix_python):
        return unix_python

    fail(f"Cannot find Python executable in venv at {venv_dir}")


def ensure_pip(venv_python):
    """Make sure pip is available. Only bootstraps if missing."""
    print("pip: checking...")

    result = subprocess.run(
        [venv_python, "-m", "pip", "--version"],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        print(f"  {result.stdout.strip()}")
        return

    # pip missing — bootstrap it
    print("  Not found, bootstrapping...")
    result = subprocess.run(
        [venv_python, "-m", "ensurepip", "--upgrade"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        fail(f"Failed to install pip:\n{result.stderr}")

    result = subprocess.run(
        [venv_python, "-m", "pip", "--version"],
        capture_output=True, text=True
    )
    print(f"  {result.stdout.strip()} (just bootstrapped)")


def ensure_pytest(venv_python):
    """Make sure pytest is installed. Only installs if missing."""
    print("pytest: checking...")

    result = subprocess.run(
        [venv_python, "-m", "pytest", "--version"],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        version_line = result.stdout.strip().split("\n")[0]
        print(f"  {version_line} (already installed)")
        return

    print("  Not found, installing...")
    result = subprocess.run(
        [venv_python, "-m", "pip", "install", "pytest", "--quiet"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        fail(f"Failed to install pytest:\n{result.stderr}")

    result = subprocess.run(
        [venv_python, "-m", "pytest", "--version"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        fail("Installed pytest but it's not working")

    version_line = result.stdout.strip().split("\n")[0]
    print(f"  {version_line} (just installed)")


def install_existing_deps(venv_python, project_dir):
    """Install dependencies from existing requirements files.

    Only called when the venv was just created — existing venvs are
    assumed to already have their dependencies.
    """
    req_txt = os.path.join(project_dir, "requirements.txt")
    if os.path.isfile(req_txt):
        print("requirements.txt: found, installing into new venv...")
        result = subprocess.run(
            [venv_python, "-m", "pip", "install", "-r", req_txt, "--quiet"],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"  WARNING: pip install -r requirements.txt failed:\n{result.stderr}")
        else:
            print("  Installed from requirements.txt")
        return

    pyproject = os.path.join(project_dir, "pyproject.toml")
    if os.path.isfile(pyproject):
        print("pyproject.toml: found, installing into new venv...")
        result = subprocess.run(
            [venv_python, "-m", "pip", "install", "-e", project_dir, "--quiet"],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print("  Note: pip install -e . didn't work (may not be a pip-installable project)")
        else:
            print("  Installed from pyproject.toml")
        return

    print("No requirements.txt or pyproject.toml found (skipping dependency install)")


def check_handoff_directory(project_dir):
    """Check if DaytimeNighttimeHandOff/ structure is present. Reports only, does not create.

    Creation is handled by SETUP.md Step 5 so Claude can do it with full context.

    Returns one of: "complete", "partial", "missing"
    """
    handoff = os.path.join(project_dir, "DaytimeNighttimeHandOff")
    if not os.path.isdir(handoff):
        print("DaytimeNighttimeHandOff/: not found (will be created by SETUP.md Step 5)")
        return "missing"

    issues = []
    for subdir in ["WrittenByDaytime", "WrittenByNighttime", "DaytimeOnly"]:
        if not os.path.isdir(os.path.join(handoff, subdir)):
            issues.append(f"{subdir}/ missing")
    if not os.path.isfile(os.path.join(handoff, "tracker.json")):
        issues.append("tracker.json missing")
    for f in ["DaytimeOnly/project_overview.md", "DaytimeOnly/ideas_log.md"]:
        if not os.path.isfile(os.path.join(handoff, f.replace("/", os.sep))):
            issues.append(f"{f} missing")

    if issues:
        print(f"DaytimeNighttimeHandOff/: exists but incomplete ({', '.join(issues)})")
        return "partial"

    print("DaytimeNighttimeHandOff/: complete")
    return "complete"


def check_readme(project_dir, venv_dir):
    """Check README status and report. Does NOT create or modify anything.

    Returns a status string and prints environment info that Claude
    should relay to the user (or include if it creates a README later).
    """
    rel_venv = os.path.relpath(venv_dir, project_dir)
    venv_name = os.path.basename(venv_dir)

    # Determine activation commands
    if os.name == "nt":
        activate_bash = f"source {rel_venv}/Scripts/activate"
        activate_cmd = f"{rel_venv}\\Scripts\\activate.bat"
        activate_ps = f"{rel_venv}\\Scripts\\Activate.ps1"
    else:
        activate_bash = f"source {rel_venv}/bin/activate"
        activate_cmd = activate_bash
        activate_ps = activate_bash

    # Check for existing README
    for name in ["README.md", "readme.md", "Readme.md", "README.rst",
                 "README.txt", "README"]:
        candidate = os.path.join(project_dir, name)
        if os.path.isfile(candidate):
            content = open(candidate, encoding="utf-8").read().lower()
            env_keywords = ["virtual environment", "venv", ".venv", "activate",
                            "pip install"]
            has_env_info = any(kw in content for kw in env_keywords)

            if has_env_info:
                print(f"README: {name} (has environment info)")
                return "has_env"
            else:
                print(f"README: {name} (no environment info)")
                return "missing_env"

    print("README: none found")
    return "no_readme"


def print_summary(project_dir, venv_dir, venv_status, venv_python, readme_status, handoff_status="unknown"):
    """Print a formatted summary of the environment for Claude to relay to the user.

    Includes Python version, venv location and status, activation commands
    for bash/cmd/PowerShell, and README advisory info.

    Args:
        project_dir: Absolute path to the project root.
        venv_dir: Absolute path to the virtual environment.
        venv_status: "existing" or "created" — reported in the summary.
        venv_python: Path to the venv's Python executable (unused in output
                     but kept for signature consistency with other functions).
        readme_status: One of "has_env", "missing_env", "no_readme".
    """
    rel_venv = os.path.relpath(venv_dir, project_dir)
    venv_name = os.path.basename(venv_dir)

    # Compute activation commands for the summary
    if os.name == "nt":
        activate_bash = f"source {rel_venv}/Scripts/activate"
        activate_cmd = f"{rel_venv}\\Scripts\\activate.bat"
        activate_ps = f"{rel_venv}\\Scripts\\Activate.ps1"
    else:
        activate_bash = f"source {rel_venv}/bin/activate"
        activate_cmd = activate_bash
        activate_ps = activate_bash

    print()
    print("=" * 60)
    print("ENVIRONMENT READY")
    print(f"  Python:     {sys.version.split()[0]}")
    print(f"  venv:       {rel_venv}/ ({venv_status})")
    print(f"  pytest:     installed")
    print()
    print("ACTIVATION COMMANDS (for README / user reference):")
    print(f"  bash:       {activate_bash}")
    print(f"  cmd:        {activate_cmd}")
    print(f"  powershell: {activate_ps}")
    print()

    if readme_status == "no_readme":
        print("README: No README found.")
        print("  Consider telling Claude to create one with environment info.")
    elif readme_status == "missing_env":
        print("README: Exists but has no environment/venv information.")
        print("  Consider telling Claude to add an environment section.")
    else:
        print("README: Has environment info. No action needed.")

    print()
    if handoff_status == "missing":
        print("HANDOFF DIR: DaytimeNighttimeHandOff/ not found.")
        print("  Run SETUP.md Step 5 to create it, or tell Claude to set it up.")
    elif handoff_status == "partial":
        print("HANDOFF DIR: DaytimeNighttimeHandOff/ exists but is incomplete.")
        print("  Run SETUP.md Step 5 to fill in missing pieces.")
    else:
        print("HANDOFF DIR: DaytimeNighttimeHandOff/ complete. No action needed.")

    print("=" * 60)


def fail(message):
    """Print an error message to stderr and exit with code 1.

    Used for unrecoverable errors where proceeding would leave the
    environment in a broken state (e.g. venv creation failed, pip missing).

    Args:
        message: Description of what went wrong. Printed with an ERROR: prefix.
    """
    print(f"ERROR: {message}", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
