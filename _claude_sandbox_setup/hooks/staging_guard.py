#!/usr/bin/env python3
"""
Staging validation hook for Claude Code.
Intercepts `git commit` commands and validates what's staged before allowing the commit.

This runs as a PreToolUse hook on Bash tool calls. It only fires when the command
contains `git commit` and only enforces in nighttime mode.

How it works:
  - Claude Code sends a JSON blob to stdin describing the Bash command about to run.
  - If the command contains `git commit`, this script runs `git diff --cached --name-only`
    to inspect staged files and blocks the commit if any forbidden files are staged.
  - It also checks the audit log for recent `git add -A` or `git add .` usage.
  - Exit code 0 = allow the action.
  - Exit code 2 = block the action (reason printed to stderr).

What it catches:
  - Runtime/config files staged for commit (.claude/, audit.jsonl, nighttime.log, etc.)
  - tracker.json committed on non-main branches
  - Build artifacts (__pycache__/, *.pyc, *.bak, pip_output.txt)
  - Bulk staging commands (git add -A, git add .) detected via audit log

What it does NOT catch:
  - Files added after this hook runs but before git actually commits (unlikely in practice
    since Claude executes the commit command immediately after this hook passes).
"""
import json
import sys
import os
import subprocess
import re


# ============================================================================
# Configuration
# ============================================================================

# Files/patterns that must never be committed by the nighttime agent.
# These are runtime artifacts, not source code.
FORBIDDEN_PREFIXES = [
    ".claude/",
]

FORBIDDEN_EXACT = [
    "DaytimeNighttimeHandOff/audit.jsonl",
    "DaytimeNighttimeHandOff/nighttime.log",
    "pip_output.txt",
]

FORBIDDEN_EXTENSIONS = [
    ".bak",
    ".pyc",
]

FORBIDDEN_DIR_SEGMENTS = [
    "__pycache__",
]

# How many recent audit log entries to scan for bulk-add commands.
AUDIT_TAIL_COUNT = 5

# Hardcoded project directory — same pattern as directory_guard.py.
# Set during first-run setup. When None, falls back to cwd from stdin.
HARDCODED_PROJECT_DIR = r"C:\Users\noahc\Dropbox\NotWork\CUNYSPS\2026\Spring2026\Project"


# ============================================================================
# Validation functions (unit-testable)
# ============================================================================

def get_project_dir(cwd_from_stdin: str) -> str:
    """Return the authoritative project directory."""
    if HARDCODED_PROJECT_DIR is not None:
        return os.path.normpath(HARDCODED_PROJECT_DIR)
    return os.path.normpath(cwd_from_stdin)


def is_daytime_mode(cwd: str) -> bool:
    """Check if the current session is running in daytime (interactive) mode.

    Reads .claude/active_mode.md relative to the project directory. If the file
    contains 'Daytime Mode' in its header, this hook is disabled.
    """
    try:
        project_dir = get_project_dir(cwd)
        mode_file = os.path.join(project_dir, ".claude", "active_mode.md")
        with open(mode_file, "r", encoding="utf-8") as f:
            head = f.read(500)
        return "Daytime Mode" in head
    except Exception:
        return False


def is_forbidden_file(filepath: str) -> str | None:
    """Check if a staged file path is forbidden.

    Args:
        filepath: A relative path from git diff --cached --name-only.
                  Uses forward slashes (git normalises to this on all platforms).

    Returns:
        A reason string if forbidden, None if allowed.
    """
    # Normalise to forward slashes for consistent matching
    normalized = filepath.replace("\\", "/")

    # Check prefix-based rules (.claude/*)
    for prefix in FORBIDDEN_PREFIXES:
        if normalized.startswith(prefix):
            return f"'{filepath}' matches forbidden prefix '{prefix}'"

    # Check exact matches
    for exact in FORBIDDEN_EXACT:
        if normalized == exact:
            return f"'{filepath}' is a forbidden file"

    # Check extensions (.bak, .pyc)
    for ext in FORBIDDEN_EXTENSIONS:
        if normalized.endswith(ext):
            return f"'{filepath}' has forbidden extension '{ext}'"

    # Check directory segments (__pycache__)
    parts = normalized.split("/")
    for segment in FORBIDDEN_DIR_SEGMENTS:
        if segment in parts:
            return f"'{filepath}' contains forbidden directory '{segment}'"

    return None


def is_tracker_on_wrong_branch(staged_files: list[str], current_branch: str) -> str | None:
    """Check if tracker.json is staged on a non-main branch.

    tracker.json should only be committed on the main branch per the workflow.

    Args:
        staged_files: List of staged file paths from git diff --cached --name-only.
        current_branch: The current git branch name.

    Returns:
        A reason string if blocked, None if allowed.
    """
    tracker_path = "DaytimeNighttimeHandOff/tracker.json"
    for f in staged_files:
        normalized = f.replace("\\", "/")
        if normalized == tracker_path:
            if current_branch != "main":
                return (
                    f"tracker.json is staged but current branch is '{current_branch}'. "
                    f"tracker.json should only be committed on the main branch."
                )
    return None


def check_audit_log_for_bulk_add(project_dir: str) -> str | None:
    """Scan recent audit log entries for `git add -A` or `git add .` commands.

    These bulk-add commands stage everything indiscriminately, which can include
    forbidden files. The nighttime agent should use explicit `git add <files>`.

    Args:
        project_dir: The project root directory.

    Returns:
        A reason string if a bulk-add was found recently, None if clean.
    """
    audit_path = os.path.join(project_dir, "DaytimeNighttimeHandOff", "audit.jsonl")
    try:
        with open(audit_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except (FileNotFoundError, PermissionError):
        # No audit log = nothing to check
        return None

    # Check the last N entries
    recent = lines[-AUDIT_TAIL_COUNT:] if len(lines) >= AUDIT_TAIL_COUNT else lines

    bulk_add_pattern = re.compile(r'git\s+add\s+(-A|\.)\s*')
    for line in recent:
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue

        # The audit log stores the command in tool_input.command for Bash calls
        command = ""
        tool_input = entry.get("tool_input", {})
        if isinstance(tool_input, dict):
            command = tool_input.get("command", "")
        elif isinstance(tool_input, str):
            command = tool_input

        if bulk_add_pattern.search(command):
            return (
                "A recent `git add -A` or `git add .` was detected in the audit log. "
                "Use explicit `git add <file1> <file2> ...` instead of bulk-adding."
            )

    return None


def is_git_commit_command(command: str) -> bool:
    """Check if a bash command contains a git commit invocation.

    Args:
        command: The bash command string.

    Returns:
        True if the command contains `git commit`.
    """
    return bool(re.search(r'\bgit\s+commit\b', command))


# ============================================================================
# Hook entry point
# ============================================================================

def get_staged_files(project_dir: str) -> list[str]:
    """Run `git diff --cached --name-only` and return the list of staged files."""
    try:
        result = subprocess.run(
            ["git", "diff", "--cached", "--name-only"],
            capture_output=True, text=True, cwd=project_dir, timeout=10
        )
        if result.returncode != 0:
            return []
        return [f for f in result.stdout.strip().split("\n") if f]
    except Exception:
        return []


def get_current_branch(project_dir: str) -> str:
    """Get the current git branch name."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, cwd=project_dir, timeout=10
        )
        return result.stdout.strip()
    except Exception:
        return ""


def block(reason: str) -> None:
    """Block the tool call by exiting with code 2."""
    print(f"BLOCKED: {reason}", file=sys.stderr)
    sys.exit(2)


def main():
    """Entry point. Read the hook payload from stdin and validate staged files.

    Only fires on Bash tool calls containing `git commit`. In daytime mode
    or for non-commit commands, exits 0 immediately (allow).
    """
    try:
        data = json.load(sys.stdin)
    except Exception:
        # Can't read input — fail open
        sys.exit(0)

    tool = data.get("tool_name", "")
    inp = data.get("tool_input", {})
    cwd = data.get("cwd", os.getcwd())

    # Only intercept Bash tool calls
    if tool != "Bash":
        sys.exit(0)

    command = inp.get("command", "")

    # Only intercept git commit commands
    if not is_git_commit_command(command):
        sys.exit(0)

    # Daytime mode: human is present, skip validation
    if is_daytime_mode(cwd):
        sys.exit(0)

    project_dir = get_project_dir(cwd)

    # Collect all violations before blocking (report everything at once)
    violations = []

    # Check audit log for recent bulk-add commands
    bulk_add_reason = check_audit_log_for_bulk_add(project_dir)
    if bulk_add_reason:
        violations.append(bulk_add_reason)

    # Get staged files and check each one
    staged_files = get_staged_files(project_dir)

    for filepath in staged_files:
        reason = is_forbidden_file(filepath)
        if reason:
            violations.append(reason)

    # Check tracker.json branch rule
    current_branch = get_current_branch(project_dir)
    tracker_reason = is_tracker_on_wrong_branch(staged_files, current_branch)
    if tracker_reason:
        violations.append(tracker_reason)

    if violations:
        msg = "Staging validation failed. Fix these issues before committing:\n"
        for i, v in enumerate(violations, 1):
            msg += f"  {i}. {v}\n"
        msg += "\nTo fix: `git reset HEAD <file>` to unstage forbidden files, then commit again."
        block(msg)

    # All checks passed
    sys.exit(0)


if __name__ == "__main__":
    main()
