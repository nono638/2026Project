#!/usr/bin/env python3
"""
Directory guard hook for Claude Code.
Blocks all file access outside the project directory.

This runs at the OS level before every tool call. Claude cannot bypass it.

How it works:
  - Claude Code sends a JSON blob to stdin describing what tool is about to run.
  - This script inspects file paths and command strings.
  - Exit code 0 = allow the action.
  - Exit code 2 = block the action (reason printed to stderr).

Why PreToolUse hooks instead of PermissionRequest hooks:
  PermissionRequest hooks exist but their deny decisions are silently ignored
  by Claude Code — the deny is not enforced. PreToolUse hooks with exit code 2
  are the correct mechanism for hard blocking.
  See: https://github.com/anthropics/claude-code/issues/19298
  (If this issue has been resolved in a newer version, PermissionRequest hooks
  may now work correctly and this file should be re-evaluated.)

What it catches:
  - Read/Edit/Write/Glob/Grep with absolute paths outside the project.
  - Symbolic links that resolve outside the project (CVE-2026-25724 mitigation).
  - Bash commands containing absolute paths outside the project.
  - Multi-level cd .. traversals that could leave the project.

What it does NOT catch:
  - Commands that dynamically construct paths at runtime (e.g., powershell
    string concatenation). This is an acknowledged limitation of command-string
    inspection: NVIDIA security research found that "application-level filters
    are security theater" against dynamic path construction because shell
    built-ins can bypass allowlists at runtime.
    See: https://developer.nvidia.com/blog/practical-security-guidance-for-sandboxing-agentic-workflows-and-managing-execution-risk/
    Mitigation: CLAUDE.md soft instructions tell Claude not to do this.
  - Paths embedded in Python/Node strings within commands (multiline bypass).
  - Child process behavior (e.g., pip writing to site-packages). This is
    expected — package managers and compilers need to write outside the project.

SETUP:
  On first run, Claude writes the resolved absolute project path into
  HARDCODED_PROJECT_DIR below. Once set, this is the single source of truth
  for what "inside the project" means — it does NOT rely on cwd from stdin.
"""
import json
import sys
import os
import re


# ============================================================================
# Claude writes the real absolute path here on first run.
# Do not change this manually unless you move the project directory.
#
# Why hardcode instead of always using cwd from stdin:
#   CVE-2025-59536 demonstrated that a malicious .claude/settings.json in an
#   untrusted repository can execute arbitrary shell commands when Claude Code
#   initialises, before any hooks run. A compromised initialisation step could
#   manipulate the working directory reported to subsequent hooks. Hardcoding
#   the project path at setup time makes this hook's boundary immutable.
#   See: https://research.checkpoint.com/2026/rce-and-api-token-exfiltration-through-claude-code-project-files-cve-2025-59536/
#   (If this CVE has been patched and Claude Code's initialisation is now
#   hardened, relying on cwd from stdin may be safe. Re-evaluate as needed.)
#
# When this is None, the hook falls back to cwd from stdin (less safe).
# ============================================================================
HARDCODED_PROJECT_DIR = r"C:\Users\noahc\Dropbox\NotWork\CUNYSPS\2026\Spring2026\Project"
# ============================================================================


def get_project_dir(cwd_from_stdin):
    """Return the authoritative project directory.

    Prefers HARDCODED_PROJECT_DIR (set during first-run setup) over the cwd
    supplied by Claude Code at runtime. The hardcoded path is more secure
    because it cannot be manipulated by a malicious command that changes
    the working directory before the hook runs.

    Args:
        cwd_from_stdin: The working directory reported by Claude Code in the
                        hook payload. Used only when HARDCODED_PROJECT_DIR is None.

    Returns:
        Normalised absolute path to the project directory.
    """
    if HARDCODED_PROJECT_DIR is not None:
        return os.path.normpath(HARDCODED_PROJECT_DIR)
    return os.path.normpath(cwd_from_stdin)


def is_within_project(filepath, project_dir):
    """Check if a resolved path is within (or equal to) the project directory.

    Resolves symlinks to their real target path before checking containment.
    This prevents a symlink bypass where a symlink inside the project points
    to a file outside it — without realpath(), os.path.abspath() alone would
    report the symlink as inside the project even though its target is outside.
    This class of vulnerability was identified during development of this sandbox
    and is documented as CVE-2026-25724 in docs/DangerousClaudeFlagReadme.md.
    (Note: CVE-2026-25724 was assigned during this project's development.
    If a public advisory exists, update this comment with the URL.)

    Comparison is case-insensitive to handle Windows filesystems correctly
    (C:\\Foo and C:\\foo are the same path on NTFS).

    Args:
        filepath: The path to check. May be absolute or relative.
        project_dir: The project root directory to check containment against.

    Returns:
        True if filepath resolves to a location inside project_dir, else False.
        Returns False on any OS error (e.g. permission denied during realpath).
    """
    try:
        # realpath resolves symlinks to canonical path, then normpath cleans it
        resolved = os.path.normpath(os.path.realpath(os.path.abspath(filepath)))
        proj_resolved = os.path.normpath(os.path.realpath(project_dir))
        proj_lower = proj_resolved.lower()
        resolved_lower = resolved.lower()
        return (resolved_lower == proj_lower or
                resolved_lower.startswith(proj_lower + os.sep))
    except Exception:
        return False


def check_file_tools(tool, inp, project_dir):
    """Check Read, Edit, Write, and MultiEdit tool calls for out-of-project paths.

    Handles both absolute paths (checked directly) and relative paths that
    may escape via traversal (e.g. ../../../../etc/passwd).

    Args:
        tool: The tool name string (e.g. "Read", "Write").
        inp: The tool_input dict from the hook payload.
        project_dir: Authoritative project root path.
    """
    file_path = inp.get("file_path", "")
    if not file_path:
        return

    # Absolute paths get checked directly
    if os.path.isabs(file_path):
        if not is_within_project(file_path, project_dir):
            block(f"{file_path} is outside project directory ({project_dir})")

    # Relative paths that try to escape (../../../../etc)
    resolved = os.path.normpath(os.path.join(project_dir, file_path))
    if not is_within_project(resolved, project_dir):
        block(f"{file_path} resolves to {resolved} which is outside project directory ({project_dir})")


def check_search_tools(tool, inp, project_dir):
    """Check Glob and Grep tool calls for out-of-project search paths.

    Args:
        tool: The tool name string (e.g. "Glob", "Grep").
        inp: The tool_input dict from the hook payload.
        project_dir: Authoritative project root path.
    """
    search_path = inp.get("path", "")
    if not search_path:
        return

    if os.path.isabs(search_path):
        if not is_within_project(search_path, project_dir):
            block(f"search path {search_path} is outside project directory ({project_dir})")

    resolved = os.path.normpath(os.path.join(project_dir, search_path))
    if not is_within_project(resolved, project_dir):
        block(f"search path {search_path} resolves outside project directory ({project_dir})")


def check_bash(inp, project_dir):
    """Check Bash tool calls for command strings that reference out-of-project paths.

    Inspects the command string for:
      - Windows absolute paths (C:\\..., D:/...) not inside the project
      - Unix-style paths to system directories (/etc, /usr, /home, /Users, etc.)
      - Long chains of parent traversals (3+ levels of ../) that may escape
      - Home directory references (~/)
      - Environment variables that resolve outside the project ($HOME, %APPDATA%, etc.)

    Note: This checks the command *string*, not runtime behaviour. Paths
    constructed dynamically at runtime (e.g. via string concatenation in a
    subprocess) are not caught here — those are covered by CLAUDE.md soft rules.
    This is a known, accepted limitation: NVIDIA security research found that
    command-string inspection cannot reliably block dynamic path construction.
    See: https://developer.nvidia.com/blog/practical-security-guidance-for-sandboxing-agentic-workflows-and-managing-execution-risk/

    Args:
        inp: The tool_input dict from the hook payload.
        project_dir: Authoritative project root path.
    """
    cmd = inp.get("command", "")
    if not cmd:
        return

    # Check Windows absolute paths (C:\..., D:/..., etc.)
    for match in re.findall(r'[A-Za-z]:[/\\][^\s"\';&|>`]*', cmd):
        clean = match.strip("\"'")
        if not is_within_project(clean, project_dir):
            block(f"command references path outside project: {match}")

    # Check Unix-style absolute paths to system directories
    if re.search(
        r'(?<!\w)/(?:etc|usr|tmp|home|root|var|opt|mnt|proc|sys|boot|dev'
        r'|windows|Windows|Users|Program\ Files|ProgramData)[/\w]*',
        cmd
    ):
        block("command references system path")

    # Check for multi-level parent traversals (../../..)
    # Single cd .. within the project is fine, but chains that could escape are blocked
    dotdot_count = len(re.findall(r'\.\.[\\/]', cmd))
    if dotdot_count >= 3:
        block(f"command has {dotdot_count} parent traversals — may escape project directory")

    # Check for home directory references
    if re.search(r'(?<!\w)[~][\\/]', cmd):
        block("command references home directory (~)")

    # Check for environment variables that typically resolve outside project
    if re.search(r'(\$HOME|\$USERPROFILE|%USERPROFILE%|%APPDATA%|%LOCALAPPDATA%|%PROGRAMFILES%|%SYSTEMROOT%|%WINDIR%)', cmd):
        block("command uses environment variable that resolves outside project")


def block(reason):
    """Block the tool call by exiting with code 2.

    Claude Code treats exit code 2 from a PreToolUse hook as a hard block —
    the tool call is cancelled and the reason string is shown to Claude as
    feedback so it understands why it was blocked.

    Args:
        reason: Human-readable explanation of why the action was blocked.
    """
    print(f"BLOCKED: {reason}", file=sys.stderr)
    sys.exit(2)


def main():
    """Entry point. Read the hook payload from stdin and run the appropriate check.

    Claude Code pipes a JSON payload to stdin before every tool call.
    Schema: {"tool_name": str, "tool_input": dict, "cwd": str, ...}

    Exits 0 (allow) or 2 (block). Never exits 1 — an unexpected error
    fails open (exit 0) so a bug in this script doesn't make Claude
    completely non-functional.
    """
    try:
        data = json.load(sys.stdin)
    except Exception:
        # If we can't read the input, allow (fail open so Claude isn't stuck)
        sys.exit(0)

    tool = data.get("tool_name", "")
    inp = data.get("tool_input", {})
    cwd = data.get("cwd", os.getcwd())
    project_dir = get_project_dir(cwd)

    if tool in ("Read", "Edit", "Write", "MultiEdit"):
        check_file_tools(tool, inp, project_dir)

    elif tool in ("Glob", "Grep"):
        check_search_tools(tool, inp, project_dir)

    elif tool == "Bash":
        check_bash(inp, project_dir)

    # All checks passed
    sys.exit(0)


if __name__ == "__main__":
    main()
