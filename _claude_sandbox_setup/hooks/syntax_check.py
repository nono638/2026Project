#!/usr/bin/env python3
"""
Syntax check hook for ClaudeDayNight.

PostToolUse hook that validates file syntax immediately after Edit/Write/MultiEdit
operations. Catches errors at the point of introduction rather than 50 edits later.

Based on lessons from autonomous operation:
  "Claude introduced syntax errors, continued editing 8 more files, wasted
  20 minutes before discovering breakage."
See: https://dev.to/yurukusa/10-claude-code-hooks-i-collected-from-108-hours-of-autonomous-operation-now-open-source-5633

Supported file types:
  - .py: Python syntax check via py_compile
  - .json: JSON parse validation
  - .yaml/.yml: YAML parse validation (if PyYAML available)

Exit codes:
  Exit 0 — always. This hook provides feedback via stderr but never blocks.
  Blocking on syntax errors would prevent Claude from fixing them.
"""
import json
import sys
import os
import subprocess


def check_python(filepath: str) -> str | None:
    """Check Python file for syntax errors. Returns error message or None."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            source = f.read()
        compile(source, filepath, "exec")
        return None
    except SyntaxError as e:
        return f"Python syntax error in {os.path.basename(filepath)} line {e.lineno}: {e.msg}"
    except Exception:
        return None  # Can't read file — not a syntax issue


def check_json(filepath: str) -> str | None:
    """Check JSON file for parse errors. Returns error message or None."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            json.load(f)
        return None
    except json.JSONDecodeError as e:
        return f"JSON parse error in {os.path.basename(filepath)} line {e.lineno}: {e.msg}"
    except Exception:
        return None


def check_yaml(filepath: str) -> str | None:
    """Check YAML file for parse errors. Returns error message or None."""
    try:
        import yaml
        with open(filepath, "r", encoding="utf-8") as f:
            yaml.safe_load(f)
        return None
    except ImportError:
        return None  # PyYAML not installed — skip silently
    except yaml.YAMLError as e:
        return f"YAML parse error in {os.path.basename(filepath)}: {e}"
    except Exception:
        return None


CHECKERS: dict[str, callable] = {
    ".py": check_python,
    ".json": check_json,
    ".yaml": check_yaml,
    ".yml": check_yaml,
}


def main() -> None:
    try:
        data = json.load(sys.stdin)
    except Exception:
        sys.exit(0)

    tool = data.get("tool_name", "")
    inp = data.get("tool_input", {})

    # Only check after file-modifying tools
    if tool not in ("Edit", "Write", "MultiEdit"):
        sys.exit(0)

    filepath = inp.get("file_path", "")
    if not filepath or not os.path.isfile(filepath):
        sys.exit(0)

    _, ext = os.path.splitext(filepath)
    checker = CHECKERS.get(ext.lower())
    if checker is None:
        sys.exit(0)

    error = checker(filepath)
    if error:
        # Print to stderr so Claude sees the feedback
        print(
            f"SYNTAX CHECK: {error}. Fix this before continuing.",
            file=sys.stderr,
        )

    sys.exit(0)


if __name__ == "__main__":
    main()
