"""Helper script for nightrun.bat — avoids inline Python in batch if-blocks.

Batch files interpret ')' inside if(...) blocks as block-closers, which breaks
inline Python that contains parentheses. This script extracts all tracker.json
reading logic into a standalone file that batch can call safely.

Usage:
    python nightrun_helper.py count <tracker_path>
    python nightrun_helper.py show <tracker_path>
    python nightrun_helper.py summary <tracker_path>
"""
from __future__ import annotations

import json
import sys


def count_pending(tracker_path: str) -> None:
    """Print the number of todo + in_progress tasks."""
    try:
        with open(tracker_path) as f:
            tasks = json.load(f)
        pending = [t for t in tasks if t.get("status") in ("todo", "in_progress")]
        print(len(pending))
    except Exception:
        print(0)


def show_pending(tracker_path: str) -> None:
    """Print a formatted list of pending tasks for the pre-launch summary."""
    try:
        with open(tracker_path) as f:
            tasks = json.load(f)
        for t in tasks:
            s = t.get("status", "")
            if s in ("in_progress", "todo"):
                tid = t.get("task_id", "???")
                desc = t.get("description", "no description")
                tag = " (resuming)" if s == "in_progress" else ""
                print(f"    {tid}: {desc}{tag}")
    except Exception:
        print("    (could not read tracker)")


def show_summary(tracker_path: str) -> None:
    """Print the end-of-run summary report."""
    try:
        with open(tracker_path) as f:
            tasks = json.load(f)
        done = [t for t in tasks if t.get("status") == "done"]
        skipped = [t for t in tasks if t.get("status") == "skipped"]
        blocked = [t for t in tasks if t.get("status") == "blocked"]
        todo = [t for t in tasks if t.get("status") == "todo"]
        print(f"  Done:    {len(done)}")
        print(f"  Skipped: {len(skipped)}")
        print(f"  Blocked: {len(blocked)}  <- needs your input")
        print(f"  Todo:    {len(todo)}     <- not started")
        print()
        if done:
            print("  Completed tasks:")
            for t in done:
                branch = t.get("branch", "no branch")
                flags = t.get("flags", [])
                flag_str = f"  [{len(flags)} flag(s)]" if flags else ""
                print(f"    {t['task_id']}: {t.get('description', '')} -- branch: {branch}{flag_str}")
        if skipped:
            print("  Skipped tasks:")
            for t in skipped:
                print(f"    {t['task_id']}: {t.get('nighttime_comments', 'see result.md')}")
        if blocked:
            print("  Blocked tasks (need your input before next run):")
            for t in blocked:
                print(f"    {t['task_id']}: {t.get('blocked_reason', 'see tracker.json')}")
    except Exception as e:
        print(f"  (Could not read tracker: {e})")


def main() -> None:
    """Dispatch to the requested command."""
    if len(sys.argv) < 3:
        print("Usage: nightrun_helper.py <count|show|summary> <tracker_path>", file=sys.stderr)
        sys.exit(1)

    cmd = sys.argv[1]
    tracker_path = sys.argv[2]

    if cmd == "count":
        count_pending(tracker_path)
    elif cmd == "show":
        show_pending(tracker_path)
    elif cmd == "summary":
        show_summary(tracker_path)
    else:
        print(f"Unknown command: {cmd}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
