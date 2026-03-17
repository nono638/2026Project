#!/usr/bin/env python3
"""
Stop quality gate hook for ClaudeDayNight.

Runs when Claude is about to end the session (Stop event). Checks
DaytimeNighttimeHandOff/tracker.json and blocks the stop if there are
tasks still in "todo" status — indicating Claude stopped prematurely
before working through all pending tasks.

Why this hook exists:
  Claude sometimes decides a session is done before working through the
  full task queue (e.g., it encounters an ambiguous task, writes a plan,
  then declares itself finished). This hook catches those early exits and
  pushes Claude back to work.

What it checks:
  - "todo" tasks → blocks the stop (Claude should work through these)
  - "in_progress" tasks → allows the stop (usage cap hit mid-task;
    nightrun will relaunch and the task will be resumed via tracker.json)
  - "done" / "skipped" tasks → ignored

Exit codes:
  Exit 0 — allow the session to end normally.
  Exit 2 — block the stop; Claude Code injects the stderr message as a
            new turn, prompting Claude to continue working.

Edge case — max-turns hit with todo tasks:
  If Claude is at --max-turns when this hook fires, the injected turn
  (from exit 2) becomes an extra turn beyond the limit. Claude Code
  may or may not process it. Either way, nightrun will relaunch and
  tracker.json will show the remaining todo tasks for the next session.
  This is acceptable behaviour.
"""
import json
import sys
import os


def main():
    try:
        data = json.load(sys.stdin)
    except Exception:
        sys.exit(0)

    cwd = data.get("cwd", os.getcwd())

    # Only block during nighttime mode — daytime intentionally leaves todo tasks queued
    active_mode_path = os.path.join(cwd, ".claude", "active_mode.md")
    try:
        with open(active_mode_path, encoding="utf-8") as f:
            mode_content = f.read()
        if "Nighttime Mode" not in mode_content:
            sys.exit(0)  # Daytime mode — allow stop with pending tasks
    except Exception:
        sys.exit(0)  # Can't read mode file — don't block

    tracker_path = os.path.join(cwd, "DaytimeNighttimeHandOff", "tracker.json")

    if not os.path.exists(tracker_path):
        sys.exit(0)  # No tracker — nothing to gate on

    try:
        with open(tracker_path, encoding="utf-8") as f:
            tasks = json.load(f)
    except Exception:
        sys.exit(0)  # Can't read tracker — don't block

    todo = [t for t in tasks if t.get("status") == "todo"]

    if todo:
        ids = ", ".join(t.get("task_id", "unknown") for t in todo)
        print(
            f"STOP BLOCKED: {len(todo)} task(s) still pending: {ids}. "
            f"Work through all todo tasks before stopping. "
            f"Read tracker.json and continue with the nighttime task loop.",
            file=sys.stderr,
        )
        sys.exit(2)

    sys.exit(0)


if __name__ == "__main__":
    main()
