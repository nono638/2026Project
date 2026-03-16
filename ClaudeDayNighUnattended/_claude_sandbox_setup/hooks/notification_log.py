#!/usr/bin/env python3
"""
Notification log hook for ClaudeDayNight.

Logs Claude Code notification events to DaytimeNighttimeHandOff/nighttime.log
so that post-session review can diagnose why a session stalled or behaved
unexpectedly.

Notification types Claude Code emits:
  - permission_prompt   — Claude needs a human to approve a tool call.
                          With defaultMode: "dontAsk" this should not occur
                          (unapproved tools are auto-denied), but logged
                          here as a belt-and-suspenders measure.
  - idle_prompt         — Claude finished and is waiting for the next prompt.
                          Logging this marks the natural end of a session.
  - elicitation_dialog  — Claude is asking a clarifying question.
                          Should be blocked by no_ask_human.py, but logged
                          here if it somehow fires anyway.

Why log to nighttime.log:
  The user is asleep during a nighttime session. Desktop notifications (toast,
  msgbox) are not useful. Appending to nighttime.log means the notification
  shows up in the same file the user reads during morning review — it's in
  context with the rest of the session narrative.

Exit codes:
  Always exits 0. This hook never blocks anything.
"""
import json
import sys
import os
from datetime import datetime


def main():
    try:
        data = json.load(sys.stdin)
    except Exception:
        sys.exit(0)

    cwd = data.get("cwd", os.getcwd())
    log_path = os.path.join(cwd, "DaytimeNighttimeHandOff", "nighttime.log")

    # Only log if the handoff directory exists — don't create it
    if not os.path.exists(os.path.dirname(log_path)):
        sys.exit(0)

    # Extract notification details — field names vary by notification type
    notification_type = data.get("notification_type", data.get("type", "unknown"))
    message = data.get("message", data.get("title", data.get("prompt", "")))

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] NOTIFICATION ({notification_type}): {message}\n"

    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(log_entry)
    except Exception:
        pass  # Never fail

    sys.exit(0)


if __name__ == "__main__":
    main()
