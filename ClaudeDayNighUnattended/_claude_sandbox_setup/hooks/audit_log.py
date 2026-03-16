#!/usr/bin/env python3
"""
Audit log hook for ClaudeDayNight.

PostToolUse hook that appends a one-line JSON record for every tool call to
DaytimeNighttimeHandOff/audit.jsonl. Runs asynchronously so it never delays
Claude's execution.

Why audit logging matters for unattended sessions:
  When something goes wrong overnight and you're debugging in the morning,
  git history tells you what changed but not the sequence of tool calls that
  led there. audit.jsonl fills that gap — it's a chronological trail of
  every file read, command run, and edit made during the session.

What gets logged:
  - Timestamp (UTC ISO 8601)
  - Tool name
  - Session ID (last 8 chars — enough to correlate entries, short enough to read)
  - Brief summary of what was accessed (file path, command preview, or pattern)

What does NOT get logged:
  - Full file contents (not useful, wastes disk)
  - Full command strings > 120 chars (truncated for readability)
  - Tool output (PostToolUse has it but we ignore it to keep logs lean)

Exit codes:
  Always exits 0. This hook never blocks anything.
"""
import json
import sys
import os
from datetime import datetime, timezone


def main():
    try:
        data = json.load(sys.stdin)
    except Exception:
        sys.exit(0)

    cwd = data.get("cwd", os.getcwd())
    log_path = os.path.join(cwd, "DaytimeNighttimeHandOff", "audit.jsonl")

    # Only log if the handoff directory exists — don't create it
    if not os.path.exists(os.path.dirname(log_path)):
        sys.exit(0)

    tool_name = data.get("tool_name", "")
    tool_input = data.get("tool_input", {})
    session_id = data.get("session_id", "")

    # Build a brief, readable summary of the tool's target
    summary = {}
    if "file_path" in tool_input:
        summary["path"] = tool_input["file_path"]
    elif "command" in tool_input:
        cmd = tool_input["command"]
        summary["command"] = cmd[:120] + "..." if len(cmd) > 120 else cmd
    elif "pattern" in tool_input:
        summary["pattern"] = tool_input["pattern"]
    elif "path" in tool_input:
        summary["path"] = tool_input["path"]
    elif "query" in tool_input:
        summary["query"] = tool_input["query"]

    entry = {
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "tool": tool_name,
        "session": session_id[-8:] if session_id else "",
        **summary,
    }

    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass  # Never fail — audit logging must not block Claude

    sys.exit(0)


if __name__ == "__main__":
    main()
