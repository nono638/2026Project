#!/usr/bin/env python3
"""
Context monitor hook for ClaudeDayNight.

PostToolUse hook that tracks tool call count as a proxy for context usage
and provides graduated warnings. This prevents the #1 autonomous session
failure mode: context depletion without warning, leading to hallucination
and lost work.

Based on lessons from 108+ hours of autonomous Claude Code operation:
  "Session hit 3% context with no warning. Lost an hour of work when
  Claude started hallucinating from memory compression."
See: https://dev.to/yurukusa/10-claude-code-hooks-i-collected-from-108-hours-of-autonomous-operation-now-open-source-5633

How it works:
  Counts tool calls in a session-specific temp file. At graduated
  thresholds, prints warnings to stderr that Claude sees as feedback.
  The warnings instruct Claude to commit progress and wrap up cleanly.

  Tool call count is an imperfect proxy for context usage, but it's
  the best signal available to hooks. Each tool call adds input/output
  to context, so more calls = fuller context.

Thresholds (tuned for Opus 4.6 with 1M context):
  - 400 calls: CAUTION — context getting large, commit progress frequently
  - 600 calls: WARNING — wrap up current task, commit, update tracker
  - 800 calls: CRITICAL — finish immediately, do not start new tasks or sweeps

Exit codes:
  Always exits 0. This hook never blocks anything.
"""
import json
import sys
import os
import tempfile


# Thresholds tuned for Opus 4.6 1M context window.
# With smaller context models, lower these significantly.
THRESHOLD_CAUTION = 400
THRESHOLD_WARNING = 600
THRESHOLD_CRITICAL = 800


def get_counter_path(session_id: str) -> str:
    """Return path to the session-specific counter file in the temp directory."""
    safe_id = session_id[-12:] if session_id else "unknown"
    return os.path.join(tempfile.gettempdir(), f"claude_context_monitor_{safe_id}.count")


def main() -> None:
    try:
        data = json.load(sys.stdin)
    except Exception:
        sys.exit(0)

    session_id = data.get("session_id", "")
    counter_path = get_counter_path(session_id)

    # Read and increment counter
    count = 0
    try:
        if os.path.exists(counter_path):
            with open(counter_path, "r") as f:
                count = int(f.read().strip())
    except Exception:
        count = 0

    count += 1

    try:
        with open(counter_path, "w") as f:
            f.write(str(count))
    except Exception:
        pass  # Don't fail if we can't write the counter

    # Graduated warnings — only print at threshold boundaries to avoid spam
    if count == THRESHOLD_CRITICAL:
        print(
            f"CONTEXT CRITICAL ({count} tool calls): "
            f"Context window is very full. Commit all work and update tracker.json NOW, "
            f"then run /compact to free space. Do NOT start new tasks or sweeps. "
            f"If you have already compacted recently, finish the current task and stop.",
            file=sys.stderr,
        )
    elif count == THRESHOLD_WARNING:
        print(
            f"CONTEXT WARNING ({count} tool calls): "
            f"Context is getting full. After finishing the current task, commit, "
            f"update tracker.json, and run /compact to reclaim space. "
            f"Skip remaining sweeps if any queued tasks are still incomplete.",
            file=sys.stderr,
        )
    elif count == THRESHOLD_CAUTION:
        print(
            f"CONTEXT CAUTION ({count} tool calls): "
            f"Context is growing large. Commit progress frequently and update "
            f"tracker.json after each task. Run /compact between tasks if context "
            f"feels sluggish — it preserves a summary of what you've done so far.",
            file=sys.stderr,
        )

    sys.exit(0)


if __name__ == "__main__":
    main()
