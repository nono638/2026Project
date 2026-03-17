#!/usr/bin/env python3
"""
No-ask-human hook for unattended Claude Code sessions.

Blocks AskUserQuestion tool calls so Claude doesn't stall waiting for
a human who isn't watching. Instead, Claude is told to decide on its own
and log the decision.

Exit code 2 = block the tool call.
The reason is printed to stderr and shown to Claude as feedback.

Design note — why we block questions instead of requiring human-in-the-loop:
  OWASP GenAI Top 10 (2025) ranks "human oversight" as a key mitigation for
  agentic AI risks and recommends human-in-the-loop review for high-risk
  operations. See: https://genai.owasp.org/llmrisk/llm01-prompt-injection/

  This hook deliberately diverges from that recommendation for unattended
  use cases. The trade-off: we accept the risk of Claude making a wrong
  autonomous decision in exchange for sessions that complete without stalling.
  CLAUDE.md instructs Claude to pick the simplest safe option and log all
  autonomous decisions so the user can review them afterward.

  If you are running attended sessions (human watching), remove this hook
  and allow AskUserQuestion so Claude can get clarification before acting.
"""
import json
import sys
import os
from datetime import datetime


def main():
    """Entry point. Block AskUserQuestion calls and log the question for later review.

    Reads the Claude Code hook payload from stdin. If the tool is not
    AskUserQuestion, exits immediately with 0 (allow). If it is, logs the
    question to CLAUDE_LOG.md and exits with 2 (block), instructing Claude
    to decide on its own rather than waiting for a human response.

    Fails open (exit 0) if stdin cannot be parsed, so a malformed payload
    doesn't permanently break unattended sessions.
    """
    try:
        data = json.load(sys.stdin)
    except Exception:
        # Can't read input — allow (fail open)
        sys.exit(0)

    tool = data.get("tool_name", "")
    inp = data.get("tool_input", {})
    cwd = data.get("cwd", os.getcwd())

    if tool != "AskUserQuestion":
        sys.exit(0)

    # Daytime mode: human is present, allow questions
    try:
        mode_file = os.path.join(cwd, ".claude", "active_mode.md")
        with open(mode_file, "r", encoding="utf-8") as f:
            if "Daytime Mode" in f.read(500):
                sys.exit(0)
    except Exception:
        pass  # Can't determine mode — assume nighttime, block

    # Extract the question Claude wanted to ask
    question = inp.get("question", "(no question text)")

    # Log the question to CLAUDE_LOG.md so the user can review later
    log_path = os.path.join(cwd, "CLAUDE_LOG.md")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"\n### Unanswered question ({timestamp})\n\n{question}\n"

    try:
        # Append to existing log or create new one
        if os.path.exists(log_path):
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(log_entry)
        else:
            with open(log_path, "w", encoding="utf-8") as f:
                f.write("# Claude Log\n")
                f.write(log_entry)
    except Exception:
        pass  # Don't fail the hook if logging fails

    # Block the question — tell Claude to decide on its own
    reason = (
        "BLOCKED: The user is not available. Do not ask questions. "
        "Pick the simplest reasonable option and proceed. "
        "Log your decision and reasoning in CLAUDE_LOG.md. "
        f"Your question was logged for the user to review later: {question}"
    )
    print(reason, file=sys.stderr)
    sys.exit(2)


if __name__ == "__main__":
    main()
