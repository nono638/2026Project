#!/usr/bin/env python3
"""
Compound command guard for Claude Code.
Blocks Bash commands that chain multiple operations with &&, ;, or ||.

Compound commands trigger interactive permission prompts in Claude Code when
the combined command doesn't match the allow list. This stalls both daytime
(user has to click approve) and nighttime (session hangs) execution. The fix
is to split compound commands into separate sequential Bash tool calls.

How it works:
  - Intercepts Bash tool calls via PreToolUse hook.
  - Scans the command string for &&, ;, or || outside of quotes and subshells.
  - Exit code 0 = allow (single command or piped command).
  - Exit code 2 = block (compound command detected — tells Claude to split).

What it allows:
  - Single commands (no chaining operators).
  - Piped commands (cmd | cmd) — these are a single logical operation.
  - Operators inside quoted strings ("echo 'a && b'" is fine).
  - Operators inside subshells/command substitutions ($(...), `...`).
  - Heredocs containing operators.

What it blocks:
  - cmd1 && cmd2
  - cmd1 ; cmd2
  - cmd1 || cmd2
  Even when the individual commands would be allowed by the permissions list.
"""
import json
import re
import sys


def strip_quoted_strings(cmd: str) -> str:
    """Remove content inside single and double quotes to avoid false positives.

    Handles escaped quotes within strings. Returns the command with quoted
    content replaced by placeholder text (no operators).

    Args:
        cmd: The raw command string.

    Returns:
        Command with quoted content replaced by 'QUOTED'.
    """
    # Remove escaped quotes first so they don't confuse the regex
    cleaned = cmd.replace("\\'", "").replace('\\"', "")
    # Replace double-quoted strings
    cleaned = re.sub(r'"[^"]*"', "QUOTED", cleaned)
    # Replace single-quoted strings
    cleaned = re.sub(r"'[^']*'", "QUOTED", cleaned)
    return cleaned


def strip_subshells(cmd: str) -> str:
    """Remove content inside $(...) and `...` subshells.

    Operators inside subshells are part of the subshell's logic, not
    command chaining at the top level.

    Args:
        cmd: The command string (already quote-stripped).

    Returns:
        Command with subshell content replaced by 'SUBSHELL'.
    """
    # Remove $(...) — handles one level of nesting
    cleaned = re.sub(r'\$\([^)]*\)', "SUBSHELL", cmd)
    # Remove backtick subshells
    cleaned = re.sub(r'`[^`]*`', "SUBSHELL", cleaned)
    return cleaned


def strip_heredocs(cmd: str) -> str:
    """Remove heredoc content (<<EOF ... EOF, <<'EOF' ... EOF).

    Heredocs can contain any text including operators. The delimiter
    and content should not be scanned for compound operators.

    Args:
        cmd: The command string.

    Returns:
        Command with heredoc content replaced by 'HEREDOC'.
    """
    # Match << followed by optional quotes around delimiter, then content until delimiter
    cleaned = re.sub(r"<<-?\s*['\"]?(\w+)['\"]?.*", "HEREDOC", cmd, flags=re.DOTALL)
    return cleaned


def has_compound_operators(cmd: str) -> str | None:
    """Check if a command contains top-level compound operators.

    Strips quoted strings, subshells, and heredocs first to avoid
    false positives on operators that appear inside data rather than
    as command separators.

    Args:
        cmd: The raw command string from the Bash tool call.

    Returns:
        The operator found (&&, ;, or ||) if compound, None if clean.
    """
    # Strip heredocs first (they can span multiple lines)
    cleaned = strip_heredocs(cmd)
    # Then strip quoted strings
    cleaned = strip_quoted_strings(cleaned)
    # Then strip subshells
    cleaned = strip_subshells(cleaned)

    # Now check for compound operators in what remains
    # && — command chaining (run second only if first succeeds)
    if "&&" in cleaned:
        return "&&"

    # || — command chaining (run second only if first fails)
    # But not inside [[ ... ]] test expressions
    cleaned_no_test = re.sub(r'\[\[.*?\]\]', "", cleaned)
    if "||" in cleaned_no_test:
        return "||"

    # ; — sequential execution
    # But not inside for/while/if/case shell constructs (do; done; then; fi; esac)
    # Simple heuristic: if ; appears and it's not part of a shell keyword pattern
    if ";" in cleaned:
        # Allow shell construct semicolons: do;, then;, else;, fi;, done;, esac;
        # Also allow ;; in case statements
        construct_pattern = re.compile(
            r';\s*;|'           # ;; in case statements
            r'\bdo\s*;|'        # for/while loops
            r'\bthen\s*;|'      # if/elif
            r'\belse\s*;|'      # else
            r';\s*\bthen\b|'    # ; then
            r';\s*\bdo\b|'      # ; do
            r';\s*\bdone\b|'    # ; done
            r';\s*\bfi\b|'      # ; fi
            r';\s*\besac\b'     # ; esac
        )
        # Remove construct-related semicolons
        no_constructs = construct_pattern.sub("", cleaned)
        # If semicolons remain, it's a compound command
        if ";" in no_constructs:
            return ";"

    return None


def main():
    """Entry point. Block compound Bash commands with a helpful message.

    Reads the Claude Code hook payload from stdin. If the tool is not Bash
    or the command is not compound, exits 0 (allow). If compound operators
    are found, exits 2 (block) with instructions to split into separate calls.
    """
    try:
        data = json.load(sys.stdin)
    except Exception:
        sys.exit(0)

    tool = data.get("tool_name", "")
    if tool != "Bash":
        sys.exit(0)

    inp = data.get("tool_input", {})
    cmd = inp.get("command", "")
    if not cmd:
        sys.exit(0)

    operator = has_compound_operators(cmd)
    if operator:
        print(
            f"BLOCKED: Compound command detected (operator: '{operator}'). "
            f"Split this into separate sequential Bash tool calls — one command per call. "
            f"For git: use 'git -C <path>' instead of 'cd <path> && git ...'. "
            f"For sequential operations: make multiple Bash calls in order. "
            f"Compound commands trigger permission prompts that stall execution.",
            file=sys.stderr,
        )
        sys.exit(2)

    sys.exit(0)


if __name__ == "__main__":
    main()
