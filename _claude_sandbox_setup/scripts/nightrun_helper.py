"""Helper script for nightrun/dayrun — avoids inline Python in batch if-blocks.

Batch files interpret ')' inside if(...) blocks as block-closers, which breaks
inline Python that contains parentheses. This script extracts tracker reading,
model config resolution, and summary logic into a standalone file.

Usage:
    python nightrun_helper.py count <tracker_path>
    python nightrun_helper.py show <tracker_path>
    python nightrun_helper.py summary <tracker_path> [session_start_iso]
    python nightrun_helper.py timestamp
    python nightrun_helper.py model <setup_dir> <mode> [env_model] [env_effort]
    python nightrun_helper.py promote <setup_dir> <model> <effort>
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# Tracker helpers
# ---------------------------------------------------------------------------

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


def _completed_this_session(task: dict, session_start: datetime | None) -> bool:
    """Check if a task was completed during the current session."""
    if session_start is None:
        return True
    completed = task.get("nighttime_completed")
    if not completed:
        return False
    try:
        completed_dt = datetime.fromisoformat(completed)
        if completed_dt.tzinfo is None:
            completed_dt = completed_dt.replace(tzinfo=timezone.utc)
        if session_start.tzinfo is None:
            session_start = session_start.replace(tzinfo=timezone.utc)
        return completed_dt >= session_start
    except (ValueError, TypeError):
        return False


def show_summary(tracker_path: str, session_start_iso: str | None = None) -> None:
    """Print the end-of-run summary report.

    If session_start_iso is provided, splits done tasks into "completed
    this session" vs "previously completed".
    """
    try:
        with open(tracker_path) as f:
            tasks = json.load(f)

        session_start: datetime | None = None
        if session_start_iso:
            try:
                session_start = datetime.fromisoformat(session_start_iso)
            except ValueError:
                session_start = None

        done = [t for t in tasks if t.get("status") == "done"]
        skipped = [t for t in tasks if t.get("status") == "skipped"]
        blocked = [t for t in tasks if t.get("status") == "blocked"]
        todo = [t for t in tasks if t.get("status") == "todo"]

        done_tonight = [t for t in done if _completed_this_session(t, session_start)]
        done_previously = [t for t in done if not _completed_this_session(t, session_start)]

        print(f"  Done tonight: {len(done_tonight)}")
        if done_previously:
            print(f"  Previously:   {len(done_previously)}")
        print(f"  Skipped: {len(skipped)}")
        print(f"  Blocked: {len(blocked)}  <- needs your input")
        print(f"  Todo:    {len(todo)}     <- not started")
        print()
        if done_tonight:
            print("  Completed this session:")
            for t in done_tonight:
                branch = t.get("branch", "no branch")
                flags = t.get("flags", [])
                flag_str = f"  [{len(flags)} flag(s)]" if flags else ""
                print(f"    {t['task_id']}: {t.get('description', '')} -- branch: {branch}{flag_str}")
        if done_previously:
            print(f"  Previously completed: {len(done_previously)} task(s)")
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


def print_timestamp() -> None:
    """Print the current UTC time as an ISO timestamp for session tracking."""
    print(datetime.now(timezone.utc).isoformat())


# ---------------------------------------------------------------------------
# Model config helpers
# ---------------------------------------------------------------------------

# Hardcoded last-resort defaults — only used when both config file and
# fallback file are missing (e.g., fresh clone before first successful run).
HARDCODED_DEFAULTS = {
    "day":   {"model": "claude-opus-4-6", "effort": "high"},
    "night": {"model": "claude-opus-4-6", "effort": "medium"},
}

CONFIG_FILENAME = "model_config.json"
FALLBACK_FILENAME = ".model_defaults"


def _read_config(setup_dir: str, mode: str) -> dict | None:
    """Read model/effort from the user-editable config file.

    Returns {"model": ..., "effort": ...} or None if unreadable.
    """
    config_path = os.path.join(setup_dir, CONFIG_FILENAME)
    try:
        with open(config_path) as f:
            data = json.load(f)
        entry = data.get(mode)
        if entry and "model" in entry and "effort" in entry:
            return {"model": entry["model"], "effort": entry["effort"]}
    except Exception:
        pass
    return None


def _read_fallback(setup_dir: str, mode: str) -> dict | None:
    """Read model/effort from the auto-updated fallback file.

    The fallback file is a simple JSON dict written after each successful
    session. It provides the last-known-good model settings when the config
    file is missing or corrupted.

    Returns {"model": ..., "effort": ...} or None if unreadable.
    """
    fallback_path = os.path.join(setup_dir, FALLBACK_FILENAME)
    try:
        with open(fallback_path) as f:
            data = json.load(f)
        entry = data.get(mode)
        if entry and "model" in entry and "effort" in entry:
            return {"model": entry["model"], "effort": entry["effort"]}
    except Exception:
        pass
    return None


def resolve_model(setup_dir: str, mode: str,
                  env_model: str | None = None,
                  env_effort: str | None = None) -> None:
    """Resolve model and effort using the fallback chain, print as KEY=VALUE.

    Resolution order:
      1. Config file (_claude_sandbox_setup/model_config.json)
      2. Fallback file (_claude_sandbox_setup/.model_defaults)
      3. Environment variables (passed in as args from the shell script)
      4. Hardcoded defaults

    Prints two lines:
      MODEL=<model>
      EFFORT=<effort>
    """
    # 1. Config file (user-editable, checked into repo)
    result = _read_config(setup_dir, mode)
    if result:
        source = "config"
    else:
        # 2. Fallback file (auto-updated, gitignored)
        result = _read_fallback(setup_dir, mode)
        if result:
            source = "fallback"
        elif env_model and env_effort:
            # 3. Environment variables
            result = {"model": env_model, "effort": env_effort}
            source = "env"
        else:
            # 4. Hardcoded defaults
            result = HARDCODED_DEFAULTS.get(mode, HARDCODED_DEFAULTS["night"])
            source = "hardcoded"

    print(f"MODEL={result['model']}")
    print(f"EFFORT={result['effort']}")
    print(f"SOURCE={source}")


def promote_model(setup_dir: str, model: str, effort: str) -> None:
    """Update the fallback file after a successful session.

    Reads the current fallback file (if it exists), updates the relevant
    mode entry, and writes it back. This gives environment-level persistence
    so the fallback tracks the last successfully-used model.

    Called by the launcher scripts when a session completes with progress
    (at least one task done, no crashes).
    """
    fallback_path = os.path.join(setup_dir, FALLBACK_FILENAME)

    # Read existing fallback data (may have both day and night entries)
    data = {}
    try:
        with open(fallback_path) as f:
            data = json.load(f)
    except Exception:
        pass

    # Read config to determine which mode this model belongs to
    config = {}
    config_path = os.path.join(setup_dir, CONFIG_FILENAME)
    try:
        with open(config_path) as f:
            config = json.load(f)
    except Exception:
        pass

    # Determine which mode(s) use this model and update them
    updated = False
    for mode in ("day", "night"):
        config_entry = config.get(mode, {})
        if config_entry.get("model") == model and config_entry.get("effort") == effort:
            current = data.get(mode, {})
            if current.get("model") != model or current.get("effort") != effort:
                data[mode] = {"model": model, "effort": effort}
                updated = True

    if updated:
        try:
            with open(fallback_path, "w") as f:
                json.dump(data, f, indent=2)
                f.write("\n")
            print(f"Fallback updated: {fallback_path}")
        except Exception as e:
            print(f"Warning: could not update fallback: {e}", file=sys.stderr)
    else:
        print("Fallback already current.")


# ---------------------------------------------------------------------------
# Main dispatch
# ---------------------------------------------------------------------------

def main() -> None:
    """Dispatch to the requested command."""
    if len(sys.argv) < 2:
        print(
            "Usage: nightrun_helper.py <count|show|summary|timestamp|model|promote> [args...]",
            file=sys.stderr,
        )
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "timestamp":
        print_timestamp()
        return

    if cmd == "model":
        # model <setup_dir> <mode> [env_model] [env_effort]
        if len(sys.argv) < 4:
            print("Usage: nightrun_helper.py model <setup_dir> <day|night> [env_model] [env_effort]",
                  file=sys.stderr)
            sys.exit(1)
        setup_dir = sys.argv[2]
        mode = sys.argv[3]
        env_model = sys.argv[4] if len(sys.argv) > 4 else None
        env_effort = sys.argv[5] if len(sys.argv) > 5 else None
        resolve_model(setup_dir, mode, env_model, env_effort)
        return

    if cmd == "promote":
        # promote <setup_dir> <model> <effort>
        if len(sys.argv) < 5:
            print("Usage: nightrun_helper.py promote <setup_dir> <model> <effort>",
                  file=sys.stderr)
            sys.exit(1)
        promote_model(sys.argv[2], sys.argv[3], sys.argv[4])
        return

    if len(sys.argv) < 3:
        print("Usage: nightrun_helper.py <count|show|summary> <tracker_path> [session_start]",
              file=sys.stderr)
        sys.exit(1)

    tracker_path = sys.argv[2]

    if cmd == "count":
        count_pending(tracker_path)
    elif cmd == "show":
        show_pending(tracker_path)
    elif cmd == "summary":
        session_start = sys.argv[3] if len(sys.argv) > 3 else None
        show_summary(tracker_path, session_start)
    else:
        print(f"Unknown command: {cmd}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
