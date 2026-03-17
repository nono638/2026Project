"""Tests for _claude_sandbox_setup/scripts/nightrun_helper.py.

Covers model config resolution, fallback promotion, session-aware
timestamp comparison, and summary output formatting.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from datetime import datetime, timezone, timedelta

import pytest

# ---------------------------------------------------------------------------
# Import the helper module from a non-package path using importlib
# ---------------------------------------------------------------------------

_HELPER_PATH = os.path.join(
    os.path.dirname(__file__),
    os.pardir,
    "_claude_sandbox_setup",
    "scripts",
    "nightrun_helper.py",
)
_HELPER_PATH = os.path.normpath(_HELPER_PATH)

spec = importlib.util.spec_from_file_location("nightrun_helper", _HELPER_PATH)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def setup_dir(tmp_path):
    """Return a temporary directory acting as _claude_sandbox_setup/."""
    return str(tmp_path)


def _write_config(setup_dir: str, data: dict) -> str:
    """Write a model_config.json into setup_dir."""
    path = os.path.join(setup_dir, helper.CONFIG_FILENAME)
    with open(path, "w") as f:
        json.dump(data, f)
    return path


def _write_fallback(setup_dir: str, data: dict) -> str:
    """Write a .model_defaults file into setup_dir."""
    path = os.path.join(setup_dir, helper.FALLBACK_FILENAME)
    with open(path, "w") as f:
        json.dump(data, f)
    return path


def _write_tracker(tmp_path, tasks: list[dict]) -> str:
    """Write a tracker.json and return its path."""
    path = str(tmp_path / "tracker.json")
    with open(path, "w") as f:
        json.dump(tasks, f)
    return path


# ===================================================================
# resolve_model() — 4-level fallback chain
# ===================================================================

class TestResolveModel:
    """Test the 4-level fallback: config -> fallback -> env -> hardcoded."""

    def test_level1_config_file(self, setup_dir, capsys):
        """Config file present and valid — should use it."""
        _write_config(setup_dir, {
            "night": {"model": "claude-sonnet-4-20250514", "effort": "low"},
        })
        helper.resolve_model(setup_dir, "night")
        out = capsys.readouterr().out
        assert "MODEL=claude-sonnet-4-20250514" in out
        assert "EFFORT=low" in out
        assert "SOURCE=config" in out

    def test_level1_config_day_mode(self, setup_dir, capsys):
        """Config file has day entry — resolve for day mode."""
        _write_config(setup_dir, {
            "day": {"model": "claude-opus-4-6", "effort": "high"},
            "night": {"model": "claude-sonnet-4-20250514", "effort": "medium"},
        })
        helper.resolve_model(setup_dir, "day")
        out = capsys.readouterr().out
        assert "MODEL=claude-opus-4-6" in out
        assert "EFFORT=high" in out
        assert "SOURCE=config" in out

    def test_level2_fallback_file(self, setup_dir, capsys):
        """No config file — should fall through to fallback file."""
        _write_fallback(setup_dir, {
            "night": {"model": "claude-sonnet-4-20250514", "effort": "medium"},
        })
        helper.resolve_model(setup_dir, "night")
        out = capsys.readouterr().out
        assert "MODEL=claude-sonnet-4-20250514" in out
        assert "EFFORT=medium" in out
        assert "SOURCE=fallback" in out

    def test_level3_env_vars(self, setup_dir, capsys):
        """No config or fallback — should use env vars when both provided."""
        helper.resolve_model(setup_dir, "night",
                             env_model="claude-haiku-35", env_effort="low")
        out = capsys.readouterr().out
        assert "MODEL=claude-haiku-35" in out
        assert "EFFORT=low" in out
        assert "SOURCE=env" in out

    def test_level3_env_requires_both(self, setup_dir, capsys):
        """Env vars only used when BOTH model and effort are provided."""
        # Only model, no effort — should fall through to hardcoded
        helper.resolve_model(setup_dir, "night",
                             env_model="claude-haiku-35", env_effort=None)
        out = capsys.readouterr().out
        assert "SOURCE=hardcoded" in out

    def test_level4_hardcoded_defaults_night(self, setup_dir, capsys):
        """Nothing available — falls to hardcoded defaults for night mode."""
        helper.resolve_model(setup_dir, "night")
        out = capsys.readouterr().out
        assert f"MODEL={helper.HARDCODED_DEFAULTS['night']['model']}" in out
        assert f"EFFORT={helper.HARDCODED_DEFAULTS['night']['effort']}" in out
        assert "SOURCE=hardcoded" in out

    def test_level4_hardcoded_defaults_day(self, setup_dir, capsys):
        """Nothing available — falls to hardcoded defaults for day mode."""
        helper.resolve_model(setup_dir, "day")
        out = capsys.readouterr().out
        assert f"MODEL={helper.HARDCODED_DEFAULTS['day']['model']}" in out
        assert f"EFFORT={helper.HARDCODED_DEFAULTS['day']['effort']}" in out
        assert "SOURCE=hardcoded" in out

    def test_level4_unknown_mode_uses_night_defaults(self, setup_dir, capsys):
        """Unknown mode falls back to night hardcoded defaults."""
        helper.resolve_model(setup_dir, "unknown_mode")
        out = capsys.readouterr().out
        assert f"MODEL={helper.HARDCODED_DEFAULTS['night']['model']}" in out
        assert "SOURCE=hardcoded" in out

    def test_config_overrides_fallback(self, setup_dir, capsys):
        """Config file takes priority even when fallback file exists."""
        _write_config(setup_dir, {
            "night": {"model": "config-model", "effort": "high"},
        })
        _write_fallback(setup_dir, {
            "night": {"model": "fallback-model", "effort": "low"},
        })
        helper.resolve_model(setup_dir, "night")
        out = capsys.readouterr().out
        assert "MODEL=config-model" in out
        assert "SOURCE=config" in out

    def test_corrupt_config_falls_to_fallback(self, setup_dir, capsys):
        """Corrupt config file should fall through to fallback."""
        config_path = os.path.join(setup_dir, helper.CONFIG_FILENAME)
        with open(config_path, "w") as f:
            f.write("NOT VALID JSON {{{")
        _write_fallback(setup_dir, {
            "night": {"model": "fallback-model", "effort": "medium"},
        })
        helper.resolve_model(setup_dir, "night")
        out = capsys.readouterr().out
        assert "MODEL=fallback-model" in out
        assert "SOURCE=fallback" in out

    def test_config_missing_mode_falls_to_fallback(self, setup_dir, capsys):
        """Config exists but doesn't have the requested mode."""
        _write_config(setup_dir, {
            "day": {"model": "day-model", "effort": "high"},
            # no "night" entry
        })
        _write_fallback(setup_dir, {
            "night": {"model": "fallback-night", "effort": "medium"},
        })
        helper.resolve_model(setup_dir, "night")
        out = capsys.readouterr().out
        assert "MODEL=fallback-night" in out
        assert "SOURCE=fallback" in out

    def test_config_incomplete_entry_falls_through(self, setup_dir, capsys):
        """Config entry missing 'effort' key should fall through."""
        _write_config(setup_dir, {
            "night": {"model": "some-model"},  # no effort
        })
        helper.resolve_model(setup_dir, "night")
        out = capsys.readouterr().out
        assert "SOURCE=hardcoded" in out


# ===================================================================
# promote_model() — writing the fallback file
# ===================================================================

class TestPromoteModel:
    """Test fallback file updates after successful sessions."""

    def test_creates_fallback_when_config_matches(self, setup_dir, capsys):
        """Promote writes fallback when config entry matches the model/effort."""
        _write_config(setup_dir, {
            "night": {"model": "claude-opus-4-6", "effort": "medium"},
        })
        helper.promote_model(setup_dir, "claude-opus-4-6", "medium")
        out = capsys.readouterr().out
        assert "Fallback updated" in out

        # Verify fallback file contents
        fallback_path = os.path.join(setup_dir, helper.FALLBACK_FILENAME)
        with open(fallback_path) as f:
            data = json.load(f)
        assert data["night"]["model"] == "claude-opus-4-6"
        assert data["night"]["effort"] == "medium"

    def test_no_update_when_already_current(self, setup_dir, capsys):
        """If fallback already matches config, no write occurs."""
        _write_config(setup_dir, {
            "night": {"model": "claude-opus-4-6", "effort": "medium"},
        })
        _write_fallback(setup_dir, {
            "night": {"model": "claude-opus-4-6", "effort": "medium"},
        })
        helper.promote_model(setup_dir, "claude-opus-4-6", "medium")
        out = capsys.readouterr().out
        assert "already current" in out

    def test_no_update_when_config_doesnt_match(self, setup_dir, capsys):
        """If no config mode matches the given model/effort, no update."""
        _write_config(setup_dir, {
            "night": {"model": "different-model", "effort": "high"},
        })
        helper.promote_model(setup_dir, "claude-opus-4-6", "medium")
        out = capsys.readouterr().out
        assert "already current" in out

    def test_preserves_other_mode(self, setup_dir, capsys):
        """Updating night fallback should preserve existing day entry."""
        _write_config(setup_dir, {
            "night": {"model": "claude-opus-4-6", "effort": "medium"},
        })
        _write_fallback(setup_dir, {
            "day": {"model": "day-model", "effort": "high"},
        })
        helper.promote_model(setup_dir, "claude-opus-4-6", "medium")

        fallback_path = os.path.join(setup_dir, helper.FALLBACK_FILENAME)
        with open(fallback_path) as f:
            data = json.load(f)
        assert data["day"]["model"] == "day-model"
        assert data["night"]["model"] == "claude-opus-4-6"

    def test_no_config_file_means_no_update(self, setup_dir, capsys):
        """Without a config file, promote can't match any mode — no update."""
        helper.promote_model(setup_dir, "claude-opus-4-6", "medium")
        out = capsys.readouterr().out
        assert "already current" in out

    def test_updates_both_modes_if_both_match(self, setup_dir, capsys):
        """If both day and night config use the same model/effort, update both."""
        _write_config(setup_dir, {
            "day": {"model": "claude-opus-4-6", "effort": "high"},
            "night": {"model": "claude-opus-4-6", "effort": "high"},
        })
        helper.promote_model(setup_dir, "claude-opus-4-6", "high")

        fallback_path = os.path.join(setup_dir, helper.FALLBACK_FILENAME)
        with open(fallback_path) as f:
            data = json.load(f)
        assert data["day"]["model"] == "claude-opus-4-6"
        assert data["night"]["model"] == "claude-opus-4-6"


# ===================================================================
# _completed_this_session() — session-aware timestamp comparison
# ===================================================================

class TestCompletedThisSession:
    """Test the session boundary logic for splitting done tasks."""

    def test_no_session_start_returns_true(self):
        """When session_start is None, all tasks count as 'this session'."""
        task = {"nighttime_completed": "2026-01-15T03:00:00+00:00"}
        assert helper._completed_this_session(task, None) is True

    def test_completed_after_session_start(self):
        """Task completed after session start — belongs to this session."""
        session = datetime(2026, 1, 15, 2, 0, 0, tzinfo=timezone.utc)
        task = {"nighttime_completed": "2026-01-15T03:00:00+00:00"}
        assert helper._completed_this_session(task, session) is True

    def test_completed_before_session_start(self):
        """Task completed before session start — previously completed."""
        session = datetime(2026, 1, 15, 4, 0, 0, tzinfo=timezone.utc)
        task = {"nighttime_completed": "2026-01-15T03:00:00+00:00"}
        assert helper._completed_this_session(task, session) is False

    def test_completed_at_exact_session_start(self):
        """Completed at exactly session start — counts as this session (>=)."""
        session = datetime(2026, 1, 15, 3, 0, 0, tzinfo=timezone.utc)
        task = {"nighttime_completed": "2026-01-15T03:00:00+00:00"}
        assert helper._completed_this_session(task, session) is True

    def test_no_completed_timestamp(self):
        """Task has no nighttime_completed field — returns False."""
        session = datetime(2026, 1, 15, 2, 0, 0, tzinfo=timezone.utc)
        task = {"status": "done"}
        assert helper._completed_this_session(task, session) is False

    def test_invalid_completed_timestamp(self):
        """Garbage timestamp — returns False gracefully."""
        session = datetime(2026, 1, 15, 2, 0, 0, tzinfo=timezone.utc)
        task = {"nighttime_completed": "not-a-date"}
        assert helper._completed_this_session(task, session) is False

    def test_naive_timestamps_get_utc(self):
        """Naive timestamps (no tz) should be treated as UTC."""
        session = datetime(2026, 1, 15, 2, 0, 0)  # naive
        task = {"nighttime_completed": "2026-01-15T03:00:00"}  # naive
        assert helper._completed_this_session(task, session) is True

    def test_mixed_tz_aware_and_naive(self):
        """Aware session_start, naive completed timestamp."""
        session = datetime(2026, 1, 15, 2, 0, 0, tzinfo=timezone.utc)
        task = {"nighttime_completed": "2026-01-15T03:00:00"}  # naive
        assert helper._completed_this_session(task, session) is True


# ===================================================================
# show_summary() — tonight vs previously completed split
# ===================================================================

class TestShowSummary:
    """Test summary output formatting and session splitting."""

    def test_basic_summary_no_session_start(self, tmp_path, capsys):
        """Without session_start, all done tasks are 'tonight'."""
        tasks = [
            {"task_id": "task-001", "status": "done",
             "description": "First task", "branch": "night/task-001",
             "nighttime_completed": "2026-01-15T03:00:00+00:00"},
            {"task_id": "task-002", "status": "todo",
             "description": "Second task"},
        ]
        path = _write_tracker(tmp_path, tasks)
        helper.show_summary(path)
        out = capsys.readouterr().out
        assert "Done tonight: 1" in out
        assert "Todo:    1" in out
        assert "task-001" in out

    def test_split_tonight_vs_previously(self, tmp_path, capsys):
        """With session_start, done tasks split into tonight vs previously."""
        session_start = "2026-01-15T02:30:00+00:00"
        tasks = [
            {"task_id": "task-001", "status": "done",
             "description": "Old task", "branch": "night/task-001",
             "nighttime_completed": "2026-01-14T22:00:00+00:00"},
            {"task_id": "task-002", "status": "done",
             "description": "New task", "branch": "night/task-002",
             "nighttime_completed": "2026-01-15T03:00:00+00:00"},
        ]
        path = _write_tracker(tmp_path, tasks)
        helper.show_summary(path, session_start)
        out = capsys.readouterr().out
        assert "Done tonight: 1" in out
        assert "Previously:   1" in out
        assert "Completed this session:" in out
        assert "Previously completed:" in out
        assert "task-002" in out
        assert "task-001" in out

    def test_skipped_tasks_shown(self, tmp_path, capsys):
        """Skipped tasks appear in output."""
        tasks = [
            {"task_id": "task-001", "status": "skipped",
             "nighttime_comments": "dependency missing"},
        ]
        path = _write_tracker(tmp_path, tasks)
        helper.show_summary(path)
        out = capsys.readouterr().out
        assert "Skipped: 1" in out
        assert "dependency missing" in out

    def test_blocked_tasks_shown(self, tmp_path, capsys):
        """Blocked tasks appear with their reason."""
        tasks = [
            {"task_id": "task-001", "status": "blocked",
             "blocked_reason": "need API key"},
        ]
        path = _write_tracker(tmp_path, tasks)
        helper.show_summary(path)
        out = capsys.readouterr().out
        assert "Blocked: 1" in out
        assert "need API key" in out

    def test_flags_shown_for_tonight_tasks(self, tmp_path, capsys):
        """Flags on completed tasks are indicated in the output."""
        tasks = [
            {"task_id": "task-001", "status": "done",
             "description": "Flagged task", "branch": "night/task-001",
             "nighttime_completed": "2026-01-15T03:00:00+00:00",
             "flags": ["needs_review", "api_changed"]},
        ]
        path = _write_tracker(tmp_path, tasks)
        helper.show_summary(path)
        out = capsys.readouterr().out
        assert "2 flag(s)" in out

    def test_no_flags_no_flag_text(self, tmp_path, capsys):
        """Tasks without flags don't show flag text."""
        tasks = [
            {"task_id": "task-001", "status": "done",
             "description": "Clean task", "branch": "night/task-001",
             "nighttime_completed": "2026-01-15T03:00:00+00:00",
             "flags": []},
        ]
        path = _write_tracker(tmp_path, tasks)
        helper.show_summary(path)
        out = capsys.readouterr().out
        assert "flag" not in out

    def test_invalid_tracker_path(self, capsys):
        """Non-existent tracker prints error message."""
        helper.show_summary("/nonexistent/path/tracker.json")
        out = capsys.readouterr().out
        assert "Could not read tracker" in out

    def test_invalid_session_start_iso(self, tmp_path, capsys):
        """Invalid session_start string — all done tasks treated as tonight."""
        tasks = [
            {"task_id": "task-001", "status": "done",
             "description": "Task", "branch": "b",
             "nighttime_completed": "2026-01-15T03:00:00+00:00"},
        ]
        path = _write_tracker(tmp_path, tasks)
        helper.show_summary(path, "NOT-A-DATE")
        out = capsys.readouterr().out
        # session_start becomes None, so all done = tonight
        assert "Done tonight: 1" in out

    def test_empty_tracker(self, tmp_path, capsys):
        """Empty task list produces zero counts."""
        path = _write_tracker(tmp_path, [])
        helper.show_summary(path)
        out = capsys.readouterr().out
        assert "Done tonight: 0" in out
        assert "Skipped: 0" in out
        assert "Blocked: 0" in out
        assert "Todo:    0" in out

    def test_previously_section_hidden_when_none(self, tmp_path, capsys):
        """Previously line is omitted when there are no previously-done tasks."""
        tasks = [
            {"task_id": "task-001", "status": "done",
             "description": "Tonight's task", "branch": "b",
             "nighttime_completed": "2026-01-15T03:00:00+00:00"},
        ]
        path = _write_tracker(tmp_path, tasks)
        helper.show_summary(path, "2026-01-15T02:00:00+00:00")
        out = capsys.readouterr().out
        assert "Previously:" not in out


# ===================================================================
# count_pending() and show_pending() — tracker display helpers
# ===================================================================

class TestTrackerHelpers:
    """Test count_pending and show_pending output."""

    def test_count_pending(self, tmp_path, capsys):
        tasks = [
            {"task_id": "t1", "status": "todo"},
            {"task_id": "t2", "status": "in_progress"},
            {"task_id": "t3", "status": "done"},
            {"task_id": "t4", "status": "skipped"},
        ]
        path = _write_tracker(tmp_path, tasks)
        helper.count_pending(path)
        assert capsys.readouterr().out.strip() == "2"

    def test_count_pending_bad_path(self, capsys):
        helper.count_pending("/no/such/file")
        assert capsys.readouterr().out.strip() == "0"

    def test_show_pending(self, tmp_path, capsys):
        tasks = [
            {"task_id": "task-001", "status": "todo",
             "description": "Build feature"},
            {"task_id": "task-002", "status": "in_progress",
             "description": "Fix bug"},
            {"task_id": "task-003", "status": "done",
             "description": "Already done"},
        ]
        path = _write_tracker(tmp_path, tasks)
        helper.show_pending(path)
        out = capsys.readouterr().out
        assert "task-001: Build feature" in out
        assert "task-002: Fix bug (resuming)" in out
        assert "task-003" not in out

    def test_show_pending_bad_path(self, capsys):
        helper.show_pending("/no/such/file")
        assert "could not read tracker" in capsys.readouterr().out
