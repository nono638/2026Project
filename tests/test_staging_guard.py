"""Tests for the staging_guard hook validation functions.

Tests the core logic (forbidden file detection, tracker branch check,
audit log scanning, daytime bypass) without testing the full hook
stdin/stdout flow.
"""
import json
import os
import tempfile

import pytest

# Import the module under test from the sandbox setup hooks directory
import sys

# Add the hooks directory to the path so we can import staging_guard
HOOKS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "_claude_sandbox_setup",
    "hooks",
)
sys.path.insert(0, HOOKS_DIR)

import staging_guard


# ============================================================================
# is_forbidden_file
# ============================================================================

class TestIsForbiddenFile:
    """Tests for forbidden file detection."""

    def test_claude_settings_blocked(self):
        assert staging_guard.is_forbidden_file(".claude/settings.json") is not None

    def test_claude_hooks_blocked(self):
        assert staging_guard.is_forbidden_file(".claude/hooks/directory_guard.py") is not None

    def test_claude_active_mode_blocked(self):
        assert staging_guard.is_forbidden_file(".claude/active_mode.md") is not None

    def test_audit_jsonl_blocked(self):
        result = staging_guard.is_forbidden_file("DaytimeNighttimeHandOff/audit.jsonl")
        assert result is not None
        assert "forbidden file" in result

    def test_nighttime_log_blocked(self):
        result = staging_guard.is_forbidden_file("DaytimeNighttimeHandOff/nighttime.log")
        assert result is not None

    def test_pip_output_blocked(self):
        result = staging_guard.is_forbidden_file("pip_output.txt")
        assert result is not None

    def test_bak_file_blocked(self):
        result = staging_guard.is_forbidden_file("src/config.py.bak")
        assert result is not None
        assert ".bak" in result

    def test_pyc_file_blocked(self):
        result = staging_guard.is_forbidden_file("src/utils.pyc")
        assert result is not None
        assert ".pyc" in result

    def test_pycache_dir_blocked(self):
        result = staging_guard.is_forbidden_file("src/__pycache__/utils.cpython-311.pyc")
        assert result is not None
        assert "__pycache__" in result

    def test_normal_python_file_allowed(self):
        assert staging_guard.is_forbidden_file("src/main.py") is None

    def test_normal_test_file_allowed(self):
        assert staging_guard.is_forbidden_file("tests/test_main.py") is None

    def test_spec_file_allowed(self):
        assert staging_guard.is_forbidden_file(
            "DaytimeNighttimeHandOff/WrittenByNighttime/task-005/result.md"
        ) is None

    def test_tracker_json_allowed_here(self):
        """is_forbidden_file does not block tracker.json — that's a branch-level check."""
        assert staging_guard.is_forbidden_file("DaytimeNighttimeHandOff/tracker.json") is None

    def test_backslash_normalization(self):
        """Windows-style backslashes should still match."""
        result = staging_guard.is_forbidden_file(".claude\\hooks\\guard.py")
        assert result is not None


# ============================================================================
# is_tracker_on_wrong_branch
# ============================================================================

class TestTrackerBranchCheck:
    """Tests for tracker.json branch validation."""

    def test_tracker_on_main_allowed(self):
        staged = ["DaytimeNighttimeHandOff/tracker.json", "src/main.py"]
        assert staging_guard.is_tracker_on_wrong_branch(staged, "main") is None

    def test_tracker_on_feature_branch_blocked(self):
        staged = ["DaytimeNighttimeHandOff/tracker.json", "src/main.py"]
        result = staging_guard.is_tracker_on_wrong_branch(staged, "night/task-005-metalearner")
        assert result is not None
        assert "main" in result
        assert "night/task-005-metalearner" in result

    def test_no_tracker_on_feature_branch_allowed(self):
        staged = ["src/main.py", "src/utils.py"]
        assert staging_guard.is_tracker_on_wrong_branch(staged, "night/task-005") is None

    def test_empty_staged_files(self):
        assert staging_guard.is_tracker_on_wrong_branch([], "night/task-005") is None

    def test_tracker_backslash_normalization(self):
        staged = ["DaytimeNighttimeHandOff\\tracker.json"]
        result = staging_guard.is_tracker_on_wrong_branch(staged, "night/task-005")
        assert result is not None


# ============================================================================
# check_audit_log_for_bulk_add
# ============================================================================

class TestAuditLogBulkAdd:
    """Tests for audit log scanning for git add -A / git add . usage."""

    def _write_audit_log(self, tmpdir, entries):
        """Helper to write audit log entries."""
        handoff_dir = os.path.join(tmpdir, "DaytimeNighttimeHandOff")
        os.makedirs(handoff_dir, exist_ok=True)
        audit_path = os.path.join(handoff_dir, "audit.jsonl")
        with open(audit_path, "w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")
        return tmpdir

    def test_no_audit_log_passes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            assert staging_guard.check_audit_log_for_bulk_add(tmpdir) is None

    def test_clean_audit_log_passes(self):
        entries = [
            {"tool_input": {"command": "git add src/main.py"}},
            {"tool_input": {"command": "git add tests/test_main.py"}},
            {"tool_input": {"command": "git status"}},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_audit_log(tmpdir, entries)
            assert staging_guard.check_audit_log_for_bulk_add(tmpdir) is None

    def test_git_add_all_flag_blocked(self):
        entries = [
            {"tool_input": {"command": "git add src/main.py"}},
            {"tool_input": {"command": "git add -A"}},
            {"tool_input": {"command": "git status"}},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_audit_log(tmpdir, entries)
            result = staging_guard.check_audit_log_for_bulk_add(tmpdir)
            assert result is not None
            assert "git add -A" in result or "git add ." in result

    def test_git_add_dot_blocked(self):
        entries = [
            {"tool_input": {"command": "git add ."}},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_audit_log(tmpdir, entries)
            result = staging_guard.check_audit_log_for_bulk_add(tmpdir)
            assert result is not None

    def test_old_bulk_add_outside_window_passes(self):
        """Bulk add more than AUDIT_TAIL_COUNT entries ago should not trigger."""
        entries = [
            {"tool_input": {"command": "git add -A"}},  # entry 1 (old)
        ]
        # Add enough clean entries to push it outside the window
        for i in range(staging_guard.AUDIT_TAIL_COUNT):
            entries.append({"tool_input": {"command": f"git add src/file{i}.py"}})
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_audit_log(tmpdir, entries)
            assert staging_guard.check_audit_log_for_bulk_add(tmpdir) is None

    def test_malformed_json_lines_skipped(self):
        """Malformed audit log lines should be skipped, not crash."""
        with tempfile.TemporaryDirectory() as tmpdir:
            handoff_dir = os.path.join(tmpdir, "DaytimeNighttimeHandOff")
            os.makedirs(handoff_dir)
            audit_path = os.path.join(handoff_dir, "audit.jsonl")
            with open(audit_path, "w") as f:
                f.write("not valid json\n")
                f.write('{"tool_input": {"command": "git status"}}\n')
            assert staging_guard.check_audit_log_for_bulk_add(tmpdir) is None


# ============================================================================
# is_git_commit_command
# ============================================================================

class TestIsGitCommitCommand:
    """Tests for git commit command detection."""

    def test_simple_commit(self):
        assert staging_guard.is_git_commit_command("git commit -m 'msg'") is True

    def test_commit_with_flags(self):
        assert staging_guard.is_git_commit_command('git commit -m "$(cat <<EOF\nmsg\nEOF\n)"') is True

    def test_git_add_not_matched(self):
        assert staging_guard.is_git_commit_command("git add src/main.py") is False

    def test_git_status_not_matched(self):
        assert staging_guard.is_git_commit_command("git status") is False

    def test_git_diff_not_matched(self):
        assert staging_guard.is_git_commit_command("git diff --cached") is False

    def test_empty_command(self):
        assert staging_guard.is_git_commit_command("") is False

    def test_commit_in_chain(self):
        assert staging_guard.is_git_commit_command("git add . && git commit -m 'msg'") is True


# ============================================================================
# is_daytime_mode (with mocked file)
# ============================================================================

class TestDaytimeMode:
    """Tests for daytime mode detection / bypass."""

    def test_daytime_mode_detected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            claude_dir = os.path.join(tmpdir, ".claude")
            os.makedirs(claude_dir)
            mode_file = os.path.join(claude_dir, "active_mode.md")
            with open(mode_file, "w", encoding="utf-8") as f:
                f.write("# ClaudeDayNight \u2014 Daytime Mode\n")
            # Temporarily override HARDCODED_PROJECT_DIR
            original = staging_guard.HARDCODED_PROJECT_DIR
            try:
                staging_guard.HARDCODED_PROJECT_DIR = tmpdir
                assert staging_guard.is_daytime_mode(tmpdir) is True
            finally:
                staging_guard.HARDCODED_PROJECT_DIR = original

    def test_nighttime_mode_detected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            claude_dir = os.path.join(tmpdir, ".claude")
            os.makedirs(claude_dir)
            mode_file = os.path.join(claude_dir, "active_mode.md")
            with open(mode_file, "w", encoding="utf-8") as f:
                f.write("# ClaudeDayNight \u2014 Nighttime Mode\n")
            original = staging_guard.HARDCODED_PROJECT_DIR
            try:
                staging_guard.HARDCODED_PROJECT_DIR = tmpdir
                assert staging_guard.is_daytime_mode(tmpdir) is False
            finally:
                staging_guard.HARDCODED_PROJECT_DIR = original

    def test_missing_mode_file_defaults_nighttime(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            original = staging_guard.HARDCODED_PROJECT_DIR
            try:
                staging_guard.HARDCODED_PROJECT_DIR = tmpdir
                assert staging_guard.is_daytime_mode(tmpdir) is False
            finally:
                staging_guard.HARDCODED_PROJECT_DIR = original
