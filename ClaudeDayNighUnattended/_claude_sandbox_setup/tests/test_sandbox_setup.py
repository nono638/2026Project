"""
Tests that verify the sandbox is correctly installed in the project.

Run from the project root with:
    .venv/Scripts/python -m pytest _claude_sandbox_setup/tests/ -v

Or on Unix:
    .venv/bin/python -m pytest _claude_sandbox_setup/tests/ -v

These tests check that all sandbox components are in place and configured
correctly. They do NOT test the hooks' blocking behavior — they just verify
the files exist and are wired up correctly.
"""
import json
import os
import sys

import pytest


def get_project_root():
    """Get the project root (parent of _claude_sandbox_setup/)."""
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    setup_dir = os.path.dirname(tests_dir)
    return os.path.dirname(setup_dir)


PROJECT_ROOT = get_project_root()


# ── CLAUDE.md ──────────────────────────────────────────────────────────────


class TestClaudeMd:
    """Verify CLAUDE.md exists and uses the @import architecture."""

    def test_claude_md_exists(self):
        path = os.path.join(PROJECT_ROOT, "CLAUDE.md")
        assert os.path.isfile(path), "CLAUDE.md not found at project root"

    def test_claude_md_has_import_line(self):
        """CLAUDE.md should import .claude/active_mode.md via the @ syntax.

        The current architecture uses @.claude/active_mode.md so that dayrun/nightrun
        can swap mode-specific rules without touching CLAUDE.md itself.
        """
        path = os.path.join(PROJECT_ROOT, "CLAUDE.md")
        if not os.path.isfile(path):
            pytest.skip("CLAUDE.md not found")
        content = open(path, encoding="utf-8").read()
        assert "@.claude/active_mode.md" in content, (
            "CLAUDE.md should contain '@.claude/active_mode.md' import line. "
            "Run SETUP.md to migrate to the current architecture."
        )

    def test_active_mode_md_exists(self):
        """.claude/active_mode.md exists — this is what CLAUDE.md imports."""
        path = os.path.join(PROJECT_ROOT, ".claude", "active_mode.md")
        assert os.path.isfile(path), (
            ".claude/active_mode.md not found. "
            "Run setup or nightrun.sh to create it."
        )

    def test_active_mode_md_has_project_root(self):
        """active_mode.md has a real PROJECT_ROOT, not 'NOT YET CONFIGURED'."""
        path = os.path.join(PROJECT_ROOT, ".claude", "active_mode.md")
        if not os.path.isfile(path):
            pytest.skip(".claude/active_mode.md not found")
        content = open(path, encoding="utf-8").read()
        assert "NOT YET CONFIGURED" not in content, (
            "active_mode.md still says 'NOT YET CONFIGURED'. "
            "First-run setup hasn't completed."
        )

    def test_active_mode_md_has_correct_project_root(self):
        path = os.path.join(PROJECT_ROOT, ".claude", "active_mode.md")
        if not os.path.isfile(path):
            pytest.skip(".claude/active_mode.md not found")
        content = open(path, encoding="utf-8").read()
        normalized_root = os.path.normpath(PROJECT_ROOT).lower()
        content_lower = content.lower()
        assert (
            normalized_root.replace("\\", "/") in content_lower.replace("\\", "/")
            or normalized_root in content_lower
        ), (
            f"active_mode.md PROJECT_ROOT doesn't match actual project directory.\n"
            f"Expected path to contain: {PROJECT_ROOT}"
        )


# ── settings.json ──────────────────────────────────────────────────────────


class TestSettingsJson:
    """Verify .claude/settings.json exists and has required rules."""

    def get_settings(self):
        path = os.path.join(PROJECT_ROOT, ".claude", "settings.json")
        if not os.path.isfile(path):
            return None
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def test_settings_json_exists(self):
        path = os.path.join(PROJECT_ROOT, ".claude", "settings.json")
        assert os.path.isfile(path), ".claude/settings.json not found"

    def test_has_deny_section(self):
        settings = self.get_settings()
        if settings is None:
            pytest.skip("settings.json not found")
        assert "permissions" in settings, "No 'permissions' key in settings.json"
        assert "deny" in settings["permissions"], "No 'deny' list in permissions"

    def test_denies_rm_rf(self):
        settings = self.get_settings()
        if settings is None:
            pytest.skip("settings.json not found")
        deny = settings.get("permissions", {}).get("deny", [])
        assert any("rm -rf" in rule for rule in deny), (
            "deny list doesn't block 'rm -rf'"
        )

    def test_denies_curl(self):
        settings = self.get_settings()
        if settings is None:
            pytest.skip("settings.json not found")
        deny = settings.get("permissions", {}).get("deny", [])
        assert any("curl" in rule for rule in deny), (
            "deny list doesn't block 'curl' (network exfiltration risk)"
        )

    def test_denies_env_files(self):
        settings = self.get_settings()
        if settings is None:
            pytest.skip("settings.json not found")
        deny = settings.get("permissions", {}).get("deny", [])
        assert any(".env" in rule for rule in deny), (
            "deny list doesn't block .env file reads (secrets risk)"
        )

    def test_denies_git_push(self):
        settings = self.get_settings()
        if settings is None:
            pytest.skip("settings.json not found")
        deny = settings.get("permissions", {}).get("deny", [])
        assert any("git push" in rule for rule in deny), (
            "deny list doesn't block 'git push'"
        )


# ── Hooks (installed) ──────────────────────────────────────────────────────


class TestHooks:
    """Verify hook scripts exist, are valid Python, and are wired in settings."""

    def _hook_path(self, name):
        return os.path.join(PROJECT_ROOT, ".claude", "hooks", name)

    def _assert_valid_python(self, path):
        source = open(path, encoding="utf-8").read()
        try:
            compile(source, path, "exec")
        except SyntaxError as e:
            pytest.fail(f"{os.path.basename(path)} has a syntax error: {e}")

    def _get_all_hook_commands(self, event):
        settings_path = os.path.join(PROJECT_ROOT, ".claude", "settings.json")
        if not os.path.isfile(settings_path):
            return []
        with open(settings_path, encoding="utf-8") as f:
            settings = json.load(f)
        entries = settings.get("hooks", {}).get(event, [])
        commands = []
        for entry in entries:
            for h in entry.get("hooks", []):
                commands.append(h.get("command", ""))
        return commands

    # Existence
    def test_directory_guard_exists(self):
        assert os.path.isfile(self._hook_path("directory_guard.py")), \
            "directory_guard.py not found in .claude/hooks/"

    def test_no_ask_human_exists(self):
        assert os.path.isfile(self._hook_path("no_ask_human.py")), \
            "no_ask_human.py not found in .claude/hooks/"

    def test_audit_log_exists(self):
        assert os.path.isfile(self._hook_path("audit_log.py")), \
            "audit_log.py not found in .claude/hooks/"

    def test_stop_quality_gate_exists(self):
        assert os.path.isfile(self._hook_path("stop_quality_gate.py")), \
            "stop_quality_gate.py not found in .claude/hooks/"

    def test_notification_log_exists(self):
        assert os.path.isfile(self._hook_path("notification_log.py")), \
            "notification_log.py not found in .claude/hooks/"

    def test_context_monitor_exists(self):
        assert os.path.isfile(self._hook_path("context_monitor.py")), \
            "context_monitor.py not found in .claude/hooks/"

    def test_syntax_check_exists(self):
        assert os.path.isfile(self._hook_path("syntax_check.py")), \
            "syntax_check.py not found in .claude/hooks/"

    # Valid Python
    def test_directory_guard_is_valid_python(self):
        path = self._hook_path("directory_guard.py")
        if not os.path.isfile(path):
            pytest.skip("directory_guard.py not found")
        self._assert_valid_python(path)

    def test_no_ask_human_is_valid_python(self):
        path = self._hook_path("no_ask_human.py")
        if not os.path.isfile(path):
            pytest.skip("no_ask_human.py not found")
        self._assert_valid_python(path)

    def test_audit_log_is_valid_python(self):
        path = self._hook_path("audit_log.py")
        if not os.path.isfile(path):
            pytest.skip("audit_log.py not found")
        self._assert_valid_python(path)

    def test_stop_quality_gate_is_valid_python(self):
        path = self._hook_path("stop_quality_gate.py")
        if not os.path.isfile(path):
            pytest.skip("stop_quality_gate.py not found")
        self._assert_valid_python(path)

    def test_notification_log_is_valid_python(self):
        path = self._hook_path("notification_log.py")
        if not os.path.isfile(path):
            pytest.skip("notification_log.py not found")
        self._assert_valid_python(path)

    def test_context_monitor_is_valid_python(self):
        path = self._hook_path("context_monitor.py")
        if not os.path.isfile(path):
            pytest.skip("context_monitor.py not found")
        self._assert_valid_python(path)

    def test_syntax_check_is_valid_python(self):
        path = self._hook_path("syntax_check.py")
        if not os.path.isfile(path):
            pytest.skip("syntax_check.py not found")
        self._assert_valid_python(path)

    # Wiring
    def test_directory_guard_wired_in_settings(self):
        commands = self._get_all_hook_commands("PreToolUse")
        assert any("directory_guard" in cmd for cmd in commands), (
            "directory_guard.py is not wired up in settings.json PreToolUse hooks"
        )

    def test_directory_guard_has_hardcoded_path(self):
        path = self._hook_path("directory_guard.py")
        if not os.path.isfile(path):
            pytest.skip("directory_guard.py not found")
        content = open(path, encoding="utf-8").read()
        assert "HARDCODED_PROJECT_DIR = None" not in content, (
            "directory_guard.py still has HARDCODED_PROJECT_DIR = None. "
            "First-run setup hasn't completed."
        )


# ── Virtual Environment ────────────────────────────────────────────────────


class TestVenv:
    """Verify the virtual environment is set up."""

    def find_venv(self):
        for name in [".venv", "venv", "env"]:
            path = os.path.join(PROJECT_ROOT, name)
            if os.path.isdir(path) and os.path.isfile(
                os.path.join(path, "pyvenv.cfg")
            ):
                return path
        return None

    def test_venv_exists(self):
        venv = self.find_venv()
        assert venv is not None, (
            "No virtual environment found (.venv/, venv/, or env/). "
            "Run verify_environment.py first."
        )

    def test_venv_has_python(self):
        venv = self.find_venv()
        if venv is None:
            pytest.skip("No venv found")
        win = os.path.join(venv, "Scripts", "python.exe")
        unix = os.path.join(venv, "bin", "python")
        assert os.path.isfile(win) or os.path.isfile(unix), (
            f"venv at {venv} has no Python executable"
        )

    def test_pytest_importable(self):
        import pytest as _pt
        assert _pt is not None


# ── Python Version ─────────────────────────────────────────────────────────


class TestPythonVersion:
    def test_python_3_10_or_newer(self):
        major, minor = sys.version_info[:2]
        assert (major, minor) >= (3, 10), (
            f"Python {major}.{minor} is too old. Need 3.10+."
        )


# ── Setup Folder ───────────────────────────────────────────────────────────


class TestSetupFolder:
    """Verify the _claude_sandbox_setup/ folder is intact."""

    EXPECTED_FILES = [
        "SETUP.md",
        "HOW_TO_USE.md",
        "docs/DangerousClaudeFlagReadme.md",
        "docs/DayNightWorkflowDesign.md",
        "templates/nighttime_supplement.md",
        "templates/daytime_supplement.md",
        "templates/nighttime_settings.json",
        "templates/daytime_settings.json",
        "templates/tracker_template.json",
        "templates/handoff_structure/README.md",
        "templates/handoff_structure/DaytimeOnly/project_overview_template.md",
        "templates/handoff_structure/DaytimeOnly/inbox_template.md",
        "templates/handoff_structure/DaytimeOnly/incubating_template.md",
        "templates/commands/day.md",
        "templates/commands/night.md",
        "templates/skills/end-of-night-sweeps/SKILL.md",
        "hooks/directory_guard.py",
        "hooks/no_ask_human.py",
        "hooks/audit_log.py",
        "hooks/stop_quality_gate.py",
        "hooks/notification_log.py",
        "hooks/context_monitor.py",
        "hooks/syntax_check.py",
        "scripts/verify_environment.py",
        "scripts/dayrun.sh",
        "scripts/dayrun.bat",
        "scripts/nightrun.sh",
        "scripts/nightrun.bat",
        "scripts/repairrun.sh",
        "scripts/repairrun.bat",
        "tests/test_sandbox_setup.py",
    ]

    def test_setup_folder_exists(self):
        path = os.path.join(PROJECT_ROOT, "_claude_sandbox_setup")
        assert os.path.isdir(path), "_claude_sandbox_setup/ not found at project root"

    def test_setup_folder_complete(self):
        """All required files are present — catches partial copies."""
        base = os.path.join(PROJECT_ROOT, "_claude_sandbox_setup")
        missing = [
            f for f in self.EXPECTED_FILES
            if not os.path.isfile(os.path.join(base, f.replace("/", os.sep)))
        ]
        assert not missing, (
            "_claude_sandbox_setup/ is incomplete. Missing files:\n"
            + "\n".join(f"  {f}" for f in missing)
        )

    def test_source_hooks_complete(self):
        hooks_dir = os.path.join(PROJECT_ROOT, "_claude_sandbox_setup", "hooks")
        for name in [
            "directory_guard.py",
            "no_ask_human.py",
            "audit_log.py",
            "stop_quality_gate.py",
            "notification_log.py",
            "context_monitor.py",
            "syntax_check.py",
        ]:
            assert os.path.isfile(os.path.join(hooks_dir, name)), (
                f"Source {name} missing from _claude_sandbox_setup/hooks/"
            )


# ── Day/Night Templates ────────────────────────────────────────────────────


class TestDayNightTemplates:
    """Verify daytime/nighttime template files exist and are correctly configured."""

    def _template_path(self, *parts):
        return os.path.join(PROJECT_ROOT, "_claude_sandbox_setup", "templates", *parts)

    def test_nighttime_supplement_exists(self):
        assert os.path.isfile(self._template_path("nighttime_supplement.md"))

    def test_daytime_supplement_exists(self):
        assert os.path.isfile(self._template_path("daytime_supplement.md"))

    def test_nighttime_supplement_has_task_loop(self):
        path = self._template_path("nighttime_supplement.md")
        if not os.path.isfile(path):
            pytest.skip()
        content = open(path, encoding="utf-8").read()
        assert "tracker.json" in content
        assert "WrittenByDaytime" in content
        assert "WrittenByNighttime" in content

    def test_nighttime_supplement_has_git_state_check(self):
        """Nighttime supplement should have an explicit git state check at session start."""
        path = self._template_path("nighttime_supplement.md")
        if not os.path.isfile(path):
            pytest.skip()
        content = open(path, encoding="utf-8").read()
        assert "git status" in content, (
            "nighttime_supplement.md should include explicit git status check at session start"
        )
        assert "git branch" in content, (
            "nighttime_supplement.md should include explicit git branch check at session start"
        )

    def test_nighttime_supplement_ignores_daytime_only(self):
        path = self._template_path("nighttime_supplement.md")
        if not os.path.isfile(path):
            pytest.skip()
        content = open(path, encoding="utf-8").read()
        assert "DaytimeOnly" in content, (
            "nighttime_supplement.md should mention DaytimeOnly/ (to say ignore it)"
        )

    def test_nighttime_supplement_has_blocked_status(self):
        path = self._template_path("nighttime_supplement.md")
        if not os.path.isfile(path):
            pytest.skip()
        content = open(path, encoding="utf-8").read()
        assert "blocked" in content, (
            "nighttime_supplement.md should handle 'blocked' task status"
        )

    def test_nighttime_supplement_handles_branch_exists(self):
        """Nighttime supplement should handle the case where a branch already exists."""
        path = self._template_path("nighttime_supplement.md")
        if not os.path.isfile(path):
            pytest.skip()
        content = open(path, encoding="utf-8").read()
        assert "already exists" in content, (
            "nighttime_supplement.md should handle branch-already-exists "
            "(e.g., after repairrun resets a crashed task)"
        )

    def test_daytime_supplement_has_daytime_marker(self):
        path = self._template_path("daytime_supplement.md")
        if not os.path.isfile(path):
            pytest.skip()
        content = open(path, encoding="utf-8").read()
        assert "DAYTIME" in content.upper()

    def test_daytime_supplement_has_inbox(self):
        path = self._template_path("daytime_supplement.md")
        if not os.path.isfile(path):
            pytest.skip()
        content = open(path, encoding="utf-8").read()
        assert "inbox.md" in content, (
            "daytime_supplement.md should reference inbox.md"
        )
        assert "incubating.md" in content, (
            "daytime_supplement.md should reference incubating.md"
        )

    def test_daytime_supplement_reads_project_overview(self):
        path = self._template_path("daytime_supplement.md")
        if not os.path.isfile(path):
            pytest.skip()
        content = open(path, encoding="utf-8").read()
        assert "project_overview.md" in content

    def test_daytime_supplement_has_reviewed_tracking(self):
        """Daytime supplement should track which tasks have been reviewed to avoid re-processing."""
        path = self._template_path("daytime_supplement.md")
        if not os.path.isfile(path):
            pytest.skip()
        content = open(path, encoding="utf-8").read()
        assert "daytime_reviewed" in content, (
            "daytime_supplement.md should use daytime_reviewed to prevent "
            "re-processing done tasks across consecutive daytime sessions"
        )

    def test_daytime_supplement_has_information_routing(self):
        """Daytime supplement should have an information routing decision tree."""
        path = self._template_path("daytime_supplement.md")
        if not os.path.isfile(path):
            pytest.skip()
        content = open(path, encoding="utf-8").read()
        assert "INFORMATION ROUTING" in content, (
            "daytime_supplement.md should have an INFORMATION ROUTING section"
        )

    def test_daytime_supplement_allows_small_code_tasks(self):
        """Daytime supplement should allow small code fixes during the day."""
        path = self._template_path("daytime_supplement.md")
        if not os.path.isfile(path):
            pytest.skip()
        content = open(path, encoding="utf-8").read()
        assert "Small tasks" in content or "small code" in content.lower() or "Bug fixes" in content, (
            "daytime_supplement.md should allow small code tasks during daytime"
        )

    def test_nighttime_supplement_has_code_quality_standards(self):
        """Nighttime supplement should require type hints, docstrings, and good comments."""
        path = self._template_path("nighttime_supplement.md")
        if not os.path.isfile(path):
            pytest.skip()
        content = open(path, encoding="utf-8").read()
        assert "CODE QUALITY" in content, (
            "nighttime_supplement.md should have a CODE QUALITY STANDARDS section"
        )
        assert "type hint" in content.lower(), (
            "nighttime_supplement.md should require type hints"
        )
        assert "docstring" in content.lower(), (
            "nighttime_supplement.md should require docstrings"
        )

    def test_nighttime_supplement_references_sweeps_skill(self):
        """Nighttime supplement should reference the sweeps skill, not inline them."""
        path = self._template_path("nighttime_supplement.md")
        if not os.path.isfile(path):
            pytest.skip()
        content = open(path, encoding="utf-8").read()
        assert "end-of-night-sweeps" in content or "SKILL.md" in content, (
            "nighttime_supplement.md should reference the sweeps skill"
        )

    def test_nighttime_supplement_reads_project_overview(self):
        """Nighttime supplement should read project_overview.md for context."""
        path = self._template_path("nighttime_supplement.md")
        if not os.path.isfile(path):
            pytest.skip()
        content = open(path, encoding="utf-8").read()
        assert "project_overview.md" in content, (
            "nighttime_supplement.md should read project_overview.md for project context"
        )

    def test_nighttime_supplement_reads_architecture_decisions(self):
        """Nighttime supplement should read architecture-decisions.md."""
        path = self._template_path("nighttime_supplement.md")
        if not os.path.isfile(path):
            pytest.skip()
        content = open(path, encoding="utf-8").read()
        assert "architecture-decisions" in content, (
            "nighttime_supplement.md should read architecture-decisions.md"
        )

    def test_nighttime_supplement_uses_compact_not_clear(self):
        """Nighttime supplement should recommend /compact, not /clear, for context management."""
        path = self._template_path("nighttime_supplement.md")
        if not os.path.isfile(path):
            pytest.skip()
        content = open(path, encoding="utf-8").read()
        assert "/compact" in content, (
            "nighttime_supplement.md should recommend /compact for context management"
        )

    def test_sweeps_skill_exists(self):
        """End-of-night sweeps skill should exist."""
        path = self._template_path("skills", "end-of-night-sweeps", "SKILL.md")
        assert os.path.isfile(path), (
            "templates/skills/end-of-night-sweeps/SKILL.md not found"
        )

    def test_sweeps_skill_has_all_sweeps(self):
        """Sweeps skill should contain all 6 sweeps."""
        path = self._template_path("skills", "end-of-night-sweeps", "SKILL.md")
        if not os.path.isfile(path):
            pytest.skip()
        content = open(path, encoding="utf-8").read()
        expected = ["test suite", "Bug sweep", "DRY", "Type hints", "Dead code", "Security"]
        for sweep in expected:
            assert sweep.lower() in content.lower(), (
                f"Sweeps skill missing sweep: {sweep}"
            )

    def test_nighttime_settings_is_valid_json(self):
        path = self._template_path("nighttime_settings.json")
        if not os.path.isfile(path):
            pytest.skip()
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert "permissions" in data

    def test_daytime_settings_is_valid_json(self):
        path = self._template_path("daytime_settings.json")
        if not os.path.isfile(path):
            pytest.skip()
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert "permissions" in data

    def test_nighttime_settings_denies_web(self):
        path = self._template_path("nighttime_settings.json")
        if not os.path.isfile(path):
            pytest.skip()
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        deny = data.get("permissions", {}).get("deny", [])
        assert "WebSearch" in deny
        assert "WebFetch" in deny

    def test_nighttime_settings_has_default_mode_dont_ask(self):
        """Nighttime settings should use defaultMode: dontAsk for unattended safety."""
        path = self._template_path("nighttime_settings.json")
        if not os.path.isfile(path):
            pytest.skip()
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert data.get("permissions", {}).get("defaultMode") == "dontAsk", (
            "nighttime_settings.json should have defaultMode: dontAsk"
        )

    def test_nighttime_settings_has_context_monitor_hook(self):
        """context_monitor.py should be wired as a PostToolUse hook in nighttime settings."""
        path = self._template_path("nighttime_settings.json")
        if not os.path.isfile(path):
            pytest.skip()
        content = open(path, encoding="utf-8").read()
        assert "context_monitor" in content, (
            "nighttime_settings.json should wire context_monitor.py as a PostToolUse hook"
        )

    def test_nighttime_settings_has_syntax_check_hook(self):
        """syntax_check.py should be wired as a PostToolUse hook in nighttime settings."""
        path = self._template_path("nighttime_settings.json")
        if not os.path.isfile(path):
            pytest.skip()
        content = open(path, encoding="utf-8").read()
        assert "syntax_check" in content, (
            "nighttime_settings.json should wire syntax_check.py as a PostToolUse hook"
        )

    def test_daytime_settings_has_syntax_check_hook(self):
        """syntax_check.py should be wired in daytime settings too."""
        path = self._template_path("daytime_settings.json")
        if not os.path.isfile(path):
            pytest.skip()
        content = open(path, encoding="utf-8").read()
        assert "syntax_check" in content, (
            "daytime_settings.json should wire syntax_check.py as a PostToolUse hook"
        )

    def test_nighttime_settings_has_audit_log_hook(self):
        """audit_log.py should be wired as a PostToolUse hook in nighttime settings."""
        path = self._template_path("nighttime_settings.json")
        if not os.path.isfile(path):
            pytest.skip()
        content = open(path, encoding="utf-8").read()
        assert "audit_log" in content, (
            "nighttime_settings.json should wire audit_log.py as a PostToolUse hook"
        )

    def test_nighttime_settings_has_stop_quality_gate(self):
        """stop_quality_gate.py should be wired as a Stop hook in nighttime settings."""
        path = self._template_path("nighttime_settings.json")
        if not os.path.isfile(path):
            pytest.skip()
        content = open(path, encoding="utf-8").read()
        assert "stop_quality_gate" in content, (
            "nighttime_settings.json should wire stop_quality_gate.py as a Stop hook"
        )

    def test_nighttime_settings_denies_gh(self):
        """gh CLI can make network requests — must be denied at night."""
        path = self._template_path("nighttime_settings.json")
        if not os.path.isfile(path):
            pytest.skip()
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        deny = data.get("permissions", {}).get("deny", [])
        assert any("gh" in rule for rule in deny), (
            "nighttime_settings.json should deny Bash(gh *) (network exfiltration risk)"
        )

    def test_nighttime_settings_denies_start(self):
        """'start' can open URLs in the browser — must be denied at night."""
        path = self._template_path("nighttime_settings.json")
        if not os.path.isfile(path):
            pytest.skip()
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        deny = data.get("permissions", {}).get("deny", [])
        assert any("start" in rule for rule in deny), (
            "nighttime_settings.json should deny Bash(start *) (URL exfiltration risk)"
        )

    def test_nighttime_settings_denies_powershell(self):
        """PowerShell can bypass network deny rules via .NET — must be denied at night."""
        path = self._template_path("nighttime_settings.json")
        if not os.path.isfile(path):
            pytest.skip()
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        deny = data.get("permissions", {}).get("deny", [])
        allow = data.get("permissions", {}).get("allow", [])
        # powershell must not be in allow, or must be in deny
        ps_allowed = any("powershell" in rule.lower() and "Remove-Item" not in rule
                         and "Invoke-" not in rule and "curl" not in rule
                         and "wget" not in rule and "Format-" not in rule
                         and "Clear-" not in rule and "Set-Execution" not in rule
                         for rule in allow)
        ps_denied = any(rule in ("Bash(powershell *)", "Bash(pwsh *)",
                                  "Bash(*powershell*)", "Bash(*pwsh*)")
                        for rule in deny)
        assert not ps_allowed or ps_denied, (
            "nighttime_settings.json should not allow bare PowerShell at night "
            "(network bypass via .NET)"
        )

    def test_daytime_settings_has_explicit_default_mode(self):
        """Daytime settings should have explicit defaultMode: ask."""
        path = self._template_path("daytime_settings.json")
        if not os.path.isfile(path):
            pytest.skip()
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert data.get("permissions", {}).get("defaultMode") == "ask", (
            "daytime_settings.json should have explicit defaultMode: ask"
        )

    def test_daytime_settings_allows_web(self):
        path = self._template_path("daytime_settings.json")
        if not os.path.isfile(path):
            pytest.skip()
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        allow = data.get("permissions", {}).get("allow", [])
        assert "WebSearch" in allow
        assert "WebFetch" in allow
        assert "AskUserQuestion" in allow

    def test_daytime_settings_has_audit_log_hook(self):
        """audit_log.py should be wired in daytime settings too."""
        path = self._template_path("daytime_settings.json")
        if not os.path.isfile(path):
            pytest.skip()
        content = open(path, encoding="utf-8").read()
        assert "audit_log" in content, (
            "daytime_settings.json should wire audit_log.py as a PostToolUse hook"
        )

    def test_daytime_settings_no_ask_human_hook_absent(self):
        """Daytime settings should NOT include the no_ask_human hook."""
        path = self._template_path("daytime_settings.json")
        if not os.path.isfile(path):
            pytest.skip()
        content = open(path, encoding="utf-8").read()
        assert "no_ask_human" not in content, (
            "daytime_settings.json should not include the no_ask_human hook"
        )

    def test_tracker_template_is_valid_json(self):
        path = self._template_path("tracker_template.json")
        if not os.path.isfile(path):
            pytest.skip()
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, list)

    def test_handoff_readme_exists(self):
        assert os.path.isfile(self._template_path("handoff_structure", "README.md"))

    def test_handoff_readme_has_structure(self):
        path = self._template_path("handoff_structure", "README.md")
        if not os.path.isfile(path):
            pytest.skip()
        content = open(path, encoding="utf-8").read()
        assert "WrittenByDaytime" in content
        assert "WrittenByNighttime" in content
        assert "tracker.json" in content
        assert "DaytimeOnly" in content

    def test_handoff_readme_has_blocked_status(self):
        path = self._template_path("handoff_structure", "README.md")
        if not os.path.isfile(path):
            pytest.skip()
        content = open(path, encoding="utf-8").read()
        assert "blocked" in content, (
            "handoff_structure/README.md should document the 'blocked' task status"
        )

    def test_daytime_only_templates_exist(self):
        for name in [
            "project_overview_template.md",
            "inbox_template.md",
            "incubating_template.md",
        ]:
            path = self._template_path("handoff_structure", "DaytimeOnly", name)
            assert os.path.isfile(path), (
                f"templates/handoff_structure/DaytimeOnly/{name} not found"
            )

    def test_project_overview_template_has_change_history(self):
        path = self._template_path("handoff_structure", "DaytimeOnly", "project_overview_template.md")
        if not os.path.isfile(path):
            pytest.skip()
        content = open(path, encoding="utf-8").read()
        assert "Change History" in content

    def test_daytime_command_exists(self):
        assert os.path.isfile(self._template_path("commands", "day.md"))

    def test_nighttime_command_exists(self):
        assert os.path.isfile(self._template_path("commands", "night.md"))

    def test_nighttime_command_has_task_loop(self):
        path = self._template_path("commands", "night.md")
        if not os.path.isfile(path):
            pytest.skip()
        content = open(path, encoding="utf-8").read()
        assert "tracker.json" in content

    def test_daytime_command_has_inbox(self):
        """day.md slash command should reference the current inbox/incubating structure."""
        path = self._template_path("commands", "day.md")
        if not os.path.isfile(path):
            pytest.skip()
        content = open(path, encoding="utf-8").read()
        assert "inbox.md" in content, (
            "day.md should reference inbox.md (not the old ideas_log.md)"
        )


# ── Scripts ────────────────────────────────────────────────────────────────


class TestScripts:
    """Verify all launcher scripts exist and reference key components."""

    def _script_path(self, name):
        return os.path.join(PROJECT_ROOT, "_claude_sandbox_setup", "scripts", name)

    def test_dayrun_sh_exists(self):
        assert os.path.isfile(self._script_path("dayrun.sh")), "scripts/dayrun.sh not found"

    def test_dayrun_bat_exists(self):
        assert os.path.isfile(self._script_path("dayrun.bat")), "scripts/dayrun.bat not found"

    def test_nightrun_sh_exists(self):
        assert os.path.isfile(self._script_path("nightrun.sh")), "scripts/nightrun.sh not found"

    def test_nightrun_bat_exists(self):
        assert os.path.isfile(self._script_path("nightrun.bat")), "scripts/nightrun.bat not found"

    def test_repairrun_sh_exists(self):
        assert os.path.isfile(self._script_path("repairrun.sh")), "scripts/repairrun.sh not found"

    def test_repairrun_bat_exists(self):
        assert os.path.isfile(self._script_path("repairrun.bat")), "scripts/repairrun.bat not found"

    def test_nightrun_sh_has_preflight(self):
        path = self._script_path("nightrun.sh")
        if not os.path.isfile(path):
            pytest.skip()
        content = open(path, encoding="utf-8").read()
        assert "Pre-flight" in content or "preflight" in content.lower(), (
            "nightrun.sh should have pre-flight checks"
        )

    def test_nightrun_sh_has_git_preflight(self):
        """nightrun.sh should check for a git repository at startup."""
        path = self._script_path("nightrun.sh")
        if not os.path.isfile(path):
            pytest.skip()
        content = open(path, encoding="utf-8").read()
        assert "git rev-parse" in content, (
            "nightrun.sh should check for a git repository in pre-flight"
        )

    def test_nightrun_bat_has_git_preflight(self):
        """nightrun.bat should check for a git repository at startup."""
        path = self._script_path("nightrun.bat")
        if not os.path.isfile(path):
            pytest.skip()
        content = open(path, encoding="utf-8").read()
        assert "git rev-parse" in content, (
            "nightrun.bat should check for a git repository in pre-flight"
        )

    def test_nightrun_sh_has_tracker_backup(self):
        """nightrun.sh should back up tracker.json before each launch."""
        path = self._script_path("nightrun.sh")
        if not os.path.isfile(path):
            pytest.skip()
        content = open(path, encoding="utf-8").read()
        assert ".bak" in content, (
            "nightrun.sh should back up tracker.json before each launch"
        )

    def test_nightrun_bat_has_tracker_backup(self):
        """nightrun.bat should back up tracker.json before each launch."""
        path = self._script_path("nightrun.bat")
        if not os.path.isfile(path):
            pytest.skip()
        content = open(path, encoding="utf-8").read()
        assert ".bak" in content, (
            "nightrun.bat should back up tracker.json before each launch"
        )

    def test_nightrun_sh_has_python_fallback(self):
        """nightrun.sh should detect python3 or python, not hardcode python3."""
        path = self._script_path("nightrun.sh")
        if not os.path.isfile(path):
            pytest.skip()
        content = open(path, encoding="utf-8").read()
        assert "PYTHON=" in content or "$PYTHON" in content, (
            "nightrun.sh should use a PYTHON variable with fallback, not hardcoded python3"
        )
        assert "python3" not in content.split("PYTHON=")[0] or "command -v python3" in content, (
            "nightrun.sh should detect python availability, not assume python3 exists"
        )

    def test_repairrun_sh_has_python_fallback(self):
        """repairrun.sh should detect python3 or python, not hardcode python3."""
        path = self._script_path("repairrun.sh")
        if not os.path.isfile(path):
            pytest.skip()
        content = open(path, encoding="utf-8").read()
        assert "PYTHON=" in content or "$PYTHON" in content, (
            "repairrun.sh should use a PYTHON variable with fallback"
        )

    def test_nightrun_sh_has_max_relaunches(self):
        path = self._script_path("nightrun.sh")
        if not os.path.isfile(path):
            pytest.skip()
        content = open(path, encoding="utf-8").read()
        assert "MAX_RELAUNCHES" in content, (
            "nightrun.sh should have a max relaunch limit"
        )

    def test_nightrun_bat_has_max_relaunches(self):
        path = self._script_path("nightrun.bat")
        if not os.path.isfile(path):
            pytest.skip()
        content = open(path, encoding="utf-8").read()
        assert "MAX_RELAUNCHES" in content, (
            "nightrun.bat should have a max relaunch limit"
        )

    def test_dayrun_sh_restores_nighttime(self):
        """dayrun.sh should restore nighttime settings on exit."""
        path = self._script_path("dayrun.sh")
        if not os.path.isfile(path):
            pytest.skip()
        content = open(path, encoding="utf-8").read()
        assert "restore_nighttime" in content or "nighttime_settings" in content, (
            "dayrun.sh should restore nighttime settings on exit"
        )

    def test_repairrun_sh_resets_in_progress(self):
        path = self._script_path("repairrun.sh")
        if not os.path.isfile(path):
            pytest.skip()
        content = open(path, encoding="utf-8").read()
        assert "in_progress" in content, (
            "repairrun.sh should reset in_progress tasks"
        )

    def test_nightrun_sh_references_tracker(self):
        path = self._script_path("nightrun.sh")
        if not os.path.isfile(path):
            pytest.skip()
        content = open(path, encoding="utf-8").read()
        assert "tracker.json" in content

    def test_nightrun_bat_references_tracker(self):
        path = self._script_path("nightrun.bat")
        if not os.path.isfile(path):
            pytest.skip()
        content = open(path, encoding="utf-8").read()
        assert "tracker.json" in content
