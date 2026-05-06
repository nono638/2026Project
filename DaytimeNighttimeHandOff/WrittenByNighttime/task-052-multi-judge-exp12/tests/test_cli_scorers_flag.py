"""Pre-written tests for task-052: --scorers CLI flag for run_experiment_1/2.

The old singular `--scorer` is removed. The new `--scorers` flag:
- Accepts one or more `provider:model` strings.
- Defaults to ["anthropic:claude-haiku-4-5-20251001", "openai:gpt-5.4-mini"].
- Rejects duplicate scorers.

These tests parse argparse without actually running an experiment.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
PYTHON = sys.executable

EXPERIMENT_SCRIPTS = ["run_experiment_1.py", "run_experiment_2.py"]


@pytest.mark.parametrize("script_name", EXPERIMENT_SCRIPTS)
class TestScorersFlag:
    """Parametrized across both run_experiment_1.py and run_experiment_2.py."""

    def test_default_panel_in_help(self, script_name: str) -> None:
        """--help shows the new --scorers flag with the two-judge default."""
        result = subprocess.run(
            [PYTHON, str(PROJECT_ROOT / "scripts" / script_name), "--help"],
            capture_output=True, text=True, cwd=str(PROJECT_ROOT), timeout=30,
        )
        assert result.returncode == 0
        assert "--scorers" in result.stdout
        assert "claude-haiku-4-5-20251001" in result.stdout
        assert "gpt-5.4-mini" in result.stdout

    def test_old_singular_flag_removed(self, script_name: str) -> None:
        """The old --scorer flag must be removed (argparse error if used)."""
        result = subprocess.run(
            [
                PYTHON, str(PROJECT_ROOT / "scripts" / script_name),
                "--scorer", "google:gemini-2.5-flash",
                "--help",
            ],
            capture_output=True, text=True, cwd=str(PROJECT_ROOT), timeout=30,
        )
        # argparse errors with non-zero exit on unrecognized argument
        assert result.returncode != 0, (
            "Old --scorer flag should not be accepted. stdout: " + result.stdout[:200]
        )
        assert "unrecognized" in result.stderr.lower() or "error" in result.stderr.lower()

    def test_duplicate_scorers_rejected(self, script_name: str, tmp_path: Path) -> None:
        """--scorers a:b a:b should exit with a clear error before any work."""
        out = tmp_path / "out"
        out.mkdir()
        # Use --skip-generation to avoid touching real models, but the script should
        # exit BEFORE hitting that path — duplicate detection happens at arg parse.
        result = subprocess.run(
            [
                PYTHON, str(PROJECT_ROOT / "scripts" / script_name),
                "--skip-generation",
                "--output-dir", str(out),
                "--scorers", "openai:gpt-5.4-mini", "openai:gpt-5.4-mini",
                "--no-gallery",
            ],
            capture_output=True, text=True, cwd=str(PROJECT_ROOT), timeout=30,
        )
        assert result.returncode != 0, "Duplicate scorers should cause non-zero exit"
        combined = (result.stdout + result.stderr).lower()
        assert "duplicate" in combined or "repeated" in combined or "unique" in combined
