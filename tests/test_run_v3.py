"""Tests for scripts/run_v3.py — Experiment 0v3 orchestrator.

Validates the three-phase pipeline (deploy, generate, score) with all
external dependencies mocked: RunPodManager, setup_pod helpers, and
subprocess.run.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest


# ---------------------------------------------------------------------------
# Test: --generation-only flag on run_experiment_0.py
# ---------------------------------------------------------------------------


class TestGenerationOnlyFlag:
    """Verify the --generation-only argparse flag exists and is accepted."""

    def test_parse_args_accepts_generation_only(self) -> None:
        """parse_args() should accept --generation-only without error."""
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from scripts.run_experiment_0 import parse_args

        # parse_args reads sys.argv — mock it
        with patch("sys.argv", ["run_experiment_0.py", "--generation-only", "--n", "5"]):
            args = parse_args()
        assert args.generation_only is True

    def test_parse_args_default_generation_only_is_false(self) -> None:
        """--generation-only defaults to False when not provided."""
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from scripts.run_experiment_0 import parse_args

        with patch("sys.argv", ["run_experiment_0.py", "--n", "5"]):
            args = parse_args()
        assert args.generation_only is False


# ---------------------------------------------------------------------------
# Test: run_v3.py wrapper script
# ---------------------------------------------------------------------------


class TestRunV3HappyPath:
    """Happy path: pod creates, generation succeeds, pod terminates, scoring succeeds."""

    @patch("scripts.run_v3.subprocess.run")
    @patch("scripts.run_v3.pull_model", return_value=True)
    @patch("scripts.run_v3.wait_for_ollama", return_value=True)
    @patch("scripts.run_v3.RunPodManager")
    @patch("scripts.run_v3.os.environ.get", return_value="fake-api-key")
    def test_happy_path(
        self,
        mock_env: MagicMock,
        mock_manager_cls: MagicMock,
        mock_wait_ollama: MagicMock,
        mock_pull: MagicMock,
        mock_subprocess: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Full pipeline completes and pod is terminated."""
        manager = mock_manager_cls.return_value
        manager.get_balance.return_value = 10.00
        manager.create_pod.return_value = {"id": "pod-123"}
        manager.get_pod_url.return_value = "http://pod-123.runpod.io:11434"

        # Both subprocess calls succeed
        mock_subprocess.return_value = MagicMock(returncode=0)

        import scripts.run_v3 as run_v3

        # Override OUTPUT_DIR to tmp
        with patch.object(run_v3, "OUTPUT_DIR", tmp_path):
            run_v3.main()

        # Pod should be terminated
        manager.terminate_pod.assert_called_once_with("pod-123")
        # subprocess called twice: generation + scoring
        assert mock_subprocess.call_count == 2


class TestRunV3GenerationFailure:
    """Generation fails — pod must still be terminated."""

    @patch("scripts.run_v3.subprocess.run")
    @patch("scripts.run_v3.pull_model", return_value=True)
    @patch("scripts.run_v3.wait_for_ollama", return_value=True)
    @patch("scripts.run_v3.RunPodManager")
    @patch("scripts.run_v3.os.environ.get", return_value="fake-api-key")
    def test_generation_failure_terminates_pod(
        self,
        mock_env: MagicMock,
        mock_manager_cls: MagicMock,
        mock_wait_ollama: MagicMock,
        mock_pull: MagicMock,
        mock_subprocess: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Pod is terminated even when generation subprocess fails."""
        manager = mock_manager_cls.return_value
        manager.get_balance.return_value = 10.00
        manager.create_pod.return_value = {"id": "pod-456"}
        manager.get_pod_url.return_value = "http://pod-456.runpod.io:11434"

        # Generation fails
        mock_subprocess.return_value = MagicMock(returncode=1)

        import scripts.run_v3 as run_v3

        with patch.object(run_v3, "OUTPUT_DIR", tmp_path):
            with pytest.raises(SystemExit):
                run_v3.main()

        # Pod MUST be terminated even on failure
        manager.terminate_pod.assert_called_once_with("pod-456")


class TestRunV3LowBalance:
    """Low balance — exits before creating pod."""

    @patch("scripts.run_v3.RunPodManager")
    @patch("scripts.run_v3.os.environ.get", return_value="fake-api-key")
    def test_low_balance_exits(
        self,
        mock_env: MagicMock,
        mock_manager_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Script exits with error when balance is too low."""
        manager = mock_manager_cls.return_value
        manager.get_balance.return_value = 0.50

        import scripts.run_v3 as run_v3

        with patch.object(run_v3, "OUTPUT_DIR", tmp_path):
            with pytest.raises(SystemExit):
                run_v3.main()

        # No pod should have been created
        manager.create_pod.assert_not_called()
        # No pod to terminate
        manager.terminate_pod.assert_not_called()


class TestRunV3NoPodCreated:
    """Pod creation fails — no terminate call (pod_id is None)."""

    @patch("scripts.run_v3.RunPodManager")
    @patch("scripts.run_v3.os.environ.get", return_value="fake-api-key")
    def test_pod_creation_failure(
        self,
        mock_env: MagicMock,
        mock_manager_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """When pod creation raises an exception, terminate is not called."""
        manager = mock_manager_cls.return_value
        manager.get_balance.return_value = 10.00
        manager.create_pod.side_effect = RuntimeError("No GPUs available")

        import scripts.run_v3 as run_v3

        with patch.object(run_v3, "OUTPUT_DIR", tmp_path):
            with pytest.raises(RuntimeError, match="No GPUs available"):
                run_v3.main()

        # pod_id was never set, so terminate should not be called
        manager.terminate_pod.assert_not_called()
