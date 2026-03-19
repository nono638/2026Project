"""Tests for deploy/setup_pod.py — pod setup and model pulling.

All tests mock HTTP calls. No real API or RunPod calls are made.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from deploy.setup_pod import (
    wait_for_ollama,
    pull_model,
    verify_model,
    print_summary,
    main,
    parse_args,
)
from deploy.runpod_manager import RunPodError


class TestWaitForOllama:
    """Tests for the wait_for_ollama polling function."""

    def test_ollama_responds_immediately(self) -> None:
        """Ollama responds on first poll — should return True."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200

        with patch("deploy.setup_pod.requests.get", return_value=mock_resp):
            result = wait_for_ollama("http://fake:11434", timeout_s=10)

        assert result is True

    def test_ollama_not_responding(self) -> None:
        """Ollama never responds — should return False after timeout."""
        with patch("deploy.setup_pod.requests.get", side_effect=Exception("conn refused")):
            with patch("deploy.setup_pod.time.sleep"):  # Don't actually sleep
                result = wait_for_ollama("http://fake:11434", timeout_s=10)

        assert result is False


class TestPullModel:
    """Tests for the pull_model streaming function."""

    def test_successful_pull(self) -> None:
        """Successful streaming pull returns True."""
        # Simulate streaming response with progress lines
        progress_lines = [
            json.dumps({"status": "pulling manifest"}).encode(),
            json.dumps({"status": "downloading", "completed": 100, "total": 1000}).encode(),
            json.dumps({"status": "downloading", "completed": 1000, "total": 1000}).encode(),
            json.dumps({"status": "success"}).encode(),
        ]

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.iter_lines.return_value = iter(progress_lines)

        with patch("deploy.setup_pod.requests.post", return_value=mock_resp):
            result = pull_model("http://fake:11434", "qwen3:4b")

        assert result is True

    def test_pull_http_error(self) -> None:
        """Non-200 response from /api/pull returns False."""
        mock_resp = MagicMock()
        mock_resp.status_code = 500

        with patch("deploy.setup_pod.requests.post", return_value=mock_resp):
            result = pull_model("http://fake:11434", "bad-model")

        assert result is False

    def test_pull_timeout(self) -> None:
        """Request timeout returns False."""
        import requests as req

        with patch("deploy.setup_pod.requests.post", side_effect=req.Timeout("timed out")):
            result = pull_model("http://fake:11434", "big-model")

        assert result is False

    def test_cloudflare_streaming_handled(self) -> None:
        """Many progress lines (simulating long pull) are read without error."""
        # Simulate 100 progress lines — tests that streaming works
        progress_lines = [
            json.dumps({"status": "downloading", "completed": i * 100, "total": 10000}).encode()
            for i in range(100)
        ] + [json.dumps({"status": "success"}).encode()]

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.iter_lines.return_value = iter(progress_lines)

        with patch("deploy.setup_pod.requests.post", return_value=mock_resp):
            result = pull_model("http://fake:11434", "qwen3:4b")

        assert result is True


class TestVerifyModel:
    """Tests for the verify_model function."""

    def test_verify_success(self) -> None:
        """200 response from /api/generate returns True."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200

        with patch("deploy.setup_pod.requests.post", return_value=mock_resp):
            result = verify_model("http://fake:11434", "qwen3:4b")

        assert result is True

    def test_verify_failure(self) -> None:
        """Non-200 response returns False."""
        mock_resp = MagicMock()
        mock_resp.status_code = 404

        with patch("deploy.setup_pod.requests.post", return_value=mock_resp):
            result = verify_model("http://fake:11434", "missing-model")

        assert result is False


class TestFullSetupFlow:
    """Integration-style tests for the main() function with all mocks."""

    @patch("deploy.setup_pod.verify_model", return_value=True)
    @patch("deploy.setup_pod.pull_model", return_value=True)
    @patch("deploy.setup_pod.wait_for_ollama", return_value=True)
    @patch("deploy.setup_pod._load_env", return_value="fake-api-key")
    def test_full_setup_flow(
        self,
        mock_env: MagicMock,
        mock_wait: MagicMock,
        mock_pull: MagicMock,
        mock_verify: MagicMock,
    ) -> None:
        """Full setup: create pod, wait, pull, verify."""
        mock_manager = MagicMock()
        mock_manager.get_balance.return_value = 10.00
        mock_manager.get_spend_per_hour.return_value = 0.17
        mock_manager.create_pod.return_value = {"id": "pod123"}
        mock_manager.wait_for_ready.return_value = True
        mock_manager.get_pod_url.return_value = "https://pod123-11434.proxy.runpod.net"

        with patch("deploy.setup_pod.RunPodManager", return_value=mock_manager):
            with patch("deploy.setup_pod.parse_args") as mock_args:
                mock_args.return_value = MagicMock(
                    pod_id=None,
                    pull_only=False,
                    models=["mxbai-embed-large", "qwen3:4b"],
                    name="ragbench-gpu",
                    image="ollama/ollama",
                    volume_gb=20,
                )
                main()

        # Verify pod was created
        mock_manager.create_pod.assert_called_once()
        # Verify both models were pulled
        assert mock_pull.call_count == 2

    @patch("deploy.setup_pod.verify_model", return_value=True)
    @patch("deploy.setup_pod.pull_model", return_value=True)
    @patch("deploy.setup_pod.wait_for_ollama", return_value=True)
    @patch("deploy.setup_pod._load_env", return_value="fake-api-key")
    def test_pull_only_mode(
        self,
        mock_env: MagicMock,
        mock_wait: MagicMock,
        mock_pull: MagicMock,
        mock_verify: MagicMock,
    ) -> None:
        """Pull-only mode: no pod creation, just pull and verify."""
        mock_manager = MagicMock()
        mock_manager.get_balance.return_value = 5.00
        mock_manager.get_spend_per_hour.return_value = 0.17
        mock_manager.get_pod_url.return_value = "https://existing-pod-11434.proxy.runpod.net"

        with patch("deploy.setup_pod.RunPodManager", return_value=mock_manager):
            with patch("deploy.setup_pod.parse_args") as mock_args:
                mock_args.return_value = MagicMock(
                    pod_id="existing-pod",
                    pull_only=True,
                    models=["qwen3:0.6b"],
                    name="ragbench-gpu",
                    image="ollama/ollama",
                    volume_gb=20,
                )
                main()

        # Pod creation should NOT be called
        mock_manager.create_pod.assert_not_called()
        # Model should be pulled
        mock_pull.assert_called_once()

    @patch("deploy.setup_pod._load_env", return_value="fake-api-key")
    def test_low_balance_exits(self, mock_env: MagicMock) -> None:
        """Balance < $1.00 should exit with code 1."""
        mock_manager = MagicMock()
        mock_manager.get_balance.return_value = 0.50

        with patch("deploy.setup_pod.RunPodManager", return_value=mock_manager):
            with patch("deploy.setup_pod.parse_args") as mock_args:
                mock_args.return_value = MagicMock(
                    pod_id=None,
                    pull_only=False,
                    models=["qwen3:4b"],
                    name="ragbench-gpu",
                    image="ollama/ollama",
                    volume_gb=20,
                )
                with pytest.raises(SystemExit) as exc_info:
                    main()

        assert exc_info.value.code == 1

    @patch("deploy.setup_pod.wait_for_ollama", return_value=False)
    @patch("deploy.setup_pod._load_env", return_value="fake-api-key")
    def test_ollama_not_responding_exits(
        self,
        mock_env: MagicMock,
        mock_wait: MagicMock,
    ) -> None:
        """Ollama not responding after pod ready should exit with code 1."""
        mock_manager = MagicMock()
        mock_manager.get_balance.return_value = 10.00
        mock_manager.create_pod.return_value = {"id": "pod456"}
        mock_manager.wait_for_ready.return_value = True
        mock_manager.get_pod_url.return_value = "https://pod456-11434.proxy.runpod.net"

        with patch("deploy.setup_pod.RunPodManager", return_value=mock_manager):
            with patch("deploy.setup_pod.parse_args") as mock_args:
                mock_args.return_value = MagicMock(
                    pod_id=None,
                    pull_only=False,
                    models=["qwen3:4b"],
                    name="ragbench-gpu",
                    image="ollama/ollama",
                    volume_gb=20,
                )
                with pytest.raises(SystemExit) as exc_info:
                    main()

        assert exc_info.value.code == 1

    @patch("deploy.setup_pod.verify_model", return_value=True)
    @patch("deploy.setup_pod.wait_for_ollama", return_value=True)
    @patch("deploy.setup_pod._load_env", return_value="fake-api-key")
    def test_model_pull_failure_continues(
        self,
        mock_env: MagicMock,
        mock_wait: MagicMock,
        mock_verify: MagicMock,
    ) -> None:
        """Failed model pull should continue to next model, not exit."""
        mock_manager = MagicMock()
        mock_manager.get_balance.return_value = 10.00
        mock_manager.create_pod.return_value = {"id": "pod789"}
        mock_manager.wait_for_ready.return_value = True
        mock_manager.get_pod_url.return_value = "https://pod789-11434.proxy.runpod.net"
        mock_manager.get_spend_per_hour.return_value = 0.17

        # First model fails, second succeeds
        with patch("deploy.setup_pod.pull_model", side_effect=[False, True]):
            with patch("deploy.setup_pod.RunPodManager", return_value=mock_manager):
                with patch("deploy.setup_pod.parse_args") as mock_args:
                    mock_args.return_value = MagicMock(
                        pod_id=None,
                        pull_only=False,
                        models=["bad-model", "qwen3:4b"],
                        name="ragbench-gpu",
                        image="ollama/ollama",
                        volume_gb=20,
                    )
                    # Should NOT raise SystemExit
                    main()


class TestMissingApiKey:
    """Tests for missing RUNPOD_API_KEY."""

    def test_missing_api_key_exits(self) -> None:
        """Missing API key should print clear error and exit 1.

        Must mock load_dotenv because _load_env() calls it internally,
        which would re-populate RUNPOD_API_KEY from the .env file even
        after we clear os.environ.
        """
        from deploy.setup_pod import _load_env

        # Prevent .env file from repopulating the cleared environment
        with patch("dotenv.load_dotenv", return_value=None):
            with patch.dict("os.environ", {}, clear=True):
                with pytest.raises(SystemExit) as exc_info:
                    _load_env()

                assert exc_info.value.code == 1
