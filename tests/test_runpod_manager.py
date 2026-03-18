"""Tests for RunPod management module.

All API calls are mocked — no real RunPod account needed.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest


class TestRunPodManager:
    """Tests for RunPodManager lifecycle methods."""

    def _make_manager(self):
        from deploy.runpod_manager import RunPodManager
        return RunPodManager(api_key="test-key-123")

    # -- create_pod --

    @patch("deploy.runpod_manager.requests.post")
    def test_create_pod_success(self, mock_post: MagicMock) -> None:
        """Successful pod creation returns pod dict with correct request body."""
        mock_post.return_value = MagicMock(
            status_code=201,
            json=MagicMock(return_value={
                "id": "pod123",
                "name": "test-pod",
                "desiredStatus": "RUNNING",
            }),
        )
        mgr = self._make_manager()
        result = mgr.create_pod(name="test-pod", image_name="runpod/pytorch")

        assert result["id"] == "pod123"

        # Verify request body
        call_kwargs = mock_post.call_args
        body = call_kwargs.kwargs.get("json") or json.loads(call_kwargs.kwargs.get("data", "{}"))
        assert body["name"] == "test-pod"
        assert body["gpuTypePriority"] == "availability"
        assert isinstance(body["gpuTypeIds"], list)
        assert len(body["gpuTypeIds"]) >= 1

    @patch("deploy.runpod_manager.requests.post")
    def test_create_pod_custom_gpu_types(self, mock_post: MagicMock) -> None:
        """Custom gpu_types override defaults."""
        mock_post.return_value = MagicMock(
            status_code=201,
            json=MagicMock(return_value={"id": "pod456"}),
        )
        mgr = self._make_manager()
        custom_gpus = ["NVIDIA A100 80GB PCIe"]
        mgr.create_pod(name="big-pod", image_name="img", gpu_types=custom_gpus)

        call_kwargs = mock_post.call_args
        body = call_kwargs.kwargs.get("json") or json.loads(call_kwargs.kwargs.get("data", "{}"))
        assert body["gpuTypeIds"] == custom_gpus

    @patch("deploy.runpod_manager.requests.post")
    def test_create_pod_api_error(self, mock_post: MagicMock) -> None:
        """Non-2xx response raises RunPodError."""
        from deploy.runpod_manager import RunPodError
        mock_post.return_value = MagicMock(
            status_code=400,
            text="Bad Request: no GPUs available",
            ok=False,
        )
        mgr = self._make_manager()
        with pytest.raises(RunPodError) as exc_info:
            mgr.create_pod(name="fail-pod", image_name="img")
        assert exc_info.value.status_code == 400

    # -- terminate_pod --

    @patch("deploy.runpod_manager.requests.delete")
    def test_terminate_pod_success(self, mock_delete: MagicMock) -> None:
        """Successful termination calls DELETE with correct URL."""
        mock_delete.return_value = MagicMock(status_code=204, ok=True)
        mgr = self._make_manager()
        mgr.terminate_pod("pod123")

        url = mock_delete.call_args.args[0] if mock_delete.call_args.args else mock_delete.call_args.kwargs.get("url", "")
        assert "pod123" in str(url) or "pod123" in str(mock_delete.call_args)

    @patch("deploy.runpod_manager.requests.delete")
    def test_terminate_pod_already_gone(self, mock_delete: MagicMock) -> None:
        """404 on terminate is not an error (pod already deleted)."""
        mock_delete.return_value = MagicMock(status_code=404, ok=False, text="Not found")
        mgr = self._make_manager()
        # Should not raise
        mgr.terminate_pod("gone-pod")

    # -- get_pod --

    @patch("deploy.runpod_manager.requests.get")
    def test_get_pod_success(self, mock_get: MagicMock) -> None:
        """Successful get returns pod dict."""
        mock_get.return_value = MagicMock(
            status_code=200,
            ok=True,
            json=MagicMock(return_value={"id": "pod123", "desiredStatus": "RUNNING"}),
        )
        mgr = self._make_manager()
        result = mgr.get_pod("pod123")
        assert result is not None
        assert result["id"] == "pod123"

    @patch("deploy.runpod_manager.requests.get")
    def test_get_pod_not_found(self, mock_get: MagicMock) -> None:
        """404 returns None."""
        mock_get.return_value = MagicMock(status_code=404, ok=False, text="Not found")
        mgr = self._make_manager()
        result = mgr.get_pod("nonexistent")
        assert result is None

    # -- list_pods --

    @patch("deploy.runpod_manager.requests.get")
    def test_list_pods(self, mock_get: MagicMock) -> None:
        """List returns array of pod dicts."""
        mock_get.return_value = MagicMock(
            status_code=200,
            ok=True,
            json=MagicMock(return_value=[{"id": "a"}, {"id": "b"}]),
        )
        mgr = self._make_manager()
        result = mgr.list_pods()
        assert len(result) == 2

    # -- get_pod_url --

    def test_get_pod_url(self) -> None:
        """URL follows RunPod proxy format."""
        mgr = self._make_manager()
        url = mgr.get_pod_url("abc123", port=11434)
        assert url == "https://abc123-11434.proxy.runpod.net"

    def test_get_pod_url_default_port(self) -> None:
        """Default port is 11434 (Ollama)."""
        mgr = self._make_manager()
        url = mgr.get_pod_url("xyz789")
        assert "11434" in url

    # -- wait_for_ready --

    @patch("deploy.runpod_manager.time.sleep")
    def test_wait_for_ready_success(self, mock_sleep: MagicMock) -> None:
        """Returns True when pod becomes ready."""
        mgr = self._make_manager()
        # First call: not ready. Second call: ready.
        with patch.object(mgr, "get_pod", side_effect=[
            {"id": "pod1", "desiredStatus": "CREATED", "runtime": None},
            {"id": "pod1", "desiredStatus": "RUNNING", "runtime": {"uptimeInSeconds": 10}},
        ]):
            result = mgr.wait_for_ready("pod1", timeout_s=30, poll_interval_s=1)
        assert result is True

    @patch("deploy.runpod_manager.time.sleep")
    def test_wait_for_ready_timeout(self, mock_sleep: MagicMock) -> None:
        """Returns False when timeout exceeded."""
        mgr = self._make_manager()
        with patch.object(mgr, "get_pod", return_value={
            "id": "pod1", "desiredStatus": "CREATED", "runtime": None,
        }):
            result = mgr.wait_for_ready("pod1", timeout_s=3, poll_interval_s=1)
        assert result is False

    # -- get_balance --

    @patch("deploy.runpod_manager.requests.post")
    def test_get_balance(self, mock_post: MagicMock) -> None:
        """Balance query returns float from GraphQL response."""
        mock_post.return_value = MagicMock(
            status_code=200,
            ok=True,
            json=MagicMock(return_value={
                "data": {"myself": {"clientBalance": 42.50, "currentSpendPerHr": 0.17}},
            }),
        )
        mgr = self._make_manager()
        balance = mgr.get_balance()
        assert balance == 42.50

    # -- get_spend_per_hour --

    @patch("deploy.runpod_manager.requests.post")
    def test_get_spend_per_hour(self, mock_post: MagicMock) -> None:
        """Spend rate query returns float from GraphQL response."""
        mock_post.return_value = MagicMock(
            status_code=200,
            ok=True,
            json=MagicMock(return_value={
                "data": {"myself": {"clientBalance": 42.50, "currentSpendPerHr": 0.17}},
            }),
        )
        mgr = self._make_manager()
        rate = mgr.get_spend_per_hour()
        assert rate == 0.17

    # -- env dict conversion --

    @patch("deploy.runpod_manager.requests.post")
    def test_env_dict_converted(self, mock_post: MagicMock) -> None:
        """Dict env is converted to RunPod's list-of-dicts format."""
        mock_post.return_value = MagicMock(
            status_code=201,
            json=MagicMock(return_value={"id": "pod789"}),
        )
        mgr = self._make_manager()
        mgr.create_pod(
            name="env-test",
            image_name="img",
            env={"FOO": "bar", "BAZ": "qux"},
        )

        call_kwargs = mock_post.call_args
        body = call_kwargs.kwargs.get("json") or json.loads(call_kwargs.kwargs.get("data", "{}"))
        env_list = body.get("env", [])
        assert {"key": "FOO", "value": "bar"} in env_list
        assert {"key": "BAZ", "value": "qux"} in env_list

    # -- default GPU types --

    @patch("deploy.runpod_manager.requests.post")
    def test_default_gpu_types_used(self, mock_post: MagicMock) -> None:
        """When no gpu_types passed, defaults are used."""
        from deploy.runpod_manager import DEFAULT_GPU_TYPES
        mock_post.return_value = MagicMock(
            status_code=201,
            json=MagicMock(return_value={"id": "pod000"}),
        )
        mgr = self._make_manager()
        mgr.create_pod(name="default-test", image_name="img")

        call_kwargs = mock_post.call_args
        body = call_kwargs.kwargs.get("json") or json.loads(call_kwargs.kwargs.get("data", "{}"))
        assert body["gpuTypeIds"] == DEFAULT_GPU_TYPES
