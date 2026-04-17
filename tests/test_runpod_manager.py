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

    # -- create_pod (uses GraphQL podFindAndDeployOnDemand) --

    @patch("deploy.runpod_manager.time.sleep")
    def test_create_pod_success(self, mock_sleep: MagicMock) -> None:
        """Successful pod creation returns pod dict from GraphQL."""
        mgr = self._make_manager()
        with patch.object(mgr, "_graphql_query", return_value={
            "podFindAndDeployOnDemand": {
                "id": "pod123",
                "desiredStatus": "RUNNING",
                "machine": {"gpuDisplayName": "NVIDIA RTX A5000"},
            },
        }):
            result = mgr.create_pod(name="test-pod", image_name="runpod/pytorch")

        assert result["id"] == "pod123"
        assert result["desiredStatus"] == "RUNNING"

    @patch("deploy.runpod_manager.time.sleep")
    def test_create_pod_custom_gpu_types(self, mock_sleep: MagicMock) -> None:
        """Custom gpu_types are tried in order; first success wins."""
        mgr = self._make_manager()
        custom_gpus = ["NVIDIA A100 80GB PCIe"]
        with patch.object(mgr, "_graphql_query", return_value={
            "podFindAndDeployOnDemand": {
                "id": "pod456",
                "desiredStatus": "RUNNING",
                "machine": {"gpuDisplayName": "NVIDIA A100 80GB PCIe"},
            },
        }) as mock_gql:
            mgr.create_pod(name="big-pod", image_name="img", gpu_types=custom_gpus)

        # Verify the GraphQL query included our custom GPU
        query_str = mock_gql.call_args.args[0]
        assert "NVIDIA A100 80GB PCIe" in query_str

    @patch("deploy.runpod_manager.time.sleep")
    def test_create_pod_falls_back_to_next_gpu(self, mock_sleep: MagicMock) -> None:
        """When first GPU fails, tries the next one."""
        from deploy.runpod_manager import RunPodError
        mgr = self._make_manager()
        with patch.object(mgr, "_graphql_query", side_effect=[
            RunPodError("No machines available for NVIDIA RTX A5000"),
            {
                "podFindAndDeployOnDemand": {
                    "id": "pod789",
                    "desiredStatus": "RUNNING",
                    "machine": {"gpuDisplayName": "NVIDIA RTX A4000"},
                },
            },
        ]):
            result = mgr.create_pod(
                name="fallback-pod", image_name="img",
                gpu_types=["NVIDIA RTX A5000", "NVIDIA RTX A4000"],
            )

        assert result["id"] == "pod789"

    @patch("deploy.runpod_manager.time.sleep")
    def test_create_pod_all_gpus_fail(self, mock_sleep: MagicMock) -> None:
        """All GPUs failing raises RunPodError."""
        from deploy.runpod_manager import RunPodError
        mgr = self._make_manager()
        with patch.object(mgr, "_graphql_query", side_effect=RunPodError("No machines")):
            with pytest.raises(RunPodError, match="no GPU available"):
                mgr.create_pod(
                    name="fail-pod", image_name="img",
                    gpu_types=["GPU-A", "GPU-B"],
                )

    @patch("deploy.runpod_manager.time.sleep")
    def test_create_pod_skips_null_response(self, mock_sleep: MagicMock) -> None:
        """GraphQL returning null pod falls through to next GPU."""
        mgr = self._make_manager()
        with patch.object(mgr, "_graphql_query", side_effect=[
            {"podFindAndDeployOnDemand": None},
            {
                "podFindAndDeployOnDemand": {
                    "id": "pod999",
                    "desiredStatus": "RUNNING",
                    "machine": {"gpuDisplayName": "GPU-B"},
                },
            },
        ]) as mock_gql:
            result = mgr.create_pod(
                name="null-test", image_name="img",
                gpu_types=["GPU-A", "GPU-B"],
            )

        assert result["id"] == "pod999"
        assert mock_gql.call_count == 2

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
        # wait_for_ready uses _graphql_query (not get_pod) — mock at the right level.
        # _graphql_query returns the "data" dict; wait_for_ready looks for data["pod"].
        with patch.object(mgr, "_graphql_query", side_effect=[
            {"pod": {"id": "pod1", "desiredStatus": "CREATED", "runtime": None}},
            {"pod": {"id": "pod1", "desiredStatus": "RUNNING", "runtime": {"uptimeInSeconds": 10}}},
        ]):
            result = mgr.wait_for_ready("pod1", timeout_s=30, poll_interval_s=1)
        assert result is True

    @patch("deploy.runpod_manager.time.sleep")
    def test_wait_for_ready_timeout(self, mock_sleep: MagicMock) -> None:
        """Returns False when timeout exceeded."""
        mgr = self._make_manager()
        # Mock _graphql_query (not get_pod) — wait_for_ready calls GraphQL directly
        with patch.object(mgr, "_graphql_query", return_value={
            "pod": {"id": "pod1", "desiredStatus": "CREATED", "runtime": None},
        }):
            result = mgr.wait_for_ready("pod1", timeout_s=3, poll_interval_s=1)
        assert result is False

    @patch("deploy.runpod_manager.time.sleep")
    def test_wait_for_ready_survives_network_error(self, mock_sleep: MagicMock) -> None:
        """Network blip during polling doesn't crash — retries and succeeds."""
        import requests as req
        mgr = self._make_manager()
        with patch.object(mgr, "_graphql_query", side_effect=[
            req.ConnectionError("connection reset by peer"),
            {"pod": {"id": "pod1", "desiredStatus": "RUNNING", "runtime": {"uptimeInSeconds": 5}}},
        ]):
            result = mgr.wait_for_ready("pod1", timeout_s=30, poll_interval_s=1)
        assert result is True

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

    # -- env dict in GraphQL query --

    @patch("deploy.runpod_manager.time.sleep")
    def test_env_dict_in_graphql_query(self, mock_sleep: MagicMock) -> None:
        """Env vars are included in the GraphQL mutation."""
        mgr = self._make_manager()
        with patch.object(mgr, "_graphql_query", return_value={
            "podFindAndDeployOnDemand": {
                "id": "pod789",
                "desiredStatus": "RUNNING",
                "machine": {"gpuDisplayName": "NVIDIA RTX A5000"},
            },
        }) as mock_gql:
            mgr.create_pod(
                name="env-test",
                image_name="img",
                env={"FOO": "bar", "BAZ": "qux"},
            )

        query_str = mock_gql.call_args.args[0]
        assert 'key: "FOO"' in query_str
        assert 'value: "bar"' in query_str
        assert 'key: "BAZ"' in query_str
        assert 'value: "qux"' in query_str

    # -- default GPU types --

    @patch("deploy.runpod_manager.time.sleep")
    def test_default_gpu_types_used(self, mock_sleep: MagicMock) -> None:
        """When no gpu_types passed, defaults are tried in order."""
        from deploy.runpod_manager import DEFAULT_GPU_TYPES
        mgr = self._make_manager()
        # Succeed on the first GPU type
        with patch.object(mgr, "_graphql_query", return_value={
            "podFindAndDeployOnDemand": {
                "id": "pod000",
                "desiredStatus": "RUNNING",
                "machine": {"gpuDisplayName": DEFAULT_GPU_TYPES[0]},
            },
        }) as mock_gql:
            mgr.create_pod(name="default-test", image_name="img")

        query_str = mock_gql.call_args.args[0]
        assert DEFAULT_GPU_TYPES[0] in query_str

    # -- terminate_all_pods --

    @patch("deploy.runpod_manager.requests.get")
    @patch("deploy.runpod_manager.requests.delete")
    def test_terminate_all_pods(self, mock_delete: MagicMock, mock_get: MagicMock) -> None:
        """terminate_all_pods terminates every pod returned by list_pods."""
        mock_get.return_value = MagicMock(
            status_code=200, ok=True,
            json=MagicMock(return_value=[{"id": "a"}, {"id": "b"}]),
        )
        mock_delete.return_value = MagicMock(status_code=204, ok=True)
        mgr = self._make_manager()
        count = mgr.terminate_all_pods()
        assert count == 2
        assert mock_delete.call_count == 2

    @patch("deploy.runpod_manager.requests.get")
    def test_terminate_all_pods_empty(self, mock_get: MagicMock) -> None:
        """No pods on account returns 0."""
        mock_get.return_value = MagicMock(
            status_code=200, ok=True,
            json=MagicMock(return_value=[]),
        )
        mgr = self._make_manager()
        count = mgr.terminate_all_pods()
        assert count == 0


class TestPodIdPersistence:
    """Tests for pod_id file save/load/clear."""

    def test_save_load_clear(self, tmp_path) -> None:
        from deploy.runpod_manager import save_pod_id, load_pod_id, clear_pod_id

        assert load_pod_id(tmp_path) is None

        save_pod_id("pod-abc", tmp_path)
        assert load_pod_id(tmp_path) == "pod-abc"

        clear_pod_id(tmp_path)
        assert load_pod_id(tmp_path) is None

    def test_clear_missing_file_is_safe(self, tmp_path) -> None:
        from deploy.runpod_manager import clear_pod_id
        clear_pod_id(tmp_path)  # Should not raise


class TestCleanupStalePods:
    """Tests for the cleanup_stale_pods convenience function."""

    def test_cleans_saved_id_and_account_pods(self, tmp_path) -> None:
        from deploy.runpod_manager import (
            RunPodManager, save_pod_id, load_pod_id, cleanup_stale_pods,
        )
        save_pod_id("stale-pod", tmp_path)

        mgr = MagicMock(spec=RunPodManager)
        mgr.terminate_pod.return_value = None
        mgr.terminate_all_pods.return_value = 1

        count = cleanup_stale_pods(mgr, tmp_path)

        # stale-pod from file + 1 from terminate_all_pods
        assert count == 2
        assert load_pod_id(tmp_path) is None

    def test_survives_list_pods_failure(self, tmp_path) -> None:
        from deploy.runpod_manager import (
            RunPodManager, RunPodError, cleanup_stale_pods,
        )
        mgr = MagicMock(spec=RunPodManager)
        mgr.list_pods.side_effect = RunPodError("network error")
        mgr.terminate_all_pods.side_effect = RunPodError("network error")

        count = cleanup_stale_pods(mgr, tmp_path)
        assert count == 0
