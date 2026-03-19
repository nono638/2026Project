"""RunPod pod lifecycle management module.

Wraps RunPod's REST API for creating, terminating, and querying pods,
and their GraphQL API for account balance/spend queries. This is deployment
infrastructure for running experiments on rented GPUs — it does not touch
the experiment code in src/.

REST API docs: https://docs.runpod.io/api-reference/pods/POST/pods
GraphQL docs: https://docs.runpod.io/sdks/graphql/manage-pods
Proxy URL docs: https://docs.runpod.io/pods/configuration/expose-ports
"""

from __future__ import annotations

import logging
import time

import requests

logger = logging.getLogger(__name__)

# --- Constants ---

RUNPOD_REST_BASE = "https://rest.runpod.io/v1"
RUNPOD_GRAPHQL_URL = "https://api.runpod.io/graphql"

# Ordered by price — cheapest viable first, widely-available last resort
# Pricing as of March 2026 on RunPod Community Cloud
DEFAULT_GPU_TYPES = [
    "NVIDIA RTX A5000",                 # 24GB, ~$0.27/hr — cheapest 24GB option
    "NVIDIA RTX 4000 Ada Generation",   # 20GB, ~$0.26/hr — good fallback (exact ID may vary)
    "NVIDIA RTX A4000",                 # 16GB — may still be available, cheapest if so
    "NVIDIA GeForce RTX 4090",          # 24GB, ~$0.59/hr — widely available last resort
]

DEFAULT_PORTS = ["11434/http", "8000/http"]  # Ollama + FastAPI


class RunPodError(Exception):
    """Exception for RunPod API errors.

    Includes HTTP status code and response body for debugging.
    """

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        response_body: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


class RunPodManager:
    """Manages RunPod pod lifecycle via REST API and balance via GraphQL.

    Uses terminate+recreate instead of stop+resume because stopped pods can
    resume with zero GPUs if the machine was re-rented. Terminate+recreate
    uses the GPU fallback list and always gets a GPU.

    Args:
        api_key: RunPod API key.
        default_gpu_types: Ordered GPU fallback list. RunPod picks whichever
            is in stock first (gpuTypePriority: "availability").
    """

    def __init__(
        self,
        api_key: str,
        default_gpu_types: list[str] | None = None,
    ) -> None:
        self._api_key = api_key
        self._default_gpu_types = default_gpu_types or DEFAULT_GPU_TYPES
        # REST API uses Bearer token auth
        self._headers = {"Authorization": f"Bearer {api_key}"}

    def create_pod(
        self,
        name: str,
        image_name: str,
        ports: list[str] | None = None,
        volume_gb: int = 20,
        env: dict[str, str] | None = None,
        gpu_types: list[str] | None = None,
    ) -> dict:
        """Create a new pod with GPU fallback.

        Args:
            name: Human-readable pod name.
            image_name: Docker image to run.
            ports: Ports to expose. Defaults to Ollama (11434) + FastAPI (8000).
            volume_gb: Volume size in GB. Defaults to 20.
            env: Environment variables as a flat dict (e.g., {"KEY": "value"}).
            gpu_types: Ordered GPU fallback list. Overrides constructor default.

        Returns:
            Full pod dict from RunPod's response (includes id, desiredStatus, etc.).

        Raises:
            RunPodError: On non-2xx response from RunPod API.
        """
        gpu_list = gpu_types or self._default_gpu_types
        port_list = ports or DEFAULT_PORTS

        body = {
            "name": name,
            "imageName": image_name,
            "gpuTypeIds": gpu_list,
            "gpuTypePriority": "availability",
            "gpuCount": 1,
            "volumeInGb": volume_gb,
            "ports": port_list,
            "cloudType": "COMMUNITY",
            "env": env or {},
        }

        logger.info("Creating pod '%s' with GPU fallback: %s", name, gpu_list)
        resp = requests.post(
            f"{RUNPOD_REST_BASE}/pods",
            headers=self._headers,
            json=body,
        )

        if not resp.ok:
            raise RunPodError(
                f"Failed to create pod '{name}': {resp.text}",
                status_code=resp.status_code,
                response_body=resp.text,
            )

        pod = resp.json()
        logger.info("Pod created: id=%s", pod.get("id"))
        return pod

    def terminate_pod(self, pod_id: str) -> None:
        """Terminate (permanently delete) a pod.

        Uses terminate instead of stop because stopped pods can resume with
        zero GPUs if the machine was re-rented. 404 is silently ignored
        (pod already gone).

        Args:
            pod_id: RunPod pod ID to terminate.

        Raises:
            RunPodError: On non-2xx response (except 404).
        """
        logger.info("Terminating pod %s", pod_id)
        resp = requests.delete(
            f"{RUNPOD_REST_BASE}/pods/{pod_id}",
            headers=self._headers,
        )

        # 404 means pod is already gone — not an error
        if resp.status_code == 404:
            logger.info("Pod %s already terminated (404)", pod_id)
            return

        if not resp.ok:
            raise RunPodError(
                f"Failed to terminate pod '{pod_id}': {resp.text}",
                status_code=resp.status_code,
                response_body=resp.text,
            )

        logger.info("Pod %s terminated", pod_id)

    def get_pod(self, pod_id: str) -> dict | None:
        """Get pod status.

        Args:
            pod_id: RunPod pod ID.

        Returns:
            Pod dict, or None if the pod doesn't exist (404).

        Raises:
            RunPodError: On non-2xx response (except 404).
        """
        resp = requests.get(
            f"{RUNPOD_REST_BASE}/pods/{pod_id}",
            headers=self._headers,
        )

        if resp.status_code == 404:
            return None

        if not resp.ok:
            raise RunPodError(
                f"Failed to get pod '{pod_id}': {resp.text}",
                status_code=resp.status_code,
                response_body=resp.text,
            )

        return resp.json()

    def list_pods(self) -> list[dict]:
        """List all pods on the account.

        Returns:
            List of pod dicts.

        Raises:
            RunPodError: On non-2xx response.
        """
        resp = requests.get(
            f"{RUNPOD_REST_BASE}/pods",
            headers=self._headers,
        )

        if not resp.ok:
            raise RunPodError(
                f"Failed to list pods: {resp.text}",
                status_code=resp.status_code,
                response_body=resp.text,
            )

        return resp.json()

    def get_pod_url(self, pod_id: str, port: int = 11434) -> str:
        """Build the stable HTTP proxy URL for a pod.

        The URL is stable across pod restarts — it's based on pod ID, not IP.
        See: https://docs.runpod.io/pods/configuration/expose-ports

        Args:
            pod_id: RunPod pod ID.
            port: Port number to proxy. Defaults to 11434 (Ollama).

        Returns:
            Proxy URL string.
        """
        return f"https://{pod_id}-{port}.proxy.runpod.net"

    def wait_for_ready(
        self,
        pod_id: str,
        timeout_s: int = 300,
        poll_interval_s: int = 10,
    ) -> bool:
        """Poll until pod is running and ready.

        Uses the GraphQL API to check for the runtime field, which is only
        populated once the container is actually serving. The REST API does
        not expose this field.

        Args:
            pod_id: RunPod pod ID.
            timeout_s: Maximum seconds to wait. Defaults to 300 (5 min).
            poll_interval_s: Seconds between polls. Defaults to 10.

        Returns:
            True if pod became ready within timeout, False otherwise.
        """
        logger.info("Waiting for pod %s to be ready (timeout=%ds)", pod_id, timeout_s)
        elapsed = 0
        query = (
            '{ pod(input: {podId: "' + pod_id + '"}) {'
            "  id desiredStatus runtime { uptimeInSeconds }"
            "} }"
        )

        while elapsed < timeout_s:
            try:
                data = self._graphql_query(query)
                pod = data.get("pod")
                if pod is None:
                    logger.warning("Pod %s disappeared while waiting", pod_id)
                    return False

                runtime = pod.get("runtime")
                if pod.get("desiredStatus") == "RUNNING" and runtime is not None:
                    logger.info("Pod %s is ready (uptime: %ss)",
                                pod_id, runtime.get("uptimeInSeconds", "?"))
                    return True
            except RunPodError as exc:
                logger.debug("GraphQL poll error (retrying): %s", exc)

            time.sleep(poll_interval_s)
            elapsed += poll_interval_s

        logger.warning("Pod %s not ready after %ds", pod_id, timeout_s)
        return False

    def _graphql_query(self, query: str) -> dict:
        """Execute a GraphQL query against RunPod's API.

        Auth is via query param (not header) per RunPod's GraphQL convention.
        See: https://docs.runpod.io/sdks/graphql/manage-pods

        Args:
            query: GraphQL query string.

        Returns:
            The 'data' dict from the GraphQL response.

        Raises:
            RunPodError: On non-2xx response or missing data.
        """
        resp = requests.post(
            f"{RUNPOD_GRAPHQL_URL}?api_key={self._api_key}",
            json={"query": query},
        )

        if not resp.ok:
            raise RunPodError(
                f"GraphQL query failed: {resp.text}",
                status_code=resp.status_code,
                response_body=resp.text,
            )

        result = resp.json()
        if "data" not in result:
            raise RunPodError(
                f"GraphQL response missing 'data': {result}",
                response_body=str(result),
            )

        return result["data"]

    def get_balance(self) -> float:
        """Query remaining account balance in USD.

        Returns:
            Account balance as a float (USD).

        Raises:
            RunPodError: On API error or invalid response.
        """
        data = self._graphql_query(
            "query { myself { currentSpendPerHr clientBalance } }"
        )
        try:
            balance = data["myself"]["clientBalance"]
        except (KeyError, TypeError) as exc:
            raise RunPodError(
                f"Unexpected GraphQL response structure: {data}"
            ) from exc
        logger.info("Account balance: $%.2f", balance)
        return float(balance)

    def get_spend_per_hour(self) -> float:
        """Query current spend rate in USD per hour.

        Returns:
            Current spend rate as a float (USD/hr).

        Raises:
            RunPodError: On API error or invalid response.
        """
        data = self._graphql_query(
            "query { myself { currentSpendPerHr clientBalance } }"
        )
        try:
            rate = data["myself"]["currentSpendPerHr"]
        except (KeyError, TypeError) as exc:
            raise RunPodError(
                f"Unexpected GraphQL response structure: {data}"
            ) from exc
        logger.info("Current spend rate: $%.2f/hr", rate)
        return float(rate)
