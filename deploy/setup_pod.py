"""End-to-end RunPod pod setup for RAGBench experiments.

Creates a pod with the ollama/ollama Docker image, pulls models via Ollama's
HTTP API, and verifies everything works. No SSH required — all communication
goes through RunPod's HTTPS proxy URL.

Why ollama/ollama image: it has Ollama pre-installed and auto-starts the server.
No startup scripts, no waiting for installation. Simpler and faster.
Docs: https://hub.docker.com/r/ollama/ollama

Why 40GB container disk: the ollama/ollama image is based on NVIDIA CUDA
(~5-8GB). With the default 10GB container disk, the image can't fully extract
and the container silently fails to start (runtime stays null). RunPod's
official examples all use 40GB. Ref: https://docs.runpod.io/sdks/graphql/manage-pods

Why volumeMountPath + OLLAMA_MODELS: models are stored on the persistent
volume (/workspace) instead of the container disk. This prevents disk
exhaustion when pulling large models and survives pod restarts.

Why HTTP API for model pulling: RunPod doesn't have a command execution API.
SSH requires key setup. Ollama's HTTP API (/api/pull) does the same thing.
Docs: https://github.com/ollama/ollama/blob/main/docs/api.md#pull-a-model

Why stream:true for pulls: Cloudflare (RunPod's proxy) enforces a 100-second
connection timeout. A 2.5GB model pull takes longer than that. Streaming keeps
the connection alive.

Usage:
    python deploy/setup_pod.py                              # full setup
    python deploy/setup_pod.py --pod-id abc123 --pull-only  # pull models only
    python deploy/setup_pod.py --pod-id abc123 --pull-only --models qwen3:0.6b qwen3:1.7b
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import requests

# Ensure project root is on sys.path so deploy/ imports work
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from deploy.runpod_manager import RunPodManager, RunPodError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Default models to pull — embedding model + generation model for Experiment 0
DEFAULT_MODELS = ["mxbai-embed-large", "qwen3:4b"]

# Timeouts — community cloud image pull + Ollama boot can take a few minutes
POD_READY_TIMEOUT_S = 600
OLLAMA_POLL_TIMEOUT_S = 600
OLLAMA_POLL_INTERVAL_S = 10
MODEL_PULL_TIMEOUT_S = 600


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Set up a RunPod pod with Ollama for RAGBench experiments.",
    )
    parser.add_argument(
        "--pod-id",
        type=str,
        default=None,
        help="Use an existing pod instead of creating a new one.",
    )
    parser.add_argument(
        "--pull-only",
        action="store_true",
        help="Skip pod creation, just pull models. Requires --pod-id.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Models to pull (default: mxbai-embed-large qwen3:4b).",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="ragbench-gpu",
        help="Pod name (default: ragbench-gpu).",
    )
    parser.add_argument(
        "--image",
        type=str,
        default="ollama/ollama",
        help="Docker image (default: ollama/ollama).",
    )
    parser.add_argument(
        "--volume-gb",
        type=int,
        default=20,
        help="Volume size in GB (default: 20).",
    )

    args = parser.parse_args()

    if args.pull_only and not args.pod_id:
        parser.error("--pull-only requires --pod-id")

    return args


def _load_env() -> str:
    """Load .env file and return the RunPod API key.

    Returns:
        The RUNPOD_API_KEY value.

    Raises:
        SystemExit: If the API key is not set.
    """
    import os

    try:
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT / ".env")
    except ImportError:
        pass

    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        print("\nERROR: RUNPOD_API_KEY is not set.")
        print("Set it in your .env file or export RUNPOD_API_KEY=<your-key>")
        print("See deploy/RUNPOD_SETUP.md for setup instructions.")
        sys.exit(1)

    return api_key


def wait_for_ollama(url: str, timeout_s: int = OLLAMA_POLL_TIMEOUT_S) -> bool:
    """Poll Ollama's /api/tags endpoint until it responds.

    The pod being 'ready' in RunPod doesn't mean Ollama is serving yet —
    it may take 10-30 seconds after boot for the server to start.

    Args:
        url: Base Ollama URL (e.g., https://pod-id-11434.proxy.runpod.net).
        timeout_s: Maximum seconds to wait. Defaults to 600.

    Returns:
        True if Ollama responded within timeout, False otherwise.
    """
    elapsed = 0
    while elapsed < timeout_s:
        try:
            resp = requests.get(f"{url}/api/tags", timeout=10)
            if resp.status_code == 200:
                logger.info("Ollama is responding at %s", url)
                return True
        except Exception:
            # Broad catch — any connection error, timeout, etc. means Ollama
            # isn't ready yet. We'll retry until the timeout expires.
            pass

        time.sleep(OLLAMA_POLL_INTERVAL_S)
        elapsed += OLLAMA_POLL_INTERVAL_S
        logger.info("Waiting for Ollama... (%ds/%ds)", elapsed, timeout_s)

    return False


def pull_model(url: str, model_name: str) -> bool:
    """Pull a model via Ollama's HTTP API using streaming.

    Uses stream:true and reads progress lines to keep the connection alive,
    avoiding Cloudflare's 100-second timeout on RunPod's proxy.

    Args:
        url: Base Ollama URL.
        model_name: Model to pull (e.g., 'qwen3:4b').

    Returns:
        True if pull succeeded, False otherwise.
    """
    logger.info("Pulling model: %s", model_name)

    try:
        # Stream:true sends newline-delimited JSON progress lines
        resp = requests.post(
            f"{url}/api/pull",
            json={"name": model_name, "stream": True},
            stream=True,
            timeout=MODEL_PULL_TIMEOUT_S,
        )

        if resp.status_code != 200:
            logger.error("Pull request failed for %s: HTTP %d", model_name, resp.status_code)
            return False

        # Read and discard progress lines, printing a dot every 10 lines
        # to show activity and keep the connection alive
        line_count = 0
        last_status = ""
        for line in resp.iter_lines():
            if line:
                line_count += 1
                try:
                    progress = json.loads(line)
                    last_status = progress.get("status", "")
                except json.JSONDecodeError:
                    pass

                if line_count % 10 == 0:
                    print(".", end="", flush=True)

        print()  # Newline after dots
        logger.info("Pull complete for %s (status: %s)", model_name, last_status)
        return True

    except requests.Timeout:
        logger.error("Pull timed out for %s after %ds", model_name, MODEL_PULL_TIMEOUT_S)
        return False
    except requests.RequestException as exc:
        logger.error("Pull failed for %s: %s", model_name, exc)
        return False


def verify_model(url: str, model_name: str) -> bool:
    """Verify a model works by sending a short test prompt.

    Args:
        url: Base Ollama URL.
        model_name: Model to verify.

    Returns:
        True if the model responded with HTTP 200, False otherwise.
    """
    try:
        resp = requests.post(
            f"{url}/api/generate",
            json={"model": model_name, "prompt": "Say hello", "stream": False},
            timeout=60,
        )
        if resp.status_code == 200:
            logger.info("Model %s verified OK", model_name)
            return True
        else:
            logger.warning("Model %s verification failed: HTTP %d", model_name, resp.status_code)
            return False
    except requests.RequestException as exc:
        logger.warning("Model %s verification failed: %s", model_name, exc)
        return False


def print_summary(
    pod_id: str,
    url: str,
    balance: float | None,
    rate: float | None,
    pull_results: dict[str, bool],
) -> None:
    """Print a summary block with pod info and pull results.

    Args:
        pod_id: RunPod pod ID.
        url: Ollama proxy URL.
        balance: Account balance in USD, or None if unavailable.
        rate: Spend rate in USD/hr, or None if unavailable.
        pull_results: Dict mapping model name to pull success status.
    """
    print()
    print("=" * 40)
    print("  Pod is ready for experiments!")
    print(f"  Pod ID:     {pod_id}")
    print(f"  Ollama URL: {url}")
    if rate is not None:
        print(f"  Spend rate: ${rate:.2f}/hr")
    if balance is not None:
        print(f"  Balance:    ${balance:.2f}")
    print("=" * 40)

    # Note any failed pulls
    failed = [m for m, ok in pull_results.items() if not ok]
    if failed:
        print(f"\nWARNING: Failed to pull: {', '.join(failed)}")
        print("Re-run with: python deploy/setup_pod.py --pod-id {} --pull-only --models {}".format(
            pod_id, " ".join(failed),
        ))


def main() -> None:
    """Run the pod setup workflow."""
    args = parse_args()
    api_key = _load_env()
    manager = RunPodManager(api_key=api_key)

    pod_id = args.pod_id

    if not args.pull_only:
        # Full setup mode: check balance, create pod, wait for ready
        balance = manager.get_balance()
        print(f"Account balance: ${balance:.2f}")

        if balance < 1.00:
            print("\nERROR: Balance is below $1.00. Add credits before creating a pod.")
            sys.exit(1)

        # Create pod with ollama/ollama image.
        # OLLAMA_HOST=0.0.0.0: accept external connections (required for proxy)
        # OLLAMA_MODELS=/workspace/ollama_models: store models on persistent
        #   volume so they survive pod restarts and don't fill container disk.
        # container_disk_gb=40: ollama/ollama is a large CUDA-based image;
        #   10GB (old default) caused silent container startup failures.
        # Ref: https://docs.runpod.io/tutorials/pods/run-ollama
        try:
            pod = manager.create_pod(
                name=args.name,
                image_name=args.image,
                env={
                    "OLLAMA_HOST": "0.0.0.0",
                    "OLLAMA_MODELS": "/workspace/ollama_models",
                },
                ports=["11434/http"],
                volume_gb=args.volume_gb,
                container_disk_gb=40,
                volume_mount_path="/workspace",
            )
        except RunPodError as exc:
            print(f"\nERROR: Failed to create pod: {exc}")
            sys.exit(1)

        pod_id = pod["id"]
        print(f"Pod created: {pod_id}")

        # Wait for pod to be ready in RunPod
        if not manager.wait_for_ready(pod_id, timeout_s=POD_READY_TIMEOUT_S):
            url = f"https://www.runpod.io/console/pods/{pod_id}"
            print(f"\nERROR: Pod did not become ready within {POD_READY_TIMEOUT_S}s.")
            print(f"Check the RunPod console: {url}")
            sys.exit(1)

    # Build proxy URL
    url = manager.get_pod_url(pod_id)

    # Wait for Ollama to actually respond
    print(f"Waiting for Ollama at {url}...")
    if not wait_for_ollama(url):
        print(f"\nERROR: Ollama is not responding at {url}")
        print("Troubleshooting:")
        print("  1. Check the pod logs in the RunPod console")
        print("  2. Ensure the pod image is ollama/ollama")
        print("  3. Ensure OLLAMA_HOST=0.0.0.0 is set in pod env")
        print("  4. Ensure port 11434 is exposed as HTTP")
        if not args.pull_only:
            print(f"\nTerminating pod {pod_id} to stop billing...")
            try:
                manager.terminate_pod(pod_id)
                print("Pod terminated.")
            except RunPodError as exc:
                print(f"WARNING: Could not terminate pod: {exc}")
                print(f"Terminate manually: https://www.runpod.io/console/pods")
        sys.exit(1)

    # Pull models
    pull_results: dict[str, bool] = {}
    for model_name in args.models:
        success = pull_model(url, model_name)
        pull_results[model_name] = success

        if not success:
            print(f"WARNING: Failed to pull {model_name}, continuing...")

    # Verify models that were pulled successfully
    for model_name, pulled_ok in pull_results.items():
        if pulled_ok:
            verified = verify_model(url, model_name)
            if not verified:
                print(f"WARNING: Model {model_name} pulled but verification failed")

    # Get balance/rate for summary
    try:
        balance = manager.get_balance()
        rate = manager.get_spend_per_hour()
    except RunPodError:
        balance = None
        rate = None

    print_summary(pod_id, url, balance, rate, pull_results)


if __name__ == "__main__":
    main()
