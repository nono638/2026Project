#!/usr/bin/env python3
"""Experiment 0v3 orchestrator — deploy GPU, generate, terminate, score.

Resolves the scorer ranking discrepancy between v1 (Sonnet best, n=50) and v2
(Haiku best, n=150) by running n=500 medium+hard HotpotQA questions. Drops
gemini-3.1-pro-preview (250/day rate limit would fail on half the questions).

Three phases:
  1. Deploy RunPod GPU pod, pull models, generate answers (--generation-only)
  2. Terminate pod immediately (saves ~$1 in idle GPU during scoring)
  3. Score with cloud APIs only (--skip-generation)

The finally block ensures the pod is terminated even if generation crashes,
preventing runaway GPU billing.

Usage:
    python scripts/run_v3.py
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

from deploy.runpod_manager import RunPodManager
from deploy.setup_pod import wait_for_ollama, pull_model

# --- Configuration ---
OUTPUT_DIR = PROJECT_ROOT / "results" / "experiment_0_v3"
N = 500
SEED = 42
MODEL = "qwen3:4b"
EMBED_MODEL = "mxbai-embed-large"
# Excludes gemini-3.1-pro-preview: "2.5" matches all three gemini-2.5 models
# but not "3.1". Anthropic models matched by "haiku", "sonnet", "opus".
JUDGES = ["2.5", "haiku", "sonnet", "opus"]
PYTHON = sys.executable
# Minimum balance required to start — generation costs ~$0.50-1.00 on A5000
MIN_BALANCE = 2.00


def main() -> None:
    """Run the full Experiment 0v3 pipeline: deploy, generate, terminate, score."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Dual logging: file + stdout so the user can monitor progress and
    # also has a persistent log in the results directory.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(OUTPUT_DIR / "run_v3.log"),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger(__name__)

    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        logger.error("RUNPOD_API_KEY not set in environment")
        sys.exit(1)

    manager = RunPodManager(api_key=api_key)
    pod_id = None

    try:
        # --- Phase 1: Deploy pod ---
        balance = manager.get_balance()
        logger.info("Account balance: $%.2f", balance)
        if balance < MIN_BALANCE:
            logger.error("Balance below $%.2f — not enough for v3 generation", MIN_BALANCE)
            sys.exit(1)

        pod = manager.create_pod(
            name="ragbench-v3",
            image_name="ollama/ollama",
            env={
                "OLLAMA_HOST": "0.0.0.0",
                "OLLAMA_MODELS": "/workspace/ollama_models",
            },
            ports=["11434/http"],
            volume_gb=20,
            container_disk_gb=40,
        )
        pod_id = pod["id"]
        logger.info("Pod created: %s", pod_id)

        # Wait for pod + Ollama to be ready
        manager.wait_for_ready(pod_id, timeout_s=600)
        ollama_url = manager.get_pod_url(pod_id)
        logger.info("Ollama URL: %s", ollama_url)

        if not wait_for_ollama(ollama_url, timeout_s=300):
            logger.error("Ollama not responding at %s", ollama_url)
            sys.exit(1)

        # Pull embedding and generation models
        for model in [EMBED_MODEL, MODEL]:
            if not pull_model(ollama_url, model):
                logger.error("Failed to pull %s", model)
                sys.exit(1)

        # --- Phase 2: Generate answers (GPU needed) ---
        logger.info("Starting generation: n=%d, model=%s", N, MODEL)
        gen_cmd = [
            PYTHON, str(PROJECT_ROOT / "scripts" / "run_experiment_0.py"),
            "--n", str(N),
            "--seed", str(SEED),
            "--model", MODEL,
            "--output-dir", str(OUTPUT_DIR),
            "--ollama-host", ollama_url,
            "--generation-only",
            "--no-gallery",
        ]
        result = subprocess.run(gen_cmd, cwd=str(PROJECT_ROOT))
        if result.returncode != 0:
            logger.error("Generation failed with exit code %d", result.returncode)
            sys.exit(1)
        logger.info("Generation complete. Terminating pod to save GPU cost.")

    finally:
        # Always terminate pod — even on crash. GPU billing doesn't stop
        # until the pod is explicitly terminated.
        if pod_id:
            try:
                manager.terminate_pod(pod_id)
                logger.info("Pod %s terminated.", pod_id)
            except Exception as exc:
                logger.error("Failed to terminate pod %s: %s", pod_id, exc)
                logger.error("TERMINATE MANUALLY: https://www.runpod.io/console/pods")

    # --- Phase 3: Score (cloud APIs only, no GPU) ---
    logger.info("Starting scoring phase (cloud APIs only)...")
    score_cmd = [
        PYTHON, str(PROJECT_ROOT / "scripts" / "run_experiment_0.py"),
        "--skip-generation",
        "--output-dir", str(OUTPUT_DIR),
        "--judges", *JUDGES,
        "--no-gallery",
    ]
    result = subprocess.run(score_cmd, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        logger.error("Scoring failed with exit code %d", result.returncode)
        sys.exit(1)

    logger.info("Experiment 0v3 complete! Results in %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
