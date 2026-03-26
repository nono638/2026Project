# task-046: Experiment 0v3 — Higher-n Scorer Validation (n=500)

## What

Run Experiment 0v3 with n=500 medium+hard HotpotQA questions to resolve scorer ranking
discrepancies between v1 (n=50) and v2 (n=150). Drop gemini-3.1-pro-preview (rate-limited).
This is the tiebreaker run that determines which scorer to use for Experiments 1 & 2.

## Why

v1 and v2 produced different scorer rankings:
- v1 (n=50): Sonnet best (r=0.682), Haiku middling (r=0.368)
- v2 (n=150): Haiku best (r=0.640), Sonnet dropped (r=0.561)

n=500 provides enough statistical power to settle this with confidence. The scorer choice
for Experiments 1 & 2 (4,500+ scorer calls) depends on this result.

gemini-3.1-pro-preview is excluded because its free-tier quota is 250 requests/day — it
would fail on 250 of 500 questions. Can be backfilled later with `--judges 3.1-pro`.

## Implementation — Three Parts

### Part 1: Add `--generation-only` flag to `run_experiment_0.py`

This lets us terminate the RunPod GPU pod after generation finishes, before the ~5.5 hour
scoring phase that only uses cloud APIs.

**File:** `scripts/run_experiment_0.py`

**Changes:**
1. Add to `parse_args()`:
   ```python
   parser.add_argument("--generation-only", action="store_true",
                       help="Generate answers and save raw_answers.csv, then exit "
                            "(skip scoring). Use with --skip-generation later to score.")
   ```

2. In `main()`, after the raw answers are saved (after `answers_df.to_csv(raw_answers_path, ...)`),
   add an early exit:
   ```python
   if args.generation_only:
       print(f"\n--generation-only: answers saved to {raw_answers_path}")
       print(f"Run again with --skip-generation to score.")
       return
   ```

   This goes right after `logger.info("Saved raw answers to %s", raw_answers_path)` (around
   line 710) and before the `# Step 4: Score all answers` section.

**Do NOT** add this exit in the `--skip-generation` branch — only in the generation branch.

### Part 2: Write wrapper script `scripts/run_v3.py`

A self-contained Python script that orchestrates the full v3 pipeline:
1. Deploy RunPod pod (using `deploy.runpod_manager.RunPodManager` and `deploy.setup_pod`)
2. Generate 500 answers with `--generation-only`
3. Terminate the pod (saves ~$1 in idle GPU during scoring)
4. Score with `--skip-generation` (cloud APIs only, no GPU needed)

**File:** `scripts/run_v3.py`

**Key requirements:**
- Import `RunPodManager` from `deploy.runpod_manager` and helper functions from
  `deploy.setup_pod` (`wait_for_ollama`, `pull_model`, `verify_model`)
- Load API key from `.env` using `dotenv`
- **Always terminate the pod in a `finally` block** — even if generation crashes, stop billing
- Log all output to both stdout and `results/experiment_0_v3/run_v3.log`
- Use `subprocess.run()` to call `run_experiment_0.py` (not import — it uses `sys.exit()`)

**Script flow:**

```python
"""Experiment 0v3 orchestrator — deploy GPU, generate, terminate, score."""

import logging
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from deploy.runpod_manager import RunPodManager
from deploy.setup_pod import wait_for_ollama, pull_model

# Config
OUTPUT_DIR = PROJECT_ROOT / "results" / "experiment_0_v3"
N = 500
SEED = 42
MODEL = "qwen3:4b"
EMBED_MODEL = "mxbai-embed-large"
JUDGES = ["2.5", "haiku", "sonnet", "opus"]  # excludes gemini-3.1-pro-preview
PYTHON = sys.executable

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Set up logging to file + stdout
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
        if balance < 2.00:
            logger.error("Balance below $2.00 — not enough for v3 generation")
            sys.exit(1)

        pod = manager.create_pod(
            name="ragbench-v3",
            image_name="ollama/ollama",
            env={"OLLAMA_HOST": "0.0.0.0",
                 "OLLAMA_MODELS": "/workspace/ollama_models"},
            ports=["11434/http"],
            volume_gb=20,
            container_disk_gb=40,
        )
        pod_id = pod["id"]
        logger.info("Pod created: %s", pod_id)

        # Wait for pod + Ollama
        manager.wait_for_ready(pod_id, timeout_s=600)
        ollama_url = manager.get_pod_url(pod_id)
        logger.info("Ollama URL: %s", ollama_url)

        if not wait_for_ollama(ollama_url, timeout_s=300):
            logger.error("Ollama not responding at %s", ollama_url)
            sys.exit(1)

        # Pull models
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
        # Always terminate pod — even on crash
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
```

**Important implementation details:**
- The `finally` block MUST terminate the pod even if generation fails. GPU billing doesn't
  stop until the pod is terminated.
- Use `subprocess.run()` not `os.system()` — it returns the exit code properly.
- Pass `--no-gallery` to both commands since the gallery doesn't know about v3 yet.
- The `JUDGES = ["2.5", "haiku", "sonnet", "opus"]` filter excludes gemini-3.1-pro-preview
  because "2.5" matches all three gemini-2.5 models but not "3.1". "haiku", "sonnet", "opus"
  match the three Anthropic models.

### Part 3: Run the experiment

After implementing Parts 1 and 2, launch the wrapper script:

```bash
python scripts/run_v3.py
```

This will run as a single long process. The generation phase takes ~3.5 hours on a
RunPod A4000/A5000. The scoring phase takes ~5.5 hours using cloud APIs. Total ~9 hours.

**The process uses incremental checkpointing** (already implemented in `run_experiment_0.py`)
so if scoring crashes, re-running `python scripts/run_experiment_0.py --skip-generation
--output-dir results/experiment_0_v3 --judges 2.5 haiku sonnet opus` resumes from where
it left off.

## Files to Create

- `scripts/run_v3.py` — orchestrator script (see Part 2 above)

## Files to Modify

- `scripts/run_experiment_0.py` — add `--generation-only` flag (see Part 1 above)

## Files NOT to Touch

- `deploy/runpod_manager.py` — use as-is
- `deploy/setup_pod.py` — use as-is (import helpers from it)
- `results/experiment_0/` — v1 data, do not modify
- `results/experiment_0_v2/` — v2 data, do not modify
- `src/` — no source changes

## Testing

### Test the `--generation-only` flag

Add a test in `tests/test_experiment_0.py` (or create it) that verifies:
1. `parse_args()` accepts `--generation-only`
2. When `--generation-only` is set AND `--skip-generation` is NOT set, the script would
   exit after saving answers (mock the generation, verify the early return path)

### Test the wrapper script

Add `tests/test_run_v3.py` with tests that mock `RunPodManager` and `subprocess.run`:
1. **Happy path**: pod creates → generation succeeds → pod terminates → scoring succeeds
2. **Generation failure**: pod creates → generation fails → pod still terminates (verify
   `terminate_pod` called in `finally`)
3. **Pod creation failure**: exits with error, no terminate called (pod_id is None)
4. **Low balance**: exits with error before pod creation

## Edge Cases

- If RunPod has no GPUs available, the script exits with a clear error.
- If generation fails mid-run, the `finally` block terminates the pod. Raw answers saved
  so far can be used for a partial re-run.
- If scoring hits API rate limits, the checkpoint file preserves progress. Re-run the
  scoring command to resume.
- If the user's RunPod balance is below $2.00, the script exits before creating a pod.

## What NOT to Do

- Don't modify the scoring checkpoint logic — it already works
- Don't add gemini-3.1-pro-preview to the judges list
- Don't modify v1 or v2 result data
- Don't run the gallery generator — v3 gallery support is a separate task
