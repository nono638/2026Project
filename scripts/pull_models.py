"""Pull all Ollama models required for RAGBench experiments.

Run standalone or called by setup.py. Idempotent — skips already-pulled models.

Usage:
    python scripts/pull_models.py
"""

from __future__ import annotations

import subprocess
import sys

# All Ollama models needed for the full experiment matrix
REQUIRED_MODELS = [
    # Qwen3 family — primary experiment axis
    "qwen3:0.6b",
    "qwen3:1.7b",
    "qwen3:4b",
    "qwen3:8b",
    # Gemma 3 — cross-family validation
    "gemma3:1b",
    "gemma3:4b",
    # Embedding model
    "mxbai-embed-large",
]


def get_installed_models() -> set[str]:
    """Get the set of already-installed Ollama models.

    Returns:
        Set of model name strings (e.g., {"qwen3:0.6b", "mxbai-embed-large"}).
    """
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except FileNotFoundError:
        print("ERROR: Ollama not found. Install from https://ollama.com")
        sys.exit(1)
    except subprocess.TimeoutExpired:
        print("ERROR: Ollama not responding. Start it with: ollama serve")
        sys.exit(1)

    if result.returncode != 0:
        print(f"ERROR: 'ollama list' failed: {result.stderr.strip()}")
        sys.exit(1)

    # Parse output: first column is model name (NAME column)
    installed: set[str] = set()
    for line in result.stdout.strip().split("\n")[1:]:  # Skip header
        if line.strip():
            # Model name is the first whitespace-delimited token
            model_name = line.split()[0]
            # Ollama list shows "name:tag" format — normalize
            installed.add(model_name)

    return installed


def pull_models() -> None:
    """Pull all required models, skipping those already installed."""
    print("RAGBench Model Puller")
    print("=" * 40)

    installed = get_installed_models()
    pulled = 0
    skipped = 0
    failed = 0
    failures: list[str] = []

    for model in REQUIRED_MODELS:
        # Check if model is already installed (handle tag variations)
        # Ollama may show "qwen3:0.6b" as "qwen3:0.6b" or with extra tag info
        if any(model in m or m.startswith(model.split(":")[0] + ":" + model.split(":")[-1]) for m in installed):
            print(f"  [SKIP] {model} (already installed)")
            skipped += 1
            continue

        print(f"  [PULL] {model}...")
        try:
            result = subprocess.run(
                ["ollama", "pull", model],
                capture_output=True,
                text=True,
                timeout=600,  # 10 minutes per model
            )
            if result.returncode == 0:
                print(f"  [DONE] {model}")
                pulled += 1
            else:
                print(f"  [FAIL] {model}: {result.stderr.strip()}")
                failed += 1
                failures.append(model)
        except subprocess.TimeoutExpired:
            print(f"  [FAIL] {model}: timed out (>10 minutes)")
            failed += 1
            failures.append(model)
        except Exception as e:
            print(f"  [FAIL] {model}: {e}")
            failed += 1
            failures.append(model)

    # Summary
    print()
    print("Summary")
    print("-" * 40)
    print(f"  Pulled:  {pulled}")
    print(f"  Skipped: {skipped} (already installed)")
    print(f"  Failed:  {failed}")
    if failures:
        print(f"  Failed models: {', '.join(failures)}")
        print("  Re-run this script to retry failed models.")


if __name__ == "__main__":
    pull_models()
