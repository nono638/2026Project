"""Pull all Ollama models required for RAGBench experiments.

Run standalone or called by setup.py. Idempotent — skips already-pulled models.

Usage:
    python scripts/pull_models.py
"""

from __future__ import annotations

import subprocess
import sys

# All Ollama models needed for the full experiment matrix (task-054).
# Exp 1: 10 LLMs across Qwen 3.5 small + Qwen 3.6 large + Gemma 4.
# Exp 2: 4 LLMs (subset of Exp 1's small Qwen tier) + same embedder.
# Total unique pulls = 10 LLMs + mxbai-embed-large = 11 entries.
#
# task-053: explicit-quant pinning (`-q4_K_M`) is preferred but couldn't be
# verified against ollama.com from the nighttime sandbox. The runtime helper
# get_ollama_model_details() stamps the resolved Q* level on every row, so
# analysis records the actual quant regardless of whether the tag is bare or
# explicit. Morning-side TODO: `ollama show <tag>` on the 5090 and replace
# bare tags with `-q4_K_M` variants once confirmed published.
REQUIRED_MODELS = [
    # Qwen 3.5 small — Exp 1 + Exp 2 small-tier
    "qwen3.5:0.8b",
    "qwen3.5:2b",
    "qwen3.5:4b",
    "qwen3.5:9b",
    # Qwen 3.6 large — Exp 1 large-tier
    "qwen3.6:27b",
    "qwen3.6:35b-a3b",
    # Gemma 4 — Exp 1 cross-family
    "gemma4:e2b",
    "gemma4:e4b",
    "gemma4:26b",
    "gemma4:31b",
    # Embedding model — single F16 artifact, no quant variant published
    "mxbai-embed-large",
]


def get_installed_models() -> set[str]:
    """Get the set of already-installed Ollama models.

    Returns:
        Set of model name strings (e.g., {"qwen3.5:0.8b", "mxbai-embed-large"}).
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
        # Ollama may show "qwen3.5:0.8b" as "qwen3.5:0.8b" or with extra tag info
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
