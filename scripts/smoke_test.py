"""Quick end-to-end smoke test for RAGBench.

Verifies that all components work before starting a long experiment run.
Runs in ~30 seconds, uses minimal resources. NOT a full test suite —
this is "can the pipeline run at all" verification.

Usage:
    python scripts/smoke_test.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure project root is on sys.path so src imports work
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Models required for experiments (same list as pull_models.py)
REQUIRED_MODELS = [
    "qwen3:0.6b",
    "qwen3:1.7b",
    "qwen3:4b",
    "qwen3:8b",
    "gemma3:1b",
    "gemma3:4b",
    "mxbai-embed-large",
]

# API keys and their descriptions
API_KEYS = {
    "ANTHROPIC_API_KEY": ("Claude scorer", "required for scoring"),
    "OPENAI_API_KEY": ("RAGAS query generation", "required for synthetic queries"),
    "GOOGLE_API_KEY": ("Google text embedder", "required for Google embeddings"),
    "GOOGLE_CLOUD_PROJECT": ("Google multimodal embedder", "optional"),
}


def check_imports() -> bool:
    """Verify all core modules can be imported."""
    try:
        from src.protocols import Chunker, Embedder, Strategy, Scorer  # noqa: F401
        from src.experiment import Experiment, ExperimentResult  # noqa: F401
        from src.document import Document, load_corpus_from_csv  # noqa: F401
        from src.retriever import Retriever  # noqa: F401
        print("  [PASS] Core imports")
        return True
    except ImportError as e:
        print(f"  [FAIL] Import error: {e}")
        return False


def check_ollama() -> bool:
    """Check if Ollama is running and reachable."""
    try:
        from ollama import Client
        Client().list()
        print("  [PASS] Ollama running")
        return True
    except Exception as e:
        print(f"  [FAIL] Ollama not reachable: {e}")
        return False


def check_models() -> tuple[int, int, list[str]]:
    """Check which required models are available in Ollama.

    Returns:
        Tuple of (available_count, total_count, missing_model_names).
    """
    try:
        from ollama import Client
        response = Client().list()
        # Ollama client returns model objects with 'model' attribute
        installed = set()
        if hasattr(response, "models"):
            for m in response.models:
                installed.add(m.model if hasattr(m, "model") else str(m))
        else:
            # Fallback for different ollama client versions
            for m in response:
                installed.add(str(m))

        available = 0
        missing: list[str] = []
        for model in REQUIRED_MODELS:
            if any(model in m for m in installed):
                available += 1
            else:
                missing.append(model)

        status = f"{available}/{len(REQUIRED_MODELS)} available"
        if missing:
            status += f" (missing: {', '.join(missing)})"
        print(f"  [INFO] Models: {status}")
        return available, len(REQUIRED_MODELS), missing
    except Exception:
        print(f"  [SKIP] Models: Ollama not available, cannot check")
        return 0, len(REQUIRED_MODELS), list(REQUIRED_MODELS)


def check_embedding() -> bool:
    """Test that embedding works with OllamaEmbedder."""
    try:
        from src.embedders import OllamaEmbedder
        e = OllamaEmbedder()
        result = e.embed(["hello world"])
        assert result.shape == (1, e.dimension)
        print(f"  [PASS] Embedding works ({e.dimension}d)")
        return True
    except Exception as e:
        print(f"  [FAIL] Embedding failed: {e}")
        return False


def check_generation() -> bool:
    """Test that the smallest model can generate text."""
    try:
        from ollama import Client
        response = Client().chat(
            model="qwen3:0.6b",
            messages=[{"role": "user", "content": "Say hello"}],
        )
        print("  [PASS] Generation works (qwen3:0.6b)")
        return True
    except Exception as e:
        print(f"  [FAIL] Generation failed: {e}")
        return False


def check_api_keys() -> tuple[int, int]:
    """Check which API keys are set in the environment.

    Returns:
        Tuple of (configured_count, total_count).
    """
    # Load .env if it exists
    try:
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT / ".env")
    except ImportError:
        pass

    configured = 0
    for key, (description, importance) in API_KEYS.items():
        value = os.environ.get(key)
        if value:
            print(f"  [PASS] {key} set ({description})")
            configured += 1
        else:
            marker = "WARN" if importance.startswith("required") else "INFO"
            print(f"  [{marker}] {key} not set ({description} — {importance})")

    return configured, len(API_KEYS)


def check_corpus() -> bool:
    """Check if a corpus CSV exists in the data directory."""
    data_dir = PROJECT_ROOT / "data"
    if not data_dir.exists():
        print("  [WARN] No data/ directory found")
        return False

    csvs = list(data_dir.glob("*.csv"))
    if not csvs:
        print("  [WARN] No corpus found in data/")
        return False

    for csv_path in csvs:
        # Count rows without loading entire file
        with open(csv_path, "r", encoding="utf-8") as f:
            row_count = sum(1 for _ in f) - 1  # Subtract header
        print(f"  [PASS] Corpus found: {csv_path.name} ({row_count} rows)")

    return True


def run_smoke_test() -> None:
    """Run all smoke test checks and print a summary."""
    print()
    print("RAGBench Smoke Test")
    print("=" * 40)
    print()

    # Run all checks
    imports_ok = check_imports()

    print()
    ollama_ok = check_ollama()
    models_available, models_total, _ = check_models()

    print()
    embedding_ok = check_embedding()
    generation_ok = check_generation()

    print()
    keys_configured, keys_total = check_api_keys()

    print()
    corpus_ok = check_corpus()

    # Summary
    print()
    print("RAGBench Smoke Test Summary")
    print("=" * 40)

    def status(ok: bool) -> str:
        return "PASS" if ok else "FAIL"

    print(f"  Core imports:     {status(imports_ok)}")
    print(f"  Ollama:           {status(ollama_ok)}")
    print(f"  Models:           {models_available}/{models_total} available")
    print(f"  Embedding:        {status(embedding_ok)}")
    print(f"  Generation:       {status(generation_ok)}")
    print(f"  API keys:         {keys_configured}/{keys_total} configured")
    print(f"  Corpus:           {status(corpus_ok)}")
    print()

    # Overall readiness — YES if imports work, Ollama running, at least one model, embedding works
    ready = imports_ok and ollama_ok and models_available > 0 and embedding_ok
    has_warnings = not (generation_ok and corpus_ok and keys_configured == keys_total)

    if ready and not has_warnings:
        print("  Ready to run experiments: YES")
    elif ready:
        print("  Ready to run experiments: YES (with warnings)")
    else:
        print("  Ready to run experiments: NO")
        if not imports_ok:
            print("    - Fix import errors first (pip install -r requirements.txt)")
        if not ollama_ok:
            print("    - Start Ollama: ollama serve")
        if models_available == 0:
            print("    - Pull models: python scripts/pull_models.py")
        if not embedding_ok:
            print("    - Check Ollama and mxbai-embed-large model")


if __name__ == "__main__":
    run_smoke_test()
