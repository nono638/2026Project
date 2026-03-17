"""RAGBench bootstrap script.

Run this once on a new machine to set up the environment:
    python setup.py

Creates venv, installs dependencies, verifies Ollama, checks API keys,
and runs a smoke test.

This is NOT a setuptools setup.py — it's a standalone bootstrap script
that uses only stdlib. Named setup.py for discoverability.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
VENV_DIR = PROJECT_ROOT / ".venv"
REQUIREMENTS = PROJECT_ROOT / "requirements.txt"
ENV_EXAMPLE = PROJECT_ROOT / ".env.example"
ENV_FILE = PROJECT_ROOT / ".env"


def get_venv_python() -> str:
    """Get the path to the venv Python executable, OS-aware.

    Returns:
        Path string to the Python executable inside .venv.
    """
    if os.name == "nt":
        return str(VENV_DIR / "Scripts" / "python.exe")
    return str(VENV_DIR / "bin" / "python")


def get_venv_pip() -> list[str]:
    """Get the pip command as a list, using venv Python -m pip.

    Returns:
        Command list for running pip via the venv Python.
    """
    return [get_venv_python(), "-m", "pip"]


def check_python_version() -> bool:
    """Verify Python >= 3.11.

    Returns:
        True if version is sufficient.
    """
    major, minor = sys.version_info[:2]
    if major < 3 or (major == 3 and minor < 11):
        print(f"ERROR: Python 3.11+ required, found {major}.{minor}")
        print("Install Python 3.11+ from https://python.org")
        return False
    print(f"  [PASS] Python {major}.{minor}")
    return True


def create_venv() -> bool:
    """Create .venv if it doesn't exist.

    Returns:
        True if venv exists (created or already present).
    """
    if VENV_DIR.exists():
        print("  [SKIP] .venv already exists")
        return True

    print("  [CREATE] Creating .venv...")
    result = subprocess.run(
        [sys.executable, "-m", "venv", str(VENV_DIR)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"  [FAIL] venv creation failed: {result.stderr}")
        return False

    print("  [PASS] .venv created")
    return True


def install_dependencies() -> bool:
    """Install requirements.txt into the venv.

    Returns:
        True if installation succeeded.
    """
    if not REQUIREMENTS.exists():
        print("  [WARN] requirements.txt not found, skipping dependency install")
        return True

    print("  [INSTALL] Installing dependencies from requirements.txt...")
    result = subprocess.run(
        [*get_venv_pip(), "install", "-r", str(REQUIREMENTS)],
        capture_output=True,
        text=True,
        timeout=600,
    )
    if result.returncode != 0:
        print(f"  [FAIL] pip install failed: {result.stderr[:500]}")
        return False

    print("  [PASS] Dependencies installed")
    return True


def install_spacy_model() -> bool:
    """Download the spaCy en_core_web_sm model.

    Returns:
        True if the model is available.
    """
    print("  [INSTALL] Downloading spaCy en_core_web_sm model...")
    result = subprocess.run(
        [get_venv_python(), "-m", "spacy", "download", "en_core_web_sm"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        print(f"  [FAIL] spaCy model download failed: {result.stderr[:500]}")
        return False

    print("  [PASS] spaCy model installed")
    return True


def check_ollama() -> bool:
    """Check if Ollama is installed and running.

    Returns:
        True if Ollama is available.
    """
    if not shutil.which("ollama"):
        print("  [WARN] Ollama not found. Install from https://ollama.com")
        print("         Then run: python scripts/pull_models.py")
        return False

    result = subprocess.run(
        ["ollama", "list"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode != 0:
        print("  [WARN] Ollama installed but not running. Start with: ollama serve")
        return False

    print("  [PASS] Ollama running")
    return True


def offer_model_pull() -> None:
    """Ask user if they want to pull required models."""
    try:
        answer = input("  Pull required models now? This downloads ~12GB. (y/n): ")
    except (EOFError, KeyboardInterrupt):
        print()
        print("  [SKIP] Model pull skipped (non-interactive)")
        return

    if answer.strip().lower() in ("y", "yes"):
        print("  [PULL] Pulling models...")
        result = subprocess.run(
            [get_venv_python(), str(PROJECT_ROOT / "scripts" / "pull_models.py")],
            timeout=3600,  # 1 hour for all models
        )
        if result.returncode != 0:
            print("  [WARN] Some models may have failed. Re-run: python scripts/pull_models.py")
    else:
        print("  [SKIP] Model pull skipped. Run later: python scripts/pull_models.py")


def setup_env_file() -> None:
    """Create .env from .env.example if it doesn't exist."""
    if ENV_FILE.exists():
        print("  [SKIP] .env already exists")
        # Check which keys are set
        with open(ENV_FILE, "r") as f:
            content = f.read()
        keys = ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_CLOUD_PROJECT"]
        for key in keys:
            if f"{key}=" in content and content.split(f"{key}=")[1].split("\n")[0].strip():
                print(f"         {key}: configured")
            else:
                print(f"         {key}: not set")
        return

    if ENV_EXAMPLE.exists():
        shutil.copy(ENV_EXAMPLE, ENV_FILE)
        print("  [CREATE] Created .env from .env.example")
        print("           Edit .env with your API keys before running experiments.")
        print("           Required: ANTHROPIC_API_KEY, OPENAI_API_KEY")
        print("           Optional: GOOGLE_API_KEY, GOOGLE_CLOUD_PROJECT")
    else:
        print("  [WARN] No .env.example found, skipping .env creation")


def run_smoke_test() -> bool:
    """Run the smoke test script.

    Returns:
        True if smoke test passed.
    """
    smoke_test = PROJECT_ROOT / "scripts" / "smoke_test.py"
    if not smoke_test.exists():
        print("  [SKIP] smoke_test.py not found")
        return True

    print()
    print("Running smoke test...")
    print("-" * 40)
    result = subprocess.run(
        [get_venv_python(), str(smoke_test)],
        timeout=120,
    )
    return result.returncode == 0


def main() -> None:
    """Run the full bootstrap process."""
    print()
    print("RAGBench Setup")
    print("=" * 40)
    print()

    # Step 1: Check Python version
    print("1. Checking Python version...")
    if not check_python_version():
        sys.exit(1)

    # Step 2: Create venv
    print("\n2. Setting up virtual environment...")
    if not create_venv():
        sys.exit(1)

    # Step 3: Install dependencies
    print("\n3. Installing dependencies...")
    if not install_dependencies():
        print("   Setup cannot continue without dependencies.")
        sys.exit(1)

    # Step 4: Install spaCy model
    print("\n4. Installing spaCy model...")
    install_spacy_model()

    # Step 5: Check Ollama
    print("\n5. Checking Ollama...")
    ollama_ok = check_ollama()
    if ollama_ok:
        offer_model_pull()

    # Step 6: Setup .env
    print("\n6. Configuring environment variables...")
    setup_env_file()

    # Step 7: Smoke test
    print("\n7. Running smoke test...")
    run_smoke_test()

    # Summary
    print()
    print("=" * 40)
    print("Setup complete!")
    print()
    print("Next steps:")
    if not ollama_ok:
        print("  1. Install Ollama from https://ollama.com")
        print("  2. Run: python scripts/pull_models.py")
    if not ENV_FILE.exists() or not any(
        os.environ.get(k) for k in ["ANTHROPIC_API_KEY", "OPENAI_API_KEY"]
    ):
        print("  - Edit .env with your API keys")
    print("  - Run experiments: python scripts/run_experiment.py")
    print("  - Quick test: python scripts/run_experiment.py --quick")


if __name__ == "__main__":
    main()
