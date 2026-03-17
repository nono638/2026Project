# Spec: task-010 — Portable Setup and Bootstrap Scripts

## What

Make the project fully portable: clone the repo on a new machine, run one script,
and be ready to run experiments or launch the demo. This covers environment setup,
Ollama model pulling, API key configuration, and a smoke test that verifies everything
works before starting a long experiment run.

### New files to create:

1. `setup.py` (project root) — bootstrap script
2. `scripts/pull_models.py` — pulls all required Ollama models
3. `scripts/smoke_test.py` — quick end-to-end verification
4. `.env.example` — template for required environment variables
5. `ENVIRONMENT.md` — setup documentation for humans
6. `scripts/run_experiment.py` — entry point for running experiments

## Why

The project is developed on a small business notebook but experiments need to run on
a larger machine with a GPU. Transferability must be baked in from the outset. A new
machine should go from `git clone` to running experiments with one setup script and
minimal manual configuration.

This also serves the professor's goal of making the pipeline usable by others — if
someone else can clone and run it, the tool has real value beyond this project.

## File Details

### `setup.py` (project root)

```python
"""RAGBench bootstrap script.

Run this once on a new machine to set up the environment:
    python setup.py

Creates venv, installs dependencies, verifies Ollama, checks API keys,
and runs a smoke test.
"""
```

**NOT a setuptools setup.py** — this is a standalone bootstrap script. Name it `setup.py`
for discoverability (everyone knows to look for it), but it's just a regular Python script
that uses only stdlib.

Algorithm:
1. Check Python version >= 3.11. Print clear error if not.
2. Create `.venv` if it doesn't exist: `python -m venv .venv`
3. Install dependencies: `.venv/Scripts/pip install -r requirements.txt` (Windows)
   or `.venv/bin/pip install -r requirements.txt` (Unix). Detect OS.
4. Download spaCy model: `.venv/.../python -m spacy download en_core_web_sm`
5. Check if Ollama is installed and running:
   - Try `ollama list` — if it fails, print: "Ollama not found. Install from https://ollama.com
     then run: python scripts/pull_models.py"
   - If Ollama is running, ask user: "Pull required models now? This downloads ~15GB. (y/n)"
   - If yes, run `scripts/pull_models.py`
6. Check for `.env` file:
   - If missing, copy `.env.example` to `.env`
   - Print: "Created .env from template. Edit it with your API keys before running experiments."
   - List which keys are required vs optional
7. Run smoke test: `scripts/smoke_test.py`
8. Print summary: what succeeded, what needs manual action

**Important:** This script must work on both Windows and Unix. Use `os.name` or
`sys.platform` to detect the OS and use the right venv paths. Use `subprocess.run`
for all shell commands, never `os.system`.

### `scripts/pull_models.py`

```python
"""Pull all Ollama models required for RAGBench experiments.

Run standalone or called by setup.py. Idempotent — skips already-pulled models.
"""
```

**Models to pull:**

```python
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
```

Algorithm:
1. Run `ollama list` to get already-pulled models
2. For each model in REQUIRED_MODELS:
   - If already pulled, print "✓ {model} (already installed)" and skip
   - If not, print "Pulling {model}..." and run `ollama pull {model}`
   - Track success/failure
3. Print summary: N pulled, N skipped, N failed

**Edge cases:**
- Ollama not running → clear error: "Start Ollama first (ollama serve)"
- Pull fails (network error, disk space) → log the failure, continue with remaining
  models, report failures at the end
- Partial pull (interrupted) → Ollama handles this natively, re-running pull resumes

### `scripts/smoke_test.py`

```python
"""Quick end-to-end smoke test for RAGBench.

Verifies that all components work before starting a long experiment run.
Runs in ~30 seconds, uses minimal resources. NOT a full test suite —
this is "can the pipeline run at all" verification.
"""
```

**What it tests:**

1. **Imports** — can we import all core modules?
   ```python
   from src.protocols import Chunker, Embedder, Strategy, Scorer
   from src.experiment import Experiment, ExperimentResult
   from src.document import Document, load_corpus_from_csv
   from src.retriever import Retriever
   ```
   Print "✓ Core imports" or "✗ Import error: {detail}"

2. **Ollama connectivity** — can we reach Ollama?
   ```python
   from ollama import Client
   Client().list()
   ```
   Print "✓ Ollama running" or "✗ Ollama not reachable"

3. **Model availability** — are the required models pulled?
   Check each model in REQUIRED_MODELS against `ollama list`. Print which are
   available and which are missing. Don't fail — just warn.

4. **Embedding test** — can we embed text?
   ```python
   from src.embedders import OllamaEmbedder
   e = OllamaEmbedder()
   result = e.embed(["hello world"])
   assert result.shape == (1, e.dimension)
   ```
   Print "✓ Embedding works ({e.dimension}d)" or "✗ Embedding failed: {detail}"

5. **Generation test** — can the smallest model generate?
   ```python
   from ollama import Client
   response = Client().chat(model="qwen3:0.6b", messages=[{"role": "user", "content": "Say hello"}])
   ```
   Print "✓ Generation works (qwen3:0.6b)" or "✗ Generation failed: {detail}"

6. **API keys** — check which are set (don't print values!)
   ```python
   keys = {
       "GOOGLE_API_KEY": "Google text embedder",
       "GOOGLE_CLOUD_PROJECT": "Google multimodal embedder",
       "OPENAI_API_KEY": "RAGAS query generation",
       "ANTHROPIC_API_KEY": "Claude scorer",
   }
   ```
   For each: print "✓ {name} set ({description})" or "⚠ {name} not set ({description})"
   Mark which are required vs optional.

7. **Data** — does the corpus exist?
   Check for `data/*.csv`. Print "✓ Corpus found: {filename} ({n} rows)" or
   "⚠ No corpus found in data/"

Print a final summary:
```
RAGBench Smoke Test Summary
━━━━━━━━━━━━━━━━━━━━━━━━━━
Core imports:     ✓
Ollama:           ✓
Models:           5/7 available (missing: qwen3:8b, gemma3:4b)
Embedding:        ✓
Generation:       ✓
API keys:         3/4 configured
Corpus:           ✓

Ready to run experiments: YES (with warnings)
```

"Ready to run experiments" is YES if imports, Ollama, at least one model, and embedding
all pass. Warnings don't block — missing models just mean fewer configurations.

### `.env.example`

```
# RAGBench environment configuration
# Copy this to .env and fill in your values:
#   cp .env.example .env

# --- Required for Claude scoring ---
ANTHROPIC_API_KEY=

# --- Required for RAGAS query generation ---
OPENAI_API_KEY=

# --- Required for Google text embedder ---
GOOGLE_API_KEY=

# --- Required for Google multimodal embedder (Vertex AI) ---
GOOGLE_CLOUD_PROJECT=

# --- Optional ---
# OLLAMA_HOST=http://localhost:11434
```

### `ENVIRONMENT.md`

```markdown
# RAGBench Environment Setup

## Quick Start

```bash
git clone <repo-url>
cd RAGBench
python setup.py
```

The setup script handles everything: venv creation, dependency installation,
Ollama model pulling, and smoke testing.

## Manual Setup

If you prefer to set up manually:

### Prerequisites

- Python 3.11+
- Ollama (https://ollama.com)
- Git

### Steps

1. Create and activate virtual environment:
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Unix:
   source .venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

3. Pull Ollama models:
   ```bash
   python scripts/pull_models.py
   ```

4. Configure API keys:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

5. Verify setup:
   ```bash
   python scripts/smoke_test.py
   ```

## Running Experiments

```bash
python scripts/run_experiment.py
```

See `scripts/run_experiment.py --help` for configuration options.

## Running the Demo (FastAPI)

```bash
uvicorn src.app:app --reload
```

Then open http://localhost:8000/docs for the interactive API.
```

### `scripts/run_experiment.py`

```python
"""Entry point for running RAGBench experiments.

Usage:
    python scripts/run_experiment.py                    # full experiment
    python scripts/run_experiment.py --quick            # 2 strategies, 2 models (testing)
    python scripts/run_experiment.py --models qwen3:4b  # specific model only
"""
```

This is a thin CLI wrapper around the Experiment class. For this task, implement
only the skeleton with argument parsing — the actual wiring to Experiment depends
on task-007's Document/Query types being available. If task-007 is done, wire it up.
If not, create the CLI structure with a `# TODO: wire to Experiment` placeholder.

**Arguments:**
- `--quick` — run a minimal configuration for testing (2 strategies: naive + self_rag,
  2 models: qwen3:0.6b + qwen3:4b, semantic chunker, ollama embedder)
- `--models` — comma-separated list of model names (default: all REQUIRED_MODELS minus
  the embedding model)
- `--strategies` — comma-separated list (default: all five)
- `--corpus` — path to corpus CSV (default: `data/*.csv`, first found)
- `--output` — output directory for results (default: `results/`)
- `--sample` — number of documents to sample from corpus (default: all)

Use `argparse`. Load `.env` file at startup using `python-dotenv` (already in
requirements or add it).

## Dependencies to Install

1. `python-dotenv` — for loading .env files

**Use the install-package skill** for this.

## Files NOT to Touch

- `src/` — no source code changes
- `tests/` — no test changes
- Existing scripts in `_claude_sandbox_setup/` — leave them alone

## Tests

No separate test file for this task. The smoke_test.py IS the test — it verifies the
setup works. Run it as the final step after creating all files.

However, do verify:
- `setup.py` runs without errors on the current machine (it should detect existing venv
  and skip creation)
- `scripts/smoke_test.py` produces the summary output
- `scripts/pull_models.py` correctly detects already-installed models
- `.env.example` is a valid file (no syntax errors)

## Edge Cases

- Running on a machine that already has everything set up → setup.py should be
  idempotent, skip what's already done
- No internet → model pulling fails gracefully, everything else works
- No Ollama → setup.py warns but doesn't fail (user might install later)
- No GPU → Ollama still works (uses CPU), just slower. Don't check for GPU.
- .env already exists → don't overwrite, just verify keys are present

## Quality Checklist
- [x] Exact files to modify are listed
- [x] All edge cases are explicit
- [x] All judgment calls are made
- [x] Why is answered for every non-obvious decision
- [x] Tests cover key behaviors, not just "does it run"
- [x] Scoped to one focused session
