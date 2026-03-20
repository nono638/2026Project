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

## Virtual Environment

- Location: `.venv/`
- Python version: 3.11+
- Activate (Windows): `.venv\Scripts\activate`
- Activate (Unix): `source .venv/bin/activate`
- Install deps: `pip install -r requirements.txt`

## Required Ollama Models

| Model | Size | Purpose |
|-------|------|---------|
| qwen3:0.6b | ~400MB | Smallest experiment model |
| qwen3:1.7b | ~1GB | Small experiment model |
| qwen3:4b | ~2.5GB | Medium experiment model |
| qwen3:8b | ~5GB | Large experiment model |
| gemma3:1b | ~600MB | Cross-family validation |
| gemma3:4b | ~2.5GB | Cross-family validation |
| mxbai-embed-large | ~700MB | Embedding model |

Total download: ~12GB

## API Keys

| Key | Required For | Where to Get |
|-----|-------------|--------------|
| ANTHROPIC_API_KEY | Claude scorer | https://console.anthropic.com |
| OPENAI_API_KEY | RAGAS query generation | https://platform.openai.com |
| GOOGLE_API_KEY | Google text embedder | https://aistudio.google.com |
| GOOGLE_CLOUD_PROJECT | Google multimodal embedder | https://console.cloud.google.com |
| GEMINI_API_KEY | LLM scorer (Gemini provider) | https://aistudio.google.com |

## Change Log

### 2026-03-17 — Installed google-genai 1.67.0
- **Why:** New unified Google GenAI SDK needed for LLMScorer Gemini adapter (task-017) and embedder migration (task-016). Replaces deprecated google-generativeai.
- **Requested by:** daytime session (pre-experiment hardening)
- **Import name:** `from google import genai`
- **Notes:** Old `google-generativeai==0.8.6` still installed — will be removed after embedder migration (task-016). Both coexist for now (different import paths: `google.generativeai` vs `google.genai`).

### 2026-03-17 — Installed rank-bm25 0.2.2
- **Why:** BM25 sparse retrieval for hybrid search (task-019). Pure Python BM25Okapi implementation, same library used by LangChain and LlamaIndex internally.
- **Requested by:** task-019 (hybrid retrieval)
- **Import name:** `from rank_bm25 import BM25Okapi`
- **Notes:** Only dependency is numpy (already installed). Supports BM25Okapi, BM25L, BM25Plus variants — we use Okapi (industry default).

### 2026-03-20 — Installed textstat 0.7.13
- **Why:** Flesch-Kincaid readability scoring for document characterization features (task-032). Provides accurate syllable counting and 10+ readability indices.
- **Requested by:** daytime session (extended features spec)
- **Import name:** `import textstat`
- **Notes:** Dependencies: pyphen==0.17.2 (hyphenation), nltk (upgraded 3.9.2→3.9.3). Pure Python, no C extensions. ~177KB.
