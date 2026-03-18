# GPU Machine Setup Guide

Step-by-step instructions for getting RAGBench running on a new machine with a GPU.

---

## Prerequisites

- **Python 3.11+** installed
- **Git** installed
- **GPU with 8GB+ VRAM** (for running Qwen3 8B quantized)
- **~15GB disk** for Ollama models
- **Internet access** for cloning, pip install, model pulls, and API calls

---

## Step 1 — Clone the repo

```bash
git clone <your-repo-url> RAGBench
cd RAGBench
```

If using Dropbox sync or USB transfer instead of git, just copy the project folder.

---

## Step 2 — Run the setup script

```bash
python setup.py
```

This does everything:
1. Creates `.venv/` virtual environment
2. Installs all dependencies from `requirements.txt`
3. Downloads spaCy English model (`en_core_web_sm`)

If setup.py fails, do it manually:
```bash
python -m venv .venv

# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

---

## Step 3 — Install Ollama

Download from https://ollama.com and install.

Verify it's running:
```bash
ollama --version
```

---

## Step 4 — Pull models

```bash
# Activate venv first if not already active
python scripts/pull_models.py
```

Or manually:
```bash
ollama pull qwen3:0.6b
ollama pull qwen3:1.7b
ollama pull qwen3:4b
ollama pull qwen3:8b
ollama pull gemma3:1b
ollama pull gemma3:4b
ollama pull mxbai-embed-large
```

Total download: ~12GB. The embedding model (`mxbai-embed-large`) is used by both the
retriever and the Semantic chunker.

---

## Step 5 — Configure API keys

```bash
cp .env.example .env
```

Edit `.env` and add your keys:
```
ANTHROPIC_API_KEY=sk-ant-...      # For Claude scoring (Haiku/Sonnet/Opus)
GOOGLE_API_KEY=AI...               # For Gemini scoring (Flash/Pro)
OPENAI_API_KEY=sk-...              # For RAGAS query generation (if used)
```

You need at minimum:
- **ANTHROPIC_API_KEY** — for Claude-based scoring
- **GOOGLE_API_KEY** — for Gemini-based scoring (get free at https://aistudio.google.com)

---

## Step 6 — Smoke test

```bash
python scripts/smoke_test.py
```

This verifies:
- Python imports work
- Ollama is reachable
- At least one model responds
- Core pipeline components load

If it fails, read the error — it will tell you what's missing.

---

## Step 7 — Run Experiment 0 (scorer validation)

```bash
python scripts/run_experiment_0.py
```

This runs 50 HotpotQA examples through NaiveRAG + Qwen3 4B, then scores each answer
with all 5 LLM judges. Results saved to `results/experiment_0/`.

---

## Step 8 — Run Experiments 1 & 2

```bash
# Experiment 1: Strategy × Model Size (30 configs)
python scripts/run_experiment.py --experiment 1

# Experiment 2: Chunking × Model Size (16 configs)
python scripts/run_experiment.py --experiment 2
```

---

## Troubleshooting

**"CUDA out of memory"**: Qwen3 8B needs ~5GB VRAM. If other models are loaded, restart
Ollama (`ollama stop` or restart the service) before switching to 8B.

**"Connection refused" from Ollama**: Ollama server isn't running. Start it:
- Windows: Ollama runs as a system tray app — launch it
- Linux: `ollama serve &`

**Slow model switching**: Ollama loads/unloads models as needed. First call to a new model
takes 10-30 seconds for weight loading. Subsequent calls are fast. The experiment runner
batches by model where possible to minimize switches.

**API rate limits**: Gemini free tier has rate limits. If you hit them, the scorer retries
once after 1 second. For bulk runs, consider a brief sleep between calls or use a paid tier.
