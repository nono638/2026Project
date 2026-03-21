# RAGBench

A configurable RAG evaluation pipeline that runs the full cartesian product of RAG configurations — chunker x embedder x strategy x language model — against any corpus, scores the results with LLM-as-judge, and trains a meta-learner to predict the optimal configuration for new queries.

## What It Does

RAGBench answers the question: **which RAG setup works best for your data?**

Instead of guessing, you define the space of configurations to test, point it at a corpus, and get scored results across every combination. The experiment runner handles chunking, embedding, retrieval, generation, and scoring — you just pick the axes.

```
Documents → Chunker → Embedder → Retriever → Strategy + LLM → Answer → Scorer → Results
                ×           ×          |           ×        ×                        |
           4 options    3 options      |      5 options  6 models              ExperimentResult
                                      |                                            |
                                 hybrid/dense/sparse                    analysis, heatmaps, export
```

## Quick Start

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com) (for local LLM inference)
- API key for scoring: [Google AI Studio](https://aistudio.google.com) (free) or [Anthropic](https://console.anthropic.com)

### Setup

```bash
git clone <repo-url>
cd RAGBench
python -m venv .venv

# Windows:
.venv\Scripts\activate
# Unix:
source .venv/bin/activate

pip install -r requirements.txt
python scripts/pull_models.py
```

Copy `.env.example` to `.env` and add your API keys.

### Run Your First Experiment

```bash
# Quick test (2 strategies, 2 models, ~5 min)
python scripts/run_experiment.py --quick --scorer google:gemini-2.0-flash

# Built-in dataset (50 HotpotQA examples)
python scripts/run_experiment.py --dataset hotpotqa --sample 50 --models qwen3:4b

# Full matrix (5 strategies x 6 models x 3 chunkers = 90 configs)
python scripts/run_experiment.py --scorer google:gemini-2.0-flash
```

Results are saved to `results/` as CSV and Parquet.

### Use a Built-in Dataset

RAGBench ships with gold-standard dataset loaders:

```bash
# HotpotQA — multi-hop Wikipedia Q&A with gold answers and difficulty labels
python scripts/run_experiment.py --dataset hotpotqa --sample 100

# SQuAD 2.0 — simple factoid Q&A, good calibration baseline
python scripts/run_experiment.py --dataset squad --sample 100
```

### Use Your Own Corpus

Place a CSV file in `data/` with at least `title` and `text` columns:

```bash
python scripts/run_experiment.py --corpus data/my_docs.csv
```

## Components

Every pipeline stage is defined by a Python [Protocol](https://docs.python.org/3/library/typing.html#typing.Protocol). You can swap any component without modifying the framework — just implement the interface.

### Chunkers

Split documents into retrievable pieces.

| Chunker | Description |
|---------|-------------|
| `semantic` | Embedding-based boundary detection |
| `fixed` | Fixed character windows with overlap |
| `recursive` | Split on natural boundaries (paragraphs, sentences) |
| `sentence` | One sentence per chunk |

### Embedders

Convert text to vectors for retrieval.

| Embedder | Description |
|----------|-------------|
| `ollama` | Local via Ollama (mxbai-embed-large, 1024d) |
| `huggingface` | Local sentence-transformers models |
| `google` | Google text-embedding-005 via API (free tier) |

### Strategies

Different approaches to retrieval-augmented generation.

| Strategy | Description |
|----------|-------------|
| `naive` | Retrieve → generate (baseline) |
| `self_rag` | Generate, self-critique, regenerate if needed |
| `multi_query` | Rephrase query multiple ways, merge retrievals |
| `corrective` | Check retrieval quality, fall back to web search |
| `adaptive` | Choose strategy based on query complexity |

### LLM Backends

| Backend | Flag | Description |
|---------|------|-------------|
| Ollama | `--llm-backend ollama` (default) | Local models via Ollama |
| OpenAI-compatible | `--llm-backend openai-compat --llm-base-url URL` | LM Studio, vLLM, llama.cpp server |

### Scorers

LLM-as-judge scoring on faithfulness, relevance, and conciseness (1-5 each).

| Provider | Flag | Example |
|----------|------|---------|
| Google | `--scorer google:gemini-2.0-flash` (default) | Free tier, fast |
| Anthropic | `--scorer anthropic:claude-haiku-4-5-20251001` | Higher quality, paid |

### Retrieval Modes

| Mode | Flag | Description |
|------|------|-------------|
| Hybrid | `--retrieval-mode hybrid` (default) | Dense + BM25 with Reciprocal Rank Fusion |
| Dense | `--retrieval-mode dense` | Embedding similarity only |
| Sparse | `--retrieval-mode sparse` | BM25 keyword matching only |

### Rerankers

Optional cross-encoder reranking between retrieval and generation.

| Reranker | Flag | Model | Size |
|----------|------|-------|------|
| MiniLM | `--reranker minilm` | ms-marco-MiniLM-L-6-v2 | 22M params |
| BGE | `--reranker bge` | BAAI/bge-reranker-v2-m3 | 278M params |
| None | `--reranker none` (default) | No reranking | — |

## CLI Reference

```
python scripts/run_experiment.py [OPTIONS]

Options:
  --quick                    Minimal test run (2 strategies, 2 models)
  --models MODEL[,MODEL]     Ollama model names (default: all 6)
  --strategies STR[,STR]     Strategy names (default: all 5)
  --chunkers CHK[,CHK]       Chunker names (default: semantic,fixed,recursive)
  --embedder NAME            Embedder: ollama, huggingface, google
  --dataset NAME             Built-in dataset: hotpotqa, squad
  --corpus PATH              Path to corpus CSV
  --sample N                 Sample N documents (default: all)
  --retrieval-mode MODE      hybrid, dense, or sparse
  --llm-backend BACKEND      ollama or openai-compat
  --llm-base-url URL         Base URL for openai-compat backend
  --scorer PROVIDER:MODEL    LLM scorer (e.g., google:gemini-2.0-flash)
  --output DIR               Output directory (default: results/)
  --reranker NAME            Reranker: minilm, bge, or none (default: none)
  --reranker-top-k N         Chunks to keep after reranking (default: 3)
  --retrieval-top-k N        Chunks to retrieve before reranking (default: 5)
  --ollama-host URL          Remote Ollama server URL
  --max-cost USD             Maximum API spend ceiling (default: $10)
  --resume                   Resume interrupted experiment (experiment scripts only)
```

## Running the Research Experiments

RAGBench includes three pre-configured experiments. Each has its own script with
checkpoint/resume support.

| Experiment | Script | Matrix | Purpose |
|------------|--------|--------|---------|
| Exp 0: Scorer Validation | `run_experiment_0.py` | 50 queries x 6 judges | Validate LLM-as-judge reliability |
| Exp 1: Strategy x Model | `run_experiment_1.py` | 5 strategies x 6 models | Does strategy compensate for model size? |
| Exp 2: Chunking x Model | `run_experiment_2.py` | 4 chunkers x 4 models | How much does chunking matter? |

See [docs/running-experiments.md](docs/running-experiments.md) for detailed instructions.

## Output Format

Experiment results are saved to `results/experiment_N/raw_scores.csv`. See
[docs/output-format.md](docs/output-format.md) for a complete column reference.

## Implementing Your Own Components

Any class that matches the Protocol signature works — no inheritance needed:

```python
# Custom chunker — just implement name and chunk()
class MyChunker:
    @property
    def name(self) -> str:
        return "my-chunker"

    def chunk(self, text: str) -> list[str]:
        return text.split("\n\n")  # split on paragraphs
```

See `src/protocols.py` for all Protocol definitions.

## Project Structure

```
src/
  protocols.py          # All component interfaces (Chunker, Embedder, Strategy, Scorer, LLM)
  experiment.py         # Experiment runner + ExperimentResult analysis
  retriever.py          # FAISS-based retriever with hybrid/dense/sparse modes
  document.py           # Document dataclass + CSV loader
  query.py              # Query dataclass + persistence
  chunkers/             # 4 chunker implementations
  embedders/            # 3 embedder implementations (Ollama, HuggingFace, Google)
  strategies/           # 5 RAG strategy implementations
  scorers/              # LLMScorer with Anthropic + Google adapters
  rerankers/            # Cross-encoder reranker implementations (MiniLM, BGE)
  llms/                 # LLM backends (Ollama, OpenAI-compatible)
  datasets/             # Built-in dataset loaders (HotpotQA, SQuAD 2.0)
  query_generators/     # RAGAS, template, BEIR, human query generators
  query_filters/        # Heuristic, round-trip, cross-encoder filters
  query_analysis/       # Distribution analyzer for query set quality
  cost_guard.py         # API spend tracking with configurable ceiling
  features.py           # Meta-learner feature extraction
  train.py              # Meta-learner training (XGBoost)
  predict.py            # Meta-learner prediction
scripts/
  run_experiment.py     # Main CLI entry point
  run_experiment_0.py   # Scorer validation experiment
  run_experiment_1.py   # Strategy x model size experiment
  run_experiment_2.py   # Chunking x model size experiment
  experiment_utils.py   # Shared experiment infrastructure
  generate_gallery.py   # Findings gallery site generator
  generate_experiment0_dashboard.py  # Exp 0 interactive dashboard
  pull_models.py        # Download Ollama models
  smoke_test.py         # Verify installation
docs/
  output-format.md      # CSV column reference
  running-experiments.md # How to run experiments (local + RunPod)
tests/                  # 549 tests
```

## License

Academic project — CUNY SPS, Spring 2026.
