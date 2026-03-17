# Project Overview

> **Night instance reads this file** for context on what it's building.

---

## What This Project Is

**RAGBench** is a configurable RAG evaluation pipeline that runs the full cartesian product
of RAG configurations (chunker × embedder × strategy × language model) against any corpus,
scores the results, and trains a meta-learner to predict the optimal configuration for
new queries. Each pipeline stage is defined by a Python Protocol, making every component
swappable without modifying the framework.

The project serves two purposes: (1) a reusable evaluation tool for RAG research, and
(2) a vehicle for two specific experiments on small local language models.

The tool is the product. The experiments are the first use cases. The professor specifically
encouraged making the evaluation pipeline reusable by others, potentially as a Python
package, and potentially extending into a capstone project.

**Team:** Noah (solo)

---

## Current Goals

- [ ] Complete the query generation and filtering pipeline (tasks 007-009, 011)
- [ ] Get the project portable across machines (task 010)
- [ ] Add Google text embedding as a cloud comparison point (task 006)
- [ ] Run Experiment 1: strategy × model size (5 strategies × 6 models = 30 configs)
- [ ] Run Experiment 2: chunking × model size (3 chunkers × 4 models = 12 configs)
- [ ] Train meta-learner on experiment results
- [ ] Build web frontend for FastAPI demo
- [ ] MVP demo (March 30th)
- [ ] Final demo (May 11th)
- [ ] Writeup/whitepaper (May 15th)

---

## Scope

### In Scope

**The pipeline (generalized):**
- Pluggable 4-axis experiment system: chunkers, embedders, strategies, models
- Pluggable query generation (RAGAS, templates, human-curated, BEIR benchmarks)
- Query validation pipeline (heuristic, round-trip, cross-encoder, distribution analysis)
- ExperimentResult with analysis/visualization (heatmaps, comparisons, export)
- Portable setup scripts for cross-machine transferability

**Our experiments (specific use cases):**
- Experiment 1: 5 RAG strategies × Qwen3 (0.6B, 1.7B, 4B, 8B) + Gemma 3 (1B, 4B)
- Experiment 2: 3 chunking strategies × Qwen3 (0.6B, 1.7B, 4B, 8B)
- Embedding: mxbai-embed-large via Ollama (held constant in both experiments)
- Corpus: Wikipedia article sample
- Scoring: Claude as LLM-as-judge (faithfulness, relevance, conciseness)
- Meta-learner: XGBoost predicting optimal configuration
- FastAPI recommendation endpoint, deployed on Render

### Out of Scope
- Multimodal embedding (incubating — needs paid GCP Vertex AI)
- HyDE (different intervention type — not a fair comparison)
- Agentic RAG (implausible for sub-7B models, too many confounds)
- Cloud GPU resources (all local, 8GB VRAM laptop)
- Deep learning classifiers (XGBoost is established best for tabular at this scale)

---

## Key Technical Decisions

| Decision | Rationale | Date |
|---|---|---|
| Protocols over ABCs | Structural subtyping — users don't inherit from framework classes | 2026-03-16 |
| Multiclass classification, not ordinal | Strategy can invert size ordering — ordinal assumption is what we're testing | 2026-03-16 |
| Two focused experiments, not one full matrix | Full 4-axis matrix is computationally prohibitive; two experiments isolate interactions cleanly | 2026-03-16 |
| Every pipeline stage gets its own protocol/module | Multimodal is an end goal; tightly coupled stages break later | 2026-03-16 |
| Multiple query generators by design | Single-source evaluation is a methodological weakness | 2026-03-16 |
| Recursive chunker (512/50) as default for Experiment 1 | Most common in production and literature; most defensible baseline | 2026-03-16 |
| Gemma 3 for cross-family validation (not Llama 3.2) | Cleaner size overlap at 1B and 4B for direct comparison | 2026-03-16 |
| Google multimodal embedder moved to incubating | Requires paid GCP Vertex AI; text embedder is free | 2026-03-16 |

---

## Architecture

```
Documents → QueryGenerator → Queries → QueryFilter(s) → Validated Queries
                                                              ↓
Validated Queries × (Chunker × Embedder × Strategy × Model) → Answers
                                                              ↓
                                                     Scorer → Scores
                                                              ↓
                                                   ExperimentResult → Analysis
                                                              ↓
                                                   Meta-Learner (XGBoost)
                                                              ↓
                                                   FastAPI Endpoint
```

---

## Professor's Feedback (on proposal)

1. **Consider ordinal regression** — addressed: using multiclass because strategy can invert size ordering
2. **LLM-as-judge can be noisy** — plan to manually validate a sample of scores
3. **Make pipeline reusable** — this became the project's primary framing
4. Mentioned ellmer/vitals and chatlas/inspect ecosystems for model evaluation

---

## Change History

| Date | Change | Why | Impact on Night Tasks |
|---|---|---|---|
| 2026-03-16 | Initial project setup | Project kickoff | n/a |
| 2026-03-16 | Renamed to RAGBench | Reframed around the pipeline, not just one hypothesis | Update any references to SmallModelBigStrategy |
| 2026-03-16 | Added Experiment 2 (chunking × model size) | Research gap — no peer-reviewed chunking × model size study exists | Chunkers are a test axis, not just infrastructure |
| 2026-03-16 | Switched to Gemma 3 from Llama 3.2 | Cleaner size overlap at 1B and 4B | Update model references in specs |
| 2026-03-16 | Dropped Google multimodal embedder | Requires paid GCP billing | Moved to incubating |
| 2026-03-16 | Solo project | Curtis and Cailinn dropped out | n/a |
