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

## Product Vision

RAGBench serves two distinct audiences with the same core engine:

### 1. The Explorer (findings gallery)

A website where someone curious about RAG browses pre-computed experimental results.
Interactive visualizations showing insights like:
- When a 4B model + smart strategy outperforms an 8B model + naive RAG
- Cost/time vs quality tradeoffs: is a 60-second answer meaningfully better than a 5-second one?
- The impact of changing one variable (chunking, embedder, strategy) in isolation
- Where the quality curve flattens — "betterness thresholds" for latency and cost

This is powered by our own experiments. The data generated for the academic paper IS the
content. Users can drill down and see the effect of individual decisions.

### 2. The Builder (experiment engine)

A Python tool where researchers/engineers run their own experiment matrix. They choose:
- Corpus (their own docs or a built-in benchmark dataset)
- Configs to test (full matrix or specific comparisons)
- Constraints (latency, cost, offline-only, model size cap)
- Whether they have gold validation data

They get: scored results, analysis, visualizations, config recommendations.

### Evaluation modes

The pipeline works with or without gold-standard reference data:

- **Intrinsic evaluation** (always available): scorer rates answers against retrieved context.
  Measures: faithfulness, relevance, conciseness. No external reference needed.
- **Extrinsic evaluation** (when gold data exists): compare answers against known-correct
  facts. Measures: factual correctness (exact match, F1). Enables scorer validation.

Both modes use the same experiment runner and produce the same output structure. When
reference answers exist (`Query.reference_answer`), the analysis layer can compute
additional correctness metrics. When they don't, those columns are simply absent.
The two signals are complementary: a model can be faithful to bad context (high intrinsic,
low extrinsic) or terse but correct (low intrinsic, high extrinsic).

### Built-in datasets

RAGBench ships with gold-standard dataset loaders so users can calibrate their pipeline
before running on their own data. Primary: HotpotQA (113K multi-hop Wikipedia Q&A pairs
with gold answers and difficulty labels). Secondary: SQuAD 2.0 (simple factoid Q&A,
easy baseline for calibration). Tertiary: the project's Wikipedia sample with
template-generated queries (no gold answers — intrinsic only).

### "Best" is user-defined

RAGBench doesn't decide what "best" means. Users express priorities through the analysis
layer: best by faithfulness, best within a latency budget, cheapest config above a quality
floor, etc. The experiment data is the same — the question you ask of it changes.

---

## Current Goals

### Done
- [x] Core framework: protocols, retriever, experiment runner (tasks 001-005)
- [x] Query generation and filtering pipeline (tasks 007-009, 011)
- [x] Google text embedder, migrated to google-genai SDK (tasks 006, 016)
- [x] Portable setup scripts (task 010)
- [x] LLMScorer: provider-agnostic LLM-as-judge with Anthropic + Google adapters (tasks 012, 017)
- [x] HotpotQA dataset loader (task 013)
- [x] SQuAD 2.0 dataset loader (task 015)
- [x] Experiment timing: strategy/scorer/total latency columns + analysis (task 014)
- [x] Hybrid retrieval: dense + BM25 with RRF fusion, 3 modes (task 019)
- [x] LLM Protocol: abstract generation with Ollama + OpenAI-compatible adapters (task 020)
- [x] Experiment 0 script written: scorer validation ready to run (task 018)
- [x] All 20 tasks complete, all tests passing
- [x] Tests consolidated into main `tests/` directory (task 021)
- [x] CLI flags for chunker/embedder/dataset/retrieval-mode/llm-backend (task 022)

### Next — Experiments (requires GPU machine + Ollama)
- [ ] Experiment 0: Scorer validation — 50 HotpotQA × NaiveRAG × Qwen3-4B, scored by 5 LLM judges. Script ready (`scripts/run_experiment_0.py`).
- [ ] Experiment 1: Strategy × Model Size — 5 strategies × 6 models = 30 configs. Held constant: Recursive chunker (500/100), mxbai-embed-large.
- [ ] Experiment 2: Chunking × Model Size — 4 chunkers × 4 Qwen3 models = 16 configs. Held constant: NaiveRAG strategy, mxbai-embed-large.

### Next — Infrastructure
- [ ] Set up RunPod account, deploy GPU pod with Ollama
- [ ] RunPod management layer: auto-start/stop, GPU fallback, budget display
- [ ] Deploy experiment scripts to RunPod, pull models

### Next — Model & Endpoint
- [ ] Train meta-learner on experiment results
- [ ] FastAPI recommendation endpoint

### Next — Product
- [ ] Findings gallery: static website with pre-computed visualizations (free hosting)
- [ ] Live demo: "try it yourself" with 1B/4B models, budget-aware, auto-start/stop GPU
- [ ] Builder docs: how to run your own experiments
- [ ] Constraint-aware analysis: best config within latency/cost/size budgets

### Milestones
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
- Built-in gold-standard dataset loaders (HotpotQA, SQuAD 2.0, with room for more)
- ExperimentResult with analysis/visualization (heatmaps, comparisons, export)
- Timing and cost tracking as first-class output dimensions
- Dual evaluation: intrinsic (scorer vs context) + extrinsic (vs gold answers when available)
- Constraint-aware analysis: best config within latency/cost/size budgets
- Portable setup scripts for cross-machine transferability

**Our experiments (specific use cases):**
- Experiment 1: 5 RAG strategies × Qwen3 (0.6B, 1.7B, 4B, 8B) + Gemma 3 (1B, 4B)
- Experiment 0: Scorer validation — 5 LLM judges compared on 50 HotpotQA examples
- Experiment 2: 4 chunking strategies × Qwen3 (0.6B, 1.7B, 4B, 8B)
- Embedding: mxbai-embed-large via Ollama (held constant in both experiments)
- Primary corpus: HotpotQA (multi-hop Wikipedia, gold answers, difficulty labels)
- Secondary corpus: SQuAD 2.0 (simple factoid, easy baseline) + Wikipedia sample
- Scoring: LLMScorer — provider-agnostic (Anthropic or Google), choice informed by Experiment 0
- Meta-learner: XGBoost predicting optimal configuration
- FastAPI recommendation endpoint

**Infrastructure:**
- RunPod GPU (prepaid, ~$0.17/hr on-demand) for experiments and live demo inference
- Three-tier demo: free static frontend + traffic cop (auto-start/stop GPU) + RunPod backend
- Live demo limited to 1B/4B models for speed and cost; gallery shows full results from all models
- Budget-aware: website shows remaining credits, degrades to gallery-only when exhausted

**Deliverables:**
- Python package: the experiment engine (Builder audience)
- Findings gallery: website with pre-computed insights and visualizations (Explorer audience)
- Live demo: "try it yourself" with small models (budget-capped)
- Academic paper: experiments, methodology, results

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
| 2026-03-17 | All 12 initial tasks complete | Pipeline built end-to-end, 246 tests passing | Ready for experiments |
| 2026-03-17 | HotpotQA as primary corpus | Multi-hop Wikipedia Q&A with gold answers; stronger methodology than Wikipedia-only | Need HotpotQA loader |
| 2026-03-17 | Product vision: Explorer + Builder audiences | Pipeline is both a findings gallery and a tool for others | Adds timing, constraint-aware analysis, built-in datasets |
| 2026-03-17 | Latency/time as first-class dimension | Users care about speed vs quality tradeoffs, not just quality alone | Add timing to experiment runner |
| 2026-03-17 | ClaudeScorer → LLMScorer (provider-agnostic) | Users shouldn't be locked to one API; Gemini is 23x cheaper | Refactor scorer, add Google adapter |
| 2026-03-17 | Added Experiment 0: scorer validation | Must validate LLM-as-judge before trusting it on 2K+ runs | Compare 5 LLM judges on 50 HotpotQA examples |
| 2026-03-17 | Experiment 2 expanded to 4 chunkers | Semantic chunker is worth testing — measures embedding-aware boundaries | 16 configs instead of 12 |
| 2026-03-17 | SQuAD 2.0 as secondary gold dataset | Easy-baseline calibration alongside HotpotQA | Add SQuAD loader |
| 2026-03-17 | Hybrid retrieval (dense + BM25 + RRF) as default | Dense-only misses keyword matches; hybrid is production standard; NDCG +26-31% | Retriever upgrade, held constant for experiments |
| 2026-03-17 | LLM Protocol: Ollama + OpenAI-compatible | Strategies hardcoded Ollama; users need LM Studio/vLLM/OpenAI support | Refactor all 5 strategies to accept LLM interface |
| 2026-03-18 | All 20 tasks merged to main | Pipeline fully built: hybrid retrieval, LLM protocol, LLMScorer, SQuAD, timing, experiment 0 script | Ready for experiments once Ollama is set up |
| 2026-03-18 | All 22 tasks merged, 369 tests passing | Test consolidation + CLI flags complete | Housekeeping done |
| 2026-03-18 | RunPod as GPU provider, three-tier demo architecture | No local GPU; RunPod is prepaid (no cost risk), has API for auto-start/stop | Infrastructure setup needed before experiments |
| 2026-03-18 | Live demo limited to 1B/4B models | Fast responses for demo UX + cost control; gallery shows full results | Demo and gallery are separate concerns |
