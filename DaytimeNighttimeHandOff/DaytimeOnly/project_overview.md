# Project Overview

> **Night instance reads this file** for context on what it's building.

---

## What This Project Is

**SmallModelBigStrategy** is two things:

1. **A generalized RAG research tool** — a pluggable framework where users choose their own embedding models, chunking strategies, RAG strategies, LLM models, and corpora, then run controlled experiments comparing them. Every component is swappable via Python Protocols (duck-typed interfaces). The tool handles the experiment matrix, scoring, and analysis.

2. **A specific research project** using that tool — testing the hypothesis that RAG strategy sophistication can substitute for raw parameter count in locally-hosted language models. This is the CUNY SPS graduate project deliverable.

The tool is the product. The research is the first use case. The professor specifically encouraged making the evaluation pipeline reusable by others, potentially as a Python package, and potentially extending into a capstone project.

**Team:** Noah (solo — Curtis and Cailinn dropped out as of 2026-03-16)

---

## Current Goals

- [ ] Build the generalized RAG research tool with pluggable components
- [ ] Run our specific experiment (strategy × model size) using the tool
- [ ] Train meta-learner on experiment results
- [ ] FastAPI app for configuration recommendation
- [ ] Final demo May 11 (10-min presentation + live demo)
- [ ] Project writeup/whitepaper due May 15 (2500 words)
- [ ] Confirm with professor whether March 30 MVP is a graded deliverable

---

## Scope

### In Scope
**The tool (generalized):**
- Pluggable component system: chunkers, embedders, LLMs, RAG strategies, scorers
- Python API: users build experiments by composing components, tool runs cartesian product
- ExperimentResult with analysis/visualization (heatmaps, comparisons, pivots, export)
- Built-in components: 4 chunkers, 2 embedder backends (Ollama, HuggingFace), 5 RAG strategies, Claude scorer

**Our research (specific use case):**
- 5 RAG strategies × 4+ Qwen3 model sizes + Llama 3.2 cross-validation
- Embedding: mxbai-embed-large via Ollama (mid-range to avoid masking strategy differences)
- Corpus: wikimedia/wikipedia HuggingFace dataset
- Scoring: Claude as LLM-as-judge (faithfulness, relevance, conciseness)
- Meta-learner: XGBoost predicting optimal (chunker, embedder, strategy, model) config
- FastAPI app for configuration recommendation, deployed on Render

### Out of Scope
- HyDE (different kind of intervention — not a fair comparison)
- Agentic RAG (implausible for sub-7B models, too many confounds)
- Cloud GPU resources (all local, 8GB VRAM laptop)
- Deep learning classifiers (XGBoost is established best practice for tabular data at this scale)

### Now In Scope (promoted from stretch)
- Reusable evaluation pipeline — pluggable components, any user can run their own experiments
- Potential Python package release

### Stretch Goals
- Thinking mode as a second axis (Qwen3 standard vs thinking mode)

---

## Key Technical Decisions

| Decision | Rationale | Date |
|---|---|---|
| Protocols over ABCs | Structural subtyping (duck typing) — users don't need to inherit from framework classes. Less boilerplate for a research tool. | 2026-03-16 |
| No registry pattern | Components passed directly as instances to Experiment(). Simpler, explicit, type-checkable. YAML config layer deferred to later. | 2026-03-16 |
| Python API first, YAML later | Keep interface honest — if the API is clean, YAML is just a thin wrapper. Avoids premature abstraction. | 2026-03-16 |
| Multiclass classification, not ordinal | Core hypothesis is that strategy can invert size ordering (4B+SelfRAG > 8B+Naive). Ordinal regression would assume monotonic size→quality. See architecture-decisions.md. | 2026-03-16 |
| XGBoost over PyTorch for meta-learner | Grinsztajn et al. (NeurIPS 2022): tree-based models outperform NNs on tabular data <100K rows | 2026-03 |
| mxbai-embed-large for embeddings | Outperforms nomic-embed-text on MTEB; deliberately mid-range to avoid masking strategy differences | 2026-03 |
| Asymmetric cost function | Under-serving (bad answer) penalized more than over-serving (extra compute); reflects real user constraints | 2026-03 |
| Claude as both query classifier and judge | Acknowledged limitation — both feature and label carry Claude's biases; flagged for future work | 2026-03 |

---

## Professor's Feedback (on proposal)

1. **Consider ordinal regression** instead of multiclass classification — larger models expected to do better, so outcomes are ordered
2. **LLM-as-judge can be noisy** — manually validate a sample of scores to build confidence
3. **Stretch: make pipeline reusable** — let others drop in their own models and corpus; could release as Python package; could extend into capstone
4. Mentioned ellmer/vitals and chatlas/inspect ecosystems for model evaluation

---

## Change History

| Date | Change | Why | Impact on Night Tasks |
|---|---|---|---|
| 2026-03-16 | Initial project setup | Project kickoff | n/a |
| 2026-03-16 | Scope shift: generalized RAG research tool | Professor encouraged reusability; building pluggable tool first, then using it for our experiment. MVP date TBD with professor. | Existing skeleton will be redesigned around registry/plugin pattern |
