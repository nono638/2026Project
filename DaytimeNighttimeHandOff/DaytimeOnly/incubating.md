# Incubating

> **Managed by daytime Claude. Night instance does not read this file.**
>
> Ideas worth keeping but not yet ready to spec. Each entry must have a **next trigger** —
> a condition that would cause promotion to a nighttime task. Items without a trigger get
> one assigned or get archived at the monthly sweep (60+ days inactive = decide or archive).

---

## Format

```
## [Idea title]
**Captured:** YYYY-MM-DD
**Last reviewed:** YYYY-MM-DD
**Context:** [What this is and why it matters — enough to reconstruct the idea cold]
**Next trigger:** [What event/information/completion would cause us to act on this]
**Blocked by:** [Optional — another task or external dependency]
```

---

## Active

## Document characterization features for meta-learner — PROMOTED 2026-03-19
**Promoted to:** Implemented directly during daytime session (commit 47f9f5b). Five features added to `src/features.py`, 13 tests in `tests/test_doc_features.py`.

## Configurable feature groups for meta-learner training
**Captured:** 2026-03-19
**Last reviewed:** 2026-03-19
**Context:** The meta-learner's feature vector now has 11 features across 4 logical groups: query (2), doc-basic (2), doc-content (5, the new NER/topic/coherence features), retrieval (2). Not all users will want all groups — some may view document "peeking" as non-standard, and multimodal inputs would make text-based NER/topic features irrelevant (images/video have no NER). Decision: always compute all features during experiments (data is cheap, rerunning isn't), but make the meta-learner training step configurable about which feature groups to include. This also enables the research comparison: "doc characterization features improved accuracy by X%." XGBoost handles missing values natively, so a single model could even handle mixed text/multimodal inputs where content features are NaN.
**Next trigger:** Meta-learner training code is being written.
**Blocked by:** Nothing — but no value in building this until the training pipeline exists.

## Multimodal embedding support
**Captured:** 2026-03-16
**Last reviewed:** 2026-03-16
**Context:** Extend the embedder system to support image/video inputs via a `MultimodalEmbedder` protocol that adds `embed_images()` and `embed_mixed()` on top of the existing `Embedder`. Chunker and retriever also need multimodal input handling.
**Next trigger:** Google text embedder is merged and tested. User decides to pursue multimodal as a real feature rather than stretch goal.

## Google multimodal embedder (multimodalembedding@001)
**Captured:** 2026-03-16
**Last reviewed:** 2026-03-16
**Context:** Google's `multimodalembedding@001` on Vertex AI puts text, image, and video into a shared 1408-dim embedding space. Interesting to benchmark on text-only despite ~32 token limit. BUT requires a paid GCP Vertex AI account (~$0.0001/prediction) — the $20/month Gemini subscription doesn't cover it. Spec was written (see architecture-decisions.md) but removed from task-006 to avoid GCP billing setup.
**Next trigger:** User sets up GCP billing, or free Vertex AI access becomes available, or multimodal becomes a priority worth paying for.

## Findings gallery website + live demo
**Captured:** 2026-03-17
**Last reviewed:** 2026-03-18
**Context:** Two-part website: (1) Static findings gallery showing pre-computed experiment results — interactive visualizations, time-quality tradeoffs, when small models beat large ones. Free hosting (Render/Vercel). (2) Live "try it yourself" demo limited to 1B/4B models for speed and cost. Backed by RunPod GPU that auto-starts on request and auto-stops after idle. Budget-aware UI shows remaining credits, degrades to gallery-only when exhausted. Architecture decision captured in reference/architecture-decisions.md.
**Next trigger:** Experiment 1 results exist and have interesting findings worth presenting. MVP demo approaching (April 7).

## Constraint-aware analysis API
**Captured:** 2026-03-17
**Last reviewed:** 2026-03-17
**Context:** Extend ExperimentResult with methods like `configs_above(quality=3.5)`, `best_config(metric="quality", where={"model_size": "<4B"})`, and sorting by latency or estimated cost. Users define what "best" means by expressing priorities. The experiment data is the same — the question you ask of it changes.
**Next trigger:** Experiment results exist with timing data. The analysis layer needs to support the Builder audience.

## Additional built-in dataset loaders (NQ + FRAMES)
**Captured:** 2026-03-17
**Last reviewed:** 2026-03-17
**Context:** SQuAD promoted to task-015. Remaining candidates: Natural Questions (real Google searches — but `nq_open` has no context documents, full version is 42GB HTML; would need Wikipedia fetching layer) and FRAMES (hard multi-hop, 824 examples — only provides Wikipedia URLs, not text; also needs fetching). Both require different infrastructure than the simple HuggingFace-to-Document pattern.
**Next trigger:** User decides to invest in a Wikipedia article fetching layer, or a simpler variant of NQ/FRAMES becomes available on HuggingFace.
**Partial promotion:** SQuAD → task-015 (2026-03-17)

## LLM Protocol: abstract generation backend — PROMOTED 2026-03-17
**Promoted to:** task-020

## CLI: expose chunker/embedder/dataset selection — PROMOTED 2026-03-18
**Promoted to:** task-022 (done)

## User documentation: README, tutorial, output guide
**Captured:** 2026-03-17
**Last reviewed:** 2026-03-17
**Context:** No README.md, no getting-started tutorial, no explanation of output columns, no API docs for the FastAPI endpoint. Code has good docstrings but nothing user-facing. A researcher who isn't us can't use RAGBench without reading source code. Needs: README with quick-start, output format spec (what each CSV column means), and a "run your first experiment" walkthrough.
**Next trigger:** MVP demo approaching (April 7th) or when positioning for external users. Could be part of the findings gallery effort.

## Reranker protocol — PROMOTED 2026-03-20
**Promoted to:** task-031 (done, merged)

## Prompt template as an experimental variable (Experiment 3 candidate)
**Captured:** 2026-03-20
**Last reviewed:** 2026-03-20
**Context:** Prompt templates are the biggest uncontrolled variable in RAG. Most production systems (LangChain, LlamaIndex, enterprise RAG) use a default template chosen by folklore, not evidence. Nobody has systematically tested whether these choices matter, how much, or whether they interact with model size. Currently our strategy prompt templates are hardcoded — prompt engineering is confounded with strategy logic. Decompose templates into 2 structural dimensions: context format (plain concat / numbered list / XML tags) × instruction framing (closed-book / open-book) = 6 templates. Run with NaiveRAG only (simplest strategy = prompt IS the strategy) on one model, 50 queries. 300 runs total — cheap. Key research question: does prompt engineering give more lift than switching strategies? If NaiveRAG+good-template beats CorrectiveRAG+bad-template, that's publishable. Implementation: extract prompt template from NaiveRAG into a parameter, define 6 canonical templates, run as a small side study.
**Next trigger:** Experiments 1 & 2 are complete. User wants to design Experiment 3.

## Guardrail protocol for engine users
**Captured:** 2026-03-20
**Last reviewed:** 2026-03-20
**Context:** A `Guardrail` protocol so RAGBench users can plug in their own input/output filters (reject off-topic queries, block harmful outputs, filter PII). Separate from live demo guardrails (which are being built for the demo endpoint). This is about making the engine production-ready for others.
**Next trigger:** Positioning RAGBench for external users / Builder audience. Could coincide with user documentation effort.

## top_k and chunk_overlap as experimental axes
**Captured:** 2026-03-20
**Last reviewed:** 2026-03-20
**Context:** The engine supports both top_k and chunk_overlap as parameters, but the CLI lacks a --chunk-overlap flag and experiment scripts don't loop over these values. Adding them as axes would let experiments explore retrieval depth and chunking overlap as variables. Needs CLI flag and experiment script updates.
**Next trigger:** Designing Experiment 1 & 2 run configurations — discuss whether these are worth varying or held constant.

## Hybrid retrieval as default — PROMOTED 2026-03-17
**Promoted to:** task-019

## Project Gutenberg corpus
**Captured:** 2026-03-16
**Last reviewed:** 2026-03-16
**Context:** Build a corpus loader for Project Gutenberg books as an alternative to the Wikipedia dataset. Long-form, public domain text with very different characteristics (narrative, archaic language, chapter structure). Would test whether RAG strategies that work on short Wikipedia articles also work on book-length documents. Could also test multimodal in the future (illustrated books).
**Next trigger:** Core experiment pipeline is running end-to-end with Wikipedia. User wants a second corpus to validate generalizability.


---

## Archive

<!-- Promoted or abandoned ideas — never delete, just move here with a one-line epitaph -->

<!-- Format:
## [Idea title] — ARCHIVED YYYY-MM-DD
**Reason:** [Why set aside — promoted to task-NNN / superseded / no longer relevant]
-->
