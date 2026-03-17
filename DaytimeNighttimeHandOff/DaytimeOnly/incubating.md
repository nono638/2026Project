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

## Findings gallery website
**Captured:** 2026-03-17
**Last reviewed:** 2026-03-17
**Context:** A web-facing presentation of pre-computed experiment results. Interactive visualizations showing when small models beat large ones, time-quality tradeoff curves, the impact of changing one variable in isolation. Target audience: curious learners, not just engineers. This is what makes RAGBench a portfolio piece, not just a class project. Could be a static site generated from ExperimentResult data, or a simple React/Streamlit app.
**Next trigger:** Experiment 1 results exist and have interesting findings worth presenting. MVP demo approaching.

## Constraint-aware analysis API
**Captured:** 2026-03-17
**Last reviewed:** 2026-03-17
**Context:** Extend ExperimentResult with methods like `configs_above(quality=3.5)`, `best_config(metric="quality", where={"model_size": "<4B"})`, and sorting by latency or estimated cost. Users define what "best" means by expressing priorities. The experiment data is the same — the question you ask of it changes.
**Next trigger:** Experiment results exist with timing data. The analysis layer needs to support the Builder audience.

## Additional built-in dataset loaders
**Captured:** 2026-03-17
**Last reviewed:** 2026-03-17
**Context:** HotpotQA is the first built-in dataset. Others worth considering: SQuAD (simple factoid, good for baselines), Natural Questions (real Google searches + Wikipedia), FRAMES (hard multi-hop, 824 examples, good for stress-testing). Each needs a loader that outputs Document + Query objects with reference_answer populated. Ships with RAGBench so users can calibrate before running on their own data.
**Next trigger:** HotpotQA loader is built and working. User wants to expand the "calibrate first" workflow.

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
