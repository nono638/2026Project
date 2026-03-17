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
