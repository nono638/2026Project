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

## Document characterization features for meta-learner
**Captured:** 2026-03-19
**Last reviewed:** 2026-03-19
**Context:** The current meta-learner uses 6 features (query_length, num_named_entities, doc_length, doc_vocab_entropy, mean/var_retrieval_score). The doc-level features are shallow — just length and vocab entropy. The hypothesis is that document content characteristics predict which RAG config works best: entity-dense legal text needs different chunking/strategy than narrative prose or poetry. No published research does this (see reference/research.md "Document Characterization for RAG Configuration Selection"). Candidate features:
1. **NER density** — distinct entities per 1000 tokens (spaCy NER). High = structured/factual, low = narrative/abstract.
2. **NER repetition ratio** — total NER mentions / distinct entities. High = document revisits same entities (good for RAG — retrieval finds relevant chunks). Low = many one-off mentions (harder to retrieve coherently).
3. **Topic density** — topics per 1000 tokens via TopicRank or LDA. High = covers many subjects (harder retrieval), low = focused (easier).
4. **Embedding cluster count** — KNN/spectral clustering on chunk embeddings, count clusters. Measures semantic diversity within the document.
5. **Semantic coherence** — average cosine similarity between consecutive chunk embeddings. High = smooth flow (narrative), low = jumpy (reference material).
These go into `src/features.py` alongside existing features, become columns in experiment results, and feed the meta-learner. The key question: do these features actually predict which strategy/model/chunker wins?
**Next trigger:** Experiment 0 results exist. Ready to enrich the feature vector before Experiments 1 and 2.
**Blocked by:** Nothing — could be implemented now, but experiments need to run to validate predictive value.

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
**Next trigger:** Experiment 1 results exist and have interesting findings worth presenting. MVP demo approaching (March 30).

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
**Next trigger:** MVP demo approaching (March 30th) or when positioning for external users. Could be part of the findings gallery effort.

## Reranker protocol
**Captured:** 2026-03-17
**Last reviewed:** 2026-03-17
**Context:** Add a `Reranker` protocol (`rerank(query, chunks, top_k) -> chunks`) as an optional stage between retrieval and generation. Cross-encoder reranking is standard in production RAG and improves NDCG@10 by 7-20% depending on baseline. The interaction between reranker and embedding model is non-trivial (Rao et al. 2025: smaller embeddings can outperform larger ones with the right reranker). Relevance-only reranking can actually hurt answer quality (REBEL 2025). For our experiments, hold reranking constant (either off, or one fixed reranker like `gte-reranker-modernbert-base` 149M). Infrastructure should support swapping rerankers for future users. See `reference/research.md` "Reranking in RAG Systems" for full literature review.
**Next trigger:** Experiments 1 & 2 are complete. User wants to run Experiment 3 (reranking study) or position RAGBench for external users who expect reranking support.
**Blocked by:** Hybrid retrieval should be implemented first — rerankers can only reorder what was retrieved, so recall matters more than precision at the retrieval stage.

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
