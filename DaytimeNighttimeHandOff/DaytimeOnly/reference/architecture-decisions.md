# Architecture Decisions

> **Night instance reads this file** — keep entries clear and actionable.
>
> Records decisions made and why, so they don't get relitigated.

---

## Multiclass classification, not ordinal regression — 2026-03-16
**Decision:** The meta-learner uses multiclass classification (XGBoost), not ordinal regression.
**Rationale:** The target variable is a (strategy, model_size) pair. The professor suggested ordinal regression because larger models are generally expected to do better — but the core hypothesis is that strategy can invert size ordering (e.g., 4B+Self-RAG beats 8B+Naive). If strategy×size interactions break the ordinal assumption, ordinal regression would impose a wrong constraint. Multiclass lets the model learn arbitrary interactions.
**Alternatives considered:** Ordinal regression (professor's suggestion) — rejected because it assumes a monotonic relationship between model size and quality, which is exactly what the project aims to disprove.

## Multimodal embedder protocol as future extension, not redesign — 2026-03-16
**Decision:** Keep `Embedder` as the text-only protocol. Future multimodal support will come via a separate `MultimodalEmbedder` protocol that extends `Embedder` with `embed_images()` and `embed_mixed()`. The experiment runner will use `isinstance` checks to detect multimodal capability.
**Rationale:** Only one embedder (Google's `multimodalembedding@001`) needs multimodal today, and it's being tested text-only first. Adding a `Content` union type or rewriting `embed()` to accept mixed inputs would add complexity to every embedder for a capability only one will use. Clean separation — text-only embedders carry no dead methods.
**Alternatives considered:** (A) Union `Content` input type on `embed()` — rejected, adds complexity to all embedders. (B) Completely separate interface — rejected, would duplicate the text path.

## Google embedders: two models, two purposes — 2026-03-16
**Decision:** Add both `GoogleTextEmbedder` (text-embedding-005, 768d, 2048 tokens) and `GoogleMultimodalEmbedder` (multimodalembedding@001, 1408d, ~32 tokens text) as experiment axes. Both implement the text-only `Embedder` protocol for now.
**Rationale:** `text-embedding-005` is a serious text retrieval contender via free API. The multimodal model is interesting to benchmark on text despite its short context — seeing how cross-modal training affects text-only retrieval quality is a research finding. Two different SDKs: `google-generativeai` for text, `google-cloud-aiplatform` for multimodal.
**Alternatives considered:** Only adding the text model — rejected because the multimodal model's text performance is an interesting research question even with the 32-token limit.

## Separation of concerns: every pipeline stage gets its own protocol — 2026-03-16
**Decision:** Each stage of the experiment pipeline (document representation, query generation, retrieval, scoring, filtering) gets its own protocol in its own module. Favor more separation over less, even when it seems like overkill for the current scope.
**Rationale:** This project is designed as an experimental test bench that may expand beyond the school project. Multimodal is an end goal, which means every stage may eventually handle different input types (text, image, video). If stages are tightly coupled now, multimodal breaks everything later. Even something like round-trip filtering — currently just "does the query retrieve its source chunk?" — becomes a different operation when the source is an image.
**Concrete implications:**
- `Document` gets its own module (`src/document.py`), not a dataclass in protocols.py
- `QueryGenerator` is a protocol, separate from `Scorer`
- Round-trip filtering is a separate step/protocol, not baked into query generation
- RAGAS is one implementation, not the framework
**Alternatives considered:** Keeping things compact in protocols.py — rejected because multimodal will outgrow it and refactoring coupled code is harder than starting separated.

## Two separate experiments, not one big matrix — 2026-03-16
**Decision:** Run two focused experiments rather than one full cartesian product. Experiment 1: strategy × model size (hold chunking constant at recursive 512/50). Experiment 2: chunking × model size (hold strategy constant at Naive RAG). Each isolates one interaction.
**Rationale:** The full 4-axis matrix (3 chunkers × 3 embedders × 5 strategies × 6 models = 270 configs) is computationally prohibitive and methodologically murky — too many variables to attribute effects. Two focused experiments are cleaner science and fill two distinct literature gaps: strategy × size (under-studied) and chunking × size (unstudied). The pipeline supports the full matrix but the experiments don't need to use it all at once.
**Research gaps filled:** (1) No prior work studies strategy × model size for small (<14B) models. (2) No peer-reviewed paper compares chunking strategies with model size as a variable — all chunking studies use GPT-class models (Wang et al. 2024, Chen et al. 2023). Liu et al. 2023 ("Lost in the Middle") provides indirect evidence that small models are more sensitive to retrieval precision, which chunking directly affects.

## Multiple query generators by design, not one framework — 2026-03-16
**Decision:** Build multiple QueryGenerator implementations: RAGAS (synthetic with evolution), TemplateQueryGenerator (entity extraction + templates), BEIRQuerySet (existing benchmarks), HumanQuerySet (hand-curated CSV). The system is not "a RAGAS project" — RAGAS is one option.
**Rationale:** Different generators have different strengths. RAGAS gives citable synthetic queries. Templates give uniformity. BEIR gives human-written gold standard. Human-curated gives a validation anchor. Running the same experiment across generator types tests whether your RAG ranking is robust to query source — itself a research finding. Upstream datasets should also be adaptable (Wikipedia now, Gutenberg or domain-specific later).
**Alternatives considered:** RAGAS-only — rejected because single-source evaluation is a methodological weakness, and the QueryGenerator protocol makes adding alternatives cheap.
