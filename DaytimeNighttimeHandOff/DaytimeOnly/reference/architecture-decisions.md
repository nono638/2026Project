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

## HotpotQA as primary evaluation corpus — 2026-03-17
**Decision:** Use HotpotQA as the primary experiment corpus. Use the existing Wikipedia sample + TemplateQueryGenerator as a secondary generalization check.
**Rationale:** HotpotQA has multi-hop questions (bridge + comparison types) with difficulty labels — exactly where RAG strategies should diverge. NaiveRAG will struggle on bridge questions that require two retrieval steps; AdaptiveRAG and CorrectiveRAG should shine. Gold answers enable scorer validation (professor's concern about LLM-as-judge noise). 113K questions, Wikipedia-based, CC BY-SA 4.0, on HuggingFace. The existing Wikipedia sample with template queries serves as a generalization test: if the same strategies win on both corpora, findings are robust.
**Alternatives considered:** (A) SQuAD — rejected, single-paragraph context doesn't differentiate retrieval strategies. (B) FRAMES — rejected, too small (824 examples) and too hard (designed for GPT-4). (C) RAGBench (the external dataset) — rejected, pre-retrieved contexts prevent testing our own retrieval pipeline. (D) Wikipedia-only — rejected, no gold answers means no scorer validation.
**Sources:** https://huggingface.co/datasets/hotpotqa/hotpot_qa, https://www.evidentlyai.com/blog/rag-benchmarks

## Dual evaluation: intrinsic + extrinsic — 2026-03-17
**Decision:** The pipeline supports two evaluation modes that coexist. Intrinsic: scorer rates answers against retrieved context (always available). Extrinsic: compare answers against gold reference facts (when `Query.reference_answer` is set). Both produce columns in ExperimentResult; analysis methods check which columns exist.
**Rationale:** Most real-world RAG users don't have gold answers — intrinsic evaluation must work standalone. But when gold data exists, it enables scorer calibration and factual correctness metrics. These measure different things: intrinsic measures "did the model use context well," extrinsic measures "did it get the right answer." A model can score high intrinsic / low extrinsic (faithful to bad retrieval) or vice versa. Both signals matter.
**Alternatives considered:** (A) Require gold data — rejected, most users won't have it. (B) Extrinsic only — rejected, factual match doesn't capture answer quality.

## Latency as a first-class experiment dimension — 2026-03-17
**Decision:** Time each strategy.run() + scorer.score() call and include latency_ms in ExperimentResult output. This is a measurement, not a feature for the meta-learner (latency depends on hardware, not query features).
**Rationale:** Users care about "is the 60-second answer meaningfully better than the 5-second answer?" This is central to the Explorer findings gallery. Without timing data, we can show quality differences but not the time-quality tradeoff curve.
**Alternatives considered:** Estimate latency from model size — rejected, real measurements are easy and far more credible.

## Multiple query generators by design, not one framework — 2026-03-16
**Decision:** Build multiple QueryGenerator implementations: RAGAS (synthetic with evolution), TemplateQueryGenerator (entity extraction + templates), BEIRQuerySet (existing benchmarks), HumanQuerySet (hand-curated CSV). The system is not "a RAGAS project" — RAGAS is one option.
**Rationale:** Different generators have different strengths. RAGAS gives citable synthetic queries. Templates give uniformity. BEIR gives human-written gold standard. Human-curated gives a validation anchor. Running the same experiment across generator types tests whether your RAG ranking is robust to query source — itself a research finding. Upstream datasets should also be adaptable (Wikipedia now, Gutenberg or domain-specific later).
**Alternatives considered:** RAGAS-only — rejected because single-source evaluation is a methodological weakness, and the QueryGenerator protocol makes adding alternatives cheap.
