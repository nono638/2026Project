# Architecture Decisions

> **Night instance reads this file** — keep entries clear and actionable.
>
> Records decisions made and why, so they don't get relitigated.

---

## Multiclass classification, not ordinal regression — 2026-03-16
**Decision:** The meta-learner uses multiclass classification (XGBoost), not ordinal regression.
**Rationale:** The target variable is a (strategy, model_size) pair. The professor suggested ordinal regression because larger models are generally expected to do better — but the core hypothesis is that strategy can invert size ordering (e.g., 4B+Self-RAG beats 8B+Naive). If strategy×size interactions break the ordinal assumption, ordinal regression would impose a wrong constraint. Multiclass lets the model learn arbitrary interactions.
**Alternatives considered:** Ordinal regression (professor's suggestion) — rejected because it assumes a monotonic relationship between model size and quality, which is exactly what the project aims to disprove.
