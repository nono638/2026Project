# Inbox

> **Managed by daytime Claude. Night instance does not read this file.**
>
> Zero-friction capture for anything that comes up mid-session but isn't being acted on
> right now. No judgment required to add — if it might matter, capture it with one sentence
> of context. Cleared completely at every session-open triage.

---

<!-- Add captures below with today's date header. Format:
## YYYY-MM-DD
- [one-line capture — what it is and why it came up]
-->

<!-- inbox cleared 2026-03-27 session -->
<!-- Triaged: CostGuard loop abort → done (task-048 merged). CostGuard Opus cost → small fix, do now during daytime. -->

## 2026-05-06
- **Refresh VRAM/disk estimates in runpod-setup-guide.md.** task-054 left ~16GB / ~22GB / ~19GB estimates for the new MoE/large models. After first pull on the 5090, replace with measured values.
- **Pin remaining model tags to `-q4_K_M` once verified.** task-053 left bare tags in `pull_models.py` REQUIRED_MODELS and Exp 1/2 ALL_MODELS pending verification. Once `ollama show <tag>` confirms the published variants on the 5090, swap in explicit pins for determinism.
- **`bert_score` not installed in local daytime venv.** 13 `tests/test_bertscore.py` tests fail with `ModuleNotFoundError: No module named 'bert_score'`. Night venv has it (passed there). Either install locally (`pip install bert_score`) or accept that test_bertscore is GPU-side only — decide and document in ENVIRONMENT.md.

<!-- 2026-05-06: VERIFY OLLAMA TAGS, live get_ollama_model_details smoke test, and backfill_quant_metadata.py — promoted to 5090-migration.md Steps 3.0, 4.5, 4.6 -->
<!-- 2026-05-06: task-055 retroactive Ollama-judge scoring on v3 — promoted to 5090-migration.md Step 8 -->

