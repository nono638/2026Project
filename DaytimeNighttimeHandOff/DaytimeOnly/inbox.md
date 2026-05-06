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
- **VERIFY OLLAMA TAGS BEFORE FIRST EXP 1/2 LAUNCH ON 5090.** task-054 baked 10 new tags into the model matrix without ollama.com verification (sandbox network blocked). Tags: qwen3.5:0.8b/2b/4b/9b, qwen3.6:27b, qwen3.6:35b-a3b, gemma4:e2b/e4b/26b/31b. Run `ollama show <tag>` for each on the 5090, fix any 404s before kicking off the matrix.
- **Live Ollama smoke test pending (task-053).** Before Exp 1/2 generation runs on the 5090, run `python -c "from scripts.experiment_utils import get_ollama_model_details; print(get_ollama_model_details('qwen3.5:4b'))"` to confirm the helper actually retrieves quant + digest from a live `/api/show`. Currently only mocked in tests.
- **Refresh VRAM/disk estimates in runpod-setup-guide.md.** task-054 left ~16GB / ~22GB / ~19GB estimates for the new MoE/large models. After first pull on the 5090, replace with measured values.
- **Backfill quant metadata for Exp 0 v1/v2/v3.** task-053 added `scripts/backfill_quant_metadata.py`. Worth running once on the 5090 against the existing v1/v2/v3 results so the gallery shows real quant values instead of "unknown".
- **Pin remaining model tags to `-q4_K_M` once verified.** task-053 left bare tags in `pull_models.py` REQUIRED_MODELS and Exp 1/2 ALL_MODELS pending verification. Once `ollama show <tag>` confirms the published variants on the 5090, swap in explicit pins for determinism.
- **`bert_score` not installed in local daytime venv.** 13 `tests/test_bertscore.py` tests fail with `ModuleNotFoundError: No module named 'bert_score'`. Night venv has it (passed there). Either install locally (`pip install bert_score`) or accept that test_bertscore is GPU-side only — decide and document in ENVIRONMENT.md.
