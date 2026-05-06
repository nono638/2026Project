# Plan: task-054 update Exp 1/2 model matrices

## Files to modify
1. `scripts/run_experiment_1.py`
   - Replace `ALL_MODELS` with the 10-tag list (Qwen 3.5/3.6 + Gemma 4) per spec.
   - Update module docstring and file header to say "5 strategies × 10 models = 50 configurations".
   - Update model-size mapping if used by report code (`model_sizes` dict around line 290 mapping qwen3:* to billions). Build new mapping for the new tags.
2. `scripts/run_experiment_2.py`
   - Replace `ALL_MODELS` with 4 Qwen 3.5 tags.
   - Update docstring "Qwen3" → "Qwen 3.5".
   - Update MODEL_ORDER / MODEL_SIZES analogues if present.
3. `scripts/generate_experiment1_dashboard.py` and `generate_experiment2_dashboard.py`
   - Update `MODEL_SIZES` and `MODEL_ORDER` constants to the new tag set.
4. `deploy/setup_pod.py` — `qwen3:4b` → `qwen3.5:4b`. Coordinate with task-053's `qwen3:4b-q4_K_M` change. Result on this branch (without task-053): `qwen3.5:4b`. Daytime merge will reconcile to `qwen3.5:4b-q4_K_M` if task-053 lands too.
5. `scripts/pull_models.py` — update REQUIRED_MODELS to the 11-model list (10 distinct LLM tags + mxbai-embed-large; the 4 Exp 2 tags are a subset of Exp 1's small-tier so no extra entries needed).
6. `docs/output-format.md` — update model literal lists at lines 82, 88.
7. `docs/running-experiments.md` — `qwen3:4b` example → `qwen3.5:4b`. Update config-count statements.
8. `README.md` — search for `qwen3:` / `gemma3:` references and update.
9. `DaytimeNighttimeHandOff/DaytimeOnly/reference/runpod-setup-guide.md` — update model tables.
10. Tests — update any `tests/test_run_experiment_*` or `test_pull_models` / `test_setup_pod` model-tag assertions.

## Tag verification
Cannot reach the public Ollama registry from the nighttime sandbox (network blocked
per CLAUDE.md). The spec asks for live `curl ...ollama.com/library/...` checks. Will
flag this in result.md so morning can run the verification before first experiment
launch.

## Coordination with task-053
Both branches independently touch `setup_pod.py` (`DEFAULT_MODELS`) and
`scripts/pull_models.py`. Task-053 is on its own branch. The merge order will determine
the final value. Per spec note: "whichever lands second should fold the other's change."
Both branches land in the morning review; daytime will reconcile.
