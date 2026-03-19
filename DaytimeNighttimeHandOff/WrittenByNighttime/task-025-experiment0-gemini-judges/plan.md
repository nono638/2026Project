# Plan: task-025 — Expand Experiment 0 to All-Gemini Judges

## Files to modify
- `scripts/run_experiment_0.py` — update JUDGE_CONFIGS, docstring, cost_estimates, recommendation text, add startup log

## Files to create
- `tests/test_experiment_0.py` — 6 tests as described in spec

## Approach
1. Update JUDGE_CONFIGS: add gemini-2.5-flash-lite, drop claude-opus, reorder
2. Update docstring to reflect "up to 6 LLM judges (4 Gemini + 2 Claude)"
3. Update cost_estimates dict: add flash-lite, remove opus
4. Update recommendation text
5. Add startup log showing initialized vs skipped scorers
6. Write tests with all API mocked

## Ambiguities
- None — spec is clear.
