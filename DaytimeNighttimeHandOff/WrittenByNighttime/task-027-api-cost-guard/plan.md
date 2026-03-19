# Plan: task-027 — API Cost Guard

## Files to create
- `src/cost_guard.py` — CostGuard class + CostLimitExceeded exception
- `tests/test_cost_guard.py` — 5 tests

## Files to modify
- `src/scorers/llm.py` — add optional cost_guard param, call record_call after API calls
- `scripts/run_experiment_0.py` — add --max-cost flag, create CostGuard, wrap scoring in try/except

## Approach
1. Create src/cost_guard.py with CostGuard class, COST_PER_CALL table, CostLimitExceeded
2. Modify LLMScorer.__init__ to accept optional cost_guard, call record_call in score()
3. Add --max-cost flag and try/except wrapping in run_experiment_0.py
4. Write 5 tests for CostGuard
5. Merge task-025 branch first since we modify run_experiment_0.py

## Ambiguities
- None — spec is clear.
