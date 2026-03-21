# Plan: task-038 — Constraint-Aware Analysis API

## Approach

1. Add constraint parsing helper `_parse_constraint` to `src/experiment.py` — copied from
   train.py's approach, not imported (per spec: separate concerns).
2. Add 7 methods to ExperimentResult class:
   - `filter(constraints)` → new ExperimentResult
   - `best_config(metric, maximize, constraints)` → dict (rewrite of existing)
   - `configs_above(metric, threshold)` → ExperimentResult
   - `configs_below(metric, threshold)` → ExperimentResult
   - `budget_analysis(quality_metric, cost_metric, budget)` → DataFrame
   - `pareto_front(quality_metric, cost_metric)` → DataFrame
   - `rank(metric, ascending, top_n)` → DataFrame
3. Update 3 existing tests that expect best_config() to return a tuple.

## Files to Modify
- `src/experiment.py` — add methods + constraint parsing
- `tests/test_core.py` — update test_best_config
- `tests/test_e2e_smoke.py` — update best_config assertion
- `tests/test_integration.py` — update best_config assertion

## Ambiguities
- The `_validate_column` helper: spec says raise KeyError with available columns.
  Will implement this as a shared helper used by all methods.
