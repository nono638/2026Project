# task-038: Constraint-Aware Analysis API

## Summary

Extend `ExperimentResult` in `src/experiment.py` with methods for filtering results
by constraints, finding the best config within budgets, and computing Pareto-optimal
configurations. This is the Builder audience's primary tool for answering "what's the
best config for MY constraints?"

Currently ExperimentResult has `best_config(metric)` which returns the global best.
Users need: best config within a latency budget, best config above a quality floor,
cheapest config that meets a quality threshold, and the full Pareto frontier of
non-dominated configs.

## Requirements

1. **`filter(constraints: dict) -> ExperimentResult`**: Return a new ExperimentResult
   with rows matching all constraints. Constraint syntax matches train.py:
   `{"quality": ">3.5", "total_latency_ms": "<5000", "model": "qwen3:4b"}`.
   Operators: `>`, `>=`, `<`, `<=`, `==`, `!=`. String values without operators do
   exact match. Returns empty ExperimentResult if no rows match.

2. **`best_config(metric, maximize=True, constraints=None) -> dict`**: Enhanced version
   of existing best_config(). Returns a dict (not tuple) with all 4 axis values plus
   the metric's mean value. Accepts optional constraints dict. `maximize=False` for
   metrics like latency where lower is better. When constraints yield no matching rows,
   raise `ValueError` with a clear message.

3. **`configs_above(metric, threshold) -> ExperimentResult`**: Return an
   ExperimentResult containing only rows where the per-config mean of `metric` meets
   or exceeds `threshold`. Groups by (chunker, embedder, strategy, model), computes
   mean, filters, then returns matching rows.

4. **`configs_below(metric, threshold) -> ExperimentResult`**: Same as configs_above
   but `<=`. Useful for latency filtering ("configs under 2 seconds").

5. **`budget_analysis(quality_metric, cost_metric, budget, maximize_quality=True) -> pd.DataFrame`**:
   For each config, compute mean quality and mean cost. Filter to configs where mean
   cost ≤ budget. Return DataFrame sorted by quality (best first) with columns:
   chunker, embedder, strategy, model, mean_{quality_metric}, mean_{cost_metric}.
   If no configs meet budget, return empty DataFrame.

6. **`pareto_front(quality_metric, cost_metric, maximize_quality=True, minimize_cost=True) -> pd.DataFrame`**:
   Compute the Pareto frontier — configs where no other config is strictly better on
   both metrics. Return DataFrame sorted by quality descending, with columns: chunker,
   embedder, strategy, model, mean_{quality_metric}, mean_{cost_metric}.

7. **`rank(metric, ascending=False, top_n=None) -> pd.DataFrame`**: Rank all configs
   by mean metric value. Returns DataFrame with rank, chunker, embedder, strategy,
   model, mean, std, count. `top_n` limits output. `ascending=True` for "lowest first"
   (latency ranking).

## Files to Modify

- `src/experiment.py` — Add methods to ExperimentResult class:
  - `filter()` (new)
  - `best_config()` (rewrite — currently returns tuple, change to dict with optional
    constraints and maximize flag)
  - `configs_above()` (new)
  - `configs_below()` (new)
  - `budget_analysis()` (new)
  - `pareto_front()` (new)
  - `rank()` (new)

## Files to Read (context, do not modify)

- `src/model/train.py` — `_parse_constraint()` and `_apply_constraints()` for constraint
  parsing syntax to reuse. Do NOT import from train.py — copy the parsing logic into
  experiment.py as a module-level helper, or better: extract to a shared utility if the
  duplication bothers you. But keeping it simple and self-contained in experiment.py is
  preferred.

## New Dependencies

None.

## Edge Cases

- **Empty DataFrame**: All methods should handle gracefully — return empty results, not
  crash.
- **Missing metric column**: Raise `KeyError` with message like
  `"Column 'latency_ms' not found. Available: [...]"`.
- **Constraint on non-existent column**: Same `KeyError`.
- **No configs meet constraints**: `filter()` returns empty ExperimentResult.
  `best_config()` raises `ValueError`. `budget_analysis()` returns empty DataFrame.
- **All configs are Pareto-optimal**: Return all of them (happens when metrics don't
  trade off).
- **Single config in data**: Pareto front is just that config. Rank is just that config.
- **NaN values in metric column**: Drop NaN rows before computing means. If a config
  has all NaN for a metric, exclude it from rankings.
- **best_config backward compat**: The old signature was `best_config(metric="quality")`
  returning a tuple. The new signature adds `maximize` and `constraints` as keyword-only
  args. BUT the return type changes from tuple to dict. This is an intentional breaking
  change — the old return type was not useful (just a tuple with no labels). Any existing
  code calling `best_config()` will need to update. Check if anything in the codebase
  calls `best_config()` and update it.

## Decisions Made

- **Constraint syntax from train.py**: Reuse the same `>3.5`, `<=1000`, `==qwen3:4b`
  syntax. **Why:** Consistency across the codebase. Users learn one syntax.
- **Copy constraint parsing, don't import from train.py**: The parsing logic is ~20 lines.
  Importing from train.py couples analysis to the model training module. **Why:** These
  are separate concerns that happen to share a syntax.
- **best_config returns dict, not tuple**: The old tuple return was unlabeled and required
  knowing the axis order. A dict with `{"chunker": ..., "embedder": ..., "strategy": ...,
  "model": ..., "mean_quality": ...}` is self-documenting. **Why:** Better UX for the
  Builder audience.
- **Pareto front uses per-config means**: Not per-row values. Each config is one point
  in (quality, cost) space, represented by its mean across all queries. **Why:** Per-row
  Pareto would be meaningless (individual query results aren't comparable configs).
- **No visualization methods in this task**: The analysis API returns DataFrames. Plotting
  (Pareto scatter, budget bar chart) is a separate concern for the gallery. **Why:** Keep
  this task focused on the data API.

## What NOT to Touch

- `src/model/train.py` — Do not modify. Just read the constraint syntax for reference.
- ExperimentResult's existing methods (`compare`, `pivot`, `summary`, `compare_strategies`,
  `compare_models`, `heatmap`, `per_query`, `strategy_vs_size`, `latency_report`,
  `time_vs_quality`, `to_csv`, `to_parquet`, `from_parquet`, `merge`) — do not modify
  these. Only modify `best_config`.
- `scripts/` — This task only touches `src/experiment.py`.

## Testing Approach

Tests in `DaytimeNighttimeHandOff/WrittenByDaytime/task-038-constraint-analysis/tests/`.

Build a synthetic DataFrame with known values for deterministic assertions:
- 3 strategies × 2 models × 5 queries = 30 rows
- Known quality scores, latency values, config labels
- Tests:
  - `test_filter_single_constraint` — filter by quality > threshold
  - `test_filter_multiple_constraints` — quality > X AND latency < Y
  - `test_filter_string_match` — model == "qwen3:4b"
  - `test_filter_empty_result` — impossible constraint returns empty
  - `test_best_config_returns_dict` — check dict keys and values
  - `test_best_config_maximize` — highest quality
  - `test_best_config_minimize` — lowest latency
  - `test_best_config_with_constraints` — best quality where latency < budget
  - `test_best_config_no_match_raises` — ValueError on impossible constraints
  - `test_configs_above` — configs with mean quality above threshold
  - `test_configs_below` — configs with mean latency below threshold
  - `test_budget_analysis` — best quality within cost budget
  - `test_budget_analysis_no_match` — empty DataFrame when budget too low
  - `test_pareto_front` — verify only non-dominated configs returned
  - `test_pareto_front_single_config` — single config is its own Pareto front
  - `test_rank_descending` — ranked by quality
  - `test_rank_ascending` — ranked by latency
  - `test_rank_top_n` — limited output
  - `test_empty_dataframe` — all methods handle gracefully
  - `test_missing_column_raises` — KeyError with helpful message
- Run with `pytest tests/test_constraint_analysis.py -v`
