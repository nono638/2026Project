# Plan: task-014 — Add Timing to Experiment Runner

## Approach

1. Modify `src/experiment.py`:
   - Add `import time` at top
   - Wrap `strategy.run()` with `time.perf_counter()` to get `strategy_latency_ms`
   - Wrap `scorer.score()` with `time.perf_counter()` to get `scorer_latency_ms`
   - Compute `total_latency_ms = strategy_latency_ms + scorer_latency_ms`
   - Add all three to each row dict
   - Update progress print to include timing
2. Add `latency_report()` method to `ExperimentResult`:
   - Groups by (strategy, model), computes mean/std/min/max of total_latency_ms
   - Returns empty DataFrame if no data
3. Add `time_vs_quality()` method to `ExperimentResult`:
   - Shows mean quality + mean latency per config
   - Returns empty DataFrame if no data

## Files to Modify
- `src/experiment.py`

## Ambiguities
- None — spec is clear.
