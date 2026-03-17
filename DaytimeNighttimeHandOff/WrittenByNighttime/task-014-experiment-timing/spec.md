# task-014: Add Timing to Experiment Runner

## Summary

Add latency measurement to the experiment runner so each row in ExperimentResult includes
how long the strategy and scorer took. This makes latency a first-class output dimension
alongside quality scores, enabling time-vs-quality tradeoff analysis ("is a 60-second
answer meaningfully better than a 5-second one?").

## Requirements

1. Each row in the ExperimentResult DataFrame includes three new columns:
   - `strategy_latency_ms` — time for `strategy.run()` in milliseconds
   - `scorer_latency_ms` — time for `scorer.score()` in milliseconds
   - `total_latency_ms` — sum of the two
2. Timing uses `time.perf_counter()` for high-resolution measurement.
3. The progress line in `Experiment.run()` includes the timing:
   `[3/30] naive / qwen3:0.6b / What is... (1234ms)`
4. ExperimentResult gets a new analysis method `latency_report()` that prints:
   - Mean/std/min/max latency grouped by (strategy, model)
   - Sorted by mean total_latency_ms ascending (fastest first)
5. ExperimentResult gets a `time_vs_quality()` method that prints a table showing
   mean quality and mean latency per config, sorted by quality descending, so users
   can see the tradeoff.
6. All existing tests continue to pass — the new columns are additive.

## Files to Modify

- `src/experiment.py` — In `Experiment.run()`:
  - Add `import time` at top
  - Wrap `strategy.run()` call with `time.perf_counter()` start/end
  - Wrap `scorer.score()` call with `time.perf_counter()` start/end
  - Add `strategy_latency_ms`, `scorer_latency_ms`, `total_latency_ms` to each row dict
  - Update the progress print to include timing
  - Add `latency_report()` and `time_vs_quality()` methods to `ExperimentResult`

## New Dependencies

None.

## Edge Cases

- **Empty corpus:** `Experiment.run()` already returns empty DataFrame for empty corpus.
  The new columns simply won't exist. `latency_report()` and `time_vs_quality()` should
  handle empty DataFrame gracefully (print "No data." and return).
- **Very fast operations:** Some mock strategies return instantly (<1ms). This is fine —
  `perf_counter` has sub-microsecond resolution. Don't add artificial floors.
- **Scorer raises ScorerError:** If scoring fails, the row won't be added (existing
  behavior). No timing concerns.

## Decisions Made

- **Milliseconds, not seconds.** Why: Most RAG calls are 1-60 seconds. Milliseconds
  avoid floats like 0.003 and make the numbers easier to read in tables and visualizations.
- **Separate strategy and scorer timing.** Why: Users need to know if latency comes from
  generation (strategy) or evaluation (scorer). In production, you'd skip the scorer
  entirely, so strategy latency is the real user-facing metric. Scorer latency matters
  for experiment planning (how long will the full matrix take?).
- **`time.perf_counter()` over `time.time()`.** Why: `perf_counter` is monotonic and
  has the highest available resolution. `time.time()` can jump on clock adjustments.
- **Not a feature for the meta-learner.** Why: Latency depends on hardware, not query
  features. A model that's fast on one machine is fast on all machines for the same query.
  It's a measurement for human decision-making, not a training feature.

## What NOT to Touch

- `src/protocols.py` — No protocol changes.
- `src/retriever.py` — Retriever build time is interesting but is per-(doc, chunker, embedder)
  triple, not per-row. Out of scope for this task.
- Existing analysis methods (`compare()`, `heatmap()`, etc.) — These stay unchanged.
- Existing test files — They should still pass with the added columns.

## Testing Approach

Pre-written tests use mock components (same pattern as test_e2e_smoke.py). Tests verify:
- New columns exist in ExperimentResult output
- Timing values are non-negative
- `latency_report()` produces grouped output
- `time_vs_quality()` produces a combined table
- Empty DataFrame handling
