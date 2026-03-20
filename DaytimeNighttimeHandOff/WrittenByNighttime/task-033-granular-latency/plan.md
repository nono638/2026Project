# Plan: task-033 — Granular Latency Breakdown

## Approach

Modify only `src/experiment.py`. Three changes:

### 1. Add `_TimedRetriever` wrapper class (before `Experiment`)
- Wraps a real `Retriever`, delegates all attribute access via `__getattr__`
- Overrides `retrieve()` to accumulate wall-clock time via `time.perf_counter()`
- Exposes `retrieval_ms` property and `reset()` method

### 2. Modify `Experiment.run()` inner loop
- Time the reranking block (lines 188-208) with `perf_counter`
- Wrap `retriever` in `_TimedRetriever` before passing to `strategy.run()`
- Reset timer before each strategy call, read `retrieval_ms` after
- Compute: `retrieval_latency_ms = timed.retrieval_ms`
- Compute: `generation_latency_ms = strategy_latency_ms - retrieval_latency_ms`
- Set `reranking_latency_ms = None` when no reranker, else measured time
- Add all three to row dict

### 3. Update `ExperimentResult.latency_report()`
- Include `retrieval_latency_ms`, `generation_latency_ms`, `reranking_latency_ms`
- Use multi-column groupby agg for all latency columns present in the DataFrame

## Files Modified
- `src/experiment.py` (only file)

## Ambiguities
- None identified — spec is clear.
