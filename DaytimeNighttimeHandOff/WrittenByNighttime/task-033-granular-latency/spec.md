# task-033: Granular latency breakdown — per-stage timing columns

## Summary

Split the monolithic `strategy_latency_ms` into per-stage timing: `retrieval_latency_ms`,
`generation_latency_ms`, and `reranking_latency_ms`. This tells users which pipeline stage
is the bottleneck (embedding+search vs LLM inference vs reranking) without changing any
protocols or strategy implementations.

## Requirements

1. Add a `_TimedRetriever` wrapper class in `src/experiment.py` that transparently delegates
   to a real `Retriever` but accumulates wall-clock time spent in `retrieve()` calls.
2. In `Experiment.run()`, wrap the cached retriever in `_TimedRetriever` before passing to
   `strategy.run()`. Reset the timer before each strategy call, read it after.
3. Add three new columns to each result row:
   - `retrieval_latency_ms` — total time spent in `retriever.retrieve()` calls during
     `strategy.run()`. Includes query embedding + index search + BM25 + RRF fusion.
     Accumulates across multiple retrieve() calls (MultiQueryRAG calls 4x, etc.).
   - `generation_latency_ms` — `strategy_latency_ms - retrieval_latency_ms`. Approximates
     LLM inference time plus any strategy overhead (prompt construction, filtering calls).
   - `reranking_latency_ms` — time for the pre-strategy reranking step (lines 188-208 in
     current code). `None` when no reranker is configured. `0.0` should not occur — if a
     reranker is present, it takes nonzero time.
4. Keep `strategy_latency_ms`, `scorer_latency_ms`, and `total_latency_ms` unchanged for
   backwards compatibility. `strategy_latency_ms` = `retrieval_latency_ms + generation_latency_ms`.
5. Update `ExperimentResult.latency_report()` to include the new columns in its output —
   group by (strategy, model) and show mean/std/min/max for all latency columns.

## Files to Modify

- `src/experiment.py`:
  - Add `_TimedRetriever` class (new, ~25 lines, at module level before `Experiment`)
  - In `Experiment.run()`, inner loop (lines ~216-226 current): wrap retriever, reset timer,
    compute new columns from timer state after strategy.run()
  - Add `reranking_latency_ms` timing around the reranking block (lines ~188-208 current)
  - Add `retrieval_latency_ms`, `generation_latency_ms`, `reranking_latency_ms` to row dict
  - In `ExperimentResult.latency_report()`: update to include new columns

## New Dependencies

None — only uses `time.perf_counter()` which is already imported.

## Edge Cases

- **No reranker configured:** `reranking_latency_ms` = `None` (not 0.0 — distinguishes
  "not used" from "used but instant")
- **Strategy makes zero retrieval calls** (AdaptiveRAG simple path, SelfRAG no-retrieval
  path): `retrieval_latency_ms` = 0.0, `generation_latency_ms` = `strategy_latency_ms`.
  This is correct — the strategy chose not to retrieve.
- **Empty corpus / no queries:** experiment returns empty DataFrame, no timing columns
  needed (existing behavior, unchanged)
- **TimedRetriever attribute access:** strategies may access `retriever.chunks` or other
  attributes. `_TimedRetriever.__getattr__` must delegate to the inner retriever.
- **TimedRetriever and isinstance checks:** `Experiment.__init__` validates the retriever
  at construction time (before wrapping). The wrapper is applied per-call in `run()`, so
  no isinstance check sees it. No issue.

## Decisions Made

- **Wrapper pattern, not Retriever modification:** Don't modify `Retriever` or `src/retriever.py`
  at all. The timing wrapper lives in `experiment.py` and is an implementation detail of
  the experiment runner. **Why:** Retriever is a concrete class used in many places. Adding
  timing state to it pollutes the interface. A thin wrapper applied only during timed runs
  is cleaner and reversible.
- **Accumulate, don't instrument sub-calls:** We time total `retrieve()` duration, not
  embedding-vs-search separately. **Why:** Breaking those apart requires modifying Retriever
  internals. The combined retrieval time is the useful signal — users can see "retrieval
  takes 200ms, generation takes 3s, bottleneck is generation." If sub-retrieval timing is
  needed later, it's a separate task.
- **reranking_latency_ms is None, not 0.0, when no reranker:** **Why:** Distinguishes "not
  configured" from "configured but somehow instant." Consistent with how `mean_rerank_score`
  and `var_rerank_score` are already handled.
- **Don't change total_latency_ms formula:** Keep it as `strategy_latency_ms + scorer_latency_ms`.
  **Why:** Backwards compatibility. The reranking step is currently for features/metadata
  only (strategies do their own retrieval), so it's not part of the user-facing pipeline
  latency. If/when reranking is wired into the strategy path, total_latency_ms can be
  updated then.

## What NOT to Touch

- `src/retriever.py` — no changes to the Retriever class
- `src/protocols.py` — no protocol changes
- `src/strategies/*.py` — no strategy changes
- Any existing latency columns (`strategy_latency_ms`, `scorer_latency_ms`, `total_latency_ms`) —
  values must remain identical to what they'd be without this change

## Testing Approach

Pre-written tests in `task-033-granular-latency/tests/test_granular_latency.py`:

1. `_TimedRetriever` accumulates time across multiple retrieve() calls
2. `_TimedRetriever` resets timer correctly
3. `_TimedRetriever` delegates attribute access to inner retriever
4. Experiment output contains all three new columns
5. `generation_latency_ms` equals `strategy_latency_ms - retrieval_latency_ms` (within float tolerance)
6. `reranking_latency_ms` is None when no reranker
7. `reranking_latency_ms` is a positive float when reranker is present
8. `latency_report()` includes new columns

Run: `python -m pytest DaytimeNighttimeHandOff/WrittenByDaytime/task-033-granular-latency/tests/ -v`
