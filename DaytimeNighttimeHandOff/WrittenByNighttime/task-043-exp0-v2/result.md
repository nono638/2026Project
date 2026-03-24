# Result: task-043 — Experiment 0 v2

**Status:** done
**Completed:** 2026-03-24T00:15:20

## Commits
- `1b43147f1774dce2d7a91533294d5258c600e995` — night: task-043 Experiment 0 v2 with diagnostics, reranker, answer quality

## Test Results
- Command run: `python -m pytest DaytimeNighttimeHandOff/WrittenByDaytime/task-043-exp0-v2/tests/ -v`
- Outcome: 18 passed, 0 failed
- Failures: none

- Command run: `python -m pytest tests/ --ignore=DaytimeNighttimeHandOff/ -v`
- Outcome: 576 passed, 0 failed
- Failures: none

## Decisions Made
1. **_RerankedRetriever passes top_k=None to underlying retriever**: The wrapper retrieves all candidates the retriever would normally return (its default top_k), then the reranker selects the best `reranker_top_k`. This matches the spec's "retrieve 10, keep 3" intent — the retrieval_top_k CLI flag controls the retriever's default.

2. **answer_quality checks "poor" before "good"**: Per spec, any single metric below threshold flags poor, then all must be above for good. This means an edge case where something is both poor AND good by different metrics always resolves to "poor" (the OR condition fires first).

3. **Gallery generates v2 as a separate page**: Instead of embedding v1 and v2 in one page (which would make it very long), v2 gets its own `experiment_0_v2.html` page. The experiments_info list includes both for the index navigation.

4. **Deleted local compute_f1 and exact_match from run_experiment_0.py**: Now imported from experiment_utils as spec requested, avoiding duplication.

## Flags for Morning Review
- The gallery v2 page reuses `build_experiment0_figures()` from `generate_experiment0_dashboard.py` for scorer charts. If that function fails on v2 data (different column names), the v2-specific charts (answer_quality, failure_stage) will still render but scorer comparison charts will be missing.
- The `_generate_experiment_0_v2` function adds a "0v2" entry to experiments_info — the index page template may need tweaking if it expects integer experiment numbers.

## Attempted Approaches
None — implementation was straightforward.
