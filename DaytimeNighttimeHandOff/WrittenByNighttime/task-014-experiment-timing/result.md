# Result: task-014 — Add Timing to Experiment Runner
**Status:** done
**Completed:** 2026-03-18T00:05:00

## Commits
- `<pending>` — night: task-014 add timing to experiment runner

## Test Results
- Command run: `python -m pytest DaytimeNighttimeHandOff/WrittenByDaytime/task-014-experiment-timing/tests/test_experiment_timing.py -v`
- Outcome: 10 passed, 0 failed
- Failures: none
- Regression check: `python -m pytest tests/ --ignore=DaytimeNighttimeHandOff/ -v` — 246 passed

## Decisions Made
- Placed timing measurement before the progress print so the print can include latency.
- `latency_report()` sorts by mean ascending (fastest first) — natural for scanning.
- `time_vs_quality()` sorts by mean_quality descending — natural for finding the best tradeoff.
- Both methods return empty DataFrame (not None) for empty input — consistent with existing patterns.

## Flags for Morning Review
None.

## Attempted Approaches (if skipped/blocked)
n/a
