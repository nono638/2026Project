# Result: task-052 — Multi-judge support for Exp 1/2
**Status:** done
**Completed:** 2026-05-06T12:15:03

## Commits
(filled in after commit on branch)

## Test Results
- Pre-written task tests: `pytest DaytimeNighttimeHandOff/WrittenByDaytime/task-052-multi-judge-exp12/tests/ -v` → 12 passed, 0 failed
- Full regression: `pytest tests/ --ignore=DaytimeNighttimeHandOff/ -q` → 641 passed, 12 skipped, 0 failed (including dashboard tests after fixture updates)

## Decisions Made
- **`allow_abbrev=False` plus pre-flight `--scorer` check.** Argparse normally treats `--scorer` as a unique prefix of `--scorers` (silent abbreviation). Setting `allow_abbrev=False` blocks that, but the pre-written test calls `--scorer X --help` and expects non-zero exit; argparse processes `--help` first and exits 0 even when other args are unknown. Added an explicit `if "--scorer" in sys.argv: sys.exit(2)` pre-flight in both run_experiment_1.py and run_experiment_2.py so the deprecated flag fails immediately.
- **`scorer_latency_ms_total` instead of `scorer_latency_ms`** in the row schema — the spec drops bare `scorer_latency_ms` (now per-judge) but the row still benefits from a summed total for `total_latency_ms`. Documented in code; reports/dashboards do not use this column directly.
- **Dashboard `agg(quality=("consensus_quality","mean"))` renamed to `agg(consensus_quality=…)`** to keep the output column name consistent with downstream lookups, after the bulk `"quality"` → `"consensus_quality"` rename in `generate_experiment{1,2}_dashboard.py`.
- **Synthetic-data fixtures in `tests/test_experiment{1,2}_dashboard.py` updated** to emit per-judge prefixed columns + `consensus_quality` instead of the legacy bare `quality`. This was implied by spec requirement 4 (drop bare cols) — the dashboards' guard `if "consensus_quality" not in df.columns` made the prior fixtures unusable.

## Flags for Morning Review
- The `--skip-generation` re-score path drops legacy `faithfulness/relevance/conciseness/quality/scorer_latency_ms` columns from any prior single-judge CSV before writing back. If you need to retain those legacy columns for archival, capture them off-disk before running `--skip-generation`.
- `consensus_quality` is computed as a NaN-safe simple mean. If a future panel adds a much weaker judge, weighting may need to change; the per-judge columns are preserved so this can be re-derived without re-scoring.
- Pre-flight `--scorer` rejection prints a plain message to stderr; if you ever want stylized argparse-style output, the message could move into a custom argparse error.

## Attempted Approaches (if skipped/blocked)
N/A — completed.
