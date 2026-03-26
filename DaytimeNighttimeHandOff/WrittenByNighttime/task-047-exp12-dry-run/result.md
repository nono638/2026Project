# Result: task-047 — Dry-run validation of Experiment 1 & 2 scripts
**Status:** done
**Completed:** 2026-03-26T01:03:34

## Commits
- `<see branch>` — night: task-047 dry-run validation for experiment 1 & 2

## Test Results
- Command run: `python -m pytest tests/ --ignore=DaytimeNighttimeHandOff/ -v`
- Outcome: 620 passed, 15 failed
- Failures:
  - 13 BERTScore tests — pre-existing, `bert_score` module not installed
  - 2 dry-run scorer tests (`test_scorer_columns_exist`) — expected, no network access for Gemini API calls
  - No regressions introduced by this task

## Decisions Made
1. **Fixed `cost_limit_hit` UnboundLocalError** in both `run_experiment_1.py` and `run_experiment_2.py`. The variable was only initialized inside the `else` (generation) branch but referenced after the if/else. Added `cost_limit_hit = False` before the branch so `--skip-generation` path works. This was a real bug, not in the spec, but blocking the dry run.
2. **Synthetic data uses HotpotQA fallback**: `create_dry_run_data.py` tries real HotpotQA questions first, falls back to hardcoded samples. Both paths work.
3. **`@pytest.mark.slow` tests use `pytest.skip()` on API failure** rather than hard-failing, so they gracefully degrade in no-network environments.

## Flags for Morning Review
- The `cost_limit_hit` bug fix is important — without it, `--skip-generation` mode crashes on both experiment scripts. This affects any future experiment run using the two-phase (generate then score) workflow.
- BERTScore module (`bert_score`) is not installed in the venv. 13 tests fail because of this. Consider adding it to requirements.txt or marking those tests as conditional.

## Attempted Approaches
None — implementation was straightforward after discovering the bug.
