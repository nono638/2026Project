# Plan: task-048 — Experiment 0 resilient resume

## Files to modify
- `scripts/run_experiment_0.py` — all three changes are in this file

## Approach

### Change 1: Generation checkpointing (lines ~693-741)
- Replace in-memory `answers` list accumulation with incremental CSV append
- Before the generation loop, check if `raw_answers.csv` exists and load existing `example_id` values
- For each generated answer, append it immediately to `raw_answers.csv`
- Use `on_bad_lines='skip'` when loading to handle corrupt rows from crashes
- Log a warning if existing answers are found (resume case)

### Change 2: Cost guard abort in `score_all_answers()` (line ~392)
- Add an explicit `except CostLimitExceeded` BEFORE the generic `except (ScorerError, Exception)`
- When caught, set a `cost_limit_hit` flag, break the inner scorer loop
- After the inner loop, check the flag and break the outer answer loop too
- Ensure partial checkpoint is saved (it already writes per-row, so partial data survives)

### Change 3: Per-judge scoring resume (in `score_all_answers()`)
- At the start of `score_all_answers()`, load existing `raw_scores.csv` + checkpoint
- Merge them (checkpoint takes priority for non-NaN values)
- For each answer row, for each scorer, check if that judge's `_quality` column already has a non-NaN value
- If it does, skip that scorer for that row
- If all judges are already scored for all rows, log and return early

### Change 4: Checkpoint retention (lines ~816-820)
- After scoring, check if all requested judge columns have non-NaN values for all rows
- Only delete checkpoint if everything is complete

## Ambiguities
- The spec says to load existing `raw_scores.csv` at the start of `score_all_answers()`. The current code loads checkpoint but not raw_scores.csv. Will merge both.
- The current `score_all_answers()` skips entire rows if they're in checkpoint. Need to change to per-judge granularity instead.
