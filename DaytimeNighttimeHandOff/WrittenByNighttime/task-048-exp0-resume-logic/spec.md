# task-048: Experiment 0 resilient resume — generation checkpointing, per-judge scoring resume, cost guard abort

## Summary

Experiment 0's `run_experiment_0.py` has three resilience gaps that caused hours of wasted
API spend during the v3 run (2026-03-26). Generation saves all answers at the end (crash
loses everything), the cost guard doesn't break the scoring loop (continues making API calls
and discarding results), and re-runs re-score all questions from scratch instead of filling
in only the missing judge scores. Experiments 1 & 2 already handle cost guard abort correctly
— this task brings Experiment 0 up to the same standard and adds the per-judge resume logic
that none of the scripts have yet.

## Requirements

1. **Generation checkpointing**: Each generated answer must be appended to `raw_answers.csv`
   immediately after generation (not batched at the end). If the process is restarted
   without `--skip-generation`, it must load existing `raw_answers.csv`, identify which
   `example_id` values are already present, and resume from the next one.

2. **Cost guard abort**: When `CostLimitExceeded` is raised during scoring, the scoring loop
   must break immediately (both the inner per-scorer loop and the outer per-answer loop).
   No further API calls should be made. Partial results must be saved to the checkpoint
   before exiting.

3. **Per-judge scoring resume**: When scoring, if `raw_scores.csv` or the checkpoint file
   already exists with scores for some judges, only call scorers whose columns are missing
   or have NaN for that row. This handles: (a) adding new judges to existing data,
   (b) filling in judges that hit the cost limit, (c) retrying failed scorers. The
   `--skip-generation` re-run should not re-score rows that already have complete scores
   for all requested judges.

4. **Checkpoint not deleted on partial completion**: Only delete `raw_scores_checkpoint.csv`
   when ALL requested judge columns have non-NaN values for ALL rows. If scoring was
   interrupted (cost guard, crash, Ctrl+C), keep the checkpoint for next resume.

5. **Backward compatible**: All existing CLI flags and output formats must continue to work.
   A clean run (no prior data) must behave identically to before. The `--judges` filter
   must still work (only score matching judges, skip others).

## Files to Modify

- `scripts/run_experiment_0.py`:
  - `main()` function, generation section (lines ~693-741): Replace the in-memory `answers`
    list accumulation with incremental CSV append. Add logic to load existing answers and
    skip already-generated example_ids on restart.
  - `score_all_answers()` function (lines ~250-423): Three changes:
    1. Catch `CostLimitExceeded` separately from `(ScorerError, Exception)` in the
       per-scorer try/except (line ~392). When caught, break both loops.
    2. Before scoring each judge for a row, check if that judge's quality column already
       has a non-NaN value (from checkpoint or prior raw_scores.csv). If so, skip that judge.
    3. Load existing `raw_scores.csv` (not just checkpoint) at the start to seed the
       scored data — merge checkpoint + existing CSV for the fullest picture.
  - Post-scoring section (lines ~816-820): Only delete checkpoint when all rows × all
    requested judges have non-NaN values.

- `src/cost_guard.py`: **No changes** — `CostLimitExceeded` already works correctly. The
  problem is in how `score_all_answers()` catches it.

## New Dependencies

None — all required packages are already installed.

## Edge Cases

- **Fresh run (no prior data)**: Behaves identically to current code. raw_answers.csv
  is created incrementally instead of at the end, but the result is the same.
- **Generation crash at question 300/500**: On restart (same output-dir, same flags),
  loads raw_answers.csv with 300 rows, resumes at example_id 300.
- **Generation restart with different --seed or --n**: The script should log a warning
  if existing raw_answers.csv has rows, noting it will skip already-generated IDs. If
  the user wants a fresh start, they use a different `--output-dir` or delete the CSV.
- **Cost guard hits mid-question (judge 3 of 6)**: Break immediately. The row has
  judges 1-3 scored, judges 4-6 missing. Checkpoint is saved. On re-run, judges 4-6
  are filled in for that row.
- **Re-run with `--judges` filter for subset**: Only those judges are scored. Existing
  columns for other judges are preserved. Missing columns for the requested judges are
  filled in.
- **Re-run adds a judge that wasn't in the original run**: New judge column is added to
  all rows. Existing columns are untouched.
- **All judges already scored for all rows**: Script detects nothing to do, logs
  "All rows already scored for requested judges", writes final CSV, exits cleanly.
- **Checkpoint exists but raw_scores.csv also exists**: Merge both, preferring non-NaN
  values. Checkpoint takes priority over raw_scores.csv for any conflicts (checkpoint
  is more recent).
- **raw_answers.csv has partial final row (corrupt from crash)**: Use
  `pd.read_csv(..., on_bad_lines='skip')` to silently drop malformed rows.

## Decisions Made

- **Append-per-row for generation, not batch-every-N**: Simplest and safest. The CSV write
  overhead (~1ms) is negligible compared to the ~10s generation time per question. **Why:**
  Any batching window is a loss window.
- **Check judge columns by NaN, not by presence**: A column can exist but have NaN (from
  a failed scorer call or cost limit). NaN means "needs scoring." A non-NaN value means
  "already scored, skip." **Why:** Handles both partial-judge and failed-judge cases
  uniformly.
- **Don't add `--resume` flag**: Unlike Exp 1 & 2 (which resume at the config level),
  Exp 0 should always resume automatically. There's no downside to checking for existing
  data. **Why:** The user forgot `--resume` during the v3 re-run and wasted 273 re-scores.
  Automatic resume is safer.
- **Keep the checkpoint file pattern**: The checkpoint (append-per-row) protects against
  mid-run crashes. The final raw_scores.csv (written once at end) is the clean output.
  **Why:** Appending to raw_scores.csv directly risks corruption; the checkpoint is
  the crash-safe buffer.
- **Import `CostLimitExceeded` in `score_all_answers`**: Catch it explicitly before the
  generic `Exception` catch. **Why:** Python catches exceptions in order; putting
  `CostLimitExceeded` first ensures it's not swallowed by the generic handler.

## What NOT to Touch

- `scripts/run_experiment_1.py` and `scripts/run_experiment_2.py` — they already handle
  cost guard abort correctly via their own `CostLimitExceeded` catch. Their config-level
  `--resume` works fine for their use case.
- `scripts/experiment_utils.py` — shared infra is fine as-is.
- `src/cost_guard.py` — the exception mechanism works correctly; only the caller is broken.
- `src/scorers/llm.py` — scorer behavior is correct.
- Report generation, gallery integration, BERTScore computation — leave unchanged.

## Testing Approach

Tests should cover:
1. **Generation resume**: Mock the generation loop. Write 3 of 5 answers to CSV. Call
   the generation logic again — verify it loads the 3 and only generates answers 4-5.
2. **Cost guard abort**: Create a mock scorer that raises `CostLimitExceeded` on the
   3rd call. Verify scoring stops immediately (not after all 500 rows). Verify partial
   checkpoint is saved.
3. **Per-judge resume**: Create a raw_scores.csv with judge A scored and judge B as NaN.
   Run scoring with both judges. Verify judge A is not re-called, judge B is called.
4. **Full-scored skip**: Create raw_scores.csv with all judges scored. Run scoring.
   Verify zero API calls made.
5. **Checkpoint merge**: Create a checkpoint with rows 1-3 and a raw_scores.csv with
   rows 1-5 (rows 4-5 having NaN for some judges). Verify merge produces correct data.
6. **Corrupt CSV handling**: Write a raw_answers.csv with a truncated last line. Verify
   it loads without error (skips the bad line).

Run with: `pytest tests/test_exp0_resume.py -v`
