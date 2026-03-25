# task-047: Experiment 1 & 2 Dry-Run Validation

## What

Verify that `run_experiment_1.py` and `run_experiment_2.py` work end-to-end by running
the scoring phase on synthetic data. This catches integration bugs (wrong columns, missing
imports, broken scorer calls, report generation failures) before we burn GPU time on real
runs.

## Why

Experiments 1 & 2 are the project's main research output — 30 configs (Exp 1) and 16
configs (Exp 2) on RunPod GPU. A bug discovered mid-run wastes hours of GPU time ($0.18/hr)
and API spend. Validating now with synthetic data costs nothing and takes minutes.

We can't test generation (needs Ollama + GPU), but we CAN test the entire scoring +
BERTScore + report pipeline by creating synthetic `raw_answers.csv` files and running
with `--skip-generation`.

## Implementation

### Step 1: Create synthetic answer data

Write a Python script `scripts/create_dry_run_data.py` that generates minimal synthetic
`raw_answers.csv` files for both experiments.

**For Experiment 1** (`results/experiment_1_dry_run/raw_answers.csv`):
- 2 questions × 2 strategies × 1 model = 4 rows
- Use real HotpotQA questions (load 2 from the dataset with seed=42)
- Use plausible fake answers (e.g., copy the gold answer with minor edits, or use
  "The answer is unknown." for some)
- Columns must match what `run_experiment_1.py` expects when loading with `--skip-generation`
  (check the `answers_df.to_dict("records")` conversion in main())
- Include the `config_label` column (e.g., "naive__qwen3_4b", "self_rag__qwen3_4b")
- Include `strategy` and `model` columns

**For Experiment 2** (`results/experiment_2_dry_run/raw_answers.csv`):
- 2 questions × 2 chunkers × 1 model = 4 rows
- Same approach as above
- Include `chunker` and `model` columns and `config_label`

To determine the exact columns needed, read the `--skip-generation` branch of each
experiment's `main()` function. The CSV must have every column that the scoring loop
reads from `ans` dict entries.

### Step 2: Run dry-run scoring

Run each experiment with `--skip-generation` on the synthetic data using the cheapest
scorer (gemini-2.5-flash-lite):

```bash
python scripts/run_experiment_1.py \
  --skip-generation \
  --output-dir results/experiment_1_dry_run \
  --scorer google:gemini-2.5-flash-lite \
  --no-gallery

python scripts/run_experiment_2.py \
  --skip-generation \
  --output-dir results/experiment_2_dry_run \
  --scorer google:gemini-2.5-flash-lite \
  --no-gallery
```

### Step 3: Validate outputs

After each run completes, verify:
1. `raw_scores.csv` exists and has the expected number of rows
2. Scorer columns exist (e.g., `google_gemini_2_5_flash_lite_quality`)
3. `gold_bertscore` column exists and has float values
4. `gold_f1` and `gold_exact_match` columns exist
5. `report.md` exists and contains expected sections
6. No Python exceptions or tracebacks in stdout/stderr

### Step 4: Write validation as a test

Create `tests/test_exp12_dry_run.py` that:
1. Generates synthetic data (inline, not from the script)
2. Runs both experiments with `--skip-generation` using subprocess
3. Asserts on the output file existence and column presence
4. Cleans up the dry-run output dirs after

Mark these tests with `@pytest.mark.slow` since they make real API calls to
gemini-2.5-flash-lite (very cheap — $0.0001/call, ~$0.0004 total for 4 rows).

### Step 5: Fix any bugs found

If either script fails during the dry run:
- Fix the bug in the experiment script
- Re-run to confirm the fix
- Document what was wrong in the commit message

Common bugs to watch for:
- Column name mismatches between generation output and scoring input
- Missing imports that only trigger during scoring (not generation)
- BERTScore crash on edge-case inputs (empty strings — should be handled by existing fix)
- Report generation assuming columns that don't exist in the data
- `config_label` format mismatches between generation and checkpoint/resume logic

## Files to Create

- `scripts/create_dry_run_data.py` — synthetic data generator
- `tests/test_exp12_dry_run.py` — validation tests

## Files to Potentially Fix

- `scripts/run_experiment_1.py` — only if bugs are found
- `scripts/run_experiment_2.py` — only if bugs are found

## Files NOT to Touch

- `scripts/run_experiment_0.py` — not part of this task
- `results/experiment_0*/` — do not modify real experiment data
- `src/` — no source changes unless a bug requires it

## Output

After this task completes:
- `results/experiment_1_dry_run/` — synthetic dry-run results (can be gitignored or deleted)
- `results/experiment_2_dry_run/` — synthetic dry-run results
- Any bug fixes committed to the experiment scripts
- Confidence that Experiments 1 & 2 will work when we run them for real on GPU

## What NOT to Do

- Don't run actual generation (no Ollama available)
- Don't use expensive scorers — flash-lite only
- Don't modify experiment designs (strategy lists, model lists, chunker lists)
- Don't commit the dry-run result directories to git (add to .gitignore)
