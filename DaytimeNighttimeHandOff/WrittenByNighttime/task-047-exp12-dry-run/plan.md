# Plan: task-047 — Dry-Run Validation of Experiment 1 & 2 Scripts

## Files to Create
- `scripts/create_dry_run_data.py` — generates synthetic raw_scores.csv for both experiments
- `tests/test_exp12_dry_run.py` — validation tests

## Files to Potentially Fix
- `scripts/run_experiment_1.py` — if bugs found
- `scripts/run_experiment_2.py` — if bugs found

## Approach

### 1. Create synthetic data script
- Load 2 HotpotQA questions (seed=42) for realistic questions/answers
- For Exp 1: 2 questions x 2 strategies x 1 model = 4 rows, columns: question, gold_answer, rag_answer, context_sent_to_llm, strategy, model, config_label + various metadata
- For Exp 2: 2 questions x 2 chunkers x 1 model = 4 rows, similar but with chunker column

### 2. Run dry-run scoring
- --skip-generation loads from raw_scores.csv (not raw_answers.csv!)
- Use --scorer google:gemini-2.5-flash-lite (cheapest)
- This requires network access — may be blocked in nighttime mode

### 3. Write tests
- Generate inline synthetic data
- Run via subprocess with --skip-generation
- Assert output file existence + column presence
- Mark @pytest.mark.slow

### Ambiguity
- Network access may be blocked by nighttime hooks for Python subprocess too.
  If so, will create scripts and tests but not execute the actual dry run.
