# Plan: task-037 — Experiment 2 Script

## Approach
Create `scripts/run_experiment_2.py` following the same pattern as run_experiment_1.py
but with the Exp 2 matrix: 4 chunkers x 4 Qwen3 models = 16 configs, NaiveRAG held
constant. Reuses all shared functions from experiment_utils.py.

## Files to Create
- `scripts/run_experiment_2.py`

## Key Differences from Exp 1
- Matrix dimension is chunker (not strategy)
- Only Qwen3 models (4 not 6)
- Only NaiveRAG strategy (held constant)
- Chunk metadata varies per chunker type
- Report focuses on chunking impact analysis
