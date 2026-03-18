# Plan: task-022 — CLI Flags for run_experiment.py

## File to modify
- `scripts/run_experiment.py` — only file that changes

## New CLI arguments
1. `--chunkers` — comma-separated, default varies by mode
2. `--embedder` — singular, default "ollama"
3. `--dataset` — hotpotqa/squad, overrides --corpus
4. `--retrieval-mode` — hybrid/dense/sparse
5. `--llm-backend` — ollama/openai-compat
6. `--llm-base-url` — for openai-compat

## Approach
1. Add arguments to parse_args()
2. Add validation logic
3. Modify build_components() to accept new args
4. Modify main() to support --dataset loading
5. Preserve default behavior when no new flags provided

## Ambiguities
- Spec shows `sample_hotpotqa(n=n, seed=42)` but the actual function signature is
  `sample_hotpotqa(docs, queries, n, seed)` — need to call load first, then sample.
- `Experiment` constructor uses `corpus=` parameter for dicts, need to check how
  built-in datasets (which return Document objects) integrate.
