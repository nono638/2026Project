# Plan: task-018 — Experiment 0 Scorer Validation Script

## Approach

Create `scripts/run_experiment_0.py` as a standalone CLI script. It depends on:
- `src/datasets/hotpotqa.py` (task-013, merged)
- `src/scorers/llm.py` (task-017 branch — LLMScorer)
- `src/strategies/naive.py` with `llm` param (task-020 branch)
- `src/llms/ollama.py` (task-020 branch)
- `src/chunkers/recursive.py` + `src/embedders/` (on main)

Since task-017 and task-020 are on unmerged branches, this script will import
from those module paths. It will work after those branches are merged.

## Files to Create
- `scripts/run_experiment_0.py` — main experiment 0 script

## Key Functions
- `compute_f1(prediction, gold)` — word-level F1
- `exact_match(prediction, gold)` — case-insensitive containment
- `main()` — orchestrates the full experiment

## Dependencies Required at Runtime
- task-017 branch (LLMScorer in src/scorers/llm.py)
- task-020 branch (OllamaLLM in src/llms/, NaiveRAG(llm=...) constructor)

## Notes
- Since this branch needs code from task-017 and task-020, I'll merge those
  branches into the task-018 branch before implementing.
