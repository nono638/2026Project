# Plan: task-036 — Experiment 1 Script + experiment_utils.py

## Approach

1. Create `scripts/experiment_utils.py` with shared infrastructure functions:
   - `compute_f1`, `exact_match`, `compute_bertscores` — extracted from run_experiment_0.py
   - `ensure_model` — check/pull Ollama model
   - `load_hotpotqa_examples` — wrapper for dataset loading
   - `generate_answer` — single-query RAG generation with metadata
   - `score_answer` — single-answer scoring wrapper
   - `load_checkpoint` — parse CSV for completed (strategy, model) tuples
   - `append_rows` — atomic CSV append with header handling
   - `format_duration` — human-readable time formatting
   - `build_scorer` — parse "provider:model" string and construct LLMScorer+CostGuard

2. Create `scripts/run_experiment_1.py` with:
   - ALL_STRATEGIES and ALL_MODELS constants
   - CLI argument parsing (all 10 flags from spec)
   - Main loop: for each (strategy, model), generate 200 answers, score, checkpoint
   - Progress display with ETA
   - Report generation (heatmap, rankings, "strategy beats size" analysis)
   - Error handling: per-query and per-config resilience

## Files to Create
- `scripts/experiment_utils.py`
- `scripts/run_experiment_1.py`

## Files to Read (for patterns)
- `scripts/run_experiment_0.py` — already read
- Source constructors — already explored via subagent

## Ambiguities
- The spec says `generate_answer` returns a dict with "answer + metadata" but doesn't specify exact keys. Will match Exp 0's answer dict structure plus strategy/model/timing columns.
- `load_hotpotqa_examples` returns "list of dicts" per spec but the actual dataset API returns (docs, queries). Will wrap to return the tuple as that's what downstream code needs.
