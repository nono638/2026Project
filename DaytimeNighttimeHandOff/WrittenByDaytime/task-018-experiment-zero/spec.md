# task-018: Experiment 0 — Scorer Validation Script

## Summary
Write a script (`scripts/run_experiment_0.py`) that validates the LLMScorer by comparing
5 LLM judges on the same 50 HotpotQA answers. The script generates RAG answers using
NaiveRAG + Qwen3 4B, scores each answer with all 5 judges, then produces a comparison
report showing inter-scorer agreement and correlation with gold-answer correctness.
This is the methodological safety net — it tells us whether cheap scorers (Gemini Flash)
give meaningfully different results than expensive ones (Claude Opus).

## Requirements
1. `scripts/run_experiment_0.py` is a standalone CLI script that runs the full experiment.
2. Loads 50 HotpotQA examples via `sample_hotpotqa(n=50, seed=42)`.
3. Runs each example through NaiveRAG + Qwen3 4B to produce 50 RAG answers.
4. Scores each answer with 5 LLM judges:
   - `LLMScorer(provider="google", model="gemini-2.5-flash")`
   - `LLMScorer(provider="google", model="gemini-2.5-pro")`
   - `LLMScorer(provider="anthropic", model="claude-haiku-4-5-20251001")`
   - `LLMScorer(provider="anthropic", model="claude-sonnet-4-20250514")`
   - `LLMScorer(provider="anthropic", model="claude-opus-4-6")`
5. Computes gold-answer correctness for each answer using simple string matching:
   - Exact match (case-insensitive): does the answer contain the gold answer?
   - F1 token overlap: word-level F1 between answer and gold answer.
6. Saves raw results to `results/experiment_0/raw_scores.csv` with columns:
   `example_id, question, gold_answer, rag_answer, gold_exact_match, gold_f1,
   {judge}_faithfulness, {judge}_relevance, {judge}_conciseness, {judge}_quality`
   where `{judge}` is each of the 5 scorer names.
7. Produces a summary report (`results/experiment_0/report.md`) containing:
   - Per-judge mean scores (faithfulness, relevance, conciseness, quality)
   - Inter-scorer correlation matrix (Pearson on quality scores)
   - Each judge's correlation with gold F1 (which judge best predicts correctness?)
   - Cost breakdown (actual API calls made × estimated per-call cost)
   - Recommendation: which scorer to use for Experiments 1 & 2
8. Prints the summary report to stdout when the script finishes.
9. Handles errors gracefully: if one scorer fails on one example (API error),
   log the error, record NaN for that scorer's scores, continue with the next example.

## Files to Create
- `scripts/run_experiment_0.py` — the main script
- `results/experiment_0/` — output directory (created by the script if missing)

## Files to Read for Context
- `src/datasets/hotpotqa.py` — `load_hotpotqa()`, `sample_hotpotqa()`
- `src/scorers/llm.py` — `LLMScorer` class (created by task-017)
- `src/strategies/naive.py` — `NaiveRAG` class
- `src/experiment.py` — for understanding how strategies and retrievers interact
- `src/retriever.py` — `Retriever` class
- `src/chunkers/recursive.py` — `RecursiveChunker` (default chunker)
- `src/embedders/` — `OllamaEmbedder` for mxbai-embed-large
- `src/document.py` — `documents_to_dicts()`

## New Dependencies
None — all required packages are installed.

## Script Structure

```python
"""Experiment 0: Scorer Validation — compare 5 LLM judges."""

import argparse
import logging
import os
import sys
from pathlib import Path

# 1. Parse args (--n=50, --seed=42, --model=qwen3:4b, --output-dir=results/experiment_0/)
# 2. Load HotpotQA sample
# 3. Set up pipeline: RecursiveChunker(500,100) + OllamaEmbedder(mxbai-embed-large)
# 4. For each example:
#    a. Build retriever (chunk + embed the document)
#    b. Run NaiveRAG to get answer
#    c. Compute gold correctness (exact match + F1)
#    d. Score with all 5 judges
# 5. Save raw CSV
# 6. Compute and save summary report

def compute_f1(prediction: str, gold: str) -> float:
    """Word-level F1 between prediction and gold answer."""
    pred_tokens = set(prediction.lower().split())
    gold_tokens = set(gold.lower().split())
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = pred_tokens & gold_tokens
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def exact_match(prediction: str, gold: str) -> bool:
    """Case-insensitive check: does prediction contain the gold answer?"""
    return gold.lower() in prediction.lower()
```

## CLI Arguments
- `--n` (int, default 50): number of HotpotQA examples
- `--seed` (int, default 42): random seed for sampling
- `--model` (str, default "qwen3:4b"): Ollama model for answer generation
- `--output-dir` (str, default "results/experiment_0"): output directory
- `--skip-generation`: if set, load previously generated answers from
  `{output-dir}/raw_answers.csv` instead of re-running NaiveRAG. Useful for
  re-scoring without re-generating (saves Ollama time).

## Edge Cases
- **Ollama not running**: detect early, print clear error message, exit 1.
- **API key missing for a scorer**: skip that scorer, log warning, continue with others.
- **API call fails mid-run**: log error, record NaN, continue.
- **Empty RAG answer**: still score it (LLMScorer handles empty answers with defaults).
- **Gold answer is very short** (e.g., "yes"): F1 will be noisy — acknowledged, not fixable.

## Decisions Made
- **NaiveRAG only for generation**: we want to test scorers, not strategies. Naive is the
  simplest — fewer confounds. **Why:** isolate what we're measuring.
- **Qwen3 4B for generation**: middle of the range. Small enough to run on CPU in ~20 min,
  large enough to produce non-trivial answers. **Why:** balance of speed and quality.
- **RecursiveChunker(500,100)**: the default for Experiment 1. Consistent. **Why:** held
  constant across all experiments.
- **String containment for exact match (not equality)**: RAG answers are usually longer
  than the gold answer. "The capital is Paris" should match gold "Paris". **Why:** strict
  equality would penalize verbose-but-correct answers.
- **--skip-generation flag**: generating 50 answers through Ollama takes 10-30 min on CPU.
  Scoring is just API calls (~2 min). Being able to re-score without re-generating saves
  time when debugging scorers. **Why:** practical for iteration.
- **NaN for failed scorer calls (not retry)**: LLMScorer already retries internally if the
  provider has retry logic. If it still fails, it's a persistent issue — NaN is better than
  blocking the entire experiment. **Why:** resilience.

## What NOT to Touch
- `src/experiment.py` — don't use the Experiment class for this. Experiment 0 is simpler
  than a full matrix run. Write a focused script that directly calls the components.
- `src/scorers/llm.py` — don't modify the scorer. Just use it.
- Any existing test files.

## Testing Approach
- Tests in `DaytimeNighttimeHandOff/WrittenByDaytime/task-018-experiment-zero/tests/test_experiment_zero.py`
- Mock Ollama calls (NaiveRAG), mock all 5 LLMScorer instances
- Test: `compute_f1()` and `exact_match()` with known inputs
- Test: CSV output has correct columns
- Test: report generation produces expected sections
- Test: graceful handling of scorer failure (NaN in output)
- Run with: `pytest DaytimeNighttimeHandOff/WrittenByDaytime/task-018-experiment-zero/tests/`
