# task-036: Experiment 1 Script — Strategy × Model Size

## Summary

Write `scripts/run_experiment_1.py` — a standalone script that runs Experiment 1:
5 RAG strategies × 6 generation models = 30 configurations, each scored on 200
HotpotQA examples. Also write `scripts/experiment_utils.py` with shared infrastructure
(checkpoint/resume, model pulling, progress tracking, scoring) that Experiment 2 will
reuse.

This is the project's core research question: does a smart RAG strategy on a small model
beat a naive strategy on a large model?

## Requirements

1. **Experiment matrix**: 5 strategies (NaiveRAG, SelfRAG, MultiQueryRAG, CorrectiveRAG,
   AdaptiveRAG) × 6 models (qwen3:0.6b, qwen3:1.7b, qwen3:4b, qwen3:8b, gemma3:1b,
   gemma3:4b) = 30 configs.
2. **Held constant**: RecursiveChunker(500, 100), OllamaEmbedder (mxbai-embed-large),
   hybrid retrieval, retrieval_top_k=5, no reranker.
3. **Dataset**: 200 HotpotQA examples, seed=42.
4. **Scorer**: Gemini 2.5 Flash (`google:gemini-2.5-flash`) — best cost/quality from Exp 0.
5. **Checkpoint/resume**: After each (strategy, model) config completes, append rows to
   `results/experiment_1/raw_scores.csv`. On restart with `--resume`, detect which configs
   are already in the CSV and skip them. This is critical — 6,000 runs will take hours
   and the script must survive interruptions.
6. **Model pre-pull**: Before running any config for a model, verify it's available in
   Ollama (via `ollama.Client.show()`). If not, pull it (via `ollama.Client.pull()`).
   Log the pull progress.
7. **Gold metrics**: Compute gold_f1 (word-overlap) and gold_exact_match (containment)
   for each answer, same as Exp 0. Compute gold_bertscore in batch at the end.
8. **Pipeline metadata columns**: Same columns as Exp 0 (chunk_type, chunk_size,
   chunk_overlap, num_chunks, embed_provider, embed_model, embed_dimension,
   retrieval_mode, retrieval_top_k, num_chunks_retrieved, context_char_length,
   reranker_model, reranker_top_k, llm_provider, llm_host, dataset_name,
   dataset_sample_seed, llm_model).
9. **Latency columns**: strategy_latency_ms per answer (time.perf_counter around
   strategy.run()). scorer_latency_ms per answer. total_latency_ms = strategy + scorer.
10. **CLI flags**:
    - `--n` (default 200): number of HotpotQA examples
    - `--seed` (default 42): random seed
    - `--output-dir` (default `results/experiment_1`)
    - `--ollama-host`: remote Ollama URL (for RunPod)
    - `--resume`: skip configs already in raw_scores.csv
    - `--max-cost` (default 10.0): API cost ceiling for scorer
    - `--models`: comma-separated subset (e.g., `--models qwen3:4b,gemma3:1b`)
    - `--strategies`: comma-separated subset
    - `--skip-generation`: re-score existing answers without re-generating
    - `--scorer` (default `google:gemini-2.5-flash`): scorer as provider:model
11. **Report**: Generate `results/experiment_1/report.md` with:
    - Strategy × Model quality heatmap (text table)
    - Per-strategy ranking (mean quality, std, count)
    - Per-model ranking
    - "Strategy beats size" analysis: cases where small_model + smart_strategy > large_model + naive
    - Latency summary
    - Cost summary
12. **Progress display**: Print `[config 5/30] SelfRAG × qwen3:4b — query 42/200` style
    progress with estimated time remaining based on elapsed time per config.

## Files to Create

- `scripts/experiment_utils.py` — Shared infrastructure:
  - `ensure_model(client, model_name)` — check/pull Ollama model
  - `load_hotpotqa_examples(n, seed)` → list of dicts with question, gold_answer, doc_text
  - `generate_answer(strategy, chunker, embedder, retriever_mode, query, doc, model, ollama_host)` → dict with answer + metadata
  - `score_answer(scorer, query, context, answer)` → dict with score metrics
  - `compute_f1(prediction, gold)` → float
  - `exact_match(prediction, gold)` → bool
  - `compute_bertscores(predictions, golds)` → list[float]
  - `load_checkpoint(csv_path)` → set of completed (strategy, model) tuples
  - `append_rows(csv_path, rows)` — append to CSV, creating header if file is new
  - `format_duration(seconds)` → human-readable string
  - `build_scorer(scorer_str, max_cost)` → LLMScorer with CostGuard
- `scripts/run_experiment_1.py` — Thin wrapper defining the Exp 1 matrix and report

## Files to Modify

None — these are new files only.

## Files to Read (context, do not modify)

- `scripts/run_experiment_0.py` — Pattern to follow for script structure
- `src/strategies/` — All 5 strategy classes and their constructors
- `src/chunkers/recursive.py` — RecursiveChunker constructor
- `src/embedders/__init__.py` — OllamaEmbedder constructor
- `src/retriever.py` — Retriever constructor and retrieve() signature
- `src/llms.py` — OllamaLLM constructor (host param)
- `src/scorers/llm.py` — LLMScorer constructor and score() signature
- `src/cost_guard.py` — CostGuard constructor
- `src/datasets/hotpotqa.py` — load_hotpotqa, sample_hotpotqa

## New Dependencies

None — all required packages are already installed.

## Edge Cases

- **Ollama model not found**: Pull it. If pull fails, log error and skip that model
  (don't crash the whole experiment).
- **Strategy.run() raises exception**: Log the error, record empty answer with NaN scores
  for that query, continue to next query.
- **Scorer fails on one answer**: Record NaN for score metrics, continue.
- **Resume with partially written row**: `append_rows` should write complete configs
  atomically — flush after each config, not after each row.
- **Empty checkpoint file**: Treat as fresh start (no configs completed).
- **--models or --strategies flag with invalid name**: Print valid options and exit.
- **Cost limit reached mid-config**: Save whatever rows are complete for that config,
  skip remaining configs, generate report from partial data.
- **BERTScore computation fails**: Log warning, skip gold_bertscore column, continue.

## Decisions Made

- **Standalone scripts, not Experiment class**: The Experiment class in src/experiment.py
  doesn't support checkpoint/resume, model pre-pulling, or the column structure we need
  (gold metrics, pipeline metadata). Writing standalone scripts matching Exp 0's pattern
  is simpler and more robust. **Why:** Exp 0 proved this pattern works for long-running
  experiments.
- **Shared experiment_utils.py**: Extract common functions used by both Exp 1 and Exp 2
  into a shared module. **Why:** Both experiments need identical checkpoint, model-pull,
  scoring, and gold-metric logic. Duplicating ~200 lines across two scripts is worse than
  a small shared module.
- **200 HotpotQA examples**: User's choice. 6,000 total runs for Exp 1 gives strong
  per-config statistics (200 data points each).
- **Gemini 2.5 Flash as scorer**: Best cost/quality from Exp 0 results.
- **Checkpoint granularity = per config (strategy, model)**: Not per-query. A config with
  200 queries takes ~5-15 minutes depending on model size. Per-query checkpointing adds
  complexity for marginal benefit. If a config is interrupted, re-run it from scratch (200
  queries, not 6,000).
- **No reranker**: Held constant for fair comparison. Reranking is a separate variable.
- **gold_bertscore computed in batch at end**: BERTScore loads a ~1.4GB model. Doing it
  once at the end is much faster than per-row. If the script crashes before the end,
  BERTScore can be added in a re-scoring pass.

## What NOT to Touch

- `src/experiment.py` — Do not modify the Experiment class or ExperimentResult.
- `scripts/run_experiment.py` — The generic runner is separate from experiment scripts.
- `scripts/run_experiment_0.py` — Read for patterns but do not modify.
- `src/` anything — These scripts use src/ modules as-is.

## Testing Approach

Tests go in `DaytimeNighttimeHandOff/WrittenByDaytime/task-036-experiment-1-script/tests/`
and are copied to `tests/` by the night instance.

- Test `experiment_utils.py` functions with mocked Ollama/scorer calls
- Test checkpoint load/save (write CSV, read back completed configs)
- Test append_rows (new file, existing file, header handling)
- Test compute_f1 and exact_match (pure functions, same as Exp 0)
- Test resume logic (mock CSV with 2 completed configs, verify they're skipped)
- Test model filtering (--models flag subset)
- Test report generation from synthetic DataFrame
- Run with `pytest tests/test_experiment_utils.py tests/test_experiment_1.py -v`
