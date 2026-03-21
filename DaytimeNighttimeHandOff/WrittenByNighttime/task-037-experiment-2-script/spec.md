# task-037: Experiment 2 Script — Chunking × Model Size

## Summary

Write `scripts/run_experiment_2.py` — a standalone script that runs Experiment 2:
4 chunking strategies × 4 Qwen3 generation models = 16 configurations, each scored
on 200 HotpotQA examples. Reuses `scripts/experiment_utils.py` from task-036.

This experiment isolates the effect of chunking strategy on RAG quality across model
sizes, holding the RAG strategy constant (NaiveRAG).

## Requirements

1. **Experiment matrix**: 4 chunkers (RecursiveChunker, FixedSizeChunker,
   SentenceChunker, SemanticChunker) × 4 models (qwen3:0.6b, qwen3:1.7b, qwen3:4b,
   qwen3:8b) = 16 configs.
2. **Held constant**: NaiveRAG strategy, OllamaEmbedder (mxbai-embed-large), hybrid
   retrieval, retrieval_top_k=5, no reranker.
3. **Dataset**: 200 HotpotQA examples, seed=42 (same sample as Exp 1 for comparability).
4. **Scorer**: Gemini 2.5 Flash (`google:gemini-2.5-flash`).
5. **Checkpoint/resume**: Same as Exp 1 — checkpoint per (chunker, model) config pair.
   On `--resume`, skip completed configs.
6. **Model pre-pull**: Same as Exp 1.
7. **Gold metrics**: gold_f1, gold_exact_match, gold_bertscore (batch at end).
8. **Pipeline metadata columns**: Same as Exp 1, but chunk_type varies per config.
   For SemanticChunker, chunk_size is not fixed — record it as the chunker's name
   property value (e.g., "semantic:mxbai-embed-large").
9. **Latency columns**: strategy_latency_ms, scorer_latency_ms, total_latency_ms.
10. **CLI flags**:
    - `--n` (default 200): number of HotpotQA examples
    - `--seed` (default 42): random seed
    - `--output-dir` (default `results/experiment_2`)
    - `--ollama-host`: remote Ollama URL
    - `--resume`: skip completed configs
    - `--max-cost` (default 10.0): API cost ceiling
    - `--models`: comma-separated subset of Qwen3 models
    - `--chunkers`: comma-separated subset
    - `--skip-generation`: re-score existing answers
    - `--scorer` (default `google:gemini-2.5-flash`): scorer as provider:model
11. **Report**: Generate `results/experiment_2/report.md` with:
    - Chunker × Model quality heatmap (text table)
    - Per-chunker ranking (mean quality, std, count)
    - Per-model ranking
    - Chunking impact analysis: which chunker gives the biggest lift on small vs large models
    - Latency summary (chunking strategies have different retrieval characteristics)
    - Cost summary
12. **Progress display**: `[config 3/16] SentenceChunker × qwen3:1.7b — query 100/200`

## Files to Create

- `scripts/run_experiment_2.py` — Thin wrapper defining the Exp 2 matrix and report.
  Imports shared functions from `scripts/experiment_utils.py` (created by task-036).

## Files to Modify

None.

## Files to Read (context, do not modify)

- `scripts/experiment_utils.py` — Shared infrastructure (created by task-036)
- `scripts/run_experiment_1.py` — Pattern to follow (created by task-036)
- `src/chunkers/recursive.py` — RecursiveChunker(chunk_size, chunk_overlap)
- `src/chunkers/fixed.py` — FixedSizeChunker(chunk_size)
- `src/chunkers/sentence.py` — SentenceChunker()
- `src/chunkers/semantic.py` — SemanticChunker() — uses embedder internally
- `src/strategies/naive.py` — NaiveRAG(llm=OllamaLLM())

## New Dependencies

None.

## Edge Cases

- **SemanticChunker produces 0 chunks for a short document**: Record empty answer with
  NaN scores, log warning, continue.
- **Same edge cases as task-036**: model pull failure, strategy exception, scorer failure,
  partial resume, cost limit.
- **Chunk metadata varies by chunker type**: RecursiveChunker has chunk_size=500,
  chunk_overlap=100. FixedSizeChunker has chunk_size=500, chunk_overlap=0.
  SentenceChunker and SemanticChunker don't have fixed sizes — record the chunker's
  `.name` property as chunk_type and leave chunk_size/chunk_overlap as the chunker's
  actual defaults.

## Decisions Made

- **Qwen3 only (4 models, not 6)**: Exp 2 isolates chunking × model size within one
  model family. Gemma is the cross-family variable in Exp 1. **Why:** Mixing families
  would confound the chunking effect with architecture differences.
- **NaiveRAG held constant**: Simplest strategy = most direct measurement of chunking
  effect. **Why:** If we used all 5 strategies, this becomes a 4×4×5=80 config experiment
  that's really just Exp 1 with extra chunkers.
- **Same 200 HotpotQA sample as Exp 1**: seed=42, n=200. **Why:** Allows direct
  comparison of results across experiments on the same questions.
- **Depends on task-036**: This task creates `run_experiment_2.py` which imports from
  `experiment_utils.py`. Task-036 creates that module. Merge task-036 first.

## What NOT to Touch

- `src/` — Use modules as-is.
- `scripts/run_experiment_0.py` — Don't modify.
- `scripts/run_experiment_1.py` — Don't modify (created by task-036).
- `scripts/experiment_utils.py` — Don't modify (created by task-036). If you need a
  function that doesn't exist, add it to experiment_utils.py as a new function (do not
  change existing function signatures).

## Testing Approach

Tests in `DaytimeNighttimeHandOff/WrittenByDaytime/task-037-experiment-2-script/tests/`.

- Test chunker instantiation (all 4 chunkers create successfully)
- Test config matrix generation (4 chunkers × 4 models = 16 configs)
- Test --chunkers and --models filtering
- Test report generation from synthetic DataFrame
- Test that SemanticChunker metadata is recorded correctly
- Run with `pytest tests/test_experiment_2.py -v`
