# task-043: Experiment 0 v2 — redo with diagnostics, reranker, answer quality, and harder questions

## Summary

Experiment 0 (scorer validation) had five oversights: (1) no capture of what the LLM saw,
(2) scorer received the full document instead of the retrieved chunks, (3) no reranker,
(4) ceiling effect from too many easy questions, (5) no composite "answer quality" column.
This task fixes all five by refactoring `run_experiment_0.py` to use the shared
`experiment_utils.generate_answer()` pipeline (gaining diagnostics for free), adding
reranker support, filtering to medium+hard questions, increasing sample size to 150,
and adding an `answer_quality` column based on BERTScore + F1 + Sonnet agreement.

Current v1 results stay untouched in `results/experiment_0/`. New results go to
`results/experiment_0_v2/`. The gallery shows both versions.

## Requirements

1. `experiment_utils.generate_answer()` accepts optional `reranker` and `reranker_top_k`
   parameters. When a reranker is provided, retrieval results are reranked before the
   strategy uses them. This works transparently with all strategies.

2. `run_experiment_0.py` uses `experiment_utils.generate_answer()` for answer generation
   instead of its own standalone `generate_answers()` function. This gives diagnostics,
   failure attribution, timing, and reranker support.

3. The scorer receives `context_sent_to_llm` (from diagnostics) as context, NOT the full
   `doc_text`. This means faithfulness is evaluated against what the LLM actually saw.
   The full `doc_text` is still saved in the CSV for reference.

4. `run_experiment_0.py` adds these CLI flags:
   - `--reranker` (choices: "bge", "minilm", "none"; default: "bge")
   - `--reranker-top-k` (int, default: 3)
   - `--retrieval-top-k` (int, default: 10) — how many chunks to retrieve before reranking
   - `--difficulty` (comma-separated, default: "medium,hard") — which HotpotQA difficulties
   - `--n` default changes from 50 to 150
   - `--output-dir` default changes to "results/experiment_0_v2"

5. Before sampling, docs/queries are filtered to only include the specified difficulties.
   The `difficulty` and `question_type` metadata fields are saved as columns in the CSV.

6. After all scoring is complete, an `answer_quality` column is computed:
   - **"good"**: `gold_bertscore >= 0.90` AND `gold_f1 >= 0.50` AND Sonnet `quality >= 4.0`
   - **"poor"**: `gold_bertscore < 0.85` OR `gold_f1 < 0.30` OR Sonnet `quality < 3.0`
   - **"questionable"**: everything else (metrics disagree)
   - If Sonnet was not run (no `anthropic_claude_sonnet_4_20250514_quality` column), log
     a warning and skip the `answer_quality` column entirely.

7. The v2 CSV includes all columns from v1 PLUS: `difficulty`, `question_type`,
   `strategy_latency_ms`, `failure_stage`, `failure_stage_confidence`,
   `failure_stage_method`, `context_sent_to_llm`, `gold_in_chunks`,
   `gold_in_retrieved`, `gold_in_context`, `reranker_model`, `reranker_top_k`,
   `answer_quality`.

8. `scripts/generate_gallery.py` is updated so the Experiment 0 page shows both v1
   and v2. Add a version toggle or two sections: "v1 (original)" and "v2 (revised)".
   v2 charts include the new `answer_quality` distribution and failure stage breakdown.

9. `results/experiment_0/` is NOT modified. v1 stays as-is.

## Files to Modify

### `scripts/experiment_utils.py`
- Add a `_RerankedRetriever` helper class (private, ~15 lines). It wraps a `Retriever`
  and a reranker. Its `.retrieve()` method calls the underlying retriever, then applies
  `reranker.rerank(query, results, top_k)`. It also exposes `.chunks` from the underlying
  retriever. This way strategies call `.retrieve()` as normal and get reranked results.
- Modify `generate_answer()` to accept two new optional params: `reranker=None` and
  `reranker_top_k=None`. When reranker is provided, wrap the Retriever in
  `_RerankedRetriever` before passing to `strategy.run()`.
- The `retrieval_top_k` already controls how many chunks the Retriever returns. When
  reranking, retrieve `retrieval_top_k` candidates (e.g., 10), rerank down to
  `reranker_top_k` (e.g., 3).

### `scripts/run_experiment_0.py`
- **Remove** the standalone `generate_answers()` function (lines ~206-269).
- **Replace** the generation loop with calls to `experiment_utils.generate_answer()`,
  passing the reranker, retrieval_top_k, etc.
- **Add** new CLI flags: `--reranker`, `--reranker-top-k`, `--retrieval-top-k`,
  `--difficulty`.
- **Change** `--n` default from 50 to 150.
- **Change** `--output-dir` default to "results/experiment_0_v2".
- **Add** difficulty/question_type filtering before `sample_hotpotqa()`.
- **Add** `difficulty` and `question_type` columns to the answers/scores CSV.
- **Change** scorer context from `doc_text` to `context_sent_to_llm`.
- **Add** `answer_quality` column computation after all scoring is done (post-processing
  step before saving the final CSV and report).
- **Add** `_build_reranker(name)` helper (same pattern as `run_experiment.py`).
- **Update** report to include answer_quality distribution and failure_stage breakdown.
- Keep `compute_f1`, `exact_match`, `compute_bertscores`, `_safe_scorer_name` — these
  are still used for scoring. `experiment_utils` also has `compute_f1` and `exact_match`
  but the v2 script can import from there to avoid duplication. Use
  `experiment_utils.compute_f1` and `experiment_utils.exact_match` and delete the local
  copies in `run_experiment_0.py`.
- Keep `compute_bertscores` in `run_experiment_0.py` — it's only used by Exp 0.

### `scripts/generate_gallery.py`
- Update the Experiment 0 page generator to detect both `results/experiment_0/` (v1)
  and `results/experiment_0_v2/` (v2).
- When both exist: show a version toggle or two labeled sections.
- v2 section adds two new charts:
  1. `answer_quality` distribution (bar chart: good/questionable/poor counts)
  2. `failure_stage` breakdown (bar chart: chunker/retrieval/filtering/generation/correct)
- Existing v1 charts (correlation matrix, cost breakdown, etc.) remain unchanged.

## Files to Read (not modify)
- `src/retriever.py` — `.retrieve()` returns `list[dict]` with `text`, `score`, `index`
- `src/rerankers/bge.py` — `.rerank(query, chunks, top_k)` returns `list[dict]` with
  `text`, `score`, `rerank_score`, `index`
- `src/diagnostics.py` — `detect_failure_stage()` interface
- `src/datasets/hotpotqa.py` — `load_hotpotqa()` and `sample_hotpotqa()` interfaces;
  Query.metadata has `difficulty` and `question_type` keys

## New Dependencies

None — all required packages (sentence-transformers for BGE, plotly for gallery) are
already installed.

## Edge Cases

- **Sonnet not run** (ANTHROPIC_API_KEY missing): skip `answer_quality` column, log
  warning "answer_quality requires Sonnet scores — column omitted".
- **Gemini 2.5 Pro fails again**: continue with other judges (existing behavior). NaN
  columns are handled.
- **Reranker returns fewer chunks than reranker_top_k**: fine — strategies handle short
  chunk lists. Diagnostics capture actual count.
- **Empty difficulty filter** (e.g., `--difficulty ""` or no matching questions): print
  error and exit.
- **--skip-generation with v2 output dir**: if `raw_answers.csv` exists in the v2 dir,
  load it. This lets users re-score without re-generating.
- **All questions are correct** (unlikely with medium+hard at 150): `answer_quality`
  still computes — it just shows all "good". The failure_stage breakdown shows "correct"
  for all.

## Decisions Made

- **BGE M3 as default reranker**: 568M params, already implemented, runs on CPU. MiniLM
  (22M) available as fallback. **Why:** substantial reranker that doesn't need GPU.
- **retrieve 10, keep 3 after reranking**: Over-retrieve gives the reranker a real
  candidate pool. 3 final chunks keeps context focused for the LLM. **Why:** standard
  practice — rerankers are most useful when selecting from a larger pool.
- **_RerankedRetriever wrapper in experiment_utils** (not modifying Retriever class):
  avoids double-reranking conflict with the Experiment class which does its own reranking.
  **Why:** smallest change, no risk to existing code paths.
- **Scorer context = context_sent_to_llm**: faithfulness should be judged against what
  the LLM saw, not the full document. **Why:** v1 bug — scorer scored against info the
  LLM never received.
- **answer_quality requires all 3 metrics**: BERTScore (semantic), F1 (lexical), and
  Sonnet (LLM judgment) must all agree an answer is good. Any single metric can be
  misleading alone. **Why:** user requirement — triangulation prevents blind spots.
- **Medium+hard only**: easy questions produce correct answers that all judges rate 5/5,
  providing no signal for scorer validation. **Why:** ceiling effect in v1 — 78% of
  scores were 5.0.
- **150 questions**: 3x the v1 sample. With medium+hard filtering, expect ~40-60 wrong/
  partial answers instead of ~13. **Why:** enough statistical power to compare judges.
- **v1 results preserved**: `results/experiment_0/` untouched. **Why:** scientific
  integrity — v1 is a published result with lessons learned.

## What NOT to Touch

- `results/experiment_0/` — v1 data stays exactly as-is
- `src/retriever.py` — do not add reranker logic to the Retriever class
- `src/experiment.py` — do not change the Experiment class's reranking behavior
- `scripts/run_experiment_1.py`, `scripts/run_experiment_2.py` — no changes
- `src/strategies/*.py` — no changes (strategies work transparently with the wrapper)

## Testing Approach

Pre-written tests in `task-043-exp0-v2/tests/`:

1. `test_reranked_retriever.py` — Tests for the `_RerankedRetriever` wrapper:
   - Wrapper calls underlying retriever then reranker
   - Wrapper passes correct top_k to reranker
   - Wrapper exposes `.chunks` from underlying retriever
   - With reranker=None, generate_answer() works as before (no wrapper)

2. `test_exp0_v2_pipeline.py` — Tests for the refactored run_experiment_0 pipeline:
   - Difficulty filtering: medium+hard only keeps correct subset
   - Difficulty filtering: invalid difficulty exits with error
   - answer_quality computation: known good → "good"
   - answer_quality computation: known bad → "poor"
   - answer_quality computation: disagreement → "questionable"
   - answer_quality skipped when Sonnet column missing
   - Scorer receives context_sent_to_llm, not doc_text
   - CSV includes difficulty and question_type columns

Run with: `pytest DaytimeNighttimeHandOff/WrittenByDaytime/task-043-exp0-v2/tests/ -v`
