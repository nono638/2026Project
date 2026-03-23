# task-042: Pipeline Diagnostics and Failure Attribution

## Summary

Add pipeline diagnostics to every strategy `run()` so experiment scripts can log what
context the LLM actually received, what chunks were retrieved vs. filtered, and
automatically attribute RAG failures to the responsible pipeline stage (chunker, retrieval,
filtering, or generation). This closes a critical observability gap discovered during
Experiment 0 analysis — we couldn't determine whether a wrong answer was caused by bad
retrieval, aggressive filtering, or model generation failure.

## Requirements

1. Each strategy's `run()` method accepts an optional `diagnostics: dict | None = None`
   parameter. When provided, the strategy populates it with intermediate pipeline data.
   When `None` (the default), behavior is unchanged — no overhead, full backward compat.

2. The diagnostics dict, when populated, contains these keys:
   - `retrieved_chunks`: list of chunk dicts from the initial `retriever.retrieve()` call
     (before any strategy-level filtering)
   - `filtered_chunks`: list of chunk texts that survived strategy-level filtering/merging
     (identical to `retrieved_chunks` texts for NaiveRAG; subset for Corrective/SelfRAG;
     merged set for MultiQuery)
   - `context_sent_to_llm`: the exact context string passed in the final generation prompt
   - `retrieval_queries`: list of all query strings used for retrieval (single-element for
     most strategies; 4 elements for MultiQueryRAG)
   - `skipped_retrieval`: bool — True when strategy skipped retrieval entirely
     (AdaptiveRAG simple path, SelfRAG "no retrieval" decision)

3. A new module `src/diagnostics.py` provides `detect_failure_stage()`:
   ```python
   def detect_failure_stage(
       gold_answer: str,
       rag_answer: str,
       all_chunks: list[str],
       retrieved_chunk_texts: list[str],
       context_sent_to_llm: str,
       skipped_retrieval: bool = False,
   ) -> str:
   ```
   Returns one of: `"none"`, `"chunker"`, `"retrieval"`, `"filtering"`, `"generation"`,
   `"no_retrieval"`, `"unknown"`.

   Logic (in order):
   - If `gold_answer` is empty/None → `"unknown"`
   - If `gold_answer` found in `rag_answer` (case-insensitive containment) → `"none"`
   - If `skipped_retrieval` is True → `"no_retrieval"` (strategy chose not to retrieve)
   - If `gold_answer` not found in ANY chunk text → `"chunker"`
   - If `gold_answer` in chunks but not in any retrieved chunk → `"retrieval"`
   - If `gold_answer` in retrieved chunks but not in `context_sent_to_llm` → `"filtering"`
   - If `gold_answer` in context but not in answer → `"generation"`

   "Found in" means case-insensitive substring match (`gold.lower() in text.lower()`).
   This is deliberately simple and heuristic — matches the existing `exact_match()` logic
   in experiment_utils.py.

4. `experiment_utils.generate_answer()` is updated to:
   - Create a `diagnostics = {}` dict
   - Pass it to `strategy.run(query.text, retriever, model, diagnostics=diagnostics)`
   - Use diagnostics to populate new return dict keys:
     - `context_sent_to_llm` (str)
     - `failure_stage` (str)
     - `gold_in_chunks` (bool)
     - `gold_in_retrieved` (bool)
     - `gold_in_context` (bool)
   - Update `context_char_length` to use the ACTUAL context from diagnostics (not the
     pre-strategy retrieval estimate that's currently there)
   - Remove the redundant `retriever.retrieve()` call on line 203 — the diagnostics
     from inside the strategy now provide this data

5. All 5 strategy implementations populate diagnostics when the parameter is not None.
   When diagnostics is None, no extra work is done — zero performance impact on existing
   callers.

## Files to Modify

- `src/protocols.py` — Add `diagnostics: dict | None = None` parameter to `Strategy.run()`
  method signature (line 64). Update docstring.

- `src/strategies/naive.py` — In `run()`: accept `diagnostics` kwarg. When not None,
  populate `retrieved_chunks`, `filtered_chunks` (same as retrieved), `context_sent_to_llm`,
  `retrieval_queries` ([query]), `skipped_retrieval` (False).

- `src/strategies/corrective.py` — In `run()`: accept `diagnostics` kwarg. Log initial
  retrieved chunks, the filtered relevant_chunks (after both rounds if reformulation
  happened), the final context string, all queries used ([query] or [query, reformulated]),
  `skipped_retrieval` (False).

- `src/strategies/multi_query.py` — In `run()`: accept `diagnostics` kwarg. Log the merged
  results from all queries, the ranked top-5 used for context, the final context string,
  all queries used (original + alternatives), `skipped_retrieval` (False).

- `src/strategies/self_rag.py` — In `run()`: accept `diagnostics` kwarg. Handle the
  "no retrieval" branch (skipped_retrieval=True, empty chunks). In the retrieval branch,
  log retrieved chunks, filtered relevant_chunks, context string, `skipped_retrieval` (False).

- `src/strategies/adaptive.py` — In `run()`: accept `diagnostics` kwarg. Pass it through
  to the three path methods (`_simple_path`, `_moderate_path`, `_complex_path`). Simple
  path sets `skipped_retrieval=True`. Moderate path logs like NaiveRAG. Complex path logs
  both retrieval rounds and the combined context.

- `src/diagnostics.py` — **CREATE**: Contains `detect_failure_stage()` function and a
  helper `_gold_in_text(gold, text)` for case-insensitive containment.

- `scripts/experiment_utils.py` — In `generate_answer()`: pass diagnostics dict to
  strategy.run(), use diagnostics to populate failure attribution columns, remove
  redundant pre-strategy retrieve() call.

## New Dependencies

None — all required packages are already installed.

## Edge Cases

- **diagnostics=None (default)**: All strategies must work exactly as before. No extra
  computation, no errors. This is the backward-compat path.
- **Empty gold answer**: `detect_failure_stage()` returns `"unknown"` — can't attribute
  without a reference.
- **Strategy skips retrieval**: AdaptiveRAG simple path and SelfRAG "no" decision both
  skip retrieval. `skipped_retrieval=True`, chunks are empty, failure_stage is
  `"no_retrieval"`.
- **Gold answer has special regex chars**: Use simple `in` operator, not regex. Safe for
  any string content.
- **Multi-round retrieval**: CorrectiveRAG may retrieve twice (original + reformulated).
  `retrieved_chunks` should contain ALL chunks from ALL rounds. `retrieval_queries` lists
  all queries used.
- **AdaptiveRAG complex path**: Two retrieval rounds. `retrieved_chunks` = union of both.
  `context_sent_to_llm` = the combined context string.
- **Empty retrieved chunks**: If retrieval returns nothing, `retrieved_chunks` is `[]`,
  `context_sent_to_llm` is `""`, failure_stage cascades correctly.
- **Generation error in generate_answer()**: If strategy.run() throws, diagnostics dict
  may be partially populated. The existing try/except in generate_answer() should set
  failure_stage to `"unknown"` and diagnostics columns to empty/NaN.

## Decisions Made

- **Mutable dict parameter over return-type change**: Chose `diagnostics: dict | None`
  kwarg over changing `run()` to return `(str, dict)`. **Why:** Avoids a breaking
  change to all callers. The dict approach is opt-in — existing code that calls
  `strategy.run(query, retriever, model)` without diagnostics works unchanged.

- **Case-insensitive substring for gold matching**: Same approach as existing
  `exact_match()` in experiment_utils.py. **Why:** Consistency with existing evaluation
  logic. More sophisticated matching (fuzzy, token-level) would add complexity without
  clear benefit for failure attribution.

- **"filtering" stage instead of "reranker"**: Current pipeline has no separate reranker
  step. What happens between retrieval and context-building is strategy-level filtering
  (CorrectiveRAG, SelfRAG) or merging (MultiQueryRAG). **Why:** Accurate naming for
  current architecture. When a reranker is added later, it can be a separate stage.

- **Don't update run_experiment_0.py**: Experiment 0 already completed and has its own
  inline generation logic. **Why:** Avoid modifying working historical code. Experiment 0
  can be re-run with diagnostics by switching to `experiment_utils.generate_answer()` in
  a future task if desired.

- **Store context_sent_to_llm in CSV**: Full context strings may be long (2500+ chars).
  **Why:** User explicitly requested seeing what was passed to the LLM. CSVs handle
  long strings fine. If bloat becomes an issue, a future task can move to a sidecar file.

- **No thread-safety concerns**: Experiment scripts run sequentially (one strategy.run()
  at a time). The mutable dict is created fresh per call. No shared state.

## What NOT to Touch

- `scripts/run_experiment_0.py` — Already ran, has its own generation logic. Leave as-is.
- `scripts/run_experiment_1.py` / `run_experiment_2.py` — These call
  `experiment_utils.generate_answer()`, so they automatically get diagnostics when utils
  is updated. No changes needed in the experiment scripts themselves.
- `src/retriever.py` — No changes needed. Strategies already call retriever.retrieve()
  internally.
- `src/scorers/` — Scoring is a separate concern from diagnostics.
- Existing test files — Don't modify existing tests. New tests go in the task test dir.

## Testing Approach

### test_diagnostics.py
- `test_detect_failure_stage_none`: Gold found in answer → `"none"`
- `test_detect_failure_stage_chunker`: Gold not in any chunk → `"chunker"`
- `test_detect_failure_stage_retrieval`: Gold in chunks but not retrieved → `"retrieval"`
- `test_detect_failure_stage_filtering`: Gold in retrieved but not in context → `"filtering"`
- `test_detect_failure_stage_generation`: Gold in context but not in answer → `"generation"`
- `test_detect_failure_stage_no_retrieval`: skipped_retrieval=True → `"no_retrieval"`
- `test_detect_failure_stage_empty_gold`: Empty gold → `"unknown"`
- `test_detect_failure_stage_case_insensitive`: Gold matching is case-insensitive

### test_strategy_diagnostics.py
- For each of the 5 strategies: mock the LLM and retriever, call `run()` with
  `diagnostics={}`, verify all expected keys are present and correctly populated.
- For each strategy: call `run()` with `diagnostics=None`, verify it still returns
  a string answer (backward compat).
- `test_adaptive_simple_path_skips_retrieval`: Verify skipped_retrieval=True
- `test_self_rag_no_retrieval_path`: Verify skipped_retrieval=True
- `test_corrective_reformulation_logs_both_rounds`: Verify multi-round chunks

### test_generate_answer_diagnostics.py
- Mock strategy and verify generate_answer() returns new diagnostic columns
- Verify failure_stage is computed correctly
- Verify error case returns "unknown" failure_stage

Run with: `python -m pytest DaytimeNighttimeHandOff/WrittenByDaytime/task-042-pipeline-diagnostics/tests/ -v`
