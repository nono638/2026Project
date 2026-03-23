# Plan: task-042 — Pipeline Diagnostics and Failure Attribution

## Approach

1. **Update `src/protocols.py`** — Add optional `diagnostics: dict | None = None` param to `Strategy.run()`.

2. **Create `src/diagnostics.py`** — `detect_failure_stage()` with case-insensitive substring matching, returns `(stage, confidence)` tuple.

3. **Update all 5 strategies** — Add `diagnostics` kwarg to `run()`. When not None, populate: `retrieved_chunks`, `filtered_chunks`, `context_sent_to_llm`, `retrieval_queries`, `skipped_retrieval`. No extra work when None.
   - `naive.py`: Simple — retrieved == filtered, single query, no skip.
   - `corrective.py`: Track both retrieval rounds if reformulation occurs. `filtered_chunks` = relevant_chunks after filtering.
   - `multi_query.py`: Track all queries (original + alternatives). `retrieved_chunks` = all merged results. `filtered_chunks` = top-5 ranked.
   - `self_rag.py`: Handle "no retrieval" branch (skipped=True, empty chunks). Retrieval branch: track retrieved, filtered relevant chunks.
   - `adaptive.py`: Pass diagnostics through to path methods. Simple → skipped=True. Moderate → like NaiveRAG. Complex → union of both retrieval rounds, combined context.

4. **Update `scripts/experiment_utils.py`** — In `generate_answer()`:
   - Create `diagnostics = {}` dict
   - Pass to `strategy.run()` via `diagnostics=diagnostics`
   - Use diagnostics to populate new columns: `context_sent_to_llm`, `failure_stage`, `failure_stage_confidence`, `failure_stage_method`, `gold_in_chunks`, `gold_in_retrieved`, `gold_in_context`
   - Update `context_char_length` to use actual context from diagnostics
   - Remove redundant `retriever.retrieve()` call on line 203
   - Handle error case: set `failure_stage = "unknown"` when strategy.run() throws

## Files Modified
- `src/protocols.py` (line 64 — Strategy.run signature)
- `src/strategies/naive.py`
- `src/strategies/corrective.py`
- `src/strategies/multi_query.py`
- `src/strategies/self_rag.py`
- `src/strategies/adaptive.py`
- `scripts/experiment_utils.py`

## Files Created
- `src/diagnostics.py`

## Ambiguities
- None significant. Spec is clear and complete.
