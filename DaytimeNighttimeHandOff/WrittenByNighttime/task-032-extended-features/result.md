# Result: task-032 — Extended Feature Columns
**Status:** done
**Completed:** 2026-03-20T13:55:47

## Commits
- (pending — will be committed after this file is written)

## Test Results
- Command run: `python -m pytest DaytimeNighttimeHandOff/WrittenByDaytime/task-032-extended-features/tests/ -v`
- Outcome: 29 passed, 2 failed
- Failures:
  - `TestQueryDocLexicalOverlap::test_full_overlap` — test uses `text * 10` ("hello world" * 10) which creates compound words like "worldhello" due to no space separator. Jaccard correctly returns 0.667 (2 matching words / 3 unique words) instead of expected 1.0. Test bug, not implementation bug.
  - `TestQueryDocLexicalOverlap::test_case_insensitive` — same issue: `text * 10` creates compound words. Jaccard correctly returns 0.75 instead of expected 1.0.

- Command run: `python -m pytest tests/ --ignore=DaytimeNighttimeHandOff/ -q`
- Outcome: 485 passed, 0 failed

## Decisions Made
- Updated `tests/test_doc_features.py` to handle new return signatures:
  - `_embedding_features()` now returns 3-tuple (topic_count, coherence, spread)
  - `_estimate_topic_count()` now returns 2-tuple (count, spread)
  - Added `_embedder.embed` mock to TestExtractFeaturesIntegration tests (needed for `_query_doc_similarity`)
  - Added new feature keys to expected_keys set
- For `build_llm_context_metadata` in experiment.py, computed context_char_length inline using `sum(len(c.get("text", "")) for c in final_chunks)` rather than extracting from build_context_metadata output, to keep the row dict construction clean.

## Flags for Morning Review
- 2 pre-written test failures are test design bugs (`text * 10` without space separator). The implementation is correct — Jaccard similarity works as specified. Consider fixing tests to use `" ".join([text] * 10)` or adjusting expectations.
- The task-031 branch was merged into this branch for dependency resolution. Both branches should be merged in order: task-031 first, then task-032.

## Attempted Approaches (if skipped/blocked)
N/A — completed successfully.
