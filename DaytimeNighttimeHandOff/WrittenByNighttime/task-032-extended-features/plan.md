# Plan: task-032 — Extended Feature Columns

## Files to Modify
1. `src/features.py` — Add 4 new feature functions, modify _embedding_features/_estimate_topic_count for spread, wire into extract_features
2. `src/metadata.py` — Add get_llm_context_window, _query_ollama_context_window, build_llm_context_metadata
3. `src/experiment.py` — Add context window metadata to row dict (requires merging task-031 branch first)
4. `tests/test_e2e_smoke.py` — Add 6 new columns to required_cols
5. `tests/test_integration.py` — Add feature key assertions
6. `tests/test_metadata.py` — Already has tests in test_context_window_metadata.py

## Approach
- This task depends on task-031 which modified experiment.py. I'll merge that branch first.
- Implement features.py changes (readability, embedding spread, query-doc similarity, lexical overlap)
- Implement metadata.py changes (context window functions)
- Update experiment.py to add context window metadata
- Update test files
- Run pre-written tests then full suite

## Ambiguities
- The spec says to use `from ollama import Client` in _query_ollama_context_window but the test mocks `src.metadata.Client`. I'll need to import Client at the function level and ensure the mock path works.
