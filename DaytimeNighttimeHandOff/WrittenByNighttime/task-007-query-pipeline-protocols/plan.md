# Plan: task-007 — Query Pipeline Protocols and Data Models

## Approach

1. Create `src/document.py` — Document dataclass, load_corpus_from_csv, sample_corpus, documents_to_dicts
2. Create `src/query.py` — Query dataclass, save_queries, load_queries, queries_to_dicts
3. Update `src/protocols.py` — add QueryGenerator and QueryFilter protocols
4. Create `src/query_generators/__init__.py` — empty package
5. Create `src/query_filters/__init__.py` — empty package
6. Create `tests/test_query_pipeline.py` — 16 tests covering data models, CSV loading, JSON persistence, bridge helpers, protocol compliance
7. Run tests

## Files to Create
- `src/document.py`
- `src/query.py`
- `src/query_generators/__init__.py`
- `src/query_filters/__init__.py`
- `tests/test_query_pipeline.py`

## Files to Modify
- `src/protocols.py` — add QueryGenerator, QueryFilter protocols + TYPE_CHECKING imports

## Ambiguities
- None — spec is very detailed with exact code snippets for all data structures
