# Plan: task-008 — RAGAS QueryGenerator, RoundTripFilter, HeuristicFilter

## Approach

1. Install `ragas` package, check installed version's API
2. Create `src/query_generators/ragas.py` — RagasQueryGenerator wrapping RAGAS TestsetGenerator
3. Create `src/query_filters/round_trip.py` — RoundTripFilter using retriever for validation
4. Create `src/query_filters/heuristic.py` — HeuristicFilter with length/question/overlap/dedup checks
5. Update `src/query_generators/__init__.py` and `src/query_filters/__init__.py`
6. Create 3 test files: test_ragas_generator.py, test_round_trip_filter.py, test_heuristic_filter.py
7. Run tests

## Files to Create
- `src/query_generators/ragas.py`
- `src/query_filters/round_trip.py`
- `src/query_filters/heuristic.py`
- `tests/test_ragas_generator.py`
- `tests/test_round_trip_filter.py`
- `tests/test_heuristic_filter.py`

## Files to Modify
- `src/query_generators/__init__.py` — add RagasQueryGenerator import
- `src/query_filters/__init__.py` — add RoundTripFilter, HeuristicFilter imports

## Ambiguities
- RAGAS API varies by version. Will install and adapt to whatever version is available.
- Spec says to use HuggingFaceEmbedder for round-trip tests but notes it's slow — will use
  HashEmbedder for most tests and HuggingFaceEmbedder only where semantic similarity matters.
