# Plan: task-011 — Cross-Encoder Filter and Distribution Analyzer

## Approach

1. Create branch, merge task-008 (depends on it for QueryFilter protocol, Document, Query)
2. Create `src/query_filters/cross_encoder.py` — CrossEncoderFilter using sentence-transformers
3. Create `src/query_analysis/__init__.py` — new package
4. Create `src/query_analysis/distribution.py` — DistributionAnalyzer
5. Update `src/query_filters/__init__.py` — add CrossEncoderFilter
6. Create `tests/test_cross_encoder_filter.py` — 9 tests (real model, no mocks)
7. Create `tests/test_distribution_analyzer.py` — 11 tests
8. Run tests

## Files to Create
- `src/query_filters/cross_encoder.py`
- `src/query_analysis/__init__.py`
- `src/query_analysis/distribution.py`
- `tests/test_cross_encoder_filter.py`
- `tests/test_distribution_analyzer.py`

## Files to Modify
- `src/query_filters/__init__.py` — add CrossEncoderFilter

## Dependencies
- sentence-transformers (already installed, includes CrossEncoder)
- task-008 branch (provides QueryFilter protocol, Document, Query, query_filters package)
- sklearn for KMeans clustering in DistributionAnalyzer (already available via scikit-learn)

## Ambiguities
- Cross-encoder tests use the real model (~25MB download on first run). Tests may be slow on first run.
- DBSCAN vs KMeans for clustering: spec says "DBSCAN or simple k-means" — will use KMeans as spec leans toward it.
