# Plan: task-001 — Core Framework

## Files to Create
1. `src/protocols.py` — Protocol definitions (Chunker, Embedder, Strategy, Scorer)
2. `src/retriever.py` — Retriever class wrapping FAISS + embedder + chunks
3. `src/experiment.py` — Experiment runner and ExperimentResult class
4. `src/features.py` — Adapted from `src/data/features.py`, new signature using Retriever
5. `tests/test_core.py` — Tests with mock Protocol implementations

## Files to Modify
- None (spec says create only, don't delete anything)

## Approach
1. Create `src/protocols.py` with all four Protocol definitions, using `from __future__ import annotations` and `TYPE_CHECKING` guard for Retriever forward reference
2. Create `src/retriever.py` with the Retriever class, handling empty chunks edge case
3. Create `src/features.py` adapted from existing `src/data/features.py` with new Retriever-based signature
4. Create `src/experiment.py` with Experiment and ExperimentResult classes, handling empty corpus edge case
5. Create `tests/test_core.py` with mock implementations and all 6 test cases from the spec

## Ambiguities
- The spec shows `extract_features` accepting a Retriever but the existing features.py has `chunks` and `index` params. Will adapt the signature as spec says, keeping the internal logic (entity counting, entropy) unchanged.
- Empty chunks in Retriever: will create an empty FAISS index (0 vectors) and return empty results from retrieve().
- The spec doesn't mention `src/__init__.py` changes — will not modify it.
