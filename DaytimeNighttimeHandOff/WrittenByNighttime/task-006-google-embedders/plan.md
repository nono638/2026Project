# Plan: task-006 — Google Text Embedder

## Approach

1. Install `google-generativeai` into the venv
2. Create `src/embedders/google_text.py` with `GoogleTextEmbedder` class
3. Update `src/embedders/__init__.py` to export it
4. Create `tests/test_google_embedders.py` with 7 tests (all mocked)
5. Run tests, verify pass

## Files to Create
- `src/embedders/google_text.py` — GoogleTextEmbedder implementation
- `tests/test_google_embedders.py` — 7 tests per spec

## Files to Modify
- `src/embedders/__init__.py` — add GoogleTextEmbedder export
- `requirements.txt` — add google-generativeai pin

## Ambiguities
- None noted — spec is very explicit about API usage, mocking strategy, and edge cases
