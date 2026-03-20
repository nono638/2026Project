# Plan: task-029 — Retry with Exponential Backoff for LLMScorer

## Approach

Modify `src/scorers/llm.py` only:

1. Add `max_retries` parameter to `__init__()` (default 3), store as `self._max_retries`
2. Add `_call_with_retry()` method with exponential backoff (2^attempt + jitter)
3. Replace the try/except in `score()` with a call to `_call_with_retry()`

## Files Modified
- `src/scorers/llm.py`

## Ambiguities
- None — spec is precise with exact code snippets.
