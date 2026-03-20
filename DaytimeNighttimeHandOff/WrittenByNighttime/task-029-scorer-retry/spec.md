# Task 029: Retry with Exponential Backoff for LLMScorer

## What

Add retry logic to `LLMScorer.score()` so that transient API errors (503 overloaded, 429 rate limit, connection errors) are retried automatically instead of immediately failing and recording NaN.

## Why

Gemini 2.5 Pro returned 503 "high demand" on ~50% of calls during Experiment 0, producing all-NaN scores. The errors were transient — the same requests succeeded minutes later. Without retry, we lose data and have to manually rerun. This is the #1 reliability issue for Experiments 1 & 2 which will make ~2000+ scorer API calls.

## Files to Modify

- `src/scorers/llm.py` — add retry logic to the `score()` method

## Exact Changes

### In `LLMScorer.score()` (line ~232)

Replace the current try/except around `self._call_llm(prompt)`:

```python
try:
    text = self._call_llm(prompt)
except Exception as exc:
    raise ScorerError(
        f"{self._provider} API call failed: {exc}"
    ) from exc
```

With a retry loop:

```python
text = self._call_with_retry(prompt)
```

### New method `_call_with_retry()`

Add a private method to `LLMScorer`:

```python
def _call_with_retry(self, prompt: str) -> str:
    import time
    import random

    max_attempts = self._max_retries + 1  # 1 initial + N retries
    for attempt in range(max_attempts):
        try:
            return self._call_llm(prompt)
        except Exception as exc:
            exc_str = str(exc).lower()
            retryable = any(s in exc_str for s in ("503", "429", "overloaded", "rate", "unavailable", "temporarily", "connection", "timeout"))

            if not retryable or attempt == max_attempts - 1:
                raise ScorerError(
                    f"{self._provider} API call failed: {exc}"
                ) from exc

            # Exponential backoff: 1s, 2s, 4s + jitter
            wait = (2 ** attempt) + random.uniform(0, 1)
            logger.warning(
                "Retryable error from %s (attempt %d/%d), waiting %.1fs: %s",
                self.name, attempt + 1, max_attempts, wait, exc,
            )
            time.sleep(wait)
```

### Constructor change

Add `max_retries` parameter to `__init__()`:

```python
def __init__(
    self,
    provider: str,
    model: str,
    api_key: str | None = None,
    cost_guard: Any | None = None,
    max_retries: int = 3,
) -> None:
```

Store as `self._max_retries = max_retries`.

## What NOT to Touch

- Don't change the adapter functions (`_anthropic_adapter`, `_google_adapter`)
- Don't change `_parse_response`, `_build_prompt`, or `score_batch`
- Don't add async/concurrent retry — keep it simple sequential
- Don't add new dependencies — `time` and `random` are stdlib

## Edge Cases

- `max_retries=0` means no retries (1 attempt only, current behavior)
- Non-retryable errors (auth, invalid model, malformed response) must NOT be retried — they fail immediately
- The retry detection is substring-based on the exception message. This is intentional — both the Anthropic and Google SDKs raise different exception types, but they all include the HTTP status code in the message string.

## Tests

Write tests in `tests/test_scorer_retry.py`:

1. **test_retry_succeeds_after_transient_error** — mock `_call_llm` to fail twice with a 503-containing exception, then succeed. Verify score is returned and 3 calls were made.
2. **test_retry_exhausted** — mock `_call_llm` to always fail with 503. Verify `ScorerError` is raised after `max_retries + 1` attempts.
3. **test_no_retry_on_auth_error** — mock `_call_llm` to fail with "401 Unauthorized". Verify `ScorerError` is raised immediately (1 call only).
4. **test_no_retry_when_disabled** — create scorer with `max_retries=0`, mock a 503 failure. Verify it fails immediately.
5. **test_retry_backoff_timing** — mock `time.sleep` and verify it's called with increasing values.

All tests should mock `_call_llm` directly on the `LLMScorer` instance (no real API calls). Use `unittest.mock.patch.object`.
