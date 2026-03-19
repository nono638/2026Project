# Task 027: API Cost Guard

## What

Add a simple cost-tracking wrapper that counts API calls and estimated spend
across LLM scoring, and aborts if a configurable ceiling is exceeded. This
protects against runaway costs from bugs, infinite loops, or unexpectedly
large experiments.

## Why

The user has limited API credits ($80 Anthropic, free-tier Google, $10 RunPod)
and is a novice at managing cloud API costs. There are currently no guardrails
— a bug in a loop or a misconfigured experiment could burn through credits
before anyone notices. Even though the current code has no retry loops, future
changes might introduce them. Defense in depth.

## Changes

### `src/cost_guard.py` (create new)

A lightweight cost tracker. Not a billing system — just a safety net.

```python
class CostGuard:
    """Tracks estimated API spend and aborts if ceiling is exceeded."""

    def __init__(self, max_cost_usd: float = 5.0):
        """
        Args:
            max_cost_usd: Maximum estimated spend before raising CostLimitExceeded.
                          Default $5.00 — safe for all planned experiments.
        """

    def record_call(self, provider: str, model: str) -> None:
        """Record one API call and check against the ceiling.

        Uses a hardcoded cost-per-call lookup table (rough estimates).
        Raises CostLimitExceeded if cumulative estimated spend exceeds max.
        """

    @property
    def total_estimated_cost(self) -> float:
        """Current cumulative estimated cost in USD."""

    @property
    def call_count(self) -> int:
        """Total number of API calls recorded."""

    def summary(self) -> str:
        """One-line summary: 'N calls, ~$X.XX estimated spend'."""


class CostLimitExceeded(Exception):
    """Raised when estimated spend exceeds the configured ceiling."""
```

**Cost-per-call lookup table** (hardcoded, rough estimates — errs on the high side):

```python
COST_PER_CALL = {
    # Google (free tier, but track anyway)
    "google:gemini-2.5-flash-lite": 0.0001,
    "google:gemini-2.5-flash": 0.0002,
    "google:gemini-2.5-pro": 0.002,
    # Anthropic
    "anthropic:claude-haiku-4-5-20251001": 0.002,
    "anthropic:claude-sonnet-4-20250514": 0.01,
}
# Default for unknown models — intentionally high to be conservative
DEFAULT_COST_PER_CALL = 0.01
```

These are rough per-call estimates for a typical scoring prompt (~500 input tokens,
~100 output tokens). They intentionally overestimate to trigger the guard early
rather than late.

### `src/scorers/llm.py`

1. Add an optional `cost_guard: CostGuard | None = None` parameter to `LLMScorer.__init__()`.
2. In the `score()` method, after a successful API call, call
   `self._cost_guard.record_call(self._provider, self._model)` if guard is set.
3. If `CostLimitExceeded` is raised, let it propagate — the experiment script
   will catch it and shut down gracefully.

### `scripts/run_experiment_0.py`

1. Add `--max-cost` CLI flag (default: `5.0`):
   ```
   --max-cost  Maximum estimated API spend in USD before aborting (default: $5.00)
   ```

2. Create a `CostGuard(max_cost_usd=args.max_cost)` and pass it to each scorer.

3. Wrap the scoring loop in a try/except for `CostLimitExceeded`:
   ```python
   try:
       results_df = score_all_answers(answers, output_dir, cost_guard=cost_guard)
   except CostLimitExceeded as exc:
       logger.error("COST LIMIT REACHED: %s", exc)
       logger.error("Saving partial results...")
       # Save whatever we have so far
   ```
   Partial results are still saved so nothing is lost.

4. At the end of the run, print the cost summary:
   ```
   API cost summary: 300 calls, ~$0.32 estimated spend (limit: $5.00)
   ```

### `tests/test_cost_guard.py` (create new)

1. `test_under_limit` — 10 calls at $0.001 each, no exception
2. `test_exceeds_limit` — set limit to $0.01, make calls until CostLimitExceeded
3. `test_unknown_model_uses_default` — unrecognized model uses $0.01 default
4. `test_summary_format` — summary string contains call count and dollar amount
5. `test_zero_limit_blocks_first_call` — limit of $0.00 raises on first call

## What NOT to touch

- Do not add cost tracking to embedder calls (Ollama is local/free, Google embedder
  is negligible)
- Do not add RunPod cost tracking here (RunPod has its own auto-shutdown)
- Do not persist cost data between runs (this is per-run only)
- Do not make the cost table configurable (hardcoded is fine for a safety net)
- Do not add billing integration or real cost calculation from API responses

## Design decisions

- **Why hardcoded costs?** This is a safety net, not a billing system. Rough
  estimates that err high are better than precise costs that require parsing
  API response headers. If the guard triggers at $4.50 instead of $5.00,
  that's fine.
- **Why default $5.00?** Experiment 0 costs ~$0.30 total. Experiments 1 & 2
  with Gemini Flash would cost under $1. $5.00 is generous enough to never
  trigger accidentally but catches genuine runaways.
- **Why not per-provider limits?** Simplicity. One number is easier to reason
  about. The user can set `--max-cost 2` if they want tighter control.
