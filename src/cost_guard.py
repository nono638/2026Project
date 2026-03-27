"""API cost guard — tracks estimated spend and aborts if ceiling is exceeded.

This is a safety net, not a billing system. Rough per-call cost estimates
intentionally err on the high side so the guard triggers early rather than
late. If the guard triggers at $4.50 instead of $5.00, that's fine.

Why hardcoded costs: precise costs would require parsing API response headers
or token counts. Rough estimates that overcount are simpler and safer for a
novice user who might not notice runaway spend.

Why default $5.00: Experiment 0 costs ~$0.30 total. Experiments 1 & 2 with
Gemini Flash would cost under $1. $5.00 is generous enough to never trigger
accidentally but catches genuine runaways.
"""

from __future__ import annotations


class CostLimitExceeded(Exception):
    """Raised when estimated spend exceeds the configured ceiling."""


# Per-call cost estimates in USD — rough, intentionally high.
# Based on ~500 input tokens + ~100 output tokens per scoring call.
COST_PER_CALL: dict[str, float] = {
    # Google (free tier, but track anyway)
    "google:gemini-2.5-flash-lite": 0.0001,
    "google:gemini-2.5-flash": 0.0002,
    "google:gemini-2.5-pro": 0.002,
    # Anthropic
    "anthropic:claude-haiku-4-5-20251001": 0.002,
    "anthropic:claude-sonnet-4-20250514": 0.01,
    "anthropic:claude-opus-4-20250514": 0.075,
}

# Intentionally high default for unknown models — conservative safety net
DEFAULT_COST_PER_CALL: float = 0.01


class CostGuard:
    """Tracks estimated API spend and aborts if ceiling is exceeded.

    Not a billing system — just a safety net with rough per-call estimates
    that err on the high side.

    Args:
        max_cost_usd: Maximum estimated spend before raising CostLimitExceeded.
                      Default $5.00 — safe for all planned experiments.
    """

    def __init__(self, max_cost_usd: float = 5.0) -> None:
        self._max_cost_usd = max_cost_usd
        self._total_cost: float = 0.0
        self._call_count: int = 0

    def record_call(self, provider: str, model: str) -> None:
        """Record one API call and check against the ceiling.

        Uses a hardcoded cost-per-call lookup table (rough estimates).
        Raises CostLimitExceeded if cumulative estimated spend exceeds max.

        Args:
            provider: API provider name (e.g., "google", "anthropic").
            model: Model identifier (e.g., "gemini-2.5-flash").
        """
        key = f"{provider}:{model}"
        cost = COST_PER_CALL.get(key, DEFAULT_COST_PER_CALL)
        self._total_cost += cost
        self._call_count += 1

        if self._total_cost > self._max_cost_usd:
            raise CostLimitExceeded(
                f"Estimated spend ${self._total_cost:.2f} exceeds "
                f"limit ${self._max_cost_usd:.2f} after {self._call_count} calls"
            )

    @property
    def total_estimated_cost(self) -> float:
        """Current cumulative estimated cost in USD."""
        return self._total_cost

    @property
    def call_count(self) -> int:
        """Total number of API calls recorded."""
        return self._call_count

    def summary(self) -> str:
        """One-line summary of API usage.

        Returns:
            String like '300 calls, ~$0.32 estimated spend'.
        """
        return f"{self._call_count} calls, ~${self._total_cost:.2f} estimated spend"
