"""Tests for the API cost guard module.

Validates cost tracking, limit enforcement, and summary formatting.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.cost_guard import CostGuard, CostLimitExceeded, COST_PER_CALL


class TestCostGuardUnderLimit:
    """Tests that calls under the limit don't raise."""

    def test_under_limit(self) -> None:
        """10 calls at a known cheap model should not raise."""
        guard = CostGuard(max_cost_usd=1.0)
        for _ in range(10):
            guard.record_call("google", "gemini-2.5-flash-lite")
        # 10 * 0.0001 = 0.001, well under $1.00
        assert guard.total_estimated_cost < 1.0
        assert guard.call_count == 10


class TestCostGuardExceedsLimit:
    """Tests that exceeding the limit raises CostLimitExceeded."""

    def test_exceeds_limit(self) -> None:
        """Set a tiny limit and make calls until CostLimitExceeded fires."""
        guard = CostGuard(max_cost_usd=0.01)
        with pytest.raises(CostLimitExceeded):
            # Each call at default cost ($0.01) should trigger on the 2nd call
            for _ in range(100):
                guard.record_call("anthropic", "claude-sonnet-4-20250514")


class TestCostGuardUnknownModel:
    """Tests that unknown models use the default cost."""

    def test_unknown_model_uses_default(self) -> None:
        """Unrecognized model should use $0.01 default per call."""
        guard = CostGuard(max_cost_usd=100.0)
        guard.record_call("openai", "gpt-4o-mini")
        # DEFAULT_COST_PER_CALL is 0.01
        assert guard.total_estimated_cost == pytest.approx(0.01)


class TestCostGuardSummary:
    """Tests for the summary string format."""

    def test_summary_format(self) -> None:
        """Summary should contain call count and dollar amount."""
        guard = CostGuard(max_cost_usd=10.0)
        guard.record_call("google", "gemini-2.5-flash")
        guard.record_call("google", "gemini-2.5-flash")
        summary = guard.summary()
        assert "2 calls" in summary
        assert "$" in summary


class TestCostGuardZeroLimit:
    """Tests that a $0.00 limit blocks the first call."""

    def test_zero_limit_blocks_first_call(self) -> None:
        """Limit of $0.00 should raise on the very first call."""
        guard = CostGuard(max_cost_usd=0.0)
        with pytest.raises(CostLimitExceeded):
            guard.record_call("google", "gemini-2.5-flash")
