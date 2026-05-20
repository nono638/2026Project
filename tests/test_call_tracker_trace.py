"""Tests for the per-row trace API in src.monitoring.call_tracker.

The going-forward fix to the "we don't know which prompts were sent"
logging gap acknowledged on the website's Methodology page. These tests
exercise the begin_row / record_complete / end_row contract used by the
experiment runner.
"""

from __future__ import annotations

import pytest

from src.monitoring import call_tracker


@pytest.fixture(autouse=True)
def _close_any_open_row():
    """Ensure no row leaks between tests — module-level state is shared."""
    yield
    call_tracker.end_row()


class TestPerRowTrace:
    """begin_row / record_complete / end_row form one ordered window."""

    def test_no_open_row_means_record_complete_is_noop(self):
        # Outside an open row, record_complete must silently no-op.
        # Strategies / scorers that fire outside the runner shouldn't
        # corrupt downstream traces or raise.
        call_tracker.record_complete(
            "chat", "qwen3.5:4b", "p", "r", 0.1, intent="generate_answer",
        )
        assert call_tracker.end_row() == []

    def test_full_cycle_captures_intents_in_order(self):
        call_tracker.begin_row()
        call_tracker.record_complete("chat", "m", "P1", "R1", 0.5, intent="rate_relevance")
        call_tracker.record_complete("chat", "m", "P2", "R2", 0.2, intent="rate_relevance")
        call_tracker.record_complete("chat", "m", "P3", "R3", 0.8, intent="generate_answer")
        trace = call_tracker.end_row()
        assert [c["intent"] for c in trace] == [
            "rate_relevance", "rate_relevance", "generate_answer",
        ]
        assert trace[-1]["prompt"] == "P3"
        assert trace[-1]["response"] == "R3"
        assert trace[-1]["latency_s"] == 0.8

    def test_unlabeled_intent_becomes_unknown(self):
        # Strategies that don't pass intent= shouldn't drop a record —
        # they get tagged "unknown" so the analyst can still see the call.
        call_tracker.begin_row()
        call_tracker.record_complete("chat", "m", "p", "r", 0.1)
        trace = call_tracker.end_row()
        assert trace[0]["intent"] == "unknown"

    def test_end_row_clears_state(self):
        call_tracker.begin_row()
        call_tracker.record_complete("chat", "m", "p", "r", 0.1, intent="x")
        call_tracker.end_row()
        # Second end_row with no fresh begin_row must return empty.
        assert call_tracker.end_row() == []
        # And record_complete after end_row must no-op.
        call_tracker.record_complete("chat", "m", "p", "r", 0.1, intent="x")
        assert call_tracker.end_row() == []

    def test_double_begin_drops_prior_partial_trace(self):
        # If a prior row was never closed (e.g., an unexpected exit path),
        # begin_row resets cleanly rather than carrying stale records
        # into the next row.
        call_tracker.begin_row()
        call_tracker.record_complete("chat", "m", "p", "r", 0.1, intent="x")
        call_tracker.begin_row()  # implicit drop
        call_tracker.record_complete("chat", "m", "p2", "r2", 0.2, intent="y")
        trace = call_tracker.end_row()
        assert len(trace) == 1
        assert trace[0]["prompt"] == "p2"


class TestCodeSha:
    """code_sha caches a git rev-parse so per-row stamping is cheap."""

    def test_returns_dict_with_full_and_short_keys(self):
        result = call_tracker.code_sha()
        assert set(result.keys()) == {"full", "short"}
        # We can't assert the SHA exists (CI may not have git) — only
        # the shape. A None value is the documented fail-open mode.
        assert result["full"] is None or isinstance(result["full"], str)
        assert result["short"] is None or isinstance(result["short"], str)
        if result["short"] is not None:
            assert len(result["short"]) == 7

    def test_short_is_prefix_of_full_when_available(self):
        result = call_tracker.code_sha()
        if result["full"] and result["short"]:
            assert result["full"].startswith(result["short"])
