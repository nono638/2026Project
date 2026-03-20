"""Tests for LLMScorer retry logic (task-029).

All tests mock _call_llm directly — no real API calls.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.scorers.llm import LLMScorer, ScorerError


@pytest.fixture
def scorer():
    """Create an LLMScorer with a mocked adapter (no real API client)."""
    with patch("src.scorers.llm._get_adapter") as mock_get:
        mock_get.return_value = MagicMock(return_value='{"faithfulness": 5, "relevance": 5, "conciseness": 5}')
        s = LLMScorer(provider="google", model="test-model", max_retries=3)
    return s


class TestRetrySucceeds:
    """Retry recovers from transient errors."""

    def test_retry_succeeds_after_transient_error(self, scorer):
        """Mock _call_llm to fail twice with 503, then succeed."""
        scorer._call_llm = MagicMock(side_effect=[
            Exception("503 UNAVAILABLE. This model is currently experiencing high demand."),
            Exception("503 UNAVAILABLE. This model is currently experiencing high demand."),
            '{"faithfulness": 4, "relevance": 5, "conciseness": 4}',
        ])

        with patch("time.sleep"):  # Don't actually wait
            result = scorer.score(query="test?", context="ctx", answer="ans")

        assert result["faithfulness"] == 4.0
        assert result["relevance"] == 5.0
        assert scorer._call_llm.call_count == 3


class TestRetryExhausted:
    """All retries fail — ScorerError raised."""

    def test_retry_exhausted(self, scorer):
        """Mock _call_llm to always fail with 503."""
        scorer._call_llm = MagicMock(
            side_effect=Exception("503 UNAVAILABLE. High demand.")
        )

        with patch("time.sleep"):
            with pytest.raises(ScorerError, match="API call failed"):
                scorer.score(query="test?", context="ctx", answer="ans")

        # max_retries=3 means 4 total attempts
        assert scorer._call_llm.call_count == 4


class TestNoRetryOnAuthError:
    """Non-retryable errors fail immediately."""

    def test_no_retry_on_auth_error(self, scorer):
        """401 Unauthorized should not be retried."""
        scorer._call_llm = MagicMock(
            side_effect=Exception("401 Unauthorized. Invalid API key.")
        )

        with patch("time.sleep") as mock_sleep:
            with pytest.raises(ScorerError):
                scorer.score(query="test?", context="ctx", answer="ans")

        assert scorer._call_llm.call_count == 1
        mock_sleep.assert_not_called()


class TestNoRetryWhenDisabled:
    """max_retries=0 means one attempt only."""

    def test_no_retry_when_disabled(self):
        """Create scorer with max_retries=0, verify immediate failure."""
        with patch("src.scorers.llm._get_adapter") as mock_get:
            mock_get.return_value = MagicMock()
            s = LLMScorer(provider="google", model="test-model", max_retries=0)

        s._call_llm = MagicMock(
            side_effect=Exception("503 UNAVAILABLE.")
        )

        with patch("time.sleep") as mock_sleep:
            with pytest.raises(ScorerError):
                s.score(query="test?", context="ctx", answer="ans")

        assert s._call_llm.call_count == 1
        mock_sleep.assert_not_called()


class TestRetryBackoffTiming:
    """Verify exponential backoff timing."""

    def test_retry_backoff_timing(self, scorer):
        """Verify sleep is called with increasing values."""
        scorer._call_llm = MagicMock(
            side_effect=Exception("429 rate limit exceeded")
        )

        with patch("time.sleep") as mock_sleep:
            with pytest.raises(ScorerError):
                scorer.score(query="test?", context="ctx", answer="ans")

        # 3 retries = 3 sleeps (attempts 0, 1, 2 fail with retry; attempt 3 fails final)
        assert mock_sleep.call_count == 3
        # Backoff: 2^0 + jitter, 2^1 + jitter, 2^2 + jitter
        calls = [c.args[0] for c in mock_sleep.call_args_list]
        assert 1.0 <= calls[0] < 2.0   # 2^0 + [0,1)
        assert 2.0 <= calls[1] < 3.0   # 2^1 + [0,1)
        assert 4.0 <= calls[2] < 5.0   # 2^2 + [0,1)
