"""Tests for retry logic in experiment_utils.

All external calls (Ollama, API scorers) are mocked — no real services needed.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiment_utils import (
    _is_transient,
    generate_answer,
    score_answer,
    ensure_model,
    MAX_RETRIES,
)


# ---------------------------------------------------------------------------
# _is_transient
# ---------------------------------------------------------------------------

class TestIsTransient:
    """Tests for transient error classification."""

    def test_connection_error(self) -> None:
        assert _is_transient(ConnectionError("connection refused")) is True

    def test_timeout_error(self) -> None:
        assert _is_transient(TimeoutError("request timed out")) is True

    def test_os_error(self) -> None:
        assert _is_transient(OSError("network unreachable")) is True

    def test_rate_limit_in_message(self) -> None:
        assert _is_transient(Exception("rate limit exceeded")) is True

    def test_429_in_message(self) -> None:
        assert _is_transient(Exception("HTTP 429 Too Many Requests")) is True

    def test_502_in_message(self) -> None:
        assert _is_transient(Exception("502 Bad Gateway")) is True

    def test_503_in_message(self) -> None:
        assert _is_transient(Exception("503 Service Unavailable")) is True

    def test_server_error_in_message(self) -> None:
        assert _is_transient(Exception("server error occurred")) is True

    def test_resource_exhausted(self) -> None:
        assert _is_transient(Exception("resource exhausted")) is True

    def test_reset_by_peer(self) -> None:
        assert _is_transient(Exception("Connection reset by peer")) is True

    def test_broken_pipe(self) -> None:
        assert _is_transient(Exception("Broken pipe")) is True

    def test_value_error_not_transient(self) -> None:
        assert _is_transient(ValueError("invalid model name")) is False

    def test_key_error_not_transient(self) -> None:
        assert _is_transient(KeyError("missing_key")) is False

    def test_generic_exception_not_transient(self) -> None:
        assert _is_transient(Exception("something went wrong")) is False

    def test_type_error_not_transient(self) -> None:
        assert _is_transient(TypeError("expected str, got int")) is False


# ---------------------------------------------------------------------------
# generate_answer retry
# ---------------------------------------------------------------------------

class TestGenerateAnswerRetry:
    """Tests for retry logic in generate_answer()."""

    def _make_mocks(self):
        """Build minimal mocks for generate_answer dependencies."""
        strategy = MagicMock()
        chunker = MagicMock()
        chunker.chunk.return_value = ["chunk1", "chunk2"]
        embedder = MagicMock()
        query = MagicMock()
        query.text = "What is Python?"
        query.reference_answer = "A programming language"
        doc = MagicMock()
        doc.text = "Python is a programming language."
        return strategy, chunker, embedder, query, doc

    @patch("scripts.experiment_utils.Retriever")
    @patch("scripts.experiment_utils.detect_failure_stage", return_value=("correct", "high"))
    @patch("scripts.experiment_utils._gold_in_text", return_value=True)
    @patch("scripts.experiment_utils.time.sleep")
    def test_retries_on_transient_then_succeeds(
        self, mock_sleep, mock_gold, mock_detect, mock_retriever,
    ) -> None:
        strategy, chunker, embedder, query, doc = self._make_mocks()
        # Fail twice with transient error, succeed on third attempt
        strategy.run.side_effect = [
            ConnectionError("connection reset"),
            TimeoutError("timed out"),
            "Python is a programming language.",
        ]

        result = generate_answer(
            strategy=strategy, chunker=chunker, embedder=embedder,
            retrieval_mode="hybrid", query=query, doc=doc, model="qwen3:4b",
        )

        assert result["answer"] == "Python is a programming language."
        assert strategy.run.call_count == 3
        assert mock_sleep.call_count == 2

    @patch("scripts.experiment_utils.Retriever")
    @patch("scripts.experiment_utils.time.sleep")
    def test_permanent_error_fails_immediately(
        self, mock_sleep, mock_retriever,
    ) -> None:
        strategy, chunker, embedder, query, doc = self._make_mocks()
        strategy.run.side_effect = ValueError("invalid model")

        result = generate_answer(
            strategy=strategy, chunker=chunker, embedder=embedder,
            retrieval_mode="hybrid", query=query, doc=doc, model="qwen3:4b",
        )

        assert result["answer"] == ""
        assert "invalid model" in result["error"]
        assert strategy.run.call_count == 1
        assert mock_sleep.call_count == 0

    @patch("scripts.experiment_utils.Retriever")
    @patch("scripts.experiment_utils.time.sleep")
    def test_exhausts_retries_returns_error(
        self, mock_sleep, mock_retriever,
    ) -> None:
        strategy, chunker, embedder, query, doc = self._make_mocks()
        # All attempts fail with transient error
        strategy.run.side_effect = ConnectionError("connection refused")

        result = generate_answer(
            strategy=strategy, chunker=chunker, embedder=embedder,
            retrieval_mode="hybrid", query=query, doc=doc, model="qwen3:4b",
        )

        assert result["answer"] == ""
        assert result["failure_stage"] == "unknown"
        assert strategy.run.call_count == MAX_RETRIES + 1
        assert mock_sleep.call_count == MAX_RETRIES

    @patch("scripts.experiment_utils.Retriever")
    @patch("scripts.experiment_utils.detect_failure_stage", return_value=("correct", "high"))
    @patch("scripts.experiment_utils._gold_in_text", return_value=True)
    @patch("scripts.experiment_utils.time.sleep")
    def test_no_retry_on_success(
        self, mock_sleep, mock_gold, mock_detect, mock_retriever,
    ) -> None:
        strategy, chunker, embedder, query, doc = self._make_mocks()
        strategy.run.return_value = "Answer"

        result = generate_answer(
            strategy=strategy, chunker=chunker, embedder=embedder,
            retrieval_mode="hybrid", query=query, doc=doc, model="qwen3:4b",
        )

        assert result["answer"] == "Answer"
        assert strategy.run.call_count == 1
        assert mock_sleep.call_count == 0

    @patch("scripts.experiment_utils.Retriever")
    @patch("scripts.experiment_utils.time.sleep")
    def test_backoff_delay_doubles(
        self, mock_sleep, mock_retriever,
    ) -> None:
        strategy, chunker, embedder, query, doc = self._make_mocks()
        strategy.run.side_effect = ConnectionError("connection refused")

        generate_answer(
            strategy=strategy, chunker=chunker, embedder=embedder,
            retrieval_mode="hybrid", query=query, doc=doc, model="qwen3:4b",
        )

        # Verify exponential backoff: 2s, 4s, 8s
        delays = [c.args[0] for c in mock_sleep.call_args_list]
        assert delays == [2.0, 4.0, 8.0]


# ---------------------------------------------------------------------------
# score_answer retry
# ---------------------------------------------------------------------------

class TestScoreAnswerRetry:
    """Tests for retry logic in score_answer()."""

    @patch("scripts.experiment_utils.time.sleep")
    def test_retries_on_transient_then_succeeds(self, mock_sleep) -> None:
        scorer = MagicMock()
        scorer.score.side_effect = [
            Exception("503 Service Unavailable"),
            {"faithfulness": 4.0, "relevance": 4.5, "conciseness": 3.5},
        ]

        result = score_answer(scorer, "question", "context", "answer")

        assert result["faithfulness"] == 4.0
        assert result["relevance"] == 4.5
        assert scorer.score.call_count == 2
        assert mock_sleep.call_count == 1

    @patch("scripts.experiment_utils.time.sleep")
    def test_permanent_error_returns_nan(self, mock_sleep) -> None:
        scorer = MagicMock()
        scorer.score.side_effect = ValueError("bad input")

        result = score_answer(scorer, "question", "context", "answer")

        import math
        assert math.isnan(result["faithfulness"])
        assert math.isnan(result["quality"])
        assert scorer.score.call_count == 1
        assert mock_sleep.call_count == 0

    @patch("scripts.experiment_utils.time.sleep")
    def test_exhausts_retries_returns_nan(self, mock_sleep) -> None:
        scorer = MagicMock()
        scorer.score.side_effect = Exception("rate limit exceeded")

        result = score_answer(scorer, "question", "context", "answer")

        import math
        assert math.isnan(result["quality"])
        assert scorer.score.call_count == MAX_RETRIES + 1

    @patch("scripts.experiment_utils.time.sleep")
    def test_no_retry_on_success(self, mock_sleep) -> None:
        scorer = MagicMock()
        scorer.score.return_value = {
            "faithfulness": 5.0, "relevance": 5.0, "conciseness": 5.0,
        }

        result = score_answer(scorer, "question", "context", "answer")

        assert result["quality"] == 5.0
        assert scorer.score.call_count == 1
        assert mock_sleep.call_count == 0


# ---------------------------------------------------------------------------
# ensure_model retry
# ---------------------------------------------------------------------------

class TestEnsureModelRetry:
    """Tests for retry logic in ensure_model()."""

    @patch("scripts.experiment_utils.time.sleep")
    def test_model_already_present_no_pull(self, mock_sleep) -> None:
        client = MagicMock()
        client.show.return_value = {"name": "qwen3:4b"}

        ensure_model(client, "qwen3:4b")

        client.show.assert_called_once_with("qwen3:4b")
        client.pull.assert_not_called()
        assert mock_sleep.call_count == 0

    @patch("scripts.experiment_utils.time.sleep")
    def test_pull_succeeds_first_try(self, mock_sleep) -> None:
        client = MagicMock()
        client.show.side_effect = Exception("not found")
        client.pull.return_value = [{"status": "success"}]

        ensure_model(client, "qwen3:4b")

        assert client.pull.call_count == 1
        assert mock_sleep.call_count == 0

    @patch("scripts.experiment_utils.time.sleep")
    def test_pull_retries_on_transient(self, mock_sleep) -> None:
        client = MagicMock()
        client.show.side_effect = Exception("not found")
        client.pull.side_effect = [
            ConnectionError("connection reset"),
            [{"status": "success"}],
        ]

        ensure_model(client, "qwen3:4b")

        assert client.pull.call_count == 2
        assert mock_sleep.call_count == 1

    @patch("scripts.experiment_utils.time.sleep")
    def test_pull_exhausts_retries_raises(self, mock_sleep) -> None:
        client = MagicMock()
        client.show.side_effect = Exception("not found")
        client.pull.side_effect = ConnectionError("connection refused")

        with pytest.raises(ConnectionError):
            ensure_model(client, "qwen3:4b")

        assert client.pull.call_count == MAX_RETRIES + 1

    @patch("scripts.experiment_utils.time.sleep")
    def test_pull_permanent_error_raises_immediately(self, mock_sleep) -> None:
        client = MagicMock()
        client.show.side_effect = Exception("not found")
        client.pull.side_effect = ValueError("invalid model name")

        with pytest.raises(ValueError):
            ensure_model(client, "qwen3:4b")

        assert client.pull.call_count == 1
        assert mock_sleep.call_count == 0
