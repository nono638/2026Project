"""Pre-written tests for task-052: multi-judge scoring helper.

These tests cover the new `score_answer_multi(scorers, query, context, answer,
existing_row=None)` helper to be added to `scripts/experiment_utils.py`.

The helper must:
- Score the answer with every scorer in the list.
- Prefix returned keys with `<safe_name>_<metric>` (no bare metric names).
- Add a `consensus_quality` key = NaN-safe mean of each judge's `_quality`.
- If `existing_row` is provided AND has non-NaN `<safe_name>_quality`, skip that
  judge (resume support); copy existing values forward.
- On per-judge failure, return NaN values for that judge and continue with the rest.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiment_utils import score_answer_multi  # noqa: E402
from scripts.run_experiment_0 import _safe_scorer_name  # noqa: E402


def _mock_scorer(provider: str, model: str, score_return: dict | Exception) -> MagicMock:
    """Build a mock LLMScorer that returns `score_return` (or raises if Exception)."""
    scorer = MagicMock()
    scorer.name = f"{provider}:{model}"
    scorer._provider = provider
    scorer._model = model
    if isinstance(score_return, Exception):
        scorer.score.side_effect = score_return
    else:
        scorer.score.return_value = score_return
    return scorer


class TestTwoJudgeHappyPath:
    """Both judges return scores; consensus is the mean of their quality values."""

    def test_two_judges_happy_path(self) -> None:
        haiku = _mock_scorer(
            "anthropic", "claude-haiku-4-5-20251001",
            {"faithfulness": 4.0, "relevance": 5.0, "conciseness": 3.0},
        )
        gpt_mini = _mock_scorer(
            "openai", "gpt-5.4-mini",
            {"faithfulness": 5.0, "relevance": 4.0, "conciseness": 4.0},
        )
        result = score_answer_multi(
            scorers=[haiku, gpt_mini],
            query="q", context="c", answer="a",
        )
        haiku_safe = _safe_scorer_name("anthropic:claude-haiku-4-5-20251001")
        gpt_safe = _safe_scorer_name("openai:gpt-5.4-mini")
        # Per-judge quality columns
        assert result[f"{haiku_safe}_quality"] == pytest.approx((4 + 5 + 3) / 3)
        assert result[f"{gpt_safe}_quality"] == pytest.approx((5 + 4 + 4) / 3)
        # Consensus = mean of the two judges' quality
        haiku_q = (4 + 5 + 3) / 3
        gpt_q = (5 + 4 + 4) / 3
        assert result["consensus_quality"] == pytest.approx((haiku_q + gpt_q) / 2)


class TestSafeNamePrefix:
    """All metric keys use `<safe_name>_<metric>` — no bare `quality`/`faithfulness` keys."""

    def test_no_bare_metric_keys(self) -> None:
        haiku = _mock_scorer(
            "anthropic", "claude-haiku-4-5-20251001",
            {"faithfulness": 4.0, "relevance": 5.0, "conciseness": 3.0},
        )
        result = score_answer_multi(scorers=[haiku], query="q", context="c", answer="a")
        for bare in ("faithfulness", "relevance", "conciseness", "quality"):
            assert bare not in result, (
                f"Bare metric key '{bare}' should not appear (only consensus_quality is unprefixed). "
                f"Got keys: {list(result.keys())}"
            )
        assert "consensus_quality" in result


class TestOneJudgeNaN:
    """If one judge fails, consensus uses the surviving judge's quality."""

    def test_one_judge_returns_nan(self) -> None:
        haiku = _mock_scorer(
            "anthropic", "claude-haiku-4-5-20251001",
            {"faithfulness": 4.0, "relevance": 4.0, "conciseness": 4.0},
        )
        gpt_mini = _mock_scorer(
            "openai", "gpt-5.4-mini",
            RuntimeError("API failure"),
        )
        result = score_answer_multi(
            scorers=[haiku, gpt_mini],
            query="q", context="c", answer="a",
        )
        haiku_safe = _safe_scorer_name("anthropic:claude-haiku-4-5-20251001")
        gpt_safe = _safe_scorer_name("openai:gpt-5.4-mini")
        # Haiku succeeds
        assert result[f"{haiku_safe}_quality"] == pytest.approx(4.0)
        # GPT mini fails -> NaN
        assert math.isnan(result[f"{gpt_safe}_quality"])
        # Consensus uses only the surviving judge
        assert result["consensus_quality"] == pytest.approx(4.0)


class TestAllJudgesNaN:
    """If all judges fail, consensus is NaN."""

    def test_all_judges_fail(self) -> None:
        haiku = _mock_scorer(
            "anthropic", "claude-haiku-4-5-20251001",
            RuntimeError("api 1"),
        )
        gpt_mini = _mock_scorer(
            "openai", "gpt-5.4-mini",
            RuntimeError("api 2"),
        )
        result = score_answer_multi(
            scorers=[haiku, gpt_mini],
            query="q", context="c", answer="a",
        )
        assert math.isnan(result["consensus_quality"])


class TestResumeSkip:
    """If existing_row has non-NaN `<safe>_quality` for a judge, that judge is skipped."""

    def test_existing_judge_is_skipped(self) -> None:
        haiku = _mock_scorer(
            "anthropic", "claude-haiku-4-5-20251001",
            {"faithfulness": 4.0, "relevance": 4.0, "conciseness": 4.0},
        )
        gpt_mini = _mock_scorer(
            "openai", "gpt-5.4-mini",
            {"faithfulness": 5.0, "relevance": 5.0, "conciseness": 5.0},
        )
        haiku_safe = _safe_scorer_name("anthropic:claude-haiku-4-5-20251001")
        # existing_row has Haiku's scores already
        existing = {
            f"{haiku_safe}_faithfulness": 3.0,
            f"{haiku_safe}_relevance": 3.0,
            f"{haiku_safe}_conciseness": 3.0,
            f"{haiku_safe}_quality": 3.0,
            f"{haiku_safe}_scorer_latency_ms": 12.5,
        }
        result = score_answer_multi(
            scorers=[haiku, gpt_mini],
            query="q", context="c", answer="a",
            existing_row=existing,
        )
        # Haiku's mock should NOT have been called
        haiku.score.assert_not_called()
        # Existing Haiku values are preserved
        assert result[f"{haiku_safe}_quality"] == pytest.approx(3.0)
        # GPT-mini was scored fresh
        gpt_safe = _safe_scorer_name("openai:gpt-5.4-mini")
        assert result[f"{gpt_safe}_quality"] == pytest.approx(5.0)

    def test_existing_nan_does_not_skip(self) -> None:
        """If `<safe>_quality` is present but NaN, the judge is re-scored."""
        haiku = _mock_scorer(
            "anthropic", "claude-haiku-4-5-20251001",
            {"faithfulness": 4.0, "relevance": 4.0, "conciseness": 4.0},
        )
        haiku_safe = _safe_scorer_name("anthropic:claude-haiku-4-5-20251001")
        existing = {f"{haiku_safe}_quality": float("nan")}
        result = score_answer_multi(
            scorers=[haiku],
            query="q", context="c", answer="a",
            existing_row=existing,
        )
        haiku.score.assert_called_once()
        assert result[f"{haiku_safe}_quality"] == pytest.approx(4.0)
