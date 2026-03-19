"""Tests for Experiment 0 utility functions and scorer initialization.

All API calls are mocked — no real LLM or API access needed.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_experiment_0 import (
    compute_f1,
    exact_match,
    _safe_scorer_name,
    score_all_answers,
    JUDGE_CONFIGS,
)


class TestComputeF1:
    """Tests for word-level F1 computation."""

    def test_high_f1_with_overlap(self) -> None:
        """'the capital is Paris' vs 'Paris' should have high F1."""
        score = compute_f1("the capital is Paris", "Paris")
        # 'paris' is in both sets. Precision = 1/4, Recall = 1/1
        # F1 = 2 * (0.25 * 1.0) / (0.25 + 1.0) = 0.4
        assert score > 0.3

    def test_no_overlap_returns_zero(self) -> None:
        """Completely different strings should give 0.0."""
        score = compute_f1("hello world", "goodbye universe")
        assert score == 0.0

    def test_empty_strings(self) -> None:
        """Empty strings should return 0.0 without error."""
        assert compute_f1("", "some text") == 0.0
        assert compute_f1("some text", "") == 0.0


class TestExactMatch:
    """Tests for case-insensitive containment check."""

    def test_contains_gold(self) -> None:
        """Gold substring in prediction should return True."""
        assert exact_match("The capital is Paris", "Paris") is True

    def test_case_insensitive(self) -> None:
        """Different casing should still match."""
        assert exact_match("the capital is PARIS", "paris") is True

    def test_no_match(self) -> None:
        """No containment should return False."""
        assert exact_match("Berlin is the capital", "Paris") is False


class TestSafeScorerName:
    """Tests for scorer name sanitization."""

    def test_google_gemini_flash(self) -> None:
        """google:gemini-2.5-flash should become google_gemini_2_5_flash."""
        assert _safe_scorer_name("google:gemini-2.5-flash") == "google_gemini_2_5_flash"

    def test_anthropic_model(self) -> None:
        """Anthropic model names with dashes and dots should be sanitized."""
        result = _safe_scorer_name("anthropic:claude-haiku-4-5-20251001")
        assert ":" not in result
        assert "-" not in result
        assert "." not in result


class TestScoreAllAnswersSkipsMissingKeys:
    """Tests that scorers with missing API keys are skipped gracefully."""

    def test_skips_missing_keys(self, tmp_path: Path) -> None:
        """Mock LLMScorer to raise on anthropic init, verify only google scorers produce columns."""
        from src.scorers.llm import ScorerError

        answers = [{
            "example_id": 0,
            "question": "What is the capital of France?",
            "gold_answer": "Paris",
            "rag_answer": "The capital of France is Paris.",
            "doc_text": "France is a country in Europe. Its capital is Paris.",
        }]

        def mock_constructor(**kwargs: Any) -> MagicMock:
            """Mock LLMScorer that fails for anthropic providers."""
            provider = kwargs.get("provider", "google")
            model = kwargs.get("model", "")
            if provider == "anthropic":
                raise ScorerError(f"ANTHROPIC_API_KEY not set for {model}")
            scorer = MagicMock()
            scorer.name = f"{provider}:{model}"
            scorer.score.return_value = {
                "faithfulness": 4.0,
                "relevance": 5.0,
                "conciseness": 3.0,
            }
            return scorer

        # Patch at the source module — score_all_answers does a local import
        with patch("src.scorers.llm.LLMScorer", side_effect=mock_constructor):
            df = score_all_answers(answers, tmp_path)

        # Check that google scorer columns exist
        google_cols = [c for c in df.columns if c.startswith("google_")]
        assert len(google_cols) > 0, "Expected google scorer columns"

        # Check that anthropic scorer columns do NOT exist
        anthropic_cols = [c for c in df.columns if c.startswith("anthropic_")]
        assert len(anthropic_cols) == 0, f"Did not expect anthropic columns, got: {anthropic_cols}"
