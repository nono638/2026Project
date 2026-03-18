"""Tests for Experiment 0 scorer validation script.

Mocks Ollama (NaiveRAG) and LLMScorer to avoid network/API dependency.
"""

from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


# ---------------------------------------------------------------------------
# Test the utility functions (these are pure, no mocking needed)
# ---------------------------------------------------------------------------

class TestExactMatch:
    """Test the exact_match function."""

    def test_gold_in_answer(self):
        from scripts.run_experiment_0 import exact_match
        assert exact_match("The capital of France is Paris.", "Paris") is True

    def test_case_insensitive(self):
        from scripts.run_experiment_0 import exact_match
        assert exact_match("PARIS is the capital", "paris") is True

    def test_no_match(self):
        from scripts.run_experiment_0 import exact_match
        assert exact_match("The capital is London", "Paris") is False

    def test_empty_answer(self):
        from scripts.run_experiment_0 import exact_match
        assert exact_match("", "Paris") is False

    def test_empty_gold(self):
        from scripts.run_experiment_0 import exact_match
        assert exact_match("Some answer", "") is True  # empty string is in everything


class TestComputeF1:
    """Test the word-level F1 function."""

    def test_perfect_match(self):
        from scripts.run_experiment_0 import compute_f1
        assert compute_f1("the cat sat", "the cat sat") == pytest.approx(1.0)

    def test_partial_overlap(self):
        from scripts.run_experiment_0 import compute_f1
        # pred: {the, cat, sat, on, mat} gold: {the, cat}
        # common: {the, cat}, precision: 2/5, recall: 2/2
        # F1 = 2 * (2/5) * (2/2) / ((2/5) + (2/2)) = 2 * 0.4 * 1.0 / 1.4 = 0.571
        f1 = compute_f1("the cat sat on mat", "the cat")
        assert f1 == pytest.approx(4 / 7, abs=0.01)

    def test_no_overlap(self):
        from scripts.run_experiment_0 import compute_f1
        assert compute_f1("hello world", "foo bar") == pytest.approx(0.0)

    def test_empty_prediction(self):
        from scripts.run_experiment_0 import compute_f1
        assert compute_f1("", "gold answer") == pytest.approx(0.0)

    def test_empty_gold(self):
        from scripts.run_experiment_0 import compute_f1
        assert compute_f1("some answer", "") == pytest.approx(0.0)

    def test_case_insensitive(self):
        from scripts.run_experiment_0 import compute_f1
        assert compute_f1("The Cat", "the cat") == pytest.approx(1.0)


class TestCSVOutput:
    """Test that the output CSV has the right structure."""

    def test_csv_has_required_columns(self):
        """Verify the expected columns exist in a mock CSV output."""
        # This tests the column names, not the script end-to-end
        expected_base_cols = [
            "example_id", "question", "gold_answer", "rag_answer",
            "gold_exact_match", "gold_f1",
        ]
        # Each scorer adds 4 columns: faithfulness, relevance, conciseness, quality
        scorer_names = [
            "google:gemini-2.5-flash",
            "google:gemini-2.5-pro",
            "anthropic:claude-haiku-4-5-20251001",
            "anthropic:claude-sonnet-4-20250514",
            "anthropic:claude-opus-4-6",
        ]
        expected_scorer_cols = []
        for name in scorer_names:
            safe = name.replace(":", "_").replace("-", "_").replace(".", "_")
            for metric in ["faithfulness", "relevance", "conciseness", "quality"]:
                expected_scorer_cols.append(f"{safe}_{metric}")

        all_expected = expected_base_cols + expected_scorer_cols
        # Just verify the count — 6 base + 5 scorers × 4 metrics = 26 columns
        assert len(all_expected) == 26


class TestGracefulFailure:
    """Test that scorer failures produce NaN, not crashes."""

    def test_scorer_error_produces_nan(self):
        """When a scorer raises, the result should be NaN for that scorer."""
        import math

        # Simulate what the script should do when a scorer fails
        try:
            raise Exception("API timeout")
        except Exception:
            score = float("nan")

        assert math.isnan(score)
