"""Tests for scripts/experiment_utils.py — shared experiment infrastructure."""

from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# Ensure project root is on path
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


class TestComputeF1:
    """Tests for word-overlap F1 computation."""

    def test_exact_match(self):
        from experiment_utils import compute_f1
        assert compute_f1("Paris", "Paris") == 1.0

    def test_partial_overlap(self):
        from experiment_utils import compute_f1
        score = compute_f1("The capital is Paris France", "Paris")
        assert 0.0 < score < 1.0

    def test_no_overlap(self):
        from experiment_utils import compute_f1
        assert compute_f1("hello world", "foo bar") == 0.0

    def test_empty_prediction(self):
        from experiment_utils import compute_f1
        assert compute_f1("", "Paris") == 0.0

    def test_empty_gold(self):
        from experiment_utils import compute_f1
        assert compute_f1("Paris", "") == 0.0

    def test_case_insensitive(self):
        from experiment_utils import compute_f1
        assert compute_f1("PARIS", "paris") == 1.0


class TestExactMatch:
    """Tests for containment-based exact match."""

    def test_contained(self):
        from experiment_utils import exact_match
        assert exact_match("The capital is Paris", "Paris") is True

    def test_not_contained(self):
        from experiment_utils import exact_match
        assert exact_match("The capital is London", "Paris") is False

    def test_case_insensitive(self):
        from experiment_utils import exact_match
        assert exact_match("the capital is PARIS", "paris") is True

    def test_exact_equality(self):
        from experiment_utils import exact_match
        assert exact_match("Paris", "Paris") is True


class TestCheckpoint:
    """Tests for checkpoint load/save via CSV."""

    def test_load_checkpoint_no_file(self):
        from experiment_utils import load_checkpoint
        result = load_checkpoint(Path("/nonexistent/path.csv"))
        assert result == set()

    def test_load_checkpoint_with_data(self, tmp_path):
        from experiment_utils import load_checkpoint
        csv_path = tmp_path / "scores.csv"
        # Write a CSV with two completed configs
        rows = [
            {"strategy": "naive", "model": "qwen3:4b", "quality": 3.5},
            {"strategy": "naive", "model": "qwen3:4b", "quality": 4.0},
            {"strategy": "self_rag", "model": "qwen3:0.6b", "quality": 2.5},
        ]
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)

        completed = load_checkpoint(csv_path)
        assert ("naive", "qwen3:4b") in completed
        assert ("self_rag", "qwen3:0.6b") in completed
        assert len(completed) == 2

    def test_load_checkpoint_empty_file(self, tmp_path):
        from experiment_utils import load_checkpoint
        csv_path = tmp_path / "scores.csv"
        csv_path.write_text("")
        result = load_checkpoint(csv_path)
        assert result == set()


class TestAppendRows:
    """Tests for CSV row appending."""

    def test_append_creates_new_file(self, tmp_path):
        from experiment_utils import append_rows
        csv_path = tmp_path / "new.csv"
        rows = [
            {"a": 1, "b": "hello"},
            {"a": 2, "b": "world"},
        ]
        append_rows(csv_path, rows)
        df = pd.read_csv(csv_path)
        assert len(df) == 2
        assert list(df.columns) == ["a", "b"]

    def test_append_to_existing(self, tmp_path):
        from experiment_utils import append_rows
        csv_path = tmp_path / "existing.csv"
        # Write initial data
        pd.DataFrame([{"a": 1, "b": "first"}]).to_csv(csv_path, index=False)
        # Append more
        append_rows(csv_path, [{"a": 2, "b": "second"}])
        df = pd.read_csv(csv_path)
        assert len(df) == 2

    def test_append_empty_rows(self, tmp_path):
        from experiment_utils import append_rows
        csv_path = tmp_path / "empty.csv"
        append_rows(csv_path, [])
        assert not csv_path.exists()


class TestFormatDuration:
    """Tests for human-readable duration formatting."""

    def test_seconds(self):
        from experiment_utils import format_duration
        assert "s" in format_duration(45)

    def test_minutes(self):
        from experiment_utils import format_duration
        result = format_duration(125)
        assert "m" in result or "min" in result

    def test_hours(self):
        from experiment_utils import format_duration
        result = format_duration(3700)
        assert "h" in result or "hr" in result


class TestBuildScorer:
    """Tests for scorer construction."""

    @patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"})
    def test_build_google_scorer(self):
        from experiment_utils import build_scorer
        scorer = build_scorer("google:gemini-2.5-flash", max_cost=5.0)
        assert scorer is not None
        assert "google" in scorer.name or "gemini" in scorer.name

    def test_build_invalid_format(self):
        from experiment_utils import build_scorer
        with pytest.raises((ValueError, SystemExit)):
            build_scorer("invalid-no-colon", max_cost=5.0)


class TestEnsureModel:
    """Tests for Ollama model check/pull."""

    def test_model_already_available(self):
        from experiment_utils import ensure_model
        client = MagicMock()
        client.show.return_value = {"name": "qwen3:4b"}
        # Should not raise
        ensure_model(client, "qwen3:4b")
        client.show.assert_called_once_with("qwen3:4b")
        client.pull.assert_not_called()

    def test_model_needs_pull(self):
        from experiment_utils import ensure_model
        client = MagicMock()
        client.show.side_effect = Exception("model not found")
        client.pull.return_value = iter([{"status": "success"}])
        ensure_model(client, "qwen3:4b")
        client.pull.assert_called_once_with("qwen3:4b", stream=True)
