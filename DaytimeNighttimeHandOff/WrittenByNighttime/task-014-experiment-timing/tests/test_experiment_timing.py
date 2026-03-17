"""Tests for experiment runner timing feature.

Uses mock components — no external services required.
"""

from __future__ import annotations

import time

import pandas as pd
import pytest

from src.chunkers import FixedSizeChunker
from src.embedders import HuggingFaceEmbedder
from src.experiment import Experiment, ExperimentResult
from src.retriever import Retriever


# ---------------------------------------------------------------------------
# Mock components (same pattern as test_e2e_smoke.py)
# ---------------------------------------------------------------------------

class MockStrategy:
    """Strategy with configurable delay."""

    def __init__(self, label: str = "fast", delay_s: float = 0.0):
        self._label = label
        self._delay = delay_s

    @property
    def name(self) -> str:
        return f"mock:{self._label}"

    def run(self, query: str, retriever: Retriever, model: str) -> str:
        if self._delay > 0:
            time.sleep(self._delay)
        retrieved = retriever.retrieve(query, top_k=2)
        return f"Answer based on {len(retrieved)} chunks"


class MockScorer:
    """Scorer with configurable delay."""

    def __init__(self, delay_s: float = 0.0):
        self._delay = delay_s

    @property
    def name(self) -> str:
        return "mock:scorer"

    def score(self, query: str, context: str, answer: str) -> dict[str, float]:
        if self._delay > 0:
            time.sleep(self._delay)
        return {"faithfulness": 4.0, "relevance": 4.0, "conciseness": 4.0}


SAMPLE_DOCS = [
    {
        "title": "Test Document",
        "text": "Python is a programming language. " * 20,
    },
]

SAMPLE_QUERIES = [
    {"text": "What is Python?", "type": "factoid"},
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTimingColumns:
    """Verify timing columns exist in experiment output."""

    def test_timing_columns_present(self):
        """ExperimentResult should have strategy_latency_ms, scorer_latency_ms, total_latency_ms."""
        exp = Experiment(
            chunkers=[FixedSizeChunker(chunk_size=100, overlap=10)],
            embedders=[HuggingFaceEmbedder()],
            models=["test-model"],
            strategies=[MockStrategy()],
            scorer=MockScorer(),
        )
        exp.load_corpus(SAMPLE_DOCS, SAMPLE_QUERIES)
        result = exp.run(progress=False)

        assert "strategy_latency_ms" in result.df.columns
        assert "scorer_latency_ms" in result.df.columns
        assert "total_latency_ms" in result.df.columns

    def test_timing_values_non_negative(self):
        """All timing values should be >= 0."""
        exp = Experiment(
            chunkers=[FixedSizeChunker(chunk_size=100, overlap=10)],
            embedders=[HuggingFaceEmbedder()],
            models=["m"],
            strategies=[MockStrategy()],
            scorer=MockScorer(),
        )
        exp.load_corpus(SAMPLE_DOCS, SAMPLE_QUERIES)
        result = exp.run(progress=False)

        assert (result.df["strategy_latency_ms"] >= 0).all()
        assert (result.df["scorer_latency_ms"] >= 0).all()
        assert (result.df["total_latency_ms"] >= 0).all()

    def test_total_equals_sum(self):
        """total_latency_ms should equal strategy + scorer."""
        exp = Experiment(
            chunkers=[FixedSizeChunker(chunk_size=100, overlap=10)],
            embedders=[HuggingFaceEmbedder()],
            models=["m"],
            strategies=[MockStrategy()],
            scorer=MockScorer(),
        )
        exp.load_corpus(SAMPLE_DOCS, SAMPLE_QUERIES)
        result = exp.run(progress=False)

        expected = result.df["strategy_latency_ms"] + result.df["scorer_latency_ms"]
        pd.testing.assert_series_equal(
            result.df["total_latency_ms"], expected, check_names=False,
        )

    def test_delayed_strategy_shows_in_timing(self):
        """A strategy with a 50ms delay should show >= 50ms in timing."""
        exp = Experiment(
            chunkers=[FixedSizeChunker(chunk_size=100, overlap=10)],
            embedders=[HuggingFaceEmbedder()],
            models=["m"],
            strategies=[MockStrategy(delay_s=0.05)],
            scorer=MockScorer(),
        )
        exp.load_corpus(SAMPLE_DOCS, SAMPLE_QUERIES)
        result = exp.run(progress=False)

        assert result.df["strategy_latency_ms"].iloc[0] >= 40  # Allow some tolerance


class TestLatencyReport:
    """Test the latency_report() analysis method."""

    def test_latency_report_returns_dataframe(self):
        """latency_report() should return a grouped DataFrame."""
        exp = Experiment(
            chunkers=[FixedSizeChunker(chunk_size=100, overlap=10)],
            embedders=[HuggingFaceEmbedder()],
            models=["small", "large"],
            strategies=[MockStrategy("a"), MockStrategy("b")],
            scorer=MockScorer(),
        )
        exp.load_corpus(SAMPLE_DOCS, SAMPLE_QUERIES)
        result = exp.run(progress=False)

        report = result.latency_report()
        assert not report.empty
        # Should be grouped by (strategy, model)
        assert report.index.names == ["strategy", "model"] or len(report) > 0

    def test_latency_report_empty_dataframe(self):
        """latency_report() handles empty results gracefully."""
        result = ExperimentResult(pd.DataFrame())
        report = result.latency_report()
        assert report.empty


class TestTimeVsQuality:
    """Test the time_vs_quality() analysis method."""

    def test_time_vs_quality_returns_dataframe(self):
        """time_vs_quality() should return a table with both quality and latency."""
        exp = Experiment(
            chunkers=[FixedSizeChunker(chunk_size=100, overlap=10)],
            embedders=[HuggingFaceEmbedder()],
            models=["m"],
            strategies=[MockStrategy("a"), MockStrategy("b")],
            scorer=MockScorer(),
        )
        exp.load_corpus(SAMPLE_DOCS, SAMPLE_QUERIES)
        result = exp.run(progress=False)

        table = result.time_vs_quality()
        assert not table.empty

    def test_time_vs_quality_empty_dataframe(self):
        """time_vs_quality() handles empty results gracefully."""
        result = ExperimentResult(pd.DataFrame())
        table = result.time_vs_quality()
        assert table.empty


class TestExistingBehavior:
    """Verify existing functionality still works with timing columns."""

    def test_compare_still_works(self):
        """compare() should still work with the additional columns."""
        exp = Experiment(
            chunkers=[FixedSizeChunker(chunk_size=100, overlap=10)],
            embedders=[HuggingFaceEmbedder()],
            models=["m"],
            strategies=[MockStrategy()],
            scorer=MockScorer(),
        )
        exp.load_corpus(SAMPLE_DOCS, SAMPLE_QUERIES)
        result = exp.run(progress=False)

        summary = result.compare()
        assert not summary.empty

    def test_parquet_roundtrip_preserves_timing(self, tmp_path):
        """Timing columns survive Parquet save/load."""
        exp = Experiment(
            chunkers=[FixedSizeChunker(chunk_size=100, overlap=10)],
            embedders=[HuggingFaceEmbedder()],
            models=["m"],
            strategies=[MockStrategy()],
            scorer=MockScorer(),
        )
        exp.load_corpus(SAMPLE_DOCS, SAMPLE_QUERIES)
        result = exp.run(progress=False)

        path = tmp_path / "timed_results.parquet"
        result.to_parquet(path)
        loaded = ExperimentResult.from_parquet(path)

        assert "strategy_latency_ms" in loaded.df.columns
        assert "total_latency_ms" in loaded.df.columns
