"""Tests for ExperimentResult analysis and visualization methods.

Uses synthetic data fixtures — no actual experiments are run.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.experiment import ExperimentResult


@pytest.fixture
def sample_results() -> ExperimentResult:
    """Build a synthetic ExperimentResult with known values."""
    rows = []
    for strategy in ["naive", "self_rag"]:
        for model in ["qwen3:0.6b", "qwen3:4b"]:
            for i in range(5):
                rows.append({
                    "doc_title": f"doc_{i}",
                    "query_text": f"query_{i}",
                    "query_type": "lookup",
                    "chunker": "semantic:mxbai-embed-large",
                    "embedder": "ollama:mxbai-embed-large",
                    "model": model,
                    "strategy": strategy,
                    "answer": "test answer",
                    "faithfulness": 3.0 + (0.5 if strategy == "self_rag" else 0),
                    "relevance": 3.5,
                    "conciseness": 4.0,
                    "quality": 3.5 + (0.5 if strategy == "self_rag" else 0),
                    "query_length": 10,
                    "num_named_entities": 1,
                    "doc_length": 500,
                    "doc_vocab_entropy": 8.5,
                    "mean_retrieval_score": 0.7,
                    "var_retrieval_score": 0.02,
                    "timestamp": "2026-03-16T12:00:00",
                })
    return ExperimentResult(pd.DataFrame(rows))


class TestCompare:
    """Test compare() method."""

    def test_compare_returns_dataframe(self, sample_results: ExperimentResult) -> None:
        """compare() should return DataFrame with expected groups."""
        summary = sample_results.compare()
        assert isinstance(summary, pd.DataFrame)
        assert "mean" in summary.columns
        # 2 strategies x 2 models = 4 groups
        assert len(summary) == 4


class TestCompareStrategies:
    """Test compare_strategies() method."""

    def test_strategy_ranking(self, sample_results: ExperimentResult) -> None:
        """self_rag should rank higher than naive in the fixture."""
        result = sample_results.compare_strategies()
        assert isinstance(result, pd.DataFrame)
        # self_rag has quality 4.0, naive has 3.5 in the fixture
        assert result.index[0] == "self_rag"


class TestCompareModels:
    """Test compare_models() method."""

    def test_model_comparison(self, sample_results: ExperimentResult) -> None:
        """compare_models() should return DataFrame grouped by model."""
        result = sample_results.compare_models()
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2


class TestPivot:
    """Test pivot() method."""

    def test_pivot_shape(self, sample_results: ExperimentResult) -> None:
        """Pivot should have correct shape."""
        pivot = sample_results.pivot("strategy", "model")
        assert pivot.shape == (2, 2)  # 2 strategies x 2 models


class TestPerQuery:
    """Test per_query() method."""

    def test_spread_calculation(self, sample_results: ExperimentResult) -> None:
        """per_query() should calculate spread correctly."""
        result = sample_results.per_query()
        assert isinstance(result, pd.DataFrame)
        assert "spread" in result.columns
        # Each query has naive (3.5) and self_rag (4.0) = spread 0.5
        assert all(result["spread"] == 0.5)


class TestStrategyVsSize:
    """Test strategy_vs_size() method."""

    def test_pivot_shape(self, sample_results: ExperimentResult) -> None:
        """strategy_vs_size should return a pivot of correct shape."""
        result = sample_results.strategy_vs_size()
        assert result.shape == (2, 2)


class TestParquetRoundtrip:
    """Test parquet save/load."""

    def test_roundtrip(self, sample_results: ExperimentResult, tmp_path: pytest.TempPathFactory) -> None:
        """Save to parquet, load back, verify equality."""
        path = tmp_path / "test.parquet"
        sample_results.to_parquet(path)
        loaded = ExperimentResult.from_parquet(path)
        pd.testing.assert_frame_equal(sample_results.df, loaded.df)


class TestCsvExport:
    """Test CSV export."""

    def test_csv_created(self, sample_results: ExperimentResult, tmp_path: pytest.TempPathFactory) -> None:
        """CSV file should be created and readable."""
        path = tmp_path / "test.csv"
        sample_results.to_csv(path)
        assert path.exists()
        loaded = pd.read_csv(path)
        assert len(loaded) == len(sample_results.df)


class TestMerge:
    """Test merge() method."""

    def test_merge_combines(self, sample_results: ExperimentResult) -> None:
        """Merging two results should double the rows."""
        merged = sample_results.merge(sample_results)
        assert len(merged.df) == 2 * len(sample_results.df)


class TestHeatmap:
    """Test heatmap generation."""

    def test_heatmap_saves_to_file(self, sample_results: ExperimentResult, tmp_path: pytest.TempPathFactory) -> None:
        """Heatmap should save a file when save_path is provided."""
        path = tmp_path / "heatmap.png"
        sample_results.heatmap("strategy", "model", save_path=path)
        assert path.exists()
        # Verify file is non-empty (actual PNG content)
        assert path.stat().st_size > 0


class TestSummary:
    """Test summary() method."""

    def test_summary_returns_dataframe(self, sample_results: ExperimentResult) -> None:
        """summary() should return a multi-level DataFrame."""
        result = sample_results.summary()
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
