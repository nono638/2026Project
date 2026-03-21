"""Tests for constraint-aware analysis API on ExperimentResult."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.experiment import ExperimentResult


def _make_test_df() -> pd.DataFrame:
    """Create a synthetic experiment DataFrame with known values.

    3 strategies × 2 models × 5 queries = 30 rows.
    Quality and latency are deterministic for easy assertions.
    """
    rows = []
    configs = [
        # (strategy, model, base_quality, base_latency)
        ("naive", "qwen3:0.6b", 2.0, 500),
        ("naive", "qwen3:4b", 3.0, 1500),
        ("self_rag", "qwen3:0.6b", 3.5, 800),
        ("self_rag", "qwen3:4b", 4.0, 2000),
        ("multi_query", "qwen3:0.6b", 3.0, 1000),
        ("multi_query", "qwen3:4b", 4.5, 3000),
    ]
    for strategy, model, base_q, base_lat in configs:
        for q in range(5):
            rows.append({
                "chunker": "recursive:500/100",
                "embedder": "ollama:mxbai-embed-large",
                "strategy": strategy,
                "model": model,
                "query_text": f"question {q}",
                "quality": base_q + (q * 0.1),  # slight variation per query
                "faithfulness": base_q,
                "relevance": base_q + 0.2,
                "conciseness": base_q - 0.1,
                "total_latency_ms": base_lat + (q * 10),
                "strategy_latency_ms": base_lat * 0.8,
                "scorer_latency_ms": base_lat * 0.2,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def result():
    """ExperimentResult with known synthetic data."""
    return ExperimentResult(_make_test_df())


@pytest.fixture
def empty_result():
    """ExperimentResult with empty DataFrame."""
    return ExperimentResult(pd.DataFrame())


# --- filter() ---

class TestFilter:
    def test_single_numeric_constraint(self, result):
        filtered = result.filter({"quality": ">3.0"})
        assert len(filtered.df) > 0
        assert (filtered.df["quality"] > 3.0).all()

    def test_multiple_constraints(self, result):
        filtered = result.filter({"quality": ">3.0", "total_latency_ms": "<2000"})
        assert len(filtered.df) > 0
        assert (filtered.df["quality"] > 3.0).all()
        assert (filtered.df["total_latency_ms"] < 2000).all()

    def test_string_exact_match(self, result):
        filtered = result.filter({"model": "qwen3:4b"})
        assert len(filtered.df) > 0
        assert (filtered.df["model"] == "qwen3:4b").all()

    def test_empty_result_on_impossible_constraint(self, result):
        filtered = result.filter({"quality": ">100.0"})
        assert len(filtered.df) == 0

    def test_empty_input(self, empty_result):
        filtered = empty_result.filter({"quality": ">3.0"})
        assert len(filtered.df) == 0

    def test_returns_experiment_result(self, result):
        filtered = result.filter({"quality": ">3.0"})
        assert isinstance(filtered, ExperimentResult)

    def test_missing_column_raises(self, result):
        with pytest.raises(KeyError):
            result.filter({"nonexistent_col": ">3.0"})


# --- best_config() ---

class TestBestConfig:
    def test_returns_dict(self, result):
        best = result.best_config("quality")
        assert isinstance(best, dict)
        assert "chunker" in best
        assert "embedder" in best
        assert "strategy" in best
        assert "model" in best

    def test_maximize_quality(self, result):
        best = result.best_config("quality", maximize=True)
        # multi_query + qwen3:4b has highest base quality (4.5)
        assert best["strategy"] == "multi_query"
        assert best["model"] == "qwen3:4b"

    def test_minimize_latency(self, result):
        best = result.best_config("total_latency_ms", maximize=False)
        # naive + qwen3:0.6b has lowest latency (500 base)
        assert best["strategy"] == "naive"
        assert best["model"] == "qwen3:0.6b"

    def test_with_constraints(self, result):
        best = result.best_config(
            "quality", maximize=True,
            constraints={"total_latency_ms": "<1500"},
        )
        # Best quality under 1500ms latency — self_rag + qwen3:0.6b (3.5 quality, 800ms)
        # or multi_query + qwen3:0.6b (3.0 quality, 1000ms)
        assert best["strategy"] == "self_rag"
        assert best["model"] == "qwen3:0.6b"

    def test_no_match_raises(self, result):
        with pytest.raises(ValueError):
            result.best_config("quality", constraints={"quality": ">100.0"})

    def test_empty_df_raises(self, empty_result):
        with pytest.raises((ValueError, KeyError)):
            empty_result.best_config("quality")

    def test_has_metric_value(self, result):
        best = result.best_config("quality")
        # Should include the mean metric value
        metric_key = [k for k in best if "quality" in k.lower() and k not in ("chunker", "embedder", "strategy", "model")]
        assert len(metric_key) > 0


# --- configs_above() / configs_below() ---

class TestConfigsAboveBelow:
    def test_configs_above(self, result):
        above = result.configs_above("quality", 3.5)
        strategies = above.df["strategy"].unique()
        # Only configs with mean quality >= 3.5
        for strat in strategies:
            config_rows = above.df[above.df["strategy"] == strat]
            for model in config_rows["model"].unique():
                mean_q = result.df[
                    (result.df["strategy"] == strat) & (result.df["model"] == model)
                ]["quality"].mean()
                assert mean_q >= 3.5

    def test_configs_below(self, result):
        below = result.configs_below("total_latency_ms", 1000)
        # Only configs with mean latency <= 1000
        assert len(below.df) > 0

    def test_configs_above_empty(self, result):
        above = result.configs_above("quality", 100.0)
        assert len(above.df) == 0

    def test_returns_experiment_result(self, result):
        above = result.configs_above("quality", 3.0)
        assert isinstance(above, ExperimentResult)


# --- budget_analysis() ---

class TestBudgetAnalysis:
    def test_within_budget(self, result):
        budget_df = result.budget_analysis("quality", "total_latency_ms", budget=1500)
        assert isinstance(budget_df, pd.DataFrame)
        if len(budget_df) > 0:
            assert "strategy" in budget_df.columns
            assert "model" in budget_df.columns

    def test_no_configs_meet_budget(self, result):
        budget_df = result.budget_analysis("quality", "total_latency_ms", budget=1)
        assert len(budget_df) == 0

    def test_sorted_by_quality(self, result):
        budget_df = result.budget_analysis("quality", "total_latency_ms", budget=5000)
        if len(budget_df) > 1:
            quality_col = [c for c in budget_df.columns if "quality" in c][0]
            values = budget_df[quality_col].tolist()
            assert values == sorted(values, reverse=True)


# --- pareto_front() ---

class TestParetoFront:
    def test_pareto_configs_are_non_dominated(self, result):
        pareto = result.pareto_front("quality", "total_latency_ms")
        assert isinstance(pareto, pd.DataFrame)
        assert len(pareto) > 0
        # Each point on the Pareto front should not be dominated by another
        quality_col = [c for c in pareto.columns if "quality" in c][0]
        cost_col = [c for c in pareto.columns if "latency" in c][0]
        for i, row in pareto.iterrows():
            for j, other in pareto.iterrows():
                if i == j:
                    continue
                # No other point should be strictly better on BOTH metrics
                assert not (
                    other[quality_col] >= row[quality_col]
                    and other[cost_col] <= row[cost_col]
                    and (other[quality_col] > row[quality_col] or other[cost_col] < row[cost_col])
                )

    def test_single_config(self):
        df = pd.DataFrame([{
            "chunker": "recursive:500/100",
            "embedder": "ollama:mxbai-embed-large",
            "strategy": "naive",
            "model": "qwen3:4b",
            "quality": 3.5,
            "total_latency_ms": 1000,
        }])
        result = ExperimentResult(df)
        pareto = result.pareto_front("quality", "total_latency_ms")
        assert len(pareto) == 1

    def test_empty_input(self, empty_result):
        pareto = empty_result.pareto_front("quality", "total_latency_ms")
        assert len(pareto) == 0


# --- rank() ---

class TestRank:
    def test_rank_descending(self, result):
        ranked = result.rank("quality", ascending=False)
        assert isinstance(ranked, pd.DataFrame)
        assert len(ranked) > 0
        mean_col = [c for c in ranked.columns if "mean" in c.lower()][0]
        values = ranked[mean_col].tolist()
        assert values == sorted(values, reverse=True)

    def test_rank_ascending(self, result):
        ranked = result.rank("total_latency_ms", ascending=True)
        mean_col = [c for c in ranked.columns if "mean" in c.lower()][0]
        values = ranked[mean_col].tolist()
        assert values == sorted(values)

    def test_rank_top_n(self, result):
        ranked = result.rank("quality", top_n=3)
        assert len(ranked) <= 3

    def test_rank_has_columns(self, result):
        ranked = result.rank("quality")
        assert "strategy" in ranked.columns
        assert "model" in ranked.columns

    def test_empty_input(self, empty_result):
        ranked = empty_result.rank("quality")
        assert len(ranked) == 0


# --- missing column ---

class TestMissingColumn:
    def test_filter_missing_col(self, result):
        with pytest.raises(KeyError):
            result.filter({"fake_metric": ">1"})

    def test_configs_above_missing_col(self, result):
        with pytest.raises(KeyError):
            result.configs_above("fake_metric", 1.0)

    def test_rank_missing_col(self, result):
        with pytest.raises(KeyError):
            result.rank("fake_metric")

    def test_best_config_missing_col(self, result):
        with pytest.raises(KeyError):
            result.best_config("fake_metric")
