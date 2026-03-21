"""Tests for scripts/run_experiment_1.py — Experiment 1: Strategy × Model Size."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


class TestExperiment1Matrix:
    """Tests for experiment matrix definition."""

    def test_all_strategies_defined(self):
        from run_experiment_1 import ALL_STRATEGIES
        assert len(ALL_STRATEGIES) == 5
        expected = {"naive", "self_rag", "multi_query", "corrective", "adaptive"}
        assert set(ALL_STRATEGIES) == expected

    def test_all_models_defined(self):
        from run_experiment_1 import ALL_MODELS
        assert len(ALL_MODELS) == 6
        assert "qwen3:0.6b" in ALL_MODELS
        assert "qwen3:8b" in ALL_MODELS
        assert "gemma3:1b" in ALL_MODELS
        assert "gemma3:4b" in ALL_MODELS

    def test_total_configs(self):
        from run_experiment_1 import ALL_STRATEGIES, ALL_MODELS
        assert len(ALL_STRATEGIES) * len(ALL_MODELS) == 30


class TestExperiment1Report:
    """Tests for report generation from synthetic data."""

    def _make_synthetic_df(self) -> pd.DataFrame:
        """Create a synthetic DataFrame matching Exp 1 output structure."""
        rows = []
        strategies = ["naive", "self_rag", "multi_query"]
        models = ["qwen3:0.6b", "qwen3:4b"]
        for strat in strategies:
            for model in models:
                for q in range(5):
                    quality = 3.0 + (0.5 if strat != "naive" else 0) + (0.3 if "4b" in model else 0)
                    rows.append({
                        "strategy": strat,
                        "model": model,
                        "chunker": "recursive:500/100",
                        "embedder": "ollama:mxbai-embed-large",
                        "query_text": f"question {q}",
                        "quality": quality,
                        "faithfulness": quality - 0.1,
                        "relevance": quality + 0.1,
                        "conciseness": quality,
                        "total_latency_ms": 1000 + (500 if "4b" in model else 0),
                        "strategy_latency_ms": 800 + (400 if "4b" in model else 0),
                        "scorer_latency_ms": 200,
                        "gold_f1": 0.6,
                        "gold_exact_match": True,
                    })
        return pd.DataFrame(rows)

    def test_report_generation(self):
        from run_experiment_1 import generate_report
        df = self._make_synthetic_df()
        report = generate_report(df)
        assert "Experiment 1" in report
        assert "Strategy" in report
        assert "Model" in report

    def test_report_contains_heatmap(self):
        from run_experiment_1 import generate_report
        df = self._make_synthetic_df()
        report = generate_report(df)
        # Should contain strategy and model names in a table
        assert "naive" in report
        assert "qwen3" in report

    def test_report_handles_empty_df(self):
        from run_experiment_1 import generate_report
        df = pd.DataFrame()
        report = generate_report(df)
        assert isinstance(report, str)


class TestExperiment1Resume:
    """Tests for resume functionality."""

    def test_resume_skips_completed(self, tmp_path):
        from experiment_utils import load_checkpoint
        csv_path = tmp_path / "raw_scores.csv"
        # Write some completed configs
        rows = []
        for q in range(3):
            rows.append({"strategy": "naive", "model": "qwen3:4b", "quality": 3.5})
        pd.DataFrame(rows).to_csv(csv_path, index=False)

        completed = load_checkpoint(csv_path)
        assert ("naive", "qwen3:4b") in completed

    def test_resume_fresh_start(self, tmp_path):
        from experiment_utils import load_checkpoint
        csv_path = tmp_path / "nonexistent.csv"
        completed = load_checkpoint(csv_path)
        assert len(completed) == 0


class TestModelFiltering:
    """Tests for --models and --strategies CLI filtering."""

    def test_filter_models(self):
        from run_experiment_1 import ALL_MODELS
        subset = [m for m in ALL_MODELS if "qwen3" in m]
        assert len(subset) == 4  # qwen3 only, no gemma

    def test_filter_strategies(self):
        from run_experiment_1 import ALL_STRATEGIES
        subset = [s for s in ALL_STRATEGIES if s in ("naive", "self_rag")]
        assert len(subset) == 2
