"""Tests for scripts/run_experiment_2.py — Experiment 2: Chunking × Model Size."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


class TestExperiment2Matrix:
    """Tests for experiment matrix definition."""

    def test_all_chunkers_defined(self):
        from run_experiment_2 import ALL_CHUNKERS
        assert len(ALL_CHUNKERS) == 4
        expected = {"recursive", "fixed", "sentence", "semantic"}
        assert set(ALL_CHUNKERS) == expected

    def test_all_models_defined(self):
        from run_experiment_2 import ALL_MODELS
        assert len(ALL_MODELS) == 4
        # Qwen3 only for Exp 2
        for m in ALL_MODELS:
            assert "qwen3" in m

    def test_total_configs(self):
        from run_experiment_2 import ALL_CHUNKERS, ALL_MODELS
        assert len(ALL_CHUNKERS) * len(ALL_MODELS) == 16

    def test_strategy_is_naive_only(self):
        from run_experiment_2 import STRATEGY
        assert STRATEGY == "naive"


class TestChunkerInstantiation:
    """Tests that all 4 chunkers can be created."""

    def test_recursive_chunker(self):
        from src.chunkers import RecursiveChunker
        c = RecursiveChunker(500, 100)
        assert "recursive" in c.name.lower()

    def test_fixed_chunker(self):
        from src.chunkers import FixedSizeChunker
        c = FixedSizeChunker(500)
        assert "fixed" in c.name.lower()

    def test_sentence_chunker(self):
        from src.chunkers import SentenceChunker
        c = SentenceChunker()
        assert "sentence" in c.name.lower()

    def test_semantic_chunker(self):
        from src.chunkers import SemanticChunker
        c = SemanticChunker()
        assert "semantic" in c.name.lower()


class TestExperiment2Report:
    """Tests for report generation from synthetic data."""

    def _make_synthetic_df(self) -> pd.DataFrame:
        """Create a synthetic DataFrame matching Exp 2 output structure."""
        rows = []
        chunkers = ["recursive:500/100", "fixed:500", "sentence", "semantic:mxbai-embed-large"]
        models = ["qwen3:0.6b", "qwen3:4b"]
        for chunker in chunkers:
            for model in models:
                for q in range(5):
                    quality = 3.0 + (0.3 if "recursive" in chunker else 0) + (0.3 if "4b" in model else 0)
                    rows.append({
                        "chunker": chunker,
                        "model": model,
                        "strategy": "naive",
                        "embedder": "ollama:mxbai-embed-large",
                        "query_text": f"question {q}",
                        "quality": quality,
                        "faithfulness": quality - 0.1,
                        "relevance": quality + 0.1,
                        "conciseness": quality,
                        "total_latency_ms": 1000,
                        "strategy_latency_ms": 800,
                        "scorer_latency_ms": 200,
                        "gold_f1": 0.6,
                        "gold_exact_match": True,
                    })
        return pd.DataFrame(rows)

    def test_report_generation(self):
        from run_experiment_2 import generate_report
        df = self._make_synthetic_df()
        report = generate_report(df)
        assert "Experiment 2" in report
        assert "Chunker" in report or "chunker" in report

    def test_report_contains_chunker_names(self):
        from run_experiment_2 import generate_report
        df = self._make_synthetic_df()
        report = generate_report(df)
        assert "recursive" in report.lower()
        assert "fixed" in report.lower()

    def test_report_handles_empty_df(self):
        from run_experiment_2 import generate_report
        df = pd.DataFrame()
        report = generate_report(df)
        assert isinstance(report, str)


class TestExperiment2Filtering:
    """Tests for --chunkers and --models filtering."""

    def test_filter_chunkers(self):
        from run_experiment_2 import ALL_CHUNKERS
        subset = [c for c in ALL_CHUNKERS if c in ("recursive", "sentence")]
        assert len(subset) == 2

    def test_filter_models(self):
        from run_experiment_2 import ALL_MODELS
        subset = [m for m in ALL_MODELS if "4b" in m or "8b" in m]
        assert len(subset) == 2
