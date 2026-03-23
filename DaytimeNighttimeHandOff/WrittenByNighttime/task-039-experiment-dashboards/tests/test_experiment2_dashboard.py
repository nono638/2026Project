"""Tests for Experiment 2 dashboard generator."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pandas as pd
import pytest

# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


# ---------------------------------------------------------------------------
# Synthetic data fixture
# ---------------------------------------------------------------------------

def _make_exp2_df(n_per_config: int = 5) -> pd.DataFrame:
    """Create a synthetic Experiment 2 DataFrame.

    4 chunkers x 3 models x n_per_config queries.
    """
    import random
    random.seed(42)

    chunkers = ["recursive", "fixed", "sentence", "semantic"]
    models = ["qwen3:0.6b", "qwen3:4b", "qwen3:8b"]
    chunker_params = {
        "recursive": (500, 100),
        "fixed": (500, 0),
        "sentence": (0, 0),   # sentence chunker doesn't use these
        "semantic": (0, 0),   # semantic chunker doesn't use these
    }
    rows = []

    for chunker in chunkers:
        cs, co = chunker_params[chunker]
        for model in models:
            for i in range(n_per_config):
                quality = random.uniform(0.2, 0.9)
                rows.append({
                    "chunker": chunker,
                    "model": model,
                    "question": f"Test question {i}?",
                    "gold_answer": f"Gold answer {i}",
                    "rag_answer": f"RAG answer for {chunker} {model} {i}",
                    "gold_f1": random.uniform(0.1, 0.8),
                    "gold_exact_match": random.choice([True, False]),
                    "gold_bertscore": random.uniform(0.3, 0.9),
                    "faithfulness": quality + random.uniform(-0.1, 0.1),
                    "relevance": quality + random.uniform(-0.1, 0.1),
                    "conciseness": quality + random.uniform(-0.1, 0.1),
                    "quality": quality,
                    "strategy_latency_ms": random.uniform(500, 5000),
                    "scorer_latency_ms": random.uniform(100, 500),
                    "total_latency_ms": random.uniform(600, 5500),
                    "chunk_type": chunker,
                    "chunk_size": cs,
                    "chunk_overlap": co,
                    "num_chunks": random.randint(3, 15),
                    "embed_provider": "ollama",
                    "embed_model": "mxbai-embed-large",
                    "embed_dimension": 1024,
                    "retrieval_mode": "hybrid",
                    "retrieval_top_k": 5,
                    "num_chunks_retrieved": 5,
                    "context_char_length": random.randint(500, 3000),
                    "reranker_model": None,
                    "reranker_top_k": None,
                    "llm_provider": "ollama",
                    "llm_host": "local",
                    "llm_model": model,
                    "dataset_name": "hotpotqa",
                    "dataset_sample_seed": 42,
                })
    return pd.DataFrame(rows)


@pytest.fixture
def exp2_df() -> pd.DataFrame:
    return _make_exp2_df()


# ---------------------------------------------------------------------------
# Import tests
# ---------------------------------------------------------------------------

class TestImports:
    def test_module_importable(self):
        mod = importlib.import_module("generate_experiment2_dashboard")
        assert mod is not None

    def test_build_figures_exists(self):
        mod = importlib.import_module("generate_experiment2_dashboard")
        assert hasattr(mod, "build_experiment2_figures")
        assert callable(mod.build_experiment2_figures)

    def test_generate_dashboard_exists(self):
        mod = importlib.import_module("generate_experiment2_dashboard")
        assert hasattr(mod, "generate_dashboard")
        assert callable(mod.generate_dashboard)


# ---------------------------------------------------------------------------
# Figure generation tests
# ---------------------------------------------------------------------------

class TestBuildFigures:
    def test_returns_list_of_tuples(self, exp2_df, tmp_path):
        csv_path = tmp_path / "raw_scores.csv"
        exp2_df.to_csv(csv_path, index=False)

        mod = importlib.import_module("generate_experiment2_dashboard")
        figures = mod.build_experiment2_figures(csv_path)

        assert isinstance(figures, list)
        assert len(figures) > 0
        for item in figures:
            assert isinstance(item, tuple)
            assert len(item) == 2
            title, fig = item
            assert isinstance(title, str)

    def test_minimum_chart_count(self, exp2_df, tmp_path):
        """Must produce at least 8 charts."""
        csv_path = tmp_path / "raw_scores.csv"
        exp2_df.to_csv(csv_path, index=False)

        mod = importlib.import_module("generate_experiment2_dashboard")
        figures = mod.build_experiment2_figures(csv_path)
        assert len(figures) >= 8

    def test_quality_heatmap_present(self, exp2_df, tmp_path):
        csv_path = tmp_path / "raw_scores.csv"
        exp2_df.to_csv(csv_path, index=False)

        mod = importlib.import_module("generate_experiment2_dashboard")
        figures = mod.build_experiment2_figures(csv_path)
        titles = [t.lower() for t, _ in figures]
        assert any("quality" in t and "heatmap" in t for t in titles), \
            f"No quality heatmap found in: {titles}"

    def test_handles_missing_gold_columns(self, exp2_df, tmp_path):
        df = exp2_df.drop(columns=["gold_f1", "gold_exact_match", "gold_bertscore"])
        csv_path = tmp_path / "raw_scores.csv"
        df.to_csv(csv_path, index=False)

        mod = importlib.import_module("generate_experiment2_dashboard")
        figures = mod.build_experiment2_figures(csv_path)
        assert len(figures) > 0

    def test_handles_single_config(self, tmp_path):
        """Should work with only 1 config's data."""
        df = _make_exp2_df(n_per_config=3)
        df = df[(df["chunker"] == "recursive") & (df["model"] == "qwen3:4b")]
        csv_path = tmp_path / "raw_scores.csv"
        df.to_csv(csv_path, index=False)

        mod = importlib.import_module("generate_experiment2_dashboard")
        figures = mod.build_experiment2_figures(csv_path)
        assert len(figures) > 0

    def test_chunk_count_analysis_present(self, exp2_df, tmp_path):
        """Exp 2 should include a chunk count analysis chart."""
        csv_path = tmp_path / "raw_scores.csv"
        exp2_df.to_csv(csv_path, index=False)

        mod = importlib.import_module("generate_experiment2_dashboard")
        figures = mod.build_experiment2_figures(csv_path)
        titles = [t.lower() for t, _ in figures]
        assert any("chunk" in t for t in titles), \
            f"No chunk-related chart found in: {titles}"


# ---------------------------------------------------------------------------
# Dashboard generation tests
# ---------------------------------------------------------------------------

class TestGenerateDashboard:
    def test_generates_html_file(self, exp2_df, tmp_path):
        csv_path = tmp_path / "raw_scores.csv"
        exp2_df.to_csv(csv_path, index=False)
        output_path = tmp_path / "dashboard.html"

        mod = importlib.import_module("generate_experiment2_dashboard")
        mod.generate_dashboard(csv_path, output_path)

        assert output_path.exists()
        html = output_path.read_text(encoding="utf-8")
        assert "<html" in html.lower()
        assert "plotly" in html.lower()

    def test_empty_csv_produces_minimal_output(self, tmp_path):
        csv_path = tmp_path / "raw_scores.csv"
        pd.DataFrame().to_csv(csv_path, index=False)
        output_path = tmp_path / "dashboard.html"

        mod = importlib.import_module("generate_experiment2_dashboard")
        mod.generate_dashboard(csv_path, output_path)

        assert output_path.exists()
        html = output_path.read_text(encoding="utf-8")
        assert "<html" in html.lower()
